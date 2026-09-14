"""
87_pitcher_similarity_by_year.py

Purpose:
--------
The REAL 13_pitcher_similarity.py, parameterized by YEAR -- all
rollup/standardization/nearest-neighbor/PCA logic UNCHANGED from
the original.

SAME ASSUMPTION as 86 -- pitch_summary.csv substituted with master_
pitch_table_{YEAR}.csv (from 77). The original ALSO separately reads
master_pitch_table_2025.csv just for p_throws -- since both inputs
resolve to the same year-parameterized file here, that second read
is redundant but harmless (kept as a separate variable for clarity
and to stay close to the original's structure).

Output:
-------
PAOM_pitcher_similarity_{YEAR}.csv
PAOM_pitcher_map_coordinates_{YEAR}.csv
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

import sys
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

INPUT_FILE = f"master_pitch_table_{YEAR}.csv"  # substitute for
                                                  # pitch_summary.csv
MASTER_FILE = f"master_pitch_table_{YEAR}.csv"   # source of p_throws

SIMILARITY_OUTPUT = f"PAOM_pitcher_similarity_{YEAR}.csv"
MAP_OUTPUT = f"PAOM_pitcher_map_coordinates_{YEAR}.csv"

MIN_TOTAL_PITCHES = 200
N_NEIGHBORS = 10


# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading pitch data for {YEAR}...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} pitcher x pitch type rows")

required_cols = [
    "player_name", "pitches", "usage",
    "velo", "release_x", "release_z", "extension"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# USAGE-WEIGHTED PITCHER-LEVEL ROLLUP
# ============================================================

print("\nRolling up to pitcher-level physical profile...")

def weighted_rollup(x):
    w = x["usage"]
    return pd.Series({
        "release_x": np.average(x["release_x"], weights=w),
        "release_z": np.average(x["release_z"], weights=w),
        "extension": np.average(x["extension"], weights=w),
        "avg_velo": np.average(x["velo"], weights=w),
        "total_pitches": x["pitches"].sum()
    })

profile = (
    df
    .groupby("player_name")
    .apply(weighted_rollup)
    .reset_index()
)

profile = profile[profile["total_pitches"] >= MIN_TOTAL_PITCHES].copy()

print(f"Pitchers meeting {MIN_TOTAL_PITCHES}-pitch minimum: {len(profile):,}")


# ============================================================
# MERGE IN THROWING HAND
# ============================================================

print("\nLoading throwing hand (p_throws)...")

master = pd.read_csv(MASTER_FILE)

hand_lookup = (
    master
    .groupby("player_name")["p_throws"]
    .first()
    .reset_index()
)

profile = profile.merge(hand_lookup, on="player_name", how="left")

missing_hand = profile["p_throws"].isna().sum()

if missing_hand > 0:
    print(
        f"WARNING: {missing_hand} pitcher(s) missing p_throws -- "
        f"dropping them."
    )
    profile = profile.dropna(subset=["p_throws"]).reset_index(drop=True)

print(profile["p_throws"].value_counts())


# ============================================================
# STANDARDIZE FEATURES
# ============================================================

features = ["release_x", "release_z", "extension", "avg_velo"]

# NaN check -- a real data-quality gap this year's data can surface
# that the original script's single 2025 test run never had to
# handle: a pitcher's rolled-up profile can come back NaN in one of
# these features if a pitch type had zero valid readings for it in
# the underlying Statcast data (more likely in an atypical season
# like 2020's shortened schedule). NearestNeighbors can't accept
# NaN input at all, so these rows are dropped here, with the count
# disclosed rather than silently lost.
n_before_nan_check = len(profile)
profile = profile.dropna(subset=features).reset_index(drop=True)
n_dropped_nan = n_before_nan_check - len(profile)

if n_dropped_nan > 0:
    print(
        f"\nNOTE: dropped {n_dropped_nan} pitcher(s) with NaN in at "
        f"least one required feature ({features}) -- likely a pitch "
        f"type with zero valid readings for that feature in the "
        f"underlying data. {len(profile):,} pitchers remain."
    )

scaler = StandardScaler()
X_scaled = scaler.fit_transform(profile[features])


# ============================================================
# NEAREST NEIGHBORS (PHYSICAL COMPS, SAME HAND ONLY)
# ============================================================

print("\nFitting nearest-neighbor comps (within each throwing hand)...")

comp_records = []

for hand, hand_group in profile.groupby("p_throws"):

    hand_indices = hand_group.index.values
    X_hand = X_scaled[hand_indices]
    hand_names = hand_group["player_name"].values

    k = min(N_NEIGHBORS + 1, len(hand_group))

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_hand)

    distances, indices = nn.kneighbors(X_hand)

    for i, player in enumerate(hand_names):
        rank = 1
        for j, dist in zip(indices[i], distances[i]):
            if hand_names[j] == player:
                continue
            comp_records.append({
                "player_name": player,
                "comp_player_name": hand_names[j],
                "season": YEAR,
                "p_throws": hand,
                "rank": rank,
                "physical_distance": dist
            })
            rank += 1
            if rank > N_NEIGHBORS:
                break

similarity_output = pd.DataFrame(comp_records)

similarity_output.to_csv(SIMILARITY_OUTPUT, index=False)

print(f"Saved {len(similarity_output):,} pitcher-comp pairs")


# ============================================================
# 2D MAP PROJECTION (PCA)
# ============================================================

print("\nFitting 2D map projection...")

pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)

print(
    f"\nExplained variance: PC1 = {pca.explained_variance_ratio_[0]:.1%}, "
    f"PC2 = {pca.explained_variance_ratio_[1]:.1%}, "
    f"total = {pca.explained_variance_ratio_.sum():.1%}"
)

if pca.explained_variance_ratio_.sum() < 0.6:
    print(
        "\nNOTE: the 2D map captures less than 60% of total physical "
        "variance -- treat proximity on the map as a rough visual "
        "guide, not a precise similarity measure."
    )

profile["map_x"] = coords[:, 0]
profile["map_y"] = coords[:, 1]
profile["season"] = YEAR

map_output = profile[
    [
        "player_name", "season", "p_throws", "total_pitches",
        "release_x", "release_z", "extension", "avg_velo",
        "map_x", "map_y"
    ]
]

map_output.to_csv(MAP_OUTPUT, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print(f"Pitcher Similarity Engine Saved ({YEAR})")
print("==============================")

print(f"\nPitchers profiled: {len(profile):,}")

print("\nSaved:")
print(f"- {SIMILARITY_OUTPUT}")
print(f"- {MAP_OUTPUT}")
