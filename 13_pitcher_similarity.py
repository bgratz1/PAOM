"""
13_pitcher_similarity.py

Purpose:
--------
Build a physical-trait similarity engine across pitchers, serving
two downstream uses:

1. "Add a pitch" recommendations -- find pitchers with similar
   release mechanics (arm slot, extension, arm strength) who throw
   a pitch this pitcher doesn't. Similar mechanics is what makes a
   donor pitch plausible for a given pitcher to actually throw.

2. The dashboard's similar-arsenals map -- a 2D projection of the
   same physical feature space, so pitchers can be plotted and
   "who's near me" is directly visual.

Features used:
--------------
release_x, release_z, extension, and a usage-weighted avg_velo --
this describes HOW the ball gets released and how hard, independent
of WHAT pitches are actually thrown. That's deliberate: similarity
should be about mechanics, not repertoire, since two pitchers with
identical release mechanics but different repertoires are exactly
the "could plausibly add this" comparison the recommendation engine
needs. Movement/pitch-type features are intentionally excluded here
-- that's what Movement and the recommendation engine's donor-pitch
matching are for.

Method:
-------
1. Usage-weighted pitcher-level rollup of release_x, release_z,
   extension, and velo from pitch_summary.csv.
2. Standardize the four features.
3. Fit NearestNeighbors SEPARATELY within each throwing hand
   (p_throws) -- comps are restricted to same-handed pitchers only,
   since a candidate pitch to "borrow" only translates practically
   from a same-handed release (mirror-image comps don't share real
   grip/shape logic). This is applied explicitly here rather than
   relying on the fact that raw release_x already implicitly
   separates hands -- explicit filtering is more robust (an
   over-the-top pitcher with release_x near zero is exactly the
   edge case where the implicit separation could misfire).
4. Fit PCA(n_components=2) on the FULL population (both hands
   together) for the 2D map -- the map is for visual exploration,
   where seeing the natural handedness clustering is itself useful,
   so it isn't restricted the way the recommendation comps are.
   p_throws is included in the map output for optional color-coding.

Output:
-------
PAOM_pitcher_similarity.csv   (long format: pitcher x comp x rank,
                                same-handed only)
PAOM_pitcher_map_coordinates.csv   (one row per pitcher, x/y +
                                     features + p_throws, both hands)
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

INPUT_FILE = "pitch_summary.csv"
MASTER_FILE = "master_pitch_table_2025.csv"   # source of p_throws

SIMILARITY_OUTPUT = "PAOM_pitcher_similarity.csv"
MAP_OUTPUT = "PAOM_pitcher_map_coordinates.csv"

MIN_TOTAL_PITCHES = 200   # pitcher-level floor for stable release stats
N_NEIGHBORS = 10           # top-K comps saved per pitcher


# ============================================================
# LOAD DATA
# ============================================================

print("Loading pitch summary...")

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
        f"dropping them, since they can't be assigned to a "
        f"same-handed comp pool."
    )
    profile = profile.dropna(subset=["p_throws"]).reset_index(drop=True)

print(profile["p_throws"].value_counts())


# ============================================================
# STANDARDIZE FEATURES
# ============================================================

features = ["release_x", "release_z", "extension", "avg_velo"]

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
                continue  # skip self-match
            comp_records.append({
                "player_name": player,
                "comp_player_name": hand_names[j],
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
        "guide, not a precise similarity measure (the nearest-"
        "neighbor table above uses the full 4D feature space and "
        "is more precise)."
    )

profile["map_x"] = coords[:, 0]
profile["map_y"] = coords[:, 1]

map_output = profile[
    [
        "player_name", "p_throws", "total_pitches",
        "release_x", "release_z", "extension", "avg_velo",
        "map_x", "map_y"
    ]
]

map_output.to_csv(MAP_OUTPUT, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("Pitcher Similarity Engine Saved")
print("==============================")

print(f"\nPitchers profiled: {len(profile):,}")

print("\nSample comps (first pitcher in the list)")
print("------------------------------------------")
sample_player = profile["player_name"].iloc[0]
print(
    similarity_output[
        similarity_output["player_name"] == sample_player
    ]
)

print("\nSaved:")
print(f"- {SIMILARITY_OUTPUT}")
print(f"- {MAP_OUTPUT}")
