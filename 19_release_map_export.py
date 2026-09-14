"""
19_release_map_export.py

Purpose:
--------
Export release-point data (release_pos_x, release_pos_z) for the
dashboard's release point map -- the visualization home for
18_release_consistency_component.py now that it's out of
paom_score. Same interpretable design as the Location Map: one
labeled mean point per pitch type with a spread ellipse, not a
raw-scatter wall of individual pitches.

No handedness split (unlike the Location Map) -- release mechanics
have no legitimate reason to vary by batter side, matching
18_release_consistency_component.py's own design rationale.

Output:
-------
PAOM_release_map_points.csv   (sampled individual pitch releases --
                                optional/detail view)
PAOM_release_map_means.csv    (mean release position + std dev per
                                pitcher x pitch type -- the primary,
                                interpretable view)
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "clean_statcast_2025.csv"

POINTS_OUTPUT = "PAOM_release_map_points.csv"
MEANS_OUTPUT = "PAOM_release_map_means.csv"

MIN_PITCHES = 50            # same floor as 18_release_consistency_component.py
MAX_SAMPLE_PER_GROUP = 250   # cap points per pitcher x pitch type

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

print("Loading cleaned Statcast...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Loaded {len(df):,} pitches")

required_cols = [
    "player_name", "pitch_type", "release_pos_x", "release_pos_z"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required_cols).copy()

print(f"Rows with complete release data: {len(df):,}")


# ============================================================
# FILTER TO PITCHER x PITCH TYPE WITH ENOUGH VOLUME
# ============================================================

pitch_counts = (
    df.groupby(["player_name", "pitch_type"])
    .size()
    .reset_index(name="n")
)

qualifying = pitch_counts[pitch_counts["n"] >= MIN_PITCHES]

df = df.merge(
    qualifying[["player_name", "pitch_type"]],
    on=["player_name", "pitch_type"],
    how="inner"
)

print(
    f"Pitcher x pitch type groups meeting {MIN_PITCHES}-pitch "
    f"minimum: {len(qualifying):,}"
)


# ============================================================
# SAMPLED POINTS (for optional detail view)
# ============================================================

print("\nSampling points for plotting...")

np.random.seed(RANDOM_SEED)

sample_frames = []

for (pitcher, pitch), group in df.groupby(["player_name", "pitch_type"]):
    n = min(len(group), MAX_SAMPLE_PER_GROUP)
    sample_frames.append(group.sample(n=n, random_state=RANDOM_SEED))

sampled = pd.concat(sample_frames, ignore_index=True)

points_output = sampled[
    ["player_name", "pitch_type", "release_pos_x", "release_pos_z"]
].copy()

points_output.to_csv(POINTS_OUTPUT, index=False)

print(f"Saved {len(points_output):,} sampled release points")


# ============================================================
# MEAN RELEASE POSITION (primary, interpretable view)
# ============================================================

means = (
    df
    .groupby(["player_name", "pitch_type"])
    .agg(
        pitches=("release_pos_x", "count"),
        mean_release_x=("release_pos_x", "mean"),
        mean_release_z=("release_pos_z", "mean"),
        std_release_x=("release_pos_x", "std"),
        std_release_z=("release_pos_z", "std")
    )
    .reset_index()
)

# std is undefined for a single-pitch group -- fill with 0 so the
# dashboard's spread ellipse degenerates to a point rather than NaN
means[["std_release_x", "std_release_z"]] = (
    means[["std_release_x", "std_release_z"]].fillna(0)
)

means.to_csv(MEANS_OUTPUT, index=False)

print(f"Saved {len(means):,} mean-release rows")


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("Release Map Export Saved")
print("==============================")

print("\nSaved:")
print(f"- {POINTS_OUTPUT}")
print(f"- {MEANS_OUTPUT}")

print("\nSample means (first pitcher)")
print("-------------------------------")
sample_player = means["player_name"].iloc[0]
print(means[means["player_name"] == sample_player])
