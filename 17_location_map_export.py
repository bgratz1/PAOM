"""
17_location_map_export.py

Purpose:
--------
Export plottable pitch LOCATION data (plate_x, plate_z -- where a
pitch actually crosses the plate) for the dashboard's strike-zone
visualization. This is a separate visual from the movement map --
movement space (HB/IVB, pitch shape) and location space (plate_x/
plate_z, where it ends up) are different axes with different
units, so they're exported and plotted separately rather than
combined onto one chart.

This exports BOTH a sampled set of individual pitch locations
(for anyone who wants to overlay raw detail) AND mean location +
spread (std dev) per pitcher x pitch type x matchup_type -- the
means file, with its std_plate_x/std_plate_z, is the primary data
for an interpretable "Savant-style" view: one labeled point per
pitch type, optionally with a spread ellipse, rather than hundreds
of overlapping raw dots. Handedness splits (same/opposite, matching
batter stand vs the pitcher's own p_throws) are included so the map
can show, e.g., "this pitcher backdoors the slider to same-handed
batters and backfoots it to opposite-handed batters" as two
distinct average locations rather than one misleadingly blended
point -- consistent with why Command was built with handedness
splits in the first place.

Output:
-------
PAOM_location_map_points.csv   (sampled individual pitch locations
                                 -- optional/detail view)
PAOM_location_map_means.csv    (mean location + std dev per pitcher
                                 x pitch type x matchup_type -- the
                                 primary, interpretable view)
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "clean_statcast_2025.csv"

POINTS_OUTPUT = "PAOM_location_map_points.csv"
MEANS_OUTPUT = "PAOM_location_map_means.csv"

MIN_PITCHES = 50           # same floor as Movement/Command
MAX_SAMPLE_PER_GROUP = 250  # cap points per pitcher x pitch type x
                             # matchup_type, to keep the file a
                             # reasonable size and the scatter
                             # readable rather than an overwhelming
                             # blob for high-usage pitches

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

print("Loading cleaned Statcast...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Loaded {len(df):,} pitches")

required_cols = [
    "player_name", "pitch_type", "p_throws", "stand",
    "plate_x", "plate_z"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required_cols).copy()

df["matchup_type"] = np.where(
    df["stand"] == df["p_throws"], "same", "opposite"
)


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
# SAMPLED POINTS (for the scatter itself)
# ============================================================

print("\nSampling points for plotting...")

np.random.seed(RANDOM_SEED)

sample_frames = []

for (pitcher, pitch, matchup), group in df.groupby(
    ["player_name", "pitch_type", "matchup_type"]
):
    n = min(len(group), MAX_SAMPLE_PER_GROUP)
    sample_frames.append(
        group.sample(n=n, random_state=RANDOM_SEED)
    )

sampled = pd.concat(sample_frames, ignore_index=True)

points_output = sampled[
    ["player_name", "pitch_type", "matchup_type", "plate_x", "plate_z"]
].copy()

points_output.to_csv(POINTS_OUTPUT, index=False)

print(f"Saved {len(points_output):,} sampled location points")


# ============================================================
# MEAN LOCATIONS (for a "typical location" marker)
# ============================================================

means = (
    df
    .groupby(["player_name", "pitch_type", "matchup_type"])
    .agg(
        pitches=("plate_x", "count"),
        mean_plate_x=("plate_x", "mean"),
        mean_plate_z=("plate_z", "mean"),
        std_plate_x=("plate_x", "std"),
        std_plate_z=("plate_z", "std")
    )
    .reset_index()
)

# std is undefined for a single-pitch group -- fill with 0 so the
# dashboard's spread ellipse degenerates to a point rather than NaN
means[["std_plate_x", "std_plate_z"]] = (
    means[["std_plate_x", "std_plate_z"]].fillna(0)
)

means.to_csv(MEANS_OUTPUT, index=False)

print(f"Saved {len(means):,} mean-location rows")


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("Location Map Export Saved")
print("==============================")

print("\nSaved:")
print(f"- {POINTS_OUTPUT}")
print(f"- {MEANS_OUTPUT}")

print("\nSample points (first pitcher, first pitch type)")
print("---------------------------------------------------")
sample_player = points_output["player_name"].iloc[0]
sample_pitch = points_output[
    points_output["player_name"] == sample_player
]["pitch_type"].iloc[0]
print(
    points_output[
        (points_output["player_name"] == sample_player)
        & (points_output["pitch_type"] == sample_pitch)
    ]
    .head(10)
)
