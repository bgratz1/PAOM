"""
25_quadrant_coverage_component.py

Purpose:
--------
Measure how many of the four movement-space quadrants (defined by
the sign of HB and IVB -- the same zero-crosshair reference already
shown on the dashboard's Movement Map) a pitcher's arsenal actually
touches. A genuinely different KIND of measure than anything
currently in Movement: discrete/binary-per-region rather than
continuous area or distance.

Why this might matter beyond what's already captured:
----------------------------------------------------------
horizontal_coverage, vertical_coverage, and hull_area can all be
driven up by two pitches that are far apart but sit in the SAME
general direction (e.g. two arm-side, rising pitches at different
intensities) -- a lot of "spread" without much true directional
variety. Quadrant coverage asks a coarser but different question:
does this arsenal actually present looks in multiple DIRECTIONS,
not just multiple DISTANCES.

Real limitation, stated upfront: this is a coarse, low-cardinality
measure (only 4 possible values). It will have much less
fine-grained differentiation than the continuous features already
in Movement -- worth knowing before deciding whether it earns a
place, not just after.

Method:
-------
1. Same MIN_PITCHES / usage-floor filtering as Movement, for a
   directly comparable population.
2. Quadrant = (sign of HB, sign of IVB) for each qualifying pitch
   type. quadrant_coverage_count = number of DISTINCT quadrants
   touched (1-4).
3. Percentile rank (handles the discreteness fine via average-rank
   tiebreaking, same as everywhere else).
4. Confidence based on n_pitch_types (same convention as Movement).

This is a CANDIDATE component, not yet added to Movement's own
feature set or to paom_score -- same "build it, validate it, then
decide" pattern used for every other component tested this way
(Velocity, fastball_relative_break, Release Consistency).

Output:
-------
PAOM_quadrant_coverage_component.csv
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "master_pitch_table_2025.csv"
OUTPUT_FILE = "PAOM_quadrant_coverage_component.csv"

MIN_PITCHES = 50               # same floor as Movement
MIN_PITCH_TYPE_USAGE = 0.05     # same usage floor as Movement

CONFIDENCE_PITCH_TYPES = 4      # same saturation point as Movement


# ============================================================
# LOAD DATA
# ============================================================

print("Loading master pitch table...")

df = pd.read_csv(INPUT_FILE)

print(f"Initial rows: {len(df):,}")

required_cols = ["player_name", "pitch_type", "pitches", "usage", "HB", "IVB"]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# FILTER (SAME CONVENTION AS MOVEMENT)
# ============================================================

df = df[df["pitches"] >= MIN_PITCHES].copy()

print(f"After {MIN_PITCHES} pitch filter: {len(df):,}")

geo_df = df[df["usage"] >= MIN_PITCH_TYPE_USAGE].copy()

print(f"After {MIN_PITCH_TYPE_USAGE:.0%} usage floor: {len(geo_df):,}")


# ============================================================
# PER-PITCHER QUADRANT COVERAGE
# ============================================================

print("\nComputing quadrant coverage...")

records = []

for pitcher, group in geo_df.groupby("player_name"):
    hb = group["HB"].values
    ivb = group["IVB"].values
    n_pitch_types = len(group)

    # quadrant = (sign of HB, sign of IVB), using >=0 as the
    # positive-side boundary convention (HB/IVB are continuous
    # physical measurements -- landing exactly on 0.000 is
    # vanishingly rare in practice, so the boundary choice barely
    # matters, just needs to be applied consistently)
    hb_sign = hb >= 0
    ivb_sign = ivb >= 0

    quadrants = set(zip(hb_sign, ivb_sign))
    quadrant_coverage_count = len(quadrants)

    records.append({
        "player_name": pitcher,
        "n_pitch_types": n_pitch_types,
        "quadrant_coverage_count": quadrant_coverage_count
    })

quadrant_df = pd.DataFrame(records)

print(f"Pitchers with quadrant coverage: {len(quadrant_df):,}")

print("\nQuadrant coverage count distribution:")
print(quadrant_df["quadrant_coverage_count"].value_counts().sort_index())


# ============================================================
# SCALE 0-100 (PERCENTILE RANK)
# ============================================================

def percentile_rank(series):
    return series.rank(pct=True) * 100

quadrant_df["quadrant_coverage_score"] = percentile_rank(
    quadrant_df["quadrant_coverage_count"]
)


# ============================================================
# CONFIDENCE ADJUSTMENT
# ============================================================

quadrant_df["confidence_score"] = (
    np.minimum(
        quadrant_df["n_pitch_types"] / CONFIDENCE_PITCH_TYPES,
        1
    )
    * 100
)

league_average = quadrant_df["quadrant_coverage_score"].mean()

quadrant_df["trusted_quadrant_coverage_score"] = (
    quadrant_df["quadrant_coverage_score"] * (quadrant_df["confidence_score"] / 100)
    + league_average * (1 - quadrant_df["confidence_score"] / 100)
)


# ============================================================
# SAVE OUTPUT
# ============================================================

final_columns = [
    "player_name",
    "n_pitch_types",
    "quadrant_coverage_count",
    "quadrant_coverage_score",
    "confidence_score",
    "trusted_quadrant_coverage_score"
]

output = (
    quadrant_df[final_columns]
    .sort_values("trusted_quadrant_coverage_score", ascending=False)
)

output.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# VALIDATION CHECKS
# ============================================================

print("\n==============================")
print("VALIDATION")
print("==============================")

corr_n_pitch_types = output["trusted_quadrant_coverage_score"].corr(
    output["n_pitch_types"]
)
print(f"\nCorrelation with n_pitch_types: {corr_n_pitch_types:.3f}")

if corr_n_pitch_types > 0.5:
    print(
        "\nHigh correlation with pitch count -- plausible given a "
        "pitcher needs enough distinct pitches to even POSSIBLY "
        "touch multiple quadrants; worth keeping in mind if this "
        "gets tested further."
    )

movement_file = "PAOM_movement_component.csv"
try:
    movement_df = pd.read_csv(movement_file)
    merge_cols = [
        "player_name", "horizontal_coverage", "vertical_coverage",
        "hull_area", "fastball_relative_break", "avg_nn_distance",
        "trusted_movement_score"
    ]
    merge_cols = [c for c in merge_cols if c in movement_df.columns]

    merged = output.merge(movement_df[merge_cols], on="player_name", how="inner")

    print("\nCorrelation with each existing Movement feature:")
    for col in merge_cols[1:-1]:
        corr = merged["trusted_quadrant_coverage_score"].corr(merged[col])
        print(f"  {col}: {corr:.3f}")

    corr_movement = merged["trusted_quadrant_coverage_score"].corr(
        merged["trusted_movement_score"]
    )
    print(f"\nCorrelation with overall trusted_movement_score: {corr_movement:.3f}")

except FileNotFoundError:
    print(f"\n({movement_file} not found -- skipping Movement comparison)")


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("Quadrant Coverage Component Saved")
print("==============================")

print(
    f"""
Pitchers scored:
{len(output):,}

Average quadrant coverage score:
{output['trusted_quadrant_coverage_score'].mean():.2f}

Average confidence:
{output['confidence_score'].mean():.2f}
"""
)

print("\nTop 20 quadrant coverage scores")
print("----------------------------------")
print(
    output[
        [
            "player_name", "n_pitch_types", "quadrant_coverage_count",
            "trusted_quadrant_coverage_score"
        ]
    ]
    .head(20)
)

print("\nSaved:")
print(f"- {OUTPUT_FILE}")
