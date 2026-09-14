"""
24_fastball_relative_break_component.py

Purpose:
--------
Measure how differentiated a pitcher's secondary pitches are from
their OWN fastball -- a different lens than centroid_dispersion,
which measures spread from the arsenal's own center of mass.

Why this is a different signal, not a rehash:
------------------------------------------------
A hitter calibrates their read off the fastball -- it's the pitch
they're braced for by default. What matters for a breaking or
offspeed pitch isn't its movement in some abstract, arsenal-centered
sense; it's how much it deviates from what the hitter is
anticipating. Two pitchers can have identical centroid_dispersion
(same average spread from their own arsenal's center) while having
very different fastball-relative separation, depending on where the
fastball itself sits within that spread.

Method:
-------
1. Anchor pitch = 'FF' if the pitcher throws it (clearing the same
   MIN_PITCHES/usage floor as everything else); falls back to 'SI'
   if no qualifying four-seam; pitcher excluded if neither
   qualifies (no fastball-type anchor to measure against).
2. For every OTHER qualifying pitch type, Euclidean distance (in
   HB/IVB space) from that pitch to the anchor.
3. UNWEIGHTED average across those distances -- matches Movement's
   own established convention (structural capability, not
   usage-experienced; this is the same philosophy documented for
   centroid_dispersion, deliberately contrasted with Command's
   usage-weighted approach).
4. Percentile rank, confidence-shrunk based on how many non-anchor
   pitch types back the average.

This is a CANDIDATE component, not yet added to paom_score or to
Movement's own PCA -- same "build it, validate it against real
outcomes, then decide" pattern used for every other component
(Velocity went through this exact process). Worth checking its
correlation with Movement's existing centroid_dispersion in
particular, to see whether it's capturing something genuinely new
or substantially overlapping.

Output:
-------
PAOM_fastball_relative_break_component.csv
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "master_pitch_table_2025.csv"
OUTPUT_FILE = "PAOM_fastball_relative_break_component.csv"

MIN_PITCHES = 50               # same floor as Movement/Velocity
MIN_PITCH_TYPE_USAGE = 0.05     # same usage floor as Movement/Velocity

ANCHOR_PRIORITY = ["FF", "SI"]  # four-seam preferred, sinker as fallback

CONFIDENCE_NON_ANCHOR_TYPES = 3  # saturation point for confidence


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
# FILTER (SAME CONVENTION AS MOVEMENT/VELOCITY)
# ============================================================

df = df[df["pitches"] >= MIN_PITCHES].copy()

print(f"After {MIN_PITCHES} pitch filter: {len(df):,}")

geo_df = df[df["usage"] >= MIN_PITCH_TYPE_USAGE].copy()

print(f"After {MIN_PITCH_TYPE_USAGE:.0%} usage floor: {len(geo_df):,}")


# ============================================================
# PER-PITCHER FASTBALL-RELATIVE BREAK
# ============================================================

print("\nComputing fastball-relative break...")

records = []
no_anchor_count = 0
single_type_count = 0

for pitcher, group in geo_df.groupby("player_name"):
    pitch_types_available = set(group["pitch_type"])

    anchor_type = None
    for candidate in ANCHOR_PRIORITY:
        if candidate in pitch_types_available:
            anchor_type = candidate
            break

    if anchor_type is None:
        no_anchor_count += 1
        continue

    anchor_row = group[group["pitch_type"] == anchor_type].iloc[0]
    anchor_hb, anchor_ivb = anchor_row["HB"], anchor_row["IVB"]

    others = group[group["pitch_type"] != anchor_type]

    if len(others) == 0:
        single_type_count += 1
        continue

    distances = np.sqrt(
        (others["HB"] - anchor_hb) ** 2 + (others["IVB"] - anchor_ivb) ** 2
    )

    records.append({
        "player_name": pitcher,
        "anchor_type": anchor_type,
        "n_pitch_types": len(group),
        "n_non_anchor_types": len(others),
        "fastball_relative_break": distances.mean()
    })

fb_break_df = pd.DataFrame(records)

print(f"Pitchers with fastball-relative break: {len(fb_break_df):,}")
print(f"Excluded (no FF/SI anchor available): {no_anchor_count:,}")
print(f"Excluded (anchor was their only qualifying pitch type): {single_type_count:,}")

print("\nAnchor type breakdown:")
print(fb_break_df["anchor_type"].value_counts())


# ============================================================
# SCALE 0-100 (PERCENTILE RANK)
# ============================================================

def percentile_rank(series):
    return series.rank(pct=True) * 100

fb_break_df["fastball_relative_break_score"] = percentile_rank(
    fb_break_df["fastball_relative_break"]
)


# ============================================================
# CONFIDENCE ADJUSTMENT
# ============================================================

fb_break_df["confidence_score"] = (
    np.minimum(
        fb_break_df["n_non_anchor_types"] / CONFIDENCE_NON_ANCHOR_TYPES,
        1
    )
    * 100
)

league_average = fb_break_df["fastball_relative_break_score"].mean()

fb_break_df["trusted_fastball_relative_break_score"] = (
    fb_break_df["fastball_relative_break_score"] * (fb_break_df["confidence_score"] / 100)
    + league_average * (1 - fb_break_df["confidence_score"] / 100)
)


# ============================================================
# SAVE OUTPUT
# ============================================================

final_columns = [
    "player_name",
    "anchor_type",
    "n_pitch_types",
    "n_non_anchor_types",
    "fastball_relative_break",
    "fastball_relative_break_score",
    "confidence_score",
    "trusted_fastball_relative_break_score"
]

output = (
    fb_break_df[final_columns]
    .sort_values("trusted_fastball_relative_break_score", ascending=False)
)

output.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# VALIDATION CHECKS
# ============================================================

print("\n==============================")
print("VALIDATION")
print("==============================")

corr_n_pitch_types = output["trusted_fastball_relative_break_score"].corr(
    output["n_pitch_types"]
)
print(f"\nCorrelation with n_pitch_types: {corr_n_pitch_types:.3f}")

movement_file = "PAOM_movement_component.csv"
try:
    movement_df = pd.read_csv(movement_file)
    merged = output.merge(
        movement_df[["player_name", "centroid_dispersion", "trusted_movement_score"]],
        on="player_name", how="inner"
    )
    corr_centroid = merged["trusted_fastball_relative_break_score"].corr(
        merged["centroid_dispersion"]
    )
    corr_movement = merged["trusted_fastball_relative_break_score"].corr(
        merged["trusted_movement_score"]
    )
    print(
        f"Correlation with Movement's centroid_dispersion "
        f"(the most similar existing feature): {corr_centroid:.3f}"
    )
    print(f"Correlation with overall trusted_movement_score: {corr_movement:.3f}")

    if abs(corr_centroid) > 0.7:
        print(
            "\nHigh correlation with centroid_dispersion -- this may "
            "be substantially re-measuring the same thing from a "
            "different anchor point, not a genuinely new signal."
        )
    else:
        print(
            "\nModerate/low correlation with centroid_dispersion -- "
            "consistent with capturing something distinct from "
            "existing Movement features."
        )
except FileNotFoundError:
    print(f"\n({movement_file} not found -- skipping Movement comparison)")


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("Fastball-Relative Break Component Saved")
print("==============================")

print(
    f"""
Pitchers scored:
{len(output):,}

Average fastball-relative break score:
{output['trusted_fastball_relative_break_score'].mean():.2f}

Average confidence:
{output['confidence_score'].mean():.2f}
"""
)

print("\nTop 20 fastball-relative break scores")
print("----------------------------------------")
print(
    output[
        [
            "player_name", "anchor_type", "n_non_anchor_types",
            "fastball_relative_break", "trusted_fastball_relative_break_score"
        ]
    ]
    .head(20)
)

print("\nSaved:")
print(f"- {OUTPUT_FILE}")
