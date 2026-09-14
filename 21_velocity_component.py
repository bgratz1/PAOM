"""
21_velocity_component.py

Purpose:
--------
Measure a pitcher's velocity-based assets -- extracted from
09_movement_component.py's PCA, where velocity_range previously
lived as one of six blended features.

Why this was pulled out, and why it's now two features:
------------------------------------------------------------
velocity_range measures something conceptually different from the
rest of Movement's features (horizontal/vertical coverage, hull
area, centroid dispersion, nearest-neighbor distance) -- those all
describe the SHAPE an arsenal occupies in break-movement space; a
wide velocity spread is about timing deception (making a hitter
defend multiple speeds), not spatial coverage.

A single-feature version (velocity_range alone) was tested first
and rejected: it correlated 0.59 with Movement (over the redundancy
threshold) and showed unstable, sign-flipping marginal weight
across both validation targets -- the same failure pattern
Tunneling showed after four attempts.

This version adds max_velo (raw arm strength -- how hard the
single fastest pitch is) as a second, genuinely distinct feature.
Two pitchers can have identical velocity_range while sitting at
very different absolute bands (88-93 vs. 96-101 mph) -- max_velo
has no obvious mechanical reason to correlate with Movement's
shape-based features the way velocity_range alone did. Being
re-tested specifically to see if this reduces the redundancy while
preserving or improving standalone signal.

Method:
-------
1. velocity_range = max velo - min velo, max_velo = the single
   fastest qualifying pitch type's average velo, both across a
   pitcher's qualifying pitch types (same MIN_PITCHES floor as
   Movement, for a directly comparable population).
2. Combine both (standardized) via PCA into one index, sign-checked
   against BOTH features (not just one), scaled 0-100 via
   percentile rank.
3. Confidence adjustment based on number of qualifying pitch types
   (same convention as Movement).

This is a CANDIDATE component, not yet added to paom_score --
matches the same "build it, validate it against real outcomes,
then decide" pattern used for every other component in this
pipeline (see 10_paom_score.py once this is tested).

Output:
-------
PAOM_velocity_component.csv
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "master_pitch_table_2025.csv"
OUTPUT_FILE = "PAOM_velocity_component.csv"

MIN_PITCHES = 50               # same floor as Movement, for a
                                 # directly comparable population
MIN_PITCH_TYPE_USAGE = 0.05     # same usage floor as Movement

CONFIDENCE_PITCH_TYPES = 4      # same saturation point as Movement

APPLY_CONFIDENCE_SHRINKAGE = True  # REVERTED. The theoretical case
                                     # for disabling this (max_velo and
                                     # velocity_range aren't degenerate
                                     # with few points the way
                                     # Movement's shape/spread features
                                     # are) held up in principle, but
                                     # NOT in the real data: pitch-
                                     # count correlation came back at
                                     # 0.481 -- right at the 0.5
                                     # threshold this project treats as
                                     # a real confound everywhere else,
                                     # more than 3x worse than
                                     # Movement's own disabled-
                                     # shrinkage result (0.14). The
                                     # leaderboard still looked
                                     # reasonable, which turned out to
                                     # be a genuine trap -- a
                                     # systematic bias in WHO tends to
                                     # rank highly doesn't necessarily
                                     # produce obviously-wrong names
                                     # the way Movement's did, so the
                                     # eye test alone wasn't sufficient
                                     # here.


# ============================================================
# LOAD DATA
# ============================================================

print("Loading master pitch table...")

df = pd.read_csv(INPUT_FILE)

print(f"Initial rows: {len(df):,}")

required_cols = ["player_name", "pitch_type", "pitches", "usage", "velo"]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# FILTER (SAME CONVENTION AS MOVEMENT)
# ============================================================

df = df[df["pitches"] >= MIN_PITCHES].copy()

print(f"After {MIN_PITCHES} pitch filter: {len(df):,}")

geo_df = df[df["usage"] >= MIN_PITCH_TYPE_USAGE].copy()

print(
    f"After {MIN_PITCH_TYPE_USAGE:.0%} usage floor "
    f"(used for velocity range): {len(geo_df):,}"
)


# ============================================================
# PER-PITCHER VELOCITY RANGE
# ============================================================

print("\nComputing velocity range...")

records = []

for pitcher, group in geo_df.groupby("player_name"):
    velo = group["velo"].values
    n_pitch_types = len(group)

    velocity_range = velo.max() - velo.min()

    records.append({
        "player_name": pitcher,
        "n_pitch_types": n_pitch_types,
        "velocity_range": velocity_range,
        "max_velo": velo.max(),
        "min_velo": velo.min()
    })

velocity_df = pd.DataFrame(records)

print(f"Pitchers with velocity range: {len(velocity_df):,}")

single_type = (velocity_df["n_pitch_types"] == 1).sum()
print(
    f"Single-pitch-type pitchers (velocity_range trivially 0 for "
    f"these): {single_type:,}"
)


# ============================================================
# COMBINE VELOCITY RANGE + MAX VELO (PCA)
# ============================================================

"""
Two genuinely distinct velocity signals, not one:
- velocity_range: how much the arsenal spreads across speed
  (timing deception -- forcing a hitter to defend multiple speeds)
- max_velo: raw arm strength (how hard the single fastest pitch
  is), independent of how much variety surrounds it -- two
  pitchers can have identical velocity_range while sitting at very
  different absolute velocity bands (88-93 vs. 96-101).

Combined via PCA (same pattern as every other multi-feature PAOM
component) rather than picking one arbitrarily. Unlike the earlier
single-feature version's correlation with Movement (0.59, driven by
velocity_range alone tracking arsenal diversity), max_velo has no
obvious mechanical reason to correlate with Movement's shape-based
features (coverage, hull area, dispersion) -- a hard max-velo
pitcher can have either a compact or a diverse arsenal shape. This
version is being tested specifically to see whether adding max_velo
reduces that redundancy while preserving or improving standalone
signal.
"""

pca_features = ["velocity_range", "max_velo"]

X = velocity_df[pca_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

print("\nExplained variance ratio")
print(pca.explained_variance_ratio_)

pc1_explained = pca.explained_variance_ratio_[0]

print(f"PC1 explains {pc1_explained:.1%} of variance")

if pc1_explained < 0.40:
    print(
        "\nWARNING: PC1 explains less than 40% of variance -- "
        "velocity_range and max_velo may be measuring genuinely "
        "independent things rather than a single 'velocity' axis. "
        "Inspect before trusting PC1 alone."
    )

velocity_df["raw_velocity_index"] = components[:, 0]

# sign-correct against BOTH features -- higher velocity_range and
# higher max_velo should both mean a HIGHER score; check alignment
# with both, not just one, since PCA doesn't guarantee direction
range_corr = velocity_df["raw_velocity_index"].corr(velocity_df["velocity_range"])

if range_corr < 0:
    velocity_df["raw_velocity_index"] *= -1

max_velo_corr_after_flip = velocity_df["raw_velocity_index"].corr(velocity_df["max_velo"])

print(
    f"\nPost-flip correlation with velocity_range: "
    f"{abs(range_corr):.3f} (should be positive)"
)
print(
    f"Post-flip correlation with max_velo: "
    f"{max_velo_corr_after_flip:.3f} (should also be positive)"
)

if max_velo_corr_after_flip < 0:
    print(
        "\nafter sign-correcting for velocity_range, the index "
        "correlates NEGATIVELY with max_velo -- PC1 can't align "
        "positively with both when they're this anti-correlated "
        "with each other (a structural property of PCA, not a "
        "bug). Falling back to a simple standardized average of "
        "the two z-scores instead, so both features get genuine "
        "positive credit rather than PCA picking one dominant "
        "direction."
    )

    range_z = (
        (velocity_df["velocity_range"] - velocity_df["velocity_range"].mean())
        / velocity_df["velocity_range"].std()
    )
    max_velo_z = (
        (velocity_df["max_velo"] - velocity_df["max_velo"].mean())
        / velocity_df["max_velo"].std()
    )

    velocity_df["raw_velocity_index"] = (range_z + max_velo_z) / 2

    print(
        f"Fallback average correlation with velocity_range: "
        f"{velocity_df['raw_velocity_index'].corr(velocity_df['velocity_range']):.3f}"
    )
    print(
        f"Fallback average correlation with max_velo: "
        f"{velocity_df['raw_velocity_index'].corr(velocity_df['max_velo']):.3f}"
    )


# ============================================================
# SCALE 0-100 (PERCENTILE RANK)
# ============================================================

def percentile_rank(series):
    return series.rank(pct=True) * 100

velocity_df["velocity_score"] = percentile_rank(velocity_df["raw_velocity_index"])


# ============================================================
# CONFIDENCE ADJUSTMENT
# ============================================================

velocity_df["confidence_score"] = (
    np.minimum(
        velocity_df["n_pitch_types"] / CONFIDENCE_PITCH_TYPES,
        1
    )
    * 100
)

league_average = velocity_df["velocity_score"].mean()

if APPLY_CONFIDENCE_SHRINKAGE:
    velocity_df["trusted_velocity_score"] = (
        velocity_df["velocity_score"] * (velocity_df["confidence_score"] / 100)
        + league_average * (1 - velocity_df["confidence_score"] / 100)
    )
else:
    velocity_df["trusted_velocity_score"] = velocity_df["velocity_score"]

print(
    f"\nConfidence shrinkage {'ENABLED' if APPLY_CONFIDENCE_SHRINKAGE else 'DISABLED (experiment)'} "
    f"for trusted_velocity_score."
)


# ============================================================
# SAVE OUTPUT
# ============================================================

final_columns = [
    "player_name",
    "n_pitch_types",
    "velocity_range",
    "max_velo",
    "min_velo",
    "raw_velocity_index",
    "velocity_score",
    "confidence_score",
    "trusted_velocity_score"
]

output = (
    velocity_df[final_columns]
    .sort_values("trusted_velocity_score", ascending=False)
)

output.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("Velocity Component Saved")
print("==============================")

print(
    f"""
Pitchers scored:
{len(output):,}

Average velocity score:
{output['trusted_velocity_score'].mean():.2f}

Average confidence:
{output['confidence_score'].mean():.2f}
"""
)

print("\nTop 20 velocity scores")
print("-------------------------")
print(
    output[
        [
            "player_name", "n_pitch_types", "velocity_range",
            "max_velo", "min_velo", "trusted_velocity_score"
        ]
    ]
    .head(20)
)

print("\n==============================")
print("VELOCITY VALIDATION")
print("==============================")

print(
    f"""
Correlation with number of pitch types:
{output['trusted_velocity_score'].corr(output['n_pitch_types'])}
"""
)

print("\nSaved:")
print(f"- {OUTPUT_FILE}")
