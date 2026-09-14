"""
78_velocity_by_year.py

Purpose:
--------
The REAL 21_velocity_component.py, parameterized by year -- replaces
71/72_velocity_component_*.py entirely, which were built before the
real script was available and had several real gaps (min-max instead
of percentile-rank scaling, wrong confidence saturation point, no
confidence shrinkage at all, missing the PCA sign-conflict fallback).
All fixed here by matching the real, confirmed methodology exactly.

Requires master_pitch_table_{YEAR}.csv (from 77).
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS -- UNCHANGED from the original
# ============================================================

import sys
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
INPUT_FILE = f"master_pitch_table_{YEAR}.csv"
OUTPUT_FILE = f"PAOM_velocity_component_{YEAR}.csv"

MIN_PITCHES = 50
MIN_PITCH_TYPE_USAGE = 0.05

CONFIDENCE_PITCH_TYPES = 4

APPLY_CONFIDENCE_SHRINKAGE = True  # confirmed necessary in the real
                                     # script -- disabling it let
                                     # pitch-count correlation climb
                                     # to 0.481 on real data


# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading master pitch table for {YEAR}...")

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
print(f"After {MIN_PITCH_TYPE_USAGE:.0%} usage floor (used for velocity range): {len(geo_df):,}")


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


# ============================================================
# COMBINE VELOCITY RANGE + MAX VELO (PCA) -- UNCHANGED
# ============================================================

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
    print("\nWARNING: PC1 explains less than 40% of variance.")

velocity_df["raw_velocity_index"] = components[:, 0]

range_corr = velocity_df["raw_velocity_index"].corr(velocity_df["velocity_range"])
if range_corr < 0:
    velocity_df["raw_velocity_index"] *= -1

max_velo_corr_after_flip = velocity_df["raw_velocity_index"].corr(velocity_df["max_velo"])

print(f"\nPost-flip correlation with velocity_range: {abs(range_corr):.3f}")
print(f"Post-flip correlation with max_velo: {max_velo_corr_after_flip:.3f}")

if max_velo_corr_after_flip < 0:
    print(
        "\nafter sign-correcting for velocity_range, the index "
        "correlates NEGATIVELY with max_velo -- falling back to a "
        "simple standardized average of the two z-scores instead."
    )

    range_z = (velocity_df["velocity_range"] - velocity_df["velocity_range"].mean()) / velocity_df["velocity_range"].std()
    max_velo_z = (velocity_df["max_velo"] - velocity_df["max_velo"].mean()) / velocity_df["max_velo"].std()

    velocity_df["raw_velocity_index"] = (range_z + max_velo_z) / 2

    print(f"Fallback average correlation with velocity_range: {velocity_df['raw_velocity_index'].corr(velocity_df['velocity_range']):.3f}")
    print(f"Fallback average correlation with max_velo: {velocity_df['raw_velocity_index'].corr(velocity_df['max_velo']):.3f}")


# ============================================================
# SCALE 0-100 (PERCENTILE RANK) -- FIXED from 71/72's min-max
# ============================================================

def percentile_rank(series):
    return series.rank(pct=True) * 100

velocity_df["velocity_score"] = percentile_rank(velocity_df["raw_velocity_index"])


# ============================================================
# CONFIDENCE ADJUSTMENT -- FIXED (real saturation point + shrinkage)
# ============================================================

velocity_df["confidence_score"] = (
    np.minimum(velocity_df["n_pitch_types"] / CONFIDENCE_PITCH_TYPES, 1) * 100
)

league_average = velocity_df["velocity_score"].mean()

if APPLY_CONFIDENCE_SHRINKAGE:
    velocity_df["trusted_velocity_score"] = (
        velocity_df["velocity_score"] * (velocity_df["confidence_score"] / 100)
        + league_average * (1 - velocity_df["confidence_score"] / 100)
    )
else:
    velocity_df["trusted_velocity_score"] = velocity_df["velocity_score"]


# ============================================================
# SAVE OUTPUT
# ============================================================

velocity_df["season"] = YEAR

final_columns = [
    "player_name", "season", "n_pitch_types", "velocity_range", "max_velo",
    "min_velo", "raw_velocity_index", "velocity_score",
    "confidence_score", "trusted_velocity_score"
]

output = velocity_df[final_columns].sort_values("trusted_velocity_score", ascending=False)

output.to_csv(OUTPUT_FILE, index=False)

print(f"\n==============================")
print(f"Velocity Component Saved ({YEAR})")
print("==============================")
print(f"\nPitchers scored: {len(output):,}")
print(f"Average velocity score: {output['trusted_velocity_score'].mean():.2f}")
print(f"Average confidence: {output['confidence_score'].mean():.2f}")
print(f"\nCorrelation with number of pitch types: {output['trusted_velocity_score'].corr(output['n_pitch_types']):.3f}")
print(f"\nSaved: {OUTPUT_FILE}")

loadings = pd.DataFrame({
    "feature": pca_features,
    "PC1_loading": pca.components_[0]
})
if range_corr < 0:
    loadings["PC1_loading"] *= -1
print("\nPCA loadings")
print(loadings)