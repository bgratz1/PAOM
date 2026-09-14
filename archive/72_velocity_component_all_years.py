"""
72_velocity_component_all_years.py

Purpose:
--------
Extends 71_velocity_component_2024.py to all six seasons (2020-2025),
matching the full range of data this thread's own Kaggle-based work
already covers -- the actual goal, per direct confirmation, is a
full historical PAOM rebuild, not a single-year sample.

Same reconstruction, same honest caveat as 71: the original 21_
velocity_component.py isn't present in this environment, only its
summary description survived ("PCA of velocity_range+max_velo;
n_pitch_types-based confidence"). This is a careful reconstruction
from that description, not a verified byte-for-byte replica.

Uses the SAME MEANINGFUL_USAGE_FLOOR (5%) convention as everywhere
else in this project for "qualifying" pitch types.

Output:
-------
velocity_component_all_years.csv -- one row per (player_id, season),
covering 2020-2025.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

ARSENAL_FILE = "pitcher_arsenal_evolution_2020_2025.csv"
OUTPUT_FILE = "velocity_component_all_years.csv"

SEASONS = list(range(2020, 2026))
MEANINGFUL_USAGE_FLOOR = 5.0

PITCH_TYPES = ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS"]


# ============================================================
# LOAD
# ============================================================

print(f"Loading arsenal data for seasons {SEASONS}...")

arsenal = pd.read_csv(ARSENAL_FILE)
arsenal_all = arsenal[arsenal["season"].isin(SEASONS)].copy()

print(f"{len(arsenal_all):,} pitcher-seasons across all years")
print(arsenal_all["season"].value_counts().sort_index())


# ============================================================
# COMPUTE max_velo AND velocity_range PER PITCHER-SEASON
# ============================================================

print("\nComputing max_velo and velocity_range from qualifying pitch types...")

records = []
for _, row in arsenal_all.iterrows():
    qualifying_speeds = []
    for pt in PITCH_TYPES:
        usage = row.get(f"{pt}_usage_pct", np.nan)
        speed = row.get(f"{pt}_avg_speed", np.nan)
        if pd.notna(usage) and usage >= MEANINGFUL_USAGE_FLOOR and pd.notna(speed):
            qualifying_speeds.append(speed)

    if len(qualifying_speeds) == 0:
        continue

    max_velo = max(qualifying_speeds)
    velocity_range = max_velo - min(qualifying_speeds)
    n_pitch_types = len(qualifying_speeds)

    records.append({
        "player_id": row["player_id"],
        "season": row["season"],
        "max_velo": max_velo,
        "velocity_range": velocity_range,
        "n_pitch_types": n_pitch_types,
    })

result_df = pd.DataFrame(records)
print(f"\n{len(result_df):,} pitcher-seasons with at least one qualifying pitch type")


# ============================================================
# PCA OF [velocity_range, max_velo], SCALED TO 0-100
#
# Fit ONCE across ALL years pooled together, not separately per
# season -- keeps the 0-100 scale directly comparable across years,
# which matters for a later blend/backtest that spans multiple
# seasons. A pitcher's velocity_score in 2021 and 2024 should mean
# the same thing on the same scale.
# ============================================================

print("\nRunning PCA on standardized [velocity_range, max_velo], pooled across all years...")

features = result_df[["velocity_range", "max_velo"]].values
feature_mean = features.mean(axis=0)
feature_std = features.std(axis=0)
feature_std[feature_std == 0] = 1.0
features_scaled = (features - feature_mean) / feature_std

pca = PCA(n_components=1)
pc1 = pca.fit_transform(features_scaled).flatten()

if np.corrcoef(pc1, result_df["max_velo"])[0, 1] < 0:
    pc1 = -pc1

pc1_min, pc1_max = pc1.min(), pc1.max()
velocity_score = 100 * (pc1 - pc1_min) / (pc1_max - pc1_min)

result_df["velocity_score"] = velocity_score
print(f"Explained variance ratio (PC1): {pca.explained_variance_ratio_[0]:.3f}")


# ============================================================
# CONFIDENCE
# ============================================================

MAX_EXPECTED_PITCH_TYPES = 5
result_df["velocity_confidence"] = (result_df["n_pitch_types"] / MAX_EXPECTED_PITCH_TYPES).clip(upper=1.0) * 100


# ============================================================
# SAVE
# ============================================================

result_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n==============================")
print(f"Saved: {OUTPUT_FILE}")
print(f"==============================")
print(f"\n{len(result_df):,} pitcher-seasons scored")
print(f"\nBy season:")
print(result_df.groupby("season").size())
print(f"\nvelocity_score distribution:")
print(result_df["velocity_score"].describe())

print(f"\nSample rows:")
print(result_df.sample(min(10, len(result_df)), random_state=1).to_string(index=False))
