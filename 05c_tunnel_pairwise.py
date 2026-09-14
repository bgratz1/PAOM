"""
05c_tunnel_pairwise.py

Purpose:
--------
Compute the actual tunnel differential for every pitch pair, using
trajectory-based decision-point positions from 03b_trajectory_summary.py
instead of release-point proxies.

Concept:
--------
A true tunnel pair looks nearly identical to the hitter at the
decision point, then diverges by the time it crosses the plate.

    decision_distance = separation at the decision point (want LOW)
    plate_distance     = separation at the plate (want HIGH)

    tunnel_differential = plate_distance / decision_distance

A high tunnel_differential means: these two pitches looked similar
early and ended up in very different places -- exactly what
"tunneling" means. A pair that's already far apart at the decision
point (hitter can tell them apart early) gets a low differential
even if they also end up far apart at the plate -- that's not
disguise, that's just two different pitches.

Output:
-------
pitch_tunnel_pairs.csv
"""

import pandas as pd
import numpy as np
from itertools import combinations


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "pitch_trajectory_summary.csv"
OUTPUT_FILE = "pitch_tunnel_pairs.csv"

# small constant to avoid division by ~0 when two pitches have
# nearly identical decision points (rare, but guards the ratio)
EPSILON = 0.05


# ============================================================
# LOAD DATA
# ============================================================

print("Loading trajectory summary...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} pitcher x pitch type rows")

required_cols = [
    "player_name",
    "pitch_type",
    "decision_x",
    "decision_z",
    "plate_x",
    "plate_z"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# BUILD PAIRS
# ============================================================

print("\nBuilding tunnel pairs...")

pairs = []

for pitcher, group in df.groupby("player_name"):

    pitch_list = group.to_dict("records")

    if len(pitch_list) < 2:
        continue

    for p1, p2 in combinations(pitch_list, 2):

        decision_distance = np.sqrt(
            (p1["decision_x"] - p2["decision_x"]) ** 2
            + (p1["decision_z"] - p2["decision_z"]) ** 2
        )

        plate_distance = np.sqrt(
            (p1["plate_x"] - p2["plate_x"]) ** 2
            + (p1["plate_z"] - p2["plate_z"]) ** 2
        )

        tunnel_differential = plate_distance / (
            decision_distance + EPSILON
        )

        pairs.append({
            "player_name": pitcher,
            "pitch_1": p1["pitch_type"],
            "pitch_2": p2["pitch_type"],
            "decision_distance": decision_distance,
            "plate_distance": plate_distance,
            "tunnel_differential": tunnel_differential
        })

tunnel_pairs = pd.DataFrame(pairs)

tunnel_pairs = tunnel_pairs.round(4)

print(f"Total tunnel pairs: {len(tunnel_pairs):,}")

print("\nTunnel differential summary:")
print(tunnel_pairs["tunnel_differential"].describe())


# ============================================================
# SAVE
# ============================================================

tunnel_pairs.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved: {OUTPUT_FILE}")

print("\nTop 20 tunnel pairs (highest differential)")
print(
    tunnel_pairs
    .sort_values("tunnel_differential", ascending=False)
    .head(20)
)
