"""
15_movement_map_export.py

Purpose:
--------
Export the actual plottable data behind each pitcher's Movement
score -- per-pitch-type points in (HB, IVB) space, plus the convex
hull boundary -- so the dashboard can render a real movement map,
not just show the summary score.

Uses the SAME filtering as 09_movement_component.py (50-pitch
minimum, 5% usage floor for geometry) so what's drawn on the map
matches exactly what the movement_score was computed from -- a
pitch type shown on the map that wasn't counted in the score would
be confusing.

Output:
-------
PAOM_movement_map_points.csv   (long format: one row per pitcher x
                                 pitch type -- HB, IVB, usage, velo)
PAOM_movement_map_hulls.csv    (long format: one row per pitcher x
                                 hull vertex, in boundary order --
                                 empty/absent for 2-pitch pitchers
                                 using the bounding-box fallback)
"""

import pandas as pd
import numpy as np

from scipy.spatial import ConvexHull, QhullError

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "master_pitch_table_2025.csv"

POINTS_OUTPUT = "PAOM_movement_map_points.csv"
HULLS_OUTPUT = "PAOM_movement_map_hulls.csv"

MIN_PITCHES = 50
MIN_PITCH_TYPE_USAGE = 0.05


# ============================================================
# LOAD DATA
# ============================================================

print("Loading master pitch table...")

df = pd.read_csv(INPUT_FILE)

print(f"Initial rows: {len(df):,}")

required_cols = [
    "player_name", "pitch_type", "pitches", "usage",
    "HB", "IVB", "velo"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# FILTER (SAME AS 09_movement_component.py)
# ============================================================

df = df[df["pitches"] >= MIN_PITCHES].copy()

print(f"After {MIN_PITCHES} pitch filter: {len(df):,}")

geo_df = df[df["usage"] >= MIN_PITCH_TYPE_USAGE].copy()

print(
    f"After {MIN_PITCH_TYPE_USAGE:.0%} usage floor "
    f"(pitch types shown on the map): {len(geo_df):,}"
)


# ============================================================
# POINTS OUTPUT -- ONE ROW PER PITCHER x PITCH TYPE
# ============================================================

points_output = geo_df[
    ["player_name", "pitch_type", "HB", "IVB", "usage", "velo", "pitches"]
].copy()

points_output.to_csv(POINTS_OUTPUT, index=False)

print(f"\nSaved {len(points_output):,} plottable points")


# ============================================================
# HULL BOUNDARIES
# ============================================================

print("\nComputing hull boundaries...")

hull_records = []

n_hull = 0
n_bbox = 0

for pitcher, group in geo_df.groupby("player_name"):

    hb = group["HB"].values
    ivb = group["IVB"].values
    n_pitch_types = len(group)

    if n_pitch_types < 3:
        # bounding-box fallback -- draw a rectangle instead of a hull
        n_bbox += 1
        hb_min, hb_max = hb.min(), hb.max()
        ivb_min, ivb_max = ivb.min(), ivb.max()

        corners = [
            (hb_min, ivb_min),
            (hb_max, ivb_min),
            (hb_max, ivb_max),
            (hb_min, ivb_max)
        ]

        for order, (x, y) in enumerate(corners):
            hull_records.append({
                "player_name": pitcher,
                "boundary_type": "bbox",
                "vertex_order": order,
                "HB": x,
                "IVB": y
            })
        continue

    points = np.column_stack([hb, ivb])

    try:
        hull = ConvexHull(points)
        n_hull += 1
        for order, vertex_idx in enumerate(hull.vertices):
            hull_records.append({
                "player_name": pitcher,
                "boundary_type": "hull",
                "vertex_order": order,
                "HB": points[vertex_idx, 0],
                "IVB": points[vertex_idx, 1]
            })
    except QhullError:
        # degenerate/collinear points -- same bounding-box fallback
        n_bbox += 1
        hb_min, hb_max = hb.min(), hb.max()
        ivb_min, ivb_max = ivb.min(), ivb.max()

        corners = [
            (hb_min, ivb_min),
            (hb_max, ivb_min),
            (hb_max, ivb_max),
            (hb_min, ivb_max)
        ]

        for order, (x, y) in enumerate(corners):
            hull_records.append({
                "player_name": pitcher,
                "boundary_type": "bbox",
                "vertex_order": order,
                "HB": x,
                "IVB": y
            })

hull_output = pd.DataFrame(hull_records)

hull_output.to_csv(HULLS_OUTPUT, index=False)

print(f"\nHull boundaries: {n_hull:,} pitchers")
print(f"Bounding-box fallback: {n_bbox:,} pitchers")
print(f"Saved {len(hull_output):,} boundary vertex rows")


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("Movement Map Export Saved")
print("==============================")

print("\nSaved:")
print(f"- {POINTS_OUTPUT}")
print(f"- {HULLS_OUTPUT}")

print("\nSample points (first pitcher)")
print("-------------------------------")
sample_player = points_output["player_name"].iloc[0]
print(points_output[points_output["player_name"] == sample_player])

print("\nSample hull (same pitcher)")
print("----------------------------")
print(hull_output[hull_output["player_name"] == sample_player])
