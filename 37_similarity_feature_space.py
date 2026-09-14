"""
37_similarity_feature_space.py

Purpose:
--------
Builds the unified per-pitcher-season feature vector the similarity-
weighted regression will actually run distances on -- combining
Movement-style geometry, Velocity-style features, and release-point
features into one standardized space, across all 2020-2025 seasons.

WHY NOT JUST RE-RUN 09_movement_component.py / 21_velocity_
component.py AGAINST 6 YEARS OF DATA: those scripts' full pipelines
(dampening, sequential orthogonalization, PCA, confidence-shrinkage)
were built and validated against ONE season's raw Statcast pull.
Re-running that entire pipeline 6 separate times would need 6 more
full-season raw pulls and 6 separate re-fits of dampening/PCA -- a
much heavier undertaking than this thread's other scripts.

INSTEAD: pitcher_arsenal_evolution_2020_2025.csv (the Kaggle file)
already has the raw per-pitch-type-per-season ingredients (usage_pct,
avg_speed, avg_pfx_x, avg_pfx_z) sitting right there. This applies
the SAME validated geometric concepts from Movement/Velocity directly
to that data -- hull area, nearest-neighbor distance, fastball-
relative break (ported faithfully from 09_movement_component.py,
including its usage-based FF/SI anchor selection), velocity range,
and max velocity -- rather than re-deriving Movement's exact fitted
scores from scratch.

REAL LIMITATION, stated plainly: the Kaggle file has usage_pct but
not raw pitch COUNTS, so there's no equivalent of Movement's own
MIN_PITCHES=50 reliability floor here -- only the same 5% usage
floor (MEANINGFUL_USAGE_FLOOR, matching 34_arsenal_change_events.py
and 09_movement_component.py's own MIN_PITCH_TYPE_USAGE) determines
which pitch types count toward a pitcher-season's geometry. A
pitcher-season built from a very small raw sample that still clears
5% usage share won't be flagged as less reliable here the way it
would be in Movement's own confidence-shrunk score.

UNITS CAVEAT (carried over from this dataset's first inspection):
avg_pfx_x/avg_pfx_z appear to be in raw Statcast feet-based units,
not necessarily the same inches-based convention Movement's HB/IVB
use. This matters less here than it would for direct comparison to
Movement's own scores, since every feature below gets standardized
(z-scored) before use -- a similarity/distance metric computed
entirely within this one consistently-sourced dataset doesn't
depend on matching an external unit convention, only on internal
consistency, which this has.

Output:
-------
pitcher_similarity_features_2020_2025.csv -- one row per (player_id,
season), with both raw and standardized (z-scored) versions of every
feature. The standardized columns are what a similarity/distance
computation should actually use.
"""

import pandas as pd
import numpy as np

from scipy.spatial import ConvexHull, QhullError

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

ARSENAL_FILE = "pitcher_arsenal_evolution_2020_2025.csv"
RELEASE_FILE = "pitcher_release_features_combined_2020_2025.csv"

OUTPUT_FILE = "pitcher_similarity_features_2020_2025.csv"

MEANINGFUL_USAGE_FLOOR = 5.0  # SAME threshold as 34_arsenal_change_
                                # events.py and 09_movement_
                                # component.py's MIN_PITCH_TYPE_USAGE
                                # -- reused deliberately for
                                # consistency across this project

PITCH_TYPES = ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS"]

ANCHOR_PRIORITY = ["FF", "SI"]  # SAME as 09_movement_component.py


# ============================================================
# HULL AREA (ported faithfully from 09_movement_component.py)
# ============================================================

def hull_area(points):
    """
    points: array of shape (n, 2) -> (pfx_x, pfx_z)

    Returns (area, method) where method is 'hull' (>=3 non-collinear
    points) or 'bbox' (fallback for 2-pitch arsenals or degenerate/
    collinear cases).
    """
    n = len(points)

    if n < 3:
        x_range = points[:, 0].max() - points[:, 0].min()
        z_range = points[:, 1].max() - points[:, 1].min()
        return x_range * z_range, "bbox"

    try:
        hull = ConvexHull(points)
        return hull.volume, "hull"
    except QhullError:
        x_range = points[:, 0].max() - points[:, 0].min()
        z_range = points[:, 1].max() - points[:, 1].min()
        return x_range * z_range, "bbox"


# ============================================================
# LOAD
# ============================================================

print("Loading arsenal evolution data...")

df = pd.read_csv(ARSENAL_FILE)

print(f"Loaded {len(df):,} pitcher-season rows")


# ============================================================
# PER-PITCHER-SEASON GEOMETRY
# ============================================================

print("\nComputing similarity features per pitcher-season...")

records = []

for _, row in df.iterrows():
    player_id = row["player_id"]
    season = row["season"]
    player_name = row["player_name"]

    # which pitch types clear the usage floor this season
    qualifying_types = []
    for pt in PITCH_TYPES:
        usage = row.get(f"{pt}_usage_pct", np.nan)
        if pd.notna(usage) and usage >= MEANINGFUL_USAGE_FLOOR:
            qualifying_types.append(pt)

    n_pitch_types = len(qualifying_types)

    if n_pitch_types == 0:
        continue  # no qualifying pitch types at all this season -- skip

    pfx_x = np.array([row[f"{pt}_avg_pfx_x"] for pt in qualifying_types])
    pfx_z = np.array([row[f"{pt}_avg_pfx_z"] for pt in qualifying_types])
    speed = np.array([row[f"{pt}_avg_speed"] for pt in qualifying_types])
    usage = np.array([row[f"{pt}_usage_pct"] for pt in qualifying_types])

    # drop this pitcher-season if any required value is missing for
    # a qualifying pitch type (usage cleared the floor but the
    # underlying movement/speed data itself is incomplete)
    if np.isnan(pfx_x).any() or np.isnan(pfx_z).any() or np.isnan(speed).any():
        continue

    velocity_range = speed.max() - speed.min()
    max_velo = speed.max()

    points = np.column_stack([pfx_x, pfx_z])
    area, hull_method = hull_area(points)

    # nearest-neighbor distance -- identical logic to
    # 09_movement_component.py
    if n_pitch_types >= 2:
        nn_distances = []
        for i in range(n_pitch_types):
            other_idx = [j for j in range(n_pitch_types) if j != i]
            d = np.sqrt(
                (pfx_x[i] - pfx_x[other_idx]) ** 2
                + (pfx_z[i] - pfx_z[other_idx]) ** 2
            )
            nn_distances.append(d.min())
        avg_nn_distance = np.mean(nn_distances)
    else:
        avg_nn_distance = 0.0

    # fastball-relative break -- identical anchor-selection logic to
    # 09_movement_component.py (usage-based FF/SI selection when
    # both present)
    available_anchors = [pt for pt in ANCHOR_PRIORITY if pt in qualifying_types]

    if len(available_anchors) == 0:
        anchor_type = None
    elif len(available_anchors) == 1:
        anchor_type = available_anchors[0]
    else:
        ff_usage = usage[qualifying_types.index("FF")]
        si_usage = usage[qualifying_types.index("SI")]
        anchor_type = "FF" if ff_usage >= si_usage else "SI"

    no_anchor_available = anchor_type is None

    if not no_anchor_available:
        anchor_idx = qualifying_types.index(anchor_type)
        anchor_x, anchor_z = pfx_x[anchor_idx], pfx_z[anchor_idx]

        other_idx = [i for i in range(n_pitch_types) if i != anchor_idx]

        if len(other_idx) > 0:
            fb_distances = np.sqrt(
                (pfx_x[other_idx] - anchor_x) ** 2
                + (pfx_z[other_idx] - anchor_z) ** 2
            )
            fastball_relative_break = fb_distances.mean()
        else:
            fastball_relative_break = 0.0
    else:
        fastball_relative_break = np.nan  # filled with population average below

    records.append({
        "player_id": player_id,
        "season": season,
        "player_name": player_name,
        "n_pitch_types": n_pitch_types,
        "hull_area": area,
        "hull_method": hull_method,
        "avg_nn_distance": avg_nn_distance,
        "fastball_relative_break": fastball_relative_break,
        "anchor_type": anchor_type,
        "no_anchor_available": no_anchor_available,
        "velocity_range": velocity_range,
        "max_velo": max_velo,
    })

geo_df = pd.DataFrame(records)

print(f"Computed geometry for {len(geo_df):,} pitcher-seasons")

n_no_anchor = geo_df["no_anchor_available"].sum()
if n_no_anchor > 0:
    league_avg_fb_break = geo_df.loc[
        ~geo_df["no_anchor_available"], "fastball_relative_break"
    ].mean()
    geo_df.loc[
        geo_df["no_anchor_available"], "fastball_relative_break"
    ] = league_avg_fb_break
    print(
        f"{n_no_anchor:,} pitcher-seasons had no qualifying FF/SI -- "
        f"filled fastball_relative_break with the league average "
        f"({league_avg_fb_break:.3f}), same convention as "
        f"09_movement_component.py"
    )


# ============================================================
# MERGE RELEASE FEATURES
# ============================================================

print("\nMerging release-point features...")

release = pd.read_csv(RELEASE_FILE)
release_cols = [c for c in ["player_id", "season", "arm_angle", "release_x", "release_z"] if c in release.columns]
release = release[release_cols]

geo_df = geo_df.merge(release, on=["player_id", "season"], how="left")

n_with_release = geo_df["release_x"].notna().sum() if "release_x" in geo_df.columns else 0
print(f"{n_with_release:,} of {len(geo_df):,} pitcher-seasons matched to release features")

n_missing_arm_angle = geo_df["arm_angle"].isna().sum() if "arm_angle" in geo_df.columns else len(geo_df)
print(
    f"{n_missing_arm_angle:,} pitcher-seasons have no arm_angle "
    f"(covered only by the smaller leaderboard, not the raw-"
    f"Statcast fallback) -- release_x/release_z should still be "
    f"present for nearly all of them"
)


# ============================================================
# STANDARDIZE
# ============================================================

print("\nStandardizing features (z-score)...")

FEATURE_COLS = [
    "hull_area", "avg_nn_distance", "fastball_relative_break",
    "velocity_range", "max_velo", "release_x", "release_z"
]
# arm_angle deliberately excluded from the default set -- present
# for only a subset of pitcher-seasons; include it in a similarity
# computation only when comparing pitchers who both have it

for col in FEATURE_COLS:
    if col not in geo_df.columns:
        continue
    mean = geo_df[col].mean()
    std = geo_df[col].std()
    geo_df[f"{col}_z"] = (geo_df[col] - mean) / std if std > 0 else 0.0

if "arm_angle" in geo_df.columns:
    mean = geo_df["arm_angle"].mean()
    std = geo_df["arm_angle"].std()
    geo_df["arm_angle_z"] = (geo_df["arm_angle"] - mean) / std if std > 0 else 0.0


# ============================================================
# SAVE
# ============================================================

geo_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n==============================")
print(f"Saved: {OUTPUT_FILE}")
print(f"==============================")
print(f"\nRows: {len(geo_df):,}")
print(f"Unique pitchers: {geo_df['player_id'].nunique():,}")
print(f"Seasons: {sorted(geo_df['season'].unique())}")
print(f"\nColumns: {geo_df.columns.tolist()}")

print(f"\nSample rows (standardized features):")
sample_cols = ["player_name", "season", "n_pitch_types"] + [f"{c}_z" for c in FEATURE_COLS if f"{c}_z" in geo_df.columns]
print(geo_df[sample_cols].head(10).to_string(index=False))
