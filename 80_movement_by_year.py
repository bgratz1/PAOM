"""
80_movement_by_year.py

Purpose:
--------
The REAL 09_movement_component.py, parameterized by YEAR -- all
dampening/PCA/confidence logic UNCHANGED from the original, only
file names and the validation-check merge target now vary by year.

Requires master_pitch_table_{YEAR}.csv (from 77).
"""

import pandas as pd
import numpy as np

from scipy.spatial import ConvexHull, QhullError

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

import sys
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

INPUT_FILE = f"master_pitch_table_{YEAR}.csv"

OUTPUT_FILE = f"PAOM_movement_component_{YEAR}.csv"

MIN_PITCHES = 50
MIN_PITCH_TYPE_USAGE = 0.05

CONFIDENCE_PITCH_TYPES = 4


# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading master pitch table for {YEAR}...")

df = pd.read_csv(INPUT_FILE)

print(f"Initial rows: {len(df):,}")


required_cols = [
    "player_name",
    "pitcher",
    "pitch_type",
    "p_throws",
    "pitches",
    "velo",
    "HB",
    "IVB",
    "usage"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# FILTER SAMPLE SIZE + USAGE FLOOR
# ============================================================

df = df[df["pitches"] >= MIN_PITCHES].copy()

print(f"After {MIN_PITCHES} pitch filter: {len(df):,}")

geo_df = df[df["usage"] >= MIN_PITCH_TYPE_USAGE].copy()

print(
    f"After {MIN_PITCH_TYPE_USAGE:.0%} usage floor "
    f"(used for geometry): {len(geo_df):,}"
)


# ============================================================
# CONVEX HULL AREA (with fallback for <3 points)
# ============================================================

def hull_area(points):
    n = len(points)

    if n < 3:
        hb_range = points[:, 0].max() - points[:, 0].min()
        ivb_range = points[:, 1].max() - points[:, 1].min()
        return hb_range * ivb_range, "bbox"

    try:
        hull = ConvexHull(points)
        return hull.volume, "hull"
    except QhullError:
        hb_range = points[:, 0].max() - points[:, 0].min()
        ivb_range = points[:, 1].max() - points[:, 1].min()
        return hb_range * ivb_range, "bbox"


# ============================================================
# PER-PITCHER GEOMETRY
# ============================================================

print("\nComputing arsenal geometry...")

records = []

ANCHOR_PRIORITY = ["FF", "SI"]

for pitcher, group in geo_df.groupby("player_name"):

    hb = group["HB"].values
    ivb = group["IVB"].values
    velo = group["velo"].values
    pitch_types = group["pitch_type"].values

    n_pitch_types = len(group)

    horizontal_coverage = hb.max() - hb.min()
    vertical_coverage = ivb.max() - ivb.min()
    velocity_range = velo.max() - velo.min()

    points = np.column_stack([hb, ivb])
    area, hull_method = hull_area(points)

    if n_pitch_types >= 2:
        nn_distances = []
        for i in range(n_pitch_types):
            other_idx = [j for j in range(n_pitch_types) if j != i]
            d = np.sqrt(
                (hb[i] - hb[other_idx]) ** 2
                + (ivb[i] - ivb[other_idx]) ** 2
            )
            nn_distances.append(d.min())
        avg_nn_distance = np.mean(nn_distances)
    else:
        avg_nn_distance = 0.0

    available_anchors = [pt for pt in ANCHOR_PRIORITY if pt in pitch_types]

    if len(available_anchors) == 0:
        anchor_type = None
    elif len(available_anchors) == 1:
        anchor_type = available_anchors[0]
    else:
        pitches_count = group["pitches"].values
        ff_pitches = pitches_count[pitch_types == "FF"][0]
        si_pitches = pitches_count[pitch_types == "SI"][0]
        anchor_type = "FF" if ff_pitches >= si_pitches else "SI"

    no_anchor_available = anchor_type is None

    if not no_anchor_available:
        anchor_idx = np.where(pitch_types == anchor_type)[0][0]
        anchor_hb, anchor_ivb = hb[anchor_idx], ivb[anchor_idx]

        other_idx = [i for i in range(n_pitch_types) if i != anchor_idx]

        if len(other_idx) > 0:
            fb_distances = np.sqrt(
                (hb[other_idx] - anchor_hb) ** 2
                + (ivb[other_idx] - anchor_ivb) ** 2
            )
            fastball_relative_break = fb_distances.mean()
        else:
            fastball_relative_break = 0.0
    else:
        fastball_relative_break = np.nan

    hb_sign = hb >= 0
    ivb_sign = ivb >= 0
    quadrant_coverage_count = len(set(zip(hb_sign, ivb_sign)))

    records.append({
        "player_name": pitcher,
        "n_pitch_types": n_pitch_types,
        "horizontal_coverage": horizontal_coverage,
        "vertical_coverage": vertical_coverage,
        "velocity_range": velocity_range,
        "hull_area": area,
        "hull_method": hull_method,
        "fastball_relative_break": fastball_relative_break,
        "anchor_type": anchor_type,
        "no_anchor_available": no_anchor_available,
        "avg_nn_distance": avg_nn_distance,
        "quadrant_coverage_count": quadrant_coverage_count
    })

movement_df = pd.DataFrame(records)

n_no_anchor = movement_df["no_anchor_available"].sum()

if n_no_anchor > 0:
    league_avg_fb_break = movement_df.loc[
        ~movement_df["no_anchor_available"], "fastball_relative_break"
    ].mean()

    movement_df.loc[
        movement_df["no_anchor_available"], "fastball_relative_break"
    ] = league_avg_fb_break

    print(
        f"\n{n_no_anchor:,} pitcher(s) had no qualifying FF or SI -- "
        f"filled fastball_relative_break with the league average "
        f"({league_avg_fb_break:.2f})."
    )

print(f"Pitchers with movement geometry: {len(movement_df):,}")

print(
    "\nHull method breakdown:\n"
    f"{movement_df['hull_method'].value_counts()}"
)


# ============================================================
# DAMPENED PITCH-COUNT ADJUSTMENT
# ============================================================

DAMPENING_FACTOR = 0.6
APPLY_DAMPENING = True

APPLY_CONFIDENCE_SHRINKAGE = True

CONFOUNDED_FEATURES = [
    "hull_area",
    "horizontal_coverage",
    "vertical_coverage",
    "fastball_relative_break",
    "avg_nn_distance"
]

print(f"\nDampening each feature individually against n_pitch_types...")

n_pitch_types_vals = movement_df["n_pitch_types"].values.astype(float)

for feature in CONFOUNDED_FEATURES:
    y = movement_df[feature].values.astype(float)

    slope, intercept = np.polyfit(n_pitch_types_vals, y, 1)
    pre_corr = np.corrcoef(y, n_pitch_types_vals)[0, 1]

    if APPLY_DAMPENING:
        predicted_trend = intercept + slope * n_pitch_types_vals
        feature_mean = y.mean()
        adjusted = y - DAMPENING_FACTOR * (predicted_trend - feature_mean)
    else:
        adjusted = y.copy()

    FLOOR = 0.01
    n_clipped = (adjusted < FLOOR).sum()
    adjusted = np.maximum(adjusted, FLOOR)

    movement_df[f"{feature}_pre_dampening"] = movement_df[feature]
    movement_df[feature] = adjusted

    post_corr = np.corrcoef(adjusted, n_pitch_types_vals)[0, 1]
    clip_note = f", {n_clipped} pitcher(s) floored at {FLOOR}" if n_clipped > 0 else ""
    damp_note = "" if APPLY_DAMPENING else " (dampening disabled)"

    print(
        f"  {feature}: correlation with n_pitch_types "
        f"{pre_corr:.3f} -> {post_corr:.3f}{damp_note}{clip_note}"
    )


# ============================================================
# SINGLE UNIFIED PCA
# ============================================================

MOVEMENT_FEATURES_FOR_PCA = [
    "hull_area",
    "horizontal_coverage",
    "vertical_coverage",
    "fastball_relative_break",
    "avg_nn_distance"
]

movement_df["confidence_score"] = (
    np.minimum(
        movement_df["n_pitch_types"] / CONFIDENCE_PITCH_TYPES,
        1
    )
    * 100
)

print(f"\nRunning single unified PCA across: {MOVEMENT_FEATURES_FOR_PCA}")

X = movement_df[MOVEMENT_FEATURES_FOR_PCA].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=len(MOVEMENT_FEATURES_FOR_PCA))
components = pca.fit_transform(X_scaled)

pc1_explained = pca.explained_variance_ratio_[0]

print(f"PC1 explains {pc1_explained:.1%} of variance")

if pc1_explained < 0.40:
    print(
        "WARNING: PC1 explains less than 40% of variance -- these "
        "features may not share a single clean axis."
    )

loadings = pd.DataFrame({
    "feature": MOVEMENT_FEATURES_FOR_PCA,
    "PC1_loading": pca.components_[0]
})
print("\nPCA loadings")
print(loadings)

raw_index = components[:, 0]

if np.corrcoef(raw_index, movement_df["hull_area"])[0, 1] < 0:
    raw_index = -raw_index
    loadings["PC1_loading"] *= -1

movement_df["raw_movement_score"] = raw_index

def percentile_rank(series):
    return series.rank(pct=True) * 100

movement_df["movement_score"] = percentile_rank(movement_df["raw_movement_score"])

league_average = movement_df["movement_score"].mean()

if APPLY_CONFIDENCE_SHRINKAGE:
    movement_df["trusted_movement_score"] = (
        movement_df["movement_score"] * (movement_df["confidence_score"] / 100)
        + league_average * (1 - movement_df["confidence_score"] / 100)
    )
else:
    movement_df["trusted_movement_score"] = movement_df["movement_score"]

print(
    f"\nConfidence shrinkage {'ENABLED' if APPLY_CONFIDENCE_SHRINKAGE else 'DISABLED (experiment)'} "
    f"for trusted_movement_score."
)


# ============================================================
# FINAL OUTPUT
# ============================================================

movement_df["season"] = YEAR

final_columns = [
    "player_name",
    "season",
    "n_pitch_types",
    "horizontal_coverage",
    "horizontal_coverage_pre_dampening",
    "vertical_coverage",
    "vertical_coverage_pre_dampening",
    "velocity_range",
    "hull_area",
    "hull_area_pre_dampening",
    "hull_method",
    "fastball_relative_break",
    "fastball_relative_break_pre_dampening",
    "anchor_type",
    "no_anchor_available",
    "avg_nn_distance",
    "avg_nn_distance_pre_dampening",
    "quadrant_coverage_count",
    "confidence_score",
    "raw_movement_score",
    "movement_score",
    "trusted_movement_score"
]

movement_output = (
    movement_df[final_columns]
    .sort_values("trusted_movement_score", ascending=False)
)

movement_output.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print(f"Movement Component Saved ({YEAR})")
print("==============================")

print(
    f"""
Pitchers scored:
{len(movement_output):,}

Average movement score:
{movement_output['trusted_movement_score'].mean():.2f}

Average confidence:
{movement_output['confidence_score'].mean():.2f}
"""
)

print("\nSaved:")
print(f"- {OUTPUT_FILE}")


# ============================================================
# VALIDATION CHECKS
# ============================================================

print("\n==============================")
print("MOVEMENT VALIDATION")
print("==============================")

print("\nCorrelation with number of pitch types:")
print(
    movement_output["trusted_movement_score"].corr(
        movement_output["n_pitch_types"]
    )
)

# merge THIS YEAR's effectiveness output specifically -- original
# hardcoded "PAOM_effectiveness_component.csv" (2025-only); year-
# parameterized here so the validation check compares same-year data
try:
    effectiveness_file = f"PAOM_effectiveness_component_{YEAR}.csv"
    effectiveness = pd.read_csv(effectiveness_file)

    eff_pitcher = (
        effectiveness
        .groupby("player_name")["trusted_effectiveness"]
        .mean()
        .reset_index()
    )

    check = movement_output.merge(
        eff_pitcher,
        on="player_name",
        how="inner"
    )

    print("\nCorrelation with effectiveness (should be low):")
    print(
        check["trusted_movement_score"].corr(
            check["trusted_effectiveness"]
        )
    )

    check.to_csv(f"PAOM_movement_effectiveness_correlation_{YEAR}.csv", index=False)

except FileNotFoundError:
    print(f"\n{effectiveness_file} not found -- skipping check.")
