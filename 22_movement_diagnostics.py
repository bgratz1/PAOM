"""
22_movement_diagnostics.py

Purpose:
--------
Diagnose whether Movement's negative weight in paom_score and its
lingering 0.42 correlation with Velocity trace back to the same
root cause: several of Movement's raw features may still be
substantially driven by n_pitch_types (raw arsenal size) rather
than genuine arsenal SHAPE quality, the same confound flagged
(but only partially fixed, via avg_nn_distance) the first time
Movement was validated.

This is READ-ONLY -- it doesn't change 09_movement_component.py or
any other file. It answers three questions so the right rework (if
any) can be chosen deliberately rather than guessed at:

1. Does movement_score/trusted_movement_score still correlate with
   n_pitch_types overall?
2. WHICH of the five raw features is driving that correlation --
   is it all five equally, or concentrated in a few (hypothesis:
   horizontal_coverage, vertical_coverage, and hull_area, which are
   range/area measures that mechanically grow with more points to
   draw a boundary around; centroid_dispersion and avg_nn_distance
   measure ARRANGEMENT, not EXTENT, so should be less size-driven)?
3. Does movement_score still meaningfully discriminate WITHIN a
   fixed n_pitch_types group (e.g. among all 4-pitch-type arsenals,
   is there real spread), or is most of its variance actually just
   "how many pitch types does this pitcher throw"?

Output:
-------
Printed diagnostics only -- no file saved. Copy the printed output
back for interpretation.
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "PAOM_movement_component.csv"

RAW_FEATURES = [
    "horizontal_coverage",
    "vertical_coverage",
    "hull_area",
    "fastball_relative_break",
    "avg_nn_distance",
    "quadrant_coverage_count"
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading movement component...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} pitchers")

required_cols = ["n_pitch_types", "movement_score", "trusted_movement_score"] + RAW_FEATURES

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}. Make sure this is "
        f"the CURRENT PAOM_movement_component.csv (post velocity_range "
        f"removal) -- re-run 09_movement_component.py if unsure."
    )


# ============================================================
# QUESTION 1: OVERALL CORRELATION WITH n_pitch_types
# ============================================================

print("\n==============================")
print("1. Overall Correlation with n_pitch_types")
print("==============================")

overall_corr_raw = df["movement_score"].corr(df["n_pitch_types"])
overall_corr_trusted = df["trusted_movement_score"].corr(df["n_pitch_types"])

print(f"\nmovement_score vs n_pitch_types:         {overall_corr_raw:.3f}")
print(f"trusted_movement_score vs n_pitch_types: {overall_corr_trusted:.3f}")

if abs(overall_corr_trusted) > 0.5:
    print(
        "\nSubstantial correlation with pitch count remains -- "
        "Movement is still significantly conflating 'how many pitch "
        "types' with 'how good is the shape.'"
    )
elif abs(overall_corr_trusted) > 0.3:
    print(
        "\nModerate correlation with pitch count -- some conflation "
        "remains, worth addressing but not the dominant signal."
    )
else:
    print(
        "\nLow correlation with pitch count -- Movement appears to "
        "be measuring something reasonably independent of raw "
        "arsenal size."
    )


# ============================================================
# QUESTION 2: WHICH RAW FEATURE(S) DRIVE IT
# ============================================================

print("\n==============================")
print("2. Each Raw Feature's Own Correlation with n_pitch_types")
print("==============================")

feature_corrs = pd.DataFrame({
    "feature": RAW_FEATURES,
    "correlation_with_n_pitch_types": [
        df[f].corr(df["n_pitch_types"]) for f in RAW_FEATURES
    ]
})

feature_corrs["abs_correlation"] = feature_corrs["correlation_with_n_pitch_types"].abs()
feature_corrs = feature_corrs.sort_values("abs_correlation", ascending=False)

print("\n", feature_corrs[["feature", "correlation_with_n_pitch_types"]])

high_confound = feature_corrs[feature_corrs["abs_correlation"] > 0.5]

if len(high_confound) > 0:
    print(
        f"\nFeature(s) with correlation > 0.5 to n_pitch_types "
        f"(size-driven, not shape-driven): "
        f"{high_confound['feature'].tolist()}"
    )
else:
    print("\nNo single feature shows a dominant size confound (all under 0.5).")


# ============================================================
# QUESTION 3: WITHIN-GROUP VARIANCE (does Movement discriminate
# beyond just pitch count?)
# ============================================================

print("\n==============================")
print("3. Within-Group Variance by n_pitch_types")
print("==============================")

print(
    "\nIf movement_score were PURELY a function of n_pitch_types, "
    "std within each group below would be ~0. Real, substantial "
    "std within a group means Movement is discriminating beyond "
    "just pitch count -- the original bar this passed when first "
    "validated."
)

within_group = (
    df.groupby("n_pitch_types")["trusted_movement_score"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

print("\n", within_group)

avg_within_std = within_group["std"].mean()
between_group_std = within_group["mean"].std()

print(f"\nAverage within-group std: {avg_within_std:.2f}")
print(f"Between-group (mean-to-mean) std: {between_group_std:.2f}")

if avg_within_std < between_group_std * 0.5:
    print(
        "\nWithin-group spread is small relative to between-group "
        "spread -- most of Movement's variance IS pitch count, "
        "with comparatively little real discrimination within a "
        "fixed arsenal size. Residualizing against n_pitch_types "
        "is likely worth doing."
    )
else:
    print(
        "\nWithin-group spread is substantial relative to "
        "between-group spread -- Movement is still discriminating "
        "meaningfully within a fixed pitch count, not just "
        "measuring arsenal size."
    )


# ============================================================
# QUESTION 4: DO THE FIVE FEATURES REDUNDANTLY MEASURE EACH OTHER?
# ============================================================

"""
Every prior check in this script looked at each feature's
relationship to n_pitch_types. This checks something different --
whether the five features are substantially redundant WITH EACH
OTHER, independent of pitch count. hull_area and centroid_dispersion
in particular are both fundamentally "how spread out is the
arsenal," just measured differently (total area vs. average
distance from center) -- plausible they're saying largely the same
thing, in which case carrying both may not add much over carrying
one. Uses the DAMPENED/adjusted feature values (the ones actually
feeding the PCA), not the raw pre-adjustment ones.
"""

print("\n==============================")
print("4. Inter-Feature Redundancy Check")
print("==============================")

feature_corr_matrix = df[RAW_FEATURES].corr().round(3)

print("\n", feature_corr_matrix)

high_pairs = []
for i in range(len(RAW_FEATURES)):
    for j in range(i + 1, len(RAW_FEATURES)):
        val = feature_corr_matrix.iloc[i, j]
        if abs(val) > 0.5:
            high_pairs.append((RAW_FEATURES[i], RAW_FEATURES[j], val))

if high_pairs:
    print("\nFeature pairs correlated above 0.5 (possible redundancy):")
    for f1, f2, val in high_pairs:
        print(f"  {f1} <-> {f2}: {val:.3f}")
    print(
        "\nConsider whether one of each flagged pair could be "
        "dropped without losing much real signal -- a smaller, "
        "less redundant feature set can produce a cleaner PCA axis "
        "than a larger, overlapping one."
    )
else:
    print(
        "\nNo feature pairs exceed 0.5 -- the five features appear "
        "to be capturing reasonably distinct aspects of arsenal "
        "shape, not redundantly re-measuring each other."
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("Diagnostic Complete")
print("==============================")

print(
    f"""
Overall correlation with n_pitch_types (trusted): {overall_corr_trusted:.3f}
Most size-confounded feature: {feature_corrs.iloc[0]['feature']} ({feature_corrs.iloc[0]['correlation_with_n_pitch_types']:.3f})
Least size-confounded feature: {feature_corrs.iloc[-1]['feature']} ({feature_corrs.iloc[-1]['correlation_with_n_pitch_types']:.3f})
Within-group std vs between-group std: {avg_within_std:.2f} vs {between_group_std:.2f}
"""
)
