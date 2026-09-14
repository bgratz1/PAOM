"""
41_outcome_metric_comparison.py

Purpose:
--------
39_validate_pitch_type_sensitivity.py's Check 4 (run twice now, once
with the corrected sign-consistency calculation) consistently showed
mean_reversion_coef dominating change_specific_coef for WAR, even
after a substantial similarity-metric overhaul (weighted features +
pitch-composition matching). Two feature-engineering attempts in a
row didn't move this -- worth checking whether the limitation is
about WAR specifically, or fundamental to the whole approach.

WAR is affected by everything that happens across an entire season --
other pitches, defense, luck, innings pitched -- while a single
arsenal change is a small, specific input several steps removed from
that outcome. FIP and xwOBA-against are more directly tied to what a
pitch actually does. This runs the SAME diagnostic (mean_reversion_
coef vs. change_specific_coef, magnitude and sign-consistency) across
all three outcome metrics side by side:
- If FIP/xwOBA show a meaningfully STRONGER, more sign-consistent
  change_specific_coef than WAR, that's real, actionable evidence
  the target metric matters -- worth switching the tool's primary
  target.
- If FIP/xwOBA show the SAME weak, unstable pattern, that's evidence
  the limitation is more fundamental (sample size, observational
  data, or the modeling approach itself) rather than fixable by
  choosing a different, closer outcome metric.

Same target-pitcher selection and sign-consistency calculation as
39_validate_pitch_type_sensitivity.py (imported directly, including
the corrected count-based majority sign logic -- not the earlier
buggy mean-based version).

Output:
-------
One side-by-side summary table comparing all three metrics. No file
saved -- this is a diagnostic, not a deliverable.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 38 (numeric-prefixed filename needs importlib)
# ============================================================

print("Loading training data and prediction function from 38...")

spec = importlib.util.spec_from_file_location(
    "similarity_weighted_regression", "38_similarity_weighted_regression.py"
)
swr = importlib.util.module_from_spec(spec)
sys.modules["similarity_weighted_regression"] = swr
spec.loader.exec_module(swr)

training = swr.training
similarity_features = swr.similarity_features
outcomes = swr.outcomes
predict_arsenal_change_effect = swr.predict_arsenal_change_effect
SIMILARITY_FEATURE_COLS = swr.SIMILARITY_FEATURE_COLS
MIN_GROUP_SIZE_FOR_REGRESSION = swr.MIN_GROUP_SIZE_FOR_REGRESSION


# ============================================================
# SETTINGS
# ============================================================

N_TARGET_PITCHERS = 6
CHANGE_TYPE = "ADD"

CANDIDATE_METRICS = ["war", "fip", "xwoba_against"]
AVAILABLE_METRICS = [m for m in CANDIDATE_METRICS if f"{m}_before" in training.columns]

print(f"Comparing across: {AVAILABLE_METRICS}")

if len(AVAILABLE_METRICS) < 2:
    raise ValueError(
        f"Only {len(AVAILABLE_METRICS)} outcome metric(s) available -- "
        f"need at least 2 to compare. Check pitcher_season_metrics_"
        f"2020_2025.csv has war/fip/xwoba_against columns."
    )

# NOTE ON DIRECTION: WAR is "higher is better," FIP and xwOBA-against
# are "lower is better." This doesn't affect the magnitude or sign-
# CONSISTENCY comparisons below (a coefficient that's reliably one
# sign is reliably one sign regardless of which direction counts as
# "good") -- but it matters when interpreting what a positive vs.
# negative change_specific_coef actually MEANS for each metric.


# ============================================================
# SAME SIGN-CONSISTENCY LOGIC AS 39 (corrected, count-based majority)
# ============================================================

def sign_consistency(vals):
    vals = pd.Series(vals).dropna()
    if len(vals) == 0:
        return np.nan
    signs = np.sign(vals)
    n_positive = (signs > 0).sum()
    n_negative = (signs < 0).sum()
    majority_sign = 1 if n_positive >= n_negative else -1
    return (signs == majority_sign).mean()


# ============================================================
# SELECT DIVERSE TARGET PITCHERS (same approach as 39)
# ============================================================

print(f"\nSelecting {N_TARGET_PITCHERS} diverse target pitchers (stratified by max_velo_z)...")

complete_profiles = similarity_features.dropna(subset=SIMILARITY_FEATURE_COLS).copy()
complete_profiles = complete_profiles.sort_values("max_velo_z").reset_index(drop=True)

strat_indices = np.linspace(0, len(complete_profiles) - 1, N_TARGET_PITCHERS).astype(int)
target_rows = complete_profiles.iloc[strat_indices]

for _, r in target_rows.iterrows():
    print(f"  {r['player_name']}, {r['season']}")


# ============================================================
# RUN THE GRID FOR EACH METRIC
# ============================================================

all_results = []

for metric in AVAILABLE_METRICS:
    print(f"\nRunning grid for {metric}...")

    add_counts = training[training["change_type"] == CHANGE_TYPE]["pitch_type"].value_counts()
    viable_pitch_types = add_counts[add_counts >= MIN_GROUP_SIZE_FOR_REGRESSION].index.tolist()

    for _, target_row in target_rows.iterrows():
        target_features = {c: target_row[c] for c in SIMILARITY_FEATURE_COLS}
        target_features["usage_pct_delta"] = 10.0

        target_baseline_rows = outcomes[outcomes["player_id"] == target_row["player_id"]]
        target_outcome_before = (
            target_baseline_rows[metric].iloc[0]
            if len(target_baseline_rows) > 0 and metric in target_baseline_rows.columns
            else training[f"{metric}_before"].median()
        )

        for pitch_type in viable_pitch_types:
            result = predict_arsenal_change_effect(
                target_features, pitch_type, CHANGE_TYPE,
                target_outcome_before, metric,
                target_row["player_id"], target_row["season"]
            )

            if result.get("mean_reversion_coef") is not None:
                all_results.append({
                    "metric": metric,
                    "target": f"{target_row['player_name']}_{target_row['season']}",
                    "pitch_type": pitch_type,
                    "mean_reversion_coef": result.get("mean_reversion_coef"),
                    "change_specific_coef": result.get("change_specific_coef"),
                })

results_df = pd.DataFrame(all_results)


# ============================================================
# SIDE-BY-SIDE COMPARISON ACROSS METRICS
# ============================================================

print("\n\n==============================")
print("Side-by-side comparison: does the target metric matter?")
print("==============================\n")

comparison_rows = []
for metric in AVAILABLE_METRICS:
    metric_rows = results_df[results_df["metric"] == metric]
    if len(metric_rows) == 0:
        continue

    avg_mean_reversion_mag = metric_rows["mean_reversion_coef"].abs().mean()
    avg_change_specific_mag = metric_rows["change_specific_coef"].abs().mean()

    # sign consistency computed PER TARGET then averaged, same
    # methodology as 39 -- consistency is a within-pitcher question
    # (does this pitcher's change-specific coefficient stay one sign
    # across different pitch types), not a global pool
    per_target_consistency = metric_rows.groupby("target")["change_specific_coef"].apply(sign_consistency)
    avg_sign_consistency = per_target_consistency.mean()

    ratio = avg_mean_reversion_mag / avg_change_specific_mag if avg_change_specific_mag > 0 else np.nan

    comparison_rows.append({
        "metric": metric,
        "n_predictions": len(metric_rows),
        "avg_|mean_reversion_coef|": avg_mean_reversion_mag,
        "avg_|change_specific_coef|": avg_change_specific_mag,
        "mean_reversion_to_change_ratio": ratio,
        "avg_sign_consistency": avg_sign_consistency,
    })

comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.round(4).to_string(index=False))

print(
    "\nLower mean_reversion_to_change_ratio and higher "
    "avg_sign_consistency both indicate a STRONGER, more trustworthy "
    "change-specific signal for that metric."
)

best_metric_row = comparison_df.loc[comparison_df["avg_sign_consistency"].idxmax()]
worst_ratio_metric_row = comparison_df.loc[comparison_df["mean_reversion_to_change_ratio"].idxmin()]

print(
    f"\nHighest sign consistency: {best_metric_row['metric']} "
    f"({best_metric_row['avg_sign_consistency']:.1%})"
)
print(
    f"Lowest mean-reversion-to-change ratio (most balanced): "
    f"{worst_ratio_metric_row['metric']} "
    f"({worst_ratio_metric_row['mean_reversion_to_change_ratio']:.2f}x)"
)

spread = comparison_df["avg_sign_consistency"].max() - comparison_df["avg_sign_consistency"].min()
if spread > 0.15:
    print(
        f"\nSign consistency varies by {spread:.1%} across metrics -- "
        f"meaningful evidence the CHOICE of outcome metric matters, "
        f"not just the modeling approach. Worth switching the tool's "
        f"primary target to whichever metric performed best above."
    )
else:
    print(
        f"\nSign consistency only varies by {spread:.1%} across "
        f"metrics -- all three show a similarly weak, unstable "
        f"change-specific signal. This is evidence the limitation is "
        f"more FUNDAMENTAL (sample size, observational-data noise, or "
        f"the modeling approach itself) than something a different "
        f"choice of outcome metric would fix."
    )
