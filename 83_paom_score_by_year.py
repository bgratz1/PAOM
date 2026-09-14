"""
83_paom_score_by_year.py

Purpose:
--------
The REAL 10_paom_score.py, parameterized by YEAR -- all Ridge/CV/
re-percentiling/weighted-average logic UNCHANGED from the original.

ONE REAL DEVIATION, disclosed explicitly: the original merges in a
PAOM_tunneling_component.csv as REFERENCE-ONLY data (left join, not
used in the actual score calculation -- see original docstring).
No year-parameterized tunneling pipeline exists yet, so this merge
is made OPTIONAL here -- if the file doesn't exist for a given year,
it's skipped gracefully (tunneling columns come back empty/NaN,
exactly matching what a left join against a genuinely tunneling-
ineligible pitcher would already produce) rather than blocking the
whole score calculation on a component this script doesn't actually
use for scoring.

Requires, for the given year: PAOM_effectiveness_component_{YEAR}
.csv (76), PAOM_movement_component_{YEAR}.csv (80), PAOM_command_
component_{YEAR}.csv (81), PAOM_velocity_component_{YEAR}.csv (78),
master_pitch_table_{YEAR}.csv (77), clean_statcast_{YEAR}.csv (75).
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

import sys
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

EFFECTIVENESS_FILE = f"PAOM_effectiveness_component_{YEAR}.csv"
TUNNELING_FILE = f"PAOM_tunneling_component_{YEAR}.csv"  # optional, see docstring
MOVEMENT_FILE = f"PAOM_movement_component_{YEAR}.csv"
COMMAND_FILE = f"PAOM_command_component_{YEAR}.csv"
VELOCITY_FILE = f"PAOM_velocity_component_{YEAR}.csv"
MASTER_FILE = f"master_pitch_table_{YEAR}.csv"
EVENTS_FILE = f"clean_statcast_{YEAR}.csv"

OUTPUT_FILE = f"PAOM_final_scores_{YEAR}.csv"
WEIGHTS_FILE = f"paom_combination_weights_{YEAR}.csv"
VALIDATION_COMPARISON_FILE = f"paom_validation_comparison_{YEAR}.csv"

MIN_PA_FOR_SECONDARY_TARGET = 50
N_CV_FOLDS = 5
RANDOM_SEED = 42


# ============================================================
# STEP 1: PITCHER-LEVEL EFFECTIVENESS (USAGE-WEIGHTED)
# ============================================================

print(f"Loading component files for {YEAR}...")

effectiveness = pd.read_csv(EFFECTIVENESS_FILE)

print(f"Effectiveness rows (pitcher x pitch type): {len(effectiveness):,}")

eff_pitcher = (
    effectiveness
    .groupby("player_name")
    .apply(
        lambda x: pd.Series({
            "pitcher_effectiveness": np.average(
                x["trusted_effectiveness"], weights=x["usage"]
            ),
            "effectiveness_confidence": np.average(
                x["confidence_score"], weights=x["usage"]
            )
        })
    )
    .reset_index()
)

print(f"Pitchers with effectiveness score: {len(eff_pitcher):,}")


# ============================================================
# STEP 1b: PITCHER-LEVEL COMMAND (USAGE-WEIGHTED)
# ============================================================

command = pd.read_csv(COMMAND_FILE)

print(f"Command rows (pitcher x pitch type): {len(command):,}")

command["command_usage"] = (
    command["pitches"]
    / command.groupby("player_name")["pitches"].transform("sum")
)

cmd_pitcher = (
    command
    .groupby("player_name")
    .apply(
        lambda x: pd.Series({
            "pitcher_command": np.average(
                x["trusted_command_score"], weights=x["command_usage"]
            ),
            "command_confidence": np.average(
                x["confidence_score"], weights=x["command_usage"]
            )
        })
    )
    .reset_index()
)

print(f"Pitchers with command score: {len(cmd_pitcher):,}")


# ============================================================
# STEP 1c: VELOCITY (ALREADY PITCHER-LEVEL) -- CANDIDATE COMPONENT
# ============================================================

velocity = pd.read_csv(VELOCITY_FILE)[
    ["player_name", "trusted_velocity_score", "confidence_score"]
].rename(columns={"confidence_score": "velocity_confidence"})

print(f"Pitchers with velocity score: {len(velocity):,}")


# ============================================================
# STEP 2: LOAD MOVEMENT (SINGLE SCORE) + TUNNELING (REFERENCE ONLY,
# OPTIONAL -- see docstring)
# ============================================================

movement = pd.read_csv(MOVEMENT_FILE)[
    ["player_name", "trusted_movement_score", "confidence_score"]
].rename(columns={"confidence_score": "movement_confidence"})

print(f"Pitchers with movement score: {len(movement):,}")

try:
    tunneling = pd.read_csv(TUNNELING_FILE)[
        ["player_name", "trusted_tunneling_score", "raw_tunneling_score"]
    ]
    print(
        f"Pitchers with tunneling score: {len(tunneling):,} "
        f"(reference only -- not used in paom_score)"
    )
    have_tunneling = True
except FileNotFoundError:
    print(
        f"\nNOTE: {TUNNELING_FILE} not found -- no year-parameterized "
        f"tunneling pipeline exists yet. Since tunneling is reference-"
        f"only (NOT used in the actual score calculation), proceeding "
        f"without it. trusted_tunneling_score/raw_tunneling_score will "
        f"be empty in the output."
    )
    have_tunneling = False


# ============================================================
# STEP 3: BUILD PRIMARY VALIDATION TARGET (RUN VALUE)
# ============================================================

master = pd.read_csv(MASTER_FILE)

primary_target_df = (
    master
    .groupby("player_name")
    .agg(
        total_run_value=("run_value", "sum"),
        total_pitches=("pitches", "sum")
    )
    .reset_index()
)

primary_target_df["run_value_per_100"] = (
    primary_target_df["total_run_value"]
    / primary_target_df["total_pitches"]
    * 100
)

primary_target_df["primary_target"] = -primary_target_df["run_value_per_100"]

primary_target_df = primary_target_df[["player_name", "primary_target"]]

print(f"Pitchers with primary validation target: {len(primary_target_df):,}")


# ============================================================
# STEP 3b: BUILD SECONDARY VALIDATION TARGET (K% - BB%)
# ============================================================

print("\nBuilding secondary validation target (K% - BB%)...")

events_df = pd.read_csv(EVENTS_FILE, usecols=["player_name", "events"])

events_df = events_df[events_df["events"].notna()]

strikeout_events = ["strikeout", "strikeout_double_play"]
walk_events = ["walk"]

pa_counts = events_df.groupby("player_name").size().rename("pa_count")

k_counts = (
    events_df[events_df["events"].isin(strikeout_events)]
    .groupby("player_name").size().rename("k_count")
)

bb_counts = (
    events_df[events_df["events"].isin(walk_events)]
    .groupby("player_name").size().rename("bb_count")
)

secondary_target_df = (
    pd.concat([pa_counts, k_counts, bb_counts], axis=1)
    .fillna(0)
    .reset_index()
)

secondary_target_df = secondary_target_df[
    secondary_target_df["pa_count"] >= MIN_PA_FOR_SECONDARY_TARGET
]

secondary_target_df["k_pct"] = (
    secondary_target_df["k_count"] / secondary_target_df["pa_count"]
)
secondary_target_df["bb_pct"] = (
    secondary_target_df["bb_count"] / secondary_target_df["pa_count"]
)

secondary_target_df["secondary_target"] = (
    secondary_target_df["k_pct"] - secondary_target_df["bb_pct"]
)

secondary_target_df = secondary_target_df[
    ["player_name", "secondary_target"]
]

print(
    f"Pitchers with secondary validation target "
    f"(>= {MIN_PA_FOR_SECONDARY_TARGET} PA): {len(secondary_target_df):,}"
)


# ============================================================
# STEP 4: MERGE
# ============================================================

print("\nMerging components...")

combined = eff_pitcher.merge(movement, on="player_name", how="inner")
combined = combined.merge(cmd_pitcher, on="player_name", how="inner")
combined = combined.merge(velocity, on="player_name", how="inner")

print(
    f"Pitchers with all four candidate components (Effectiveness + "
    f"Movement + Command + Velocity): {len(combined):,} "
    f"(dropped {len(eff_pitcher) - len(combined):,})"
)

if have_tunneling:
    combined = combined.merge(tunneling, on="player_name", how="left")
    print(
        f"Of those, {combined['trusted_tunneling_score'].notna().sum():,} "
        f"also have a tunneling score available for recommendation-time "
        f"matching."
    )
else:
    combined["trusted_tunneling_score"] = np.nan
    combined["raw_tunneling_score"] = np.nan


# ============================================================
# STEP 4a: PAOM CONFIDENCE (MINIMUM OF THE CANDIDATE COMPONENTS)
# ============================================================

combined["paom_confidence"] = combined[
    [
        "effectiveness_confidence", "movement_confidence",
        "command_confidence", "velocity_confidence"
    ]
].min(axis=1)

print(
    f"\nPAOM confidence (min of the candidate components) -- "
    f"mean: {combined['paom_confidence'].mean():.1f}, "
    f"median: {combined['paom_confidence'].median():.1f}"
)


# ============================================================
# STEP 4b: RE-PERCENTILE WITHIN THE PAOM-ELIGIBLE POPULATION
# ============================================================

corr_cols = [
    "pitcher_effectiveness",
    "pitcher_command",
    "trusted_velocity_score",
    "trusted_movement_score"
]

for component in corr_cols:
    combined[component] = combined[component].rank(pct=True) * 100

print(
    "\nRe-percentiled Effectiveness/Command/Velocity/Movement within "
    "the paom-eligible population."
)


# ============================================================
# STEP 5: CORRELATION CHECK
# ============================================================

print("\n==============================")
print("Component Correlation Check")
print("==============================")

corr_matrix = combined[corr_cols].corr().round(3)

print("\nCorrelation matrix:")
print(corr_matrix)

high_corr = False

for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        val = corr_matrix.iloc[i, j]
        if abs(val) > 0.5:
            high_corr = True
            print(
                f"\nWARNING: {corr_cols[i]} and {corr_cols[j]} "
                f"correlate at {val:.2f} -- combining both at "
                f"full weight may double-count shared signal."
            )

if not high_corr:
    print(
        "\nNo pairwise correlation exceeds 0.5 -- components "
        "appear to measure distinct things -- safe to combine."
    )


# ============================================================
# STEP 6: CROSS-VALIDATED RIDGE FOR BOTH TARGETS
# ============================================================

def cv_ridge_fit(X_df, y, label, sample_weight=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    y = np.asarray(y)
    w = np.asarray(sample_weight) if sample_weight is not None else None

    alpha_search = RidgeCV(alphas=np.logspace(-3, 3, 50))
    alpha_search.fit(X_scaled, y, sample_weight=w)
    best_alpha = alpha_search.alpha_

    cv = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    cv_scores = []
    for train_idx, test_idx in cv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_model = Ridge(alpha=best_alpha)

        if w is not None:
            w_train, w_test = w[train_idx], w[test_idx]
            fold_model.fit(X_train, y_train, sample_weight=w_train)
            cv_scores.append(fold_model.score(X_test, y_test, sample_weight=w_test))
        else:
            fold_model.fit(X_train, y_train)
            cv_scores.append(fold_model.score(X_test, y_test))

    cv_scores = np.array(cv_scores)

    print(f"\n{label}")
    print("-" * len(label))
    print(f"Selected alpha: {best_alpha:.3f}")
    if w is not None:
        print(f"Confidence-weighted (mean weight: {w.mean():.2f})")
    print(f"Cross-validated R2 per fold: {np.round(cv_scores, 3)}")
    print(
        f"Cross-validated R2: {cv_scores.mean():.3f} "
        f"(+/- {cv_scores.std():.3f})"
    )

    final_model = Ridge(alpha=best_alpha)

    if w is not None:
        final_model.fit(X_scaled, y, sample_weight=w)
        in_sample_r2 = final_model.score(X_scaled, y, sample_weight=w)
    else:
        final_model.fit(X_scaled, y)
        in_sample_r2 = final_model.score(X_scaled, y)

    print(
        f"(For comparison, in-sample R2 on full data: "
        f"{in_sample_r2:.3f})"
    )

    coef_df = pd.DataFrame({
        "component": X_df.columns,
        "coefficient": final_model.coef_
    })

    return {
        "scaler": scaler,
        "model": final_model,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        "in_sample_r2": in_sample_r2,
        "coefficients": coef_df
    }


print("\n==============================")
print("Cross-Validated Combination Weights")
print("==============================")

primary_model_df = combined.merge(primary_target_df, on="player_name", how="inner")
print(f"\nPitchers used for primary-target fit: {len(primary_model_df):,}")

primary_weights = primary_model_df["paom_confidence"] / 100

primary_result = cv_ridge_fit(
    primary_model_df[corr_cols],
    primary_model_df["primary_target"],
    "PRIMARY TARGET: Run value per 100 pitches",
    sample_weight=primary_weights
)

secondary_model_df = combined.merge(
    secondary_target_df, on="player_name", how="inner"
)
print(f"\nPitchers used for secondary-target fit: {len(secondary_model_df):,}")

secondary_weights = secondary_model_df["paom_confidence"] / 100

secondary_result = cv_ridge_fit(
    secondary_model_df[corr_cols],
    secondary_model_df["secondary_target"],
    "SECONDARY TARGET: K% - BB% (independent check)",
    sample_weight=secondary_weights
)


# ============================================================
# STEP 6b: COMPARE THE TWO TARGETS
# ============================================================

print("\n==============================")
print("Primary vs Secondary Target Comparison")
print("==============================")

comparison = primary_result["coefficients"].merge(
    secondary_result["coefficients"],
    on="component",
    suffixes=("_primary", "_secondary")
)

comparison["same_sign"] = (
    np.sign(comparison["coefficient_primary"])
    == np.sign(comparison["coefficient_secondary"])
)

print("\n", comparison)

print(
    f"\nCV R2 -- primary: {primary_result['cv_r2_mean']:.3f}, "
    f"secondary: {secondary_result['cv_r2_mean']:.3f}"
)

if comparison["same_sign"].all():
    print(
        "\nAll components agree in sign across both targets -- "
        "the primary result does not appear to be an artifact of "
        "target circularity."
    )
else:
    disagreeing = comparison[~comparison["same_sign"]]["component"].tolist()
    print(
        f"\nNOTE: {disagreeing} disagree in sign between the two "
        f"targets -- treat that component's weight with caution."
    )

comparison.to_csv(VALIDATION_COMPARISON_FILE, index=False)


# ============================================================
# STEP 7: FINALIZE WEIGHTS (FROM PRIMARY TARGET, FULL POPULATION)
# ============================================================

weights_df = primary_result["coefficients"].copy()
weights_df["abs_coefficient"] = weights_df["coefficient"].abs()
weights_df = weights_df.sort_values("abs_coefficient", ascending=False)

negative = weights_df[weights_df["coefficient"] < 0]

if len(negative) > 0:
    print(
        "\nNOTE: the following component(s) came back with a "
        "NEGATIVE coefficient in the primary (deployed) fit -- "
        "floored to 0 before normalizing (per explicit decision: a "
        "component with an unstable, likely noise-driven sign should "
        "contribute nothing to the score in that year, not actively "
        "subtract from it). Raw coefficient shown for transparency; "
        "the floored value is what's actually used below."
    )
    print(negative)

# FLOOR: a negative coefficient becomes 0 (contributes nothing),
# rather than being used as-is (which would let that component
# actively SUBTRACT from paom_score for pitchers who score high on
# it). The other components' weights correctly redistribute to fill
# the gap, since normalization happens AFTER flooring.
weights_df["floored_coefficient"] = weights_df["coefficient"].clip(lower=0)

coefficient_sum = weights_df["floored_coefficient"].sum()

if coefficient_sum <= 0:
    raise ValueError(
        f"Sum of floored coefficients is {coefficient_sum:.4f} (<=0) "
        f"-- cannot normalize into meaningful weights. This means "
        f"EVERY component came back negative, which would indicate a "
        f"much more serious problem with the fit than one noisy "
        f"component -- inspect before proceeding."
    )

weights_df["normalized_weight"] = (
    weights_df["floored_coefficient"] / coefficient_sum
)

print("\nNormalized weights (used as literal blend weights):")
print(weights_df[["component", "coefficient", "floored_coefficient", "normalized_weight"]])

weights_df.to_csv(WEIGHTS_FILE, index=False)

weight_lookup = dict(
    zip(weights_df["component"], weights_df["normalized_weight"])
)


# ============================================================
# STEP 8: WEIGHTED AVERAGE OF COMPONENT PERCENTILES
# ============================================================

contribution_cols = []

for component in corr_cols:
    contribution_col = f"{component}_contribution"
    combined[contribution_col] = (
        combined[component] * weight_lookup[component]
    )
    contribution_cols.append(contribution_col)

combined["paom_score"] = combined[contribution_cols].sum(axis=1)

out_of_range = (
    (combined["paom_score"] < 0) | (combined["paom_score"] > 100)
).sum()

if out_of_range > 0:
    print(f"\nNOTE: {out_of_range} pitcher(s) landed outside 0-100 before clipping.")

combined["paom_score"] = combined["paom_score"].clip(0, 100)


# ============================================================
# SAVE OUTPUT
# ============================================================

combined["season"] = YEAR

output = combined[
    [
        "player_name",
        "season",
        "pitcher_effectiveness",
        "pitcher_command",
        "trusted_velocity_score",
        "trusted_movement_score",
        "trusted_tunneling_score",
        "raw_tunneling_score"
    ]
    + contribution_cols
    + [
        "paom_score",
        "paom_confidence",
        "effectiveness_confidence",
        "movement_confidence",
        "command_confidence",
        "velocity_confidence"
    ]
].sort_values("paom_score", ascending=False)

output.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print(f"PAOM Final Score Saved ({YEAR})")
print("==============================")

print(
    f"""
Pitchers scored:
{len(output):,}

Average PAOM score:
{output['paom_score'].mean():.2f}

Primary-target cross-validated R2 (honest, out-of-sample):
{primary_result['cv_r2_mean']:.3f} (+/- {primary_result['cv_r2_std']:.3f})
"""
)

print("\nSaved:")
print(f"- {OUTPUT_FILE}")
print(f"- {WEIGHTS_FILE}")
print(f"- {VALIDATION_COMPARISON_FILE}")
