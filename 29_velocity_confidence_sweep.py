"""
29_velocity_confidence_sweep.py

Purpose:
--------
Velocity's confidence formula (min(n_pitch_types/4, 1) -- a linear
ramp that hard-caps at 4 pitch types) was, like every other
component's original formula, never actually tested against
alternatives. This sweeps BOTH the FORM of the formula and each
form's own parameter, evaluated through the same real downstream
validation used throughout this pipeline -- same methodology as
26/27/28, adapted for Velocity's structure.

n_pitch_types is a small, discrete count (roughly 1-7 in practice),
not a large continuous count like swings or pitches -- grids are
scaled down accordingly, same as Movement's sweep.

Movement is held at its OWN current default throughout (not swept
here) -- see 28_movement_confidence_sweep.py for Movement's own
sweep. Testing one component's parameter at a time, holding
everything else fixed, matches the approach already used for
Effectiveness, Command, and Movement.

Two forms tested (see 26_effectiveness_confidence_sweep.py for the
full reasoning behind these two specific forms):
1. LINEAR_CAP (current): confidence = min(n / saturation, 1) * 100
2. HYPERBOLIC (n / (n+k)): never fully caps at 100%.

For each (form, parameter) candidate:
- Recompute confidence_score from n_pitch_types using that form
- Recompute trusted_velocity_score with the SAME blend formula,
  just a different confidence input (velocity_score, the pre-
  shrinkage PCA-derived percentile, is unaffected -- only the
  shrinkage step changes)
- Merge with Effectiveness/Command/Movement (held at CURRENT
  defaults) + both validation targets
- Re-percentile within the paom-eligible population
- Run the same confidence-weighted CV-Ridge fit against both targets

Reported per candidate: CV R2 on both targets, Velocity's own
coefficient and sign-agreement, and the same degeneracy guardrail
as the prior sweeps (pitchers landing near the exact 0/100 extremes
-- particularly relevant here given the real-data evidence that
disabling Velocity's shrinkage entirely produced a 0.481 pitch-count
correlation, right at this project's 0.5 redundancy threshold; see
21_velocity_component.py's history).

Output:
-------
Printed sweep table only -- no file saved. Once a winner is picked,
update 21_velocity_component.py's actual formula to match.
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

VELOCITY_FILE = "PAOM_velocity_component.csv"
EFFECTIVENESS_FILE = "PAOM_effectiveness_component.csv"
COMMAND_FILE = "PAOM_command_component.csv"
MOVEMENT_FILE = "PAOM_movement_component.csv"
MASTER_FILE = "master_pitch_table_2025.csv"
EVENTS_FILE = "clean_statcast_2025.csv"

# n_pitch_types is a small discrete count (~1-7) -- grids scaled
# down accordingly from swings/pitches-based sweeps, same as
# Movement's sweep
LINEAR_CAP_GRID = [2, 3, 4, 5, 6, 7]
HYPERBOLIC_K_GRID = [1, 2, 3, 4, 5, 6]

MIN_PA_FOR_SECONDARY_TARGET = 50
N_CV_FOLDS = 5
RANDOM_SEED = 42

DEGENERACY_EXTREME_THRESHOLD = 0.5  # pitchers within this of 0 or 100


# ============================================================
# LOAD RAW VELOCITY DATA (form-independent)
# ============================================================

print("Loading velocity component (pre-shrinkage values)...")

velocity_raw = pd.read_csv(VELOCITY_FILE)

required_cols = ["player_name", "velocity_score", "n_pitch_types"]
missing = [c for c in required_cols if c not in velocity_raw.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}. This script needs "
        f"21_velocity_component.py's output with velocity_score "
        f"(pre-shrinkage) and n_pitch_types preserved."
    )

print(f"Loaded {len(velocity_raw):,} pitcher rows")


# ============================================================
# LOAD OTHER COMPONENTS (held fixed -- this sweep only touches Velocity)
# ============================================================

print("\nLoading other components (held at current defaults)...")

effectiveness = pd.read_csv(EFFECTIVENESS_FILE)
if "usage" in effectiveness.columns:
    effectiveness["eff_usage"] = effectiveness["usage"]
else:
    effectiveness["eff_usage"] = (
        effectiveness["pitches"]
        / effectiveness.groupby("player_name")["pitches"].transform("sum")
    )
eff_pitcher = (
    effectiveness.groupby("player_name")
    .apply(lambda x: np.average(x["trusted_effectiveness"], weights=x["eff_usage"]))
    .reset_index(name="pitcher_effectiveness")
)

command = pd.read_csv(COMMAND_FILE)
command["command_usage"] = (
    command["pitches"] / command.groupby("player_name")["pitches"].transform("sum")
)
cmd_pitcher = (
    command.groupby("player_name")
    .apply(lambda x: np.average(x["trusted_command_score"], weights=x["command_usage"]))
    .reset_index(name="pitcher_command")
)

movement = pd.read_csv(MOVEMENT_FILE)[["player_name", "trusted_movement_score"]]

print(f"Effectiveness: {len(eff_pitcher):,}, Command: {len(cmd_pitcher):,}, Movement: {len(movement):,}")


# ============================================================
# BUILD VALIDATION TARGETS
# ============================================================

print("\nBuilding validation targets...")

master = pd.read_csv(MASTER_FILE)

primary_target_df = (
    master.groupby("player_name")
    .agg(total_run_value=("run_value", "sum"), total_pitches=("pitches", "sum"))
    .reset_index()
)
primary_target_df["primary_target"] = -(
    primary_target_df["total_run_value"] / primary_target_df["total_pitches"] * 100
)
primary_target_df = primary_target_df[["player_name", "primary_target"]]

events_df = pd.read_csv(EVENTS_FILE, usecols=["player_name", "events"])
events_df = events_df[events_df["events"].notna()]

pa_counts = events_df.groupby("player_name").size().rename("pa_count")
k_counts = (
    events_df[events_df["events"].isin(["strikeout", "strikeout_double_play"])]
    .groupby("player_name").size().rename("k_count")
)
bb_counts = (
    events_df[events_df["events"] == "walk"]
    .groupby("player_name").size().rename("bb_count")
)

secondary_target_df = (
    pd.concat([pa_counts, k_counts, bb_counts], axis=1).fillna(0).reset_index()
)
secondary_target_df = secondary_target_df[
    secondary_target_df["pa_count"] >= MIN_PA_FOR_SECONDARY_TARGET
]
secondary_target_df["secondary_target"] = (
    secondary_target_df["k_count"] / secondary_target_df["pa_count"]
    - secondary_target_df["bb_count"] / secondary_target_df["pa_count"]
)
secondary_target_df = secondary_target_df[["player_name", "secondary_target"]]

print(f"Primary target: {len(primary_target_df):,}, Secondary target: {len(secondary_target_df):,}")


# ============================================================
# CV-RIDGE FIT (identical logic to 10_paom_score.py)
# ============================================================

def cv_ridge_fit(X_df, y, sample_weight=None):
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

    final_model = Ridge(alpha=best_alpha)
    if w is not None:
        final_model.fit(X_scaled, y, sample_weight=w)
    else:
        final_model.fit(X_scaled, y)

    coef_df = pd.DataFrame({"component": X_df.columns, "coefficient": final_model.coef_})

    return cv_scores.mean(), cv_scores.std(), coef_df


def percentile_rank(series):
    return series.rank(pct=True) * 100


# ============================================================
# SWEEP
# ============================================================

def run_candidate(form_name, param_value, confidence_fn):
    """
    confidence_fn: takes the n_pitch_types Series, returns a
    confidence Series (0-100).
    """
    vel = velocity_raw.copy()

    vel["confidence_score"] = confidence_fn(vel["n_pitch_types"])

    league_average = vel["velocity_score"].mean()
    vel["trusted_velocity_score"] = (
        vel["velocity_score"] * (vel["confidence_score"] / 100)
        + league_average * (1 - vel["confidence_score"] / 100)
    )

    # degeneracy guardrail
    n_extreme = (
        (vel["trusted_velocity_score"] <= DEGENERACY_EXTREME_THRESHOLD)
        | (vel["trusted_velocity_score"] >= 100 - DEGENERACY_EXTREME_THRESHOLD)
    ).sum()

    vel_pitcher = vel[["player_name", "trusted_velocity_score"]]

    combined = eff_pitcher.merge(vel_pitcher, on="player_name", how="inner")
    combined = combined.merge(cmd_pitcher, on="player_name", how="inner")
    combined = combined.merge(movement, on="player_name", how="inner")

    corr_cols = [
        "pitcher_effectiveness", "pitcher_command",
        "trusted_velocity_score", "trusted_movement_score"
    ]

    for component in corr_cols:
        combined[component] = percentile_rank(combined[component])

    primary_df = combined.merge(primary_target_df, on="player_name", how="inner")
    secondary_df = combined.merge(secondary_target_df, on="player_name", how="inner")

    primary_r2, _, primary_coefs = cv_ridge_fit(
        primary_df[corr_cols], primary_df["primary_target"]
    )
    secondary_r2, _, secondary_coefs = cv_ridge_fit(
        secondary_df[corr_cols], secondary_df["secondary_target"]
    )

    vel_coef_primary = primary_coefs.set_index("component").loc[
        "trusted_velocity_score", "coefficient"
    ]
    vel_coef_secondary = secondary_coefs.set_index("component").loc[
        "trusted_velocity_score", "coefficient"
    ]

    return {
        "form": form_name,
        "parameter": param_value,
        "avg_confidence": vel["confidence_score"].mean(),
        "n_pitchers_at_extreme": n_extreme,
        "primary_cv_r2": primary_r2,
        "secondary_cv_r2": secondary_r2,
        "vel_coef_primary": vel_coef_primary,
        "vel_coef_secondary": vel_coef_secondary,
        "sign_agrees": np.sign(vel_coef_primary) == np.sign(vel_coef_secondary)
    }


results = []

print("\n==============================")
print(f"Sweeping LINEAR_CAP saturation: {LINEAR_CAP_GRID}")
print("==============================")

for saturation in LINEAR_CAP_GRID:
    result = run_candidate(
        "linear_cap", saturation,
        lambda n, sat=saturation: np.minimum(n / sat, 1) * 100
    )
    results.append(result)
    print(
        f"linear_cap, saturation={saturation}: "
        f"avg_conf={result['avg_confidence']:.1f}, "
        f"extreme_pitchers={result['n_pitchers_at_extreme']}, "
        f"primary R2={result['primary_cv_r2']:.3f}, "
        f"secondary R2={result['secondary_cv_r2']:.3f}, "
        f"vel_coef={result['vel_coef_primary']:.4f}, "
        f"sign_agrees={result['sign_agrees']}"
    )

print("\n==============================")
print(f"Sweeping HYPERBOLIC k: {HYPERBOLIC_K_GRID}")
print("==============================")

for k in HYPERBOLIC_K_GRID:
    result = run_candidate(
        "hyperbolic", k,
        lambda n, k_val=k: (n / (n + k_val)) * 100
    )
    results.append(result)
    print(
        f"hyperbolic, k={k}: "
        f"avg_conf={result['avg_confidence']:.1f}, "
        f"extreme_pitchers={result['n_pitchers_at_extreme']}, "
        f"primary R2={result['primary_cv_r2']:.3f}, "
        f"secondary R2={result['secondary_cv_r2']:.3f}, "
        f"vel_coef={result['vel_coef_primary']:.4f}, "
        f"sign_agrees={result['sign_agrees']}"
    )


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

print("\n\n==============================")
print("Sweep Summary")
print("==============================\n")
print(results_df.to_string(index=False))

best_primary = results_df.loc[results_df["primary_cv_r2"].idxmax()]
best_secondary = results_df.loc[results_df["secondary_cv_r2"].idxmax()]

print(
    f"\nBest primary-target CV R2: {best_primary['form']} "
    f"(param={best_primary['parameter']}, R2={best_primary['primary_cv_r2']:.3f}, "
    f"extreme_pitchers={best_primary['n_pitchers_at_extreme']})"
)
print(
    f"Best secondary-target CV R2: {best_secondary['form']} "
    f"(param={best_secondary['parameter']}, R2={best_secondary['secondary_cv_r2']:.3f}, "
    f"extreme_pitchers={best_secondary['n_pitchers_at_extreme']})"
)

# guardrail: warn if the best-R2 candidate also shows a spike in
# extreme-valued pitchers relative to the CURRENT default (linear_cap,
# 4) -- particularly important here given real-data evidence that
# disabling Velocity's shrinkage entirely produced a 0.481 pitch-count
# correlation (see 21_velocity_component.py's history)
baseline = results_df[
    (results_df["form"] == "linear_cap") & (results_df["parameter"] == 4)
]
if len(baseline) > 0:
    baseline_extreme = baseline.iloc[0]["n_pitchers_at_extreme"]
    if best_primary["n_pitchers_at_extreme"] > baseline_extreme * 1.5:
        print(
            f"\nWARNING: the best primary-R2 candidate shows "
            f"{best_primary['n_pitchers_at_extreme']} pitchers at the "
            f"0/100 extremes, notably more than the current default's "
            f"{baseline_extreme} -- this may be the same 'R2 improved "
            f"because shrinkage got weaker, not because the formula "
            f"got better' trap seen with Velocity's own confidence-"
            f"removal experiment. Inspect before adopting."
        )

# secondary guardrail: check whether the winning parameter is at the
# EDGE of the tested range rather than a genuine interior optimum
grid_edges = {
    "linear_cap": (min(LINEAR_CAP_GRID), max(LINEAR_CAP_GRID)),
    "hyperbolic": (min(HYPERBOLIC_K_GRID), max(HYPERBOLIC_K_GRID))
}
edge_min, edge_max = grid_edges[best_primary["form"]]
if best_primary["parameter"] in (edge_min, edge_max):
    print(
        f"\nNOTE: the best primary-R2 candidate ({best_primary['form']}, "
        f"param={best_primary['parameter']}) sits at the EDGE of the "
        f"tested grid, not an interior optimum -- consider extending "
        f"the grid further before trusting this as a real peak."
    )

stable_sign = results_df[results_df["sign_agrees"]]
print(
    f"\n{len(stable_sign)} of {len(results_df)} candidates showed "
    f"sign-stable Velocity coefficients across both targets."
)
