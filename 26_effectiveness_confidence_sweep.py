"""
26_effectiveness_confidence_sweep.py

Purpose:
--------
Effectiveness's confidence formula (min(swings/500, 1) -- a linear
ramp that hard-caps at 500 swings) was a reasonable-sounding round
number, never actually tested against alternatives. This sweeps
BOTH the FORM of the formula and each form's own parameter,
evaluated through the same real downstream validation used
everywhere else in this pipeline (confidence-weighted CV-Ridge fit
against both validation targets) -- not just picking a number
within an assumed shape, but testing whether the shape itself is
right.

Two forms tested:
------------------
1. LINEAR_CAP (current): confidence = min(n / saturation, 1) * 100.
   Ramps linearly, then hard-caps at 100% once saturation is
   reached -- treats "just cleared the saturation point" identically
   to "way beyond it," which isn't grounded in how estimation
   uncertainty actually behaves.

2. HYPERBOLIC (n / (n+k)): the standard sabermetric "stabilization
   point" form -- derived from actual reliability theory (true
   variance vs. error variance), not chosen for convenience. Never
   fully caps at 100% (always leaves a small residual shrinkage),
   and ramps faster early / more gradually later, better matching
   how standard error actually shrinks with sample size (~1/sqrt(n),
   not linearly). k is interpretable as "the swing count at which
   confidence reaches 50%."

For each (form, parameter) candidate:
- Recompute confidence_score using that form
- Recompute trusted_effectiveness with the SAME blend formula,
  just a different confidence input
- Roll up to pitcher level (usage-weighted, matching
  10_paom_score.py's existing logic)
- Merge with Command/Movement/Velocity (held at their CURRENT/
  default confidence formulas -- this sweep only touches
  Effectiveness) + both validation targets
- Re-percentile within the paom-eligible population
- Run the same confidence-weighted CV-Ridge fit against both targets

Reported per candidate:
- CV R2 on both targets (the primary signal)
- Effectiveness's own coefficient and sign-agreement (stability)
- A DEGENERACY GUARDRAIL: how many pitchers land within 0.5 of the
  exact 0/100 extremes after rollup -- directly modeling the
  "Maton hits exactly 100.000000" failure mode already seen twice
  in this pipeline. A candidate that maximizes R2 by effectively
  disabling shrinkage (near-0 saturation, or a tiny k) should be
  caught here even if R2 alone looks good, the same lesson learned
  from the Velocity confidence-removal experiment.

Output:
-------
Printed sweep table only -- no file saved. Once a winner is picked,
update 07d_final_effectiveness.py's actual formula to match.
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

EFFECTIVENESS_FILE = "PAOM_effectiveness_component.csv"
COMMAND_FILE = "PAOM_command_component.csv"
MOVEMENT_FILE = "PAOM_movement_component.csv"
VELOCITY_FILE = "PAOM_velocity_component.csv"
MASTER_FILE = "master_pitch_table_2025.csv"
EVENTS_FILE = "clean_statcast_2025.csv"

LINEAR_CAP_GRID = [200, 350, 500, 650, 800, 1000]
HYPERBOLIC_K_GRID = [100, 200, 300, 500, 750, 1000]

MIN_PA_FOR_SECONDARY_TARGET = 50
N_CV_FOLDS = 5
RANDOM_SEED = 42

DEGENERACY_EXTREME_THRESHOLD = 0.5  # pitchers within this of 0 or 100


# ============================================================
# LOAD RAW EFFECTIVENESS DATA (form-independent)
# ============================================================

print("Loading effectiveness component (pre-shrinkage values)...")

effectiveness_raw = pd.read_csv(EFFECTIVENESS_FILE)

required_cols = ["player_name", "pitch_type", "effectiveness_score", "swings", "usage"]
missing = [c for c in required_cols if c not in effectiveness_raw.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}. This script needs "
        f"07d_final_effectiveness.py's output with effectiveness_score "
        f"(pre-shrinkage) and swings preserved."
    )

print(f"Loaded {len(effectiveness_raw):,} pitcher x pitch-type rows")


# ============================================================
# LOAD OTHER COMPONENTS (held fixed -- this sweep only touches Effectiveness)
# ============================================================

print("\nLoading other components (held at current defaults)...")

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
velocity = pd.read_csv(VELOCITY_FILE)[["player_name", "trusted_velocity_score"]]

print(f"Command: {len(cmd_pitcher):,}, Movement: {len(movement):,}, Velocity: {len(velocity):,}")


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
    confidence_fn: takes the swings Series, returns a confidence
    Series (0-100).
    """
    eff = effectiveness_raw.copy()

    eff["confidence_score"] = confidence_fn(eff["swings"])

    league_average = eff["effectiveness_score"].mean()
    eff["trusted_effectiveness"] = (
        eff["effectiveness_score"] * (eff["confidence_score"] / 100)
        + league_average * (1 - eff["confidence_score"] / 100)
    )

    eff_pitcher = (
        eff.groupby("player_name")
        .apply(lambda x: pd.Series({
            "pitcher_effectiveness": np.average(
                x["trusted_effectiveness"], weights=x["usage"]
            ),
            "effectiveness_confidence": np.average(
                x["confidence_score"], weights=x["usage"]
            )
        }))
        .reset_index()
    )

    # degeneracy guardrail: how many pitchers land within
    # DEGENERACY_EXTREME_THRESHOLD of the exact 0/100 extremes
    n_extreme = (
        (eff_pitcher["pitcher_effectiveness"] <= DEGENERACY_EXTREME_THRESHOLD)
        | (eff_pitcher["pitcher_effectiveness"] >= 100 - DEGENERACY_EXTREME_THRESHOLD)
    ).sum()

    combined = eff_pitcher.merge(movement, on="player_name", how="inner")
    combined = combined.merge(cmd_pitcher, on="player_name", how="inner")
    combined = combined.merge(velocity, on="player_name", how="inner")

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

    eff_coef_primary = primary_coefs.set_index("component").loc[
        "pitcher_effectiveness", "coefficient"
    ]
    eff_coef_secondary = secondary_coefs.set_index("component").loc[
        "pitcher_effectiveness", "coefficient"
    ]

    return {
        "form": form_name,
        "parameter": param_value,
        "avg_confidence": eff_pitcher["effectiveness_confidence"].mean(),
        "n_pitchers_at_extreme": n_extreme,
        "primary_cv_r2": primary_r2,
        "secondary_cv_r2": secondary_r2,
        "eff_coef_primary": eff_coef_primary,
        "eff_coef_secondary": eff_coef_secondary,
        "sign_agrees": np.sign(eff_coef_primary) == np.sign(eff_coef_secondary)
    }


results = []

print("\n==============================")
print(f"Sweeping LINEAR_CAP saturation: {LINEAR_CAP_GRID}")
print("==============================")

for saturation in LINEAR_CAP_GRID:
    result = run_candidate(
        "linear_cap", saturation,
        lambda swings, sat=saturation: np.minimum(swings / sat, 1) * 100
    )
    results.append(result)
    print(
        f"linear_cap, saturation={saturation}: "
        f"avg_conf={result['avg_confidence']:.1f}, "
        f"extreme_pitchers={result['n_pitchers_at_extreme']}, "
        f"primary R2={result['primary_cv_r2']:.3f}, "
        f"secondary R2={result['secondary_cv_r2']:.3f}, "
        f"eff_coef={result['eff_coef_primary']:.4f}, "
        f"sign_agrees={result['sign_agrees']}"
    )

print("\n==============================")
print(f"Sweeping HYPERBOLIC k: {HYPERBOLIC_K_GRID}")
print("==============================")

for k in HYPERBOLIC_K_GRID:
    result = run_candidate(
        "hyperbolic", k,
        lambda swings, k_val=k: (swings / (swings + k_val)) * 100
    )
    results.append(result)
    print(
        f"hyperbolic, k={k}: "
        f"avg_conf={result['avg_confidence']:.1f}, "
        f"extreme_pitchers={result['n_pitchers_at_extreme']}, "
        f"primary R2={result['primary_cv_r2']:.3f}, "
        f"secondary R2={result['secondary_cv_r2']:.3f}, "
        f"eff_coef={result['eff_coef_primary']:.4f}, "
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
# 500) -- the same "R2 went up because shrinkage got weaker" trap
# already seen with Velocity's confidence-removal experiment
baseline = results_df[
    (results_df["form"] == "linear_cap") & (results_df["parameter"] == 500)
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
            f"got better' trap seen with Velocity's confidence-removal "
            f"experiment. Inspect before adopting."
        )

stable_sign = results_df[results_df["sign_agrees"]]
print(
    f"\n{len(stable_sign)} of {len(results_df)} candidates showed "
    f"sign-stable Effectiveness coefficients across both targets."
)
