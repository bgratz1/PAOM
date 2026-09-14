"""
31_effectiveness_joint_cutoff_sweep.py

Purpose:
--------
Follow-up to 30_effectiveness_cutoff_sweep.py's one-parameter-at-a-
time approach, which held MIN_SWINGS at its OLD default (50) while
sweeping MIN_PITCHES, and held MIN_PITCHES at its OLD default (75)
while sweeping MIN_SWINGS. That found MIN_PITCHES=60 (with MIN_
SWINGS=50) a real improvement, and found MIN_SWINGS wasn't binding
AT MIN_PITCHES=75 specifically -- but never tested whether MIN_
SWINGS becomes newly binding at the NEW, lower MIN_PITCHES=60, or
whether some other joint combination does better than either
one-at-a-time result.

This tests the FULL 2D grid -- every (MIN_PITCHES, MIN_SWINGS)
combination -- to find the genuinely best joint pair, not just the
best value of one holding the other at whatever the old default
happened to be.

Same underlying re-run of the full Effectiveness pipeline per
candidate as 30 (a cutoff changes which rows feed the regression
itself), same guardrails (coverage vs. current default, degeneracy,
rate-instability), same downstream confidence-weighted CV-Ridge
validation.

Output:
-------
Printed sweep table (full grid) plus pivoted heatmap-style views for
R2 and pitcher count, for readability across a 2D grid. No file
saved -- once a decision is made, update 07d_final_effectiveness.py's
MIN_PITCHES/MIN_SWINGS to match.
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

STATCAST_FILE = "clean_statcast_2025.csv"
COMMAND_FILE = "PAOM_command_component.csv"
MOVEMENT_FILE = "PAOM_movement_component.csv"
VELOCITY_FILE = "PAOM_velocity_component.csv"
MASTER_FILE = "master_pitch_table_2025.csv"
EVENTS_FILE = "clean_statcast_2025.csv"

FEATURES = ["csw_rate", "chase_rate", "hard_hit_rate", "gb_rate"]
TARGET = "xwoba"

CONFIDENCE_K = 200  # already-validated hyperbolic parameter

# current (as of this sweep) defaults: MIN_PITCHES=60, MIN_SWINGS=50
# grid centered on the promising MIN_PITCHES=60 result, extended in
# both directions; MIN_SWINGS grid extended both directions from 50
# to properly test whether it becomes binding at lower pitch floors
MIN_PITCHES_GRID = [40, 50, 60, 75, 90]
MIN_SWINGS_GRID = [20, 30, 40, 50, 65]

MIN_PA_FOR_SECONDARY_TARGET = 50
N_CV_FOLDS = 5
RANDOM_SEED = 42

DEGENERACY_EXTREME_THRESHOLD = 0.5  # pitchers within this of 0 or 100


# ============================================================
# LOAD RAW STATCAST DATA (form-independent, filtered once for
# cleanliness -- NOT for the cutoffs themselves, those vary per candidate)
# ============================================================

print("Loading raw Statcast data...")

raw = pd.read_csv(STATCAST_FILE)

required_flags = ["is_swing", "is_whiff", "is_called_strike", "is_hard_hit", "is_chase"]
missing_flags = [c for c in required_flags if c not in raw.columns]
if missing_flags:
    raise ValueError(
        f"Missing required columns: {missing_flags}. This script needs "
        f"the same raw Statcast input 07d_final_effectiveness.py uses, "
        f"not a pre-aggregated file."
    )

raw = raw[raw["pitch_type"].notna()]
raw = raw[raw["estimated_woba_using_speedangle"].notna()]

print(f"Rows after cleaning: {len(raw):,}")

print("\nAggregating pitcher x pitch type (once, before any cutoff)...")

pitch_summary_full = (
    raw.groupby(["player_name", "pitcher", "pitch_type", "pitch_name"])
    .agg(
        pitches=("pitch_type", "count"),
        xwoba=("estimated_woba_using_speedangle", "mean"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        called_strikes=("is_called_strike", "sum"),
        hard_hits=("is_hard_hit", "sum"),
        chase_swings=("is_chase", "sum"),
        outside_zone=("is_outside_zone", "sum"),
        balls_in_play=("is_in_play", "sum"),
        ground_balls=("launch_angle", lambda x: (x < 10).sum())
    )
    .reset_index()
)

pitch_summary_full["whiff_rate"] = pitch_summary_full["whiffs"] / pitch_summary_full["swings"]
pitch_summary_full["csw_rate"] = (
    (pitch_summary_full["whiffs"] + pitch_summary_full["called_strikes"])
    / pitch_summary_full["pitches"]
)
pitch_summary_full["chase_rate"] = (
    pitch_summary_full["chase_swings"] / pitch_summary_full["outside_zone"]
)
pitch_summary_full["hard_hit_rate"] = pitch_summary_full["hard_hits"] / pitch_summary_full["pitches"]
pitch_summary_full["gb_rate"] = pitch_summary_full["ground_balls"] / pitch_summary_full["balls_in_play"]

pitch_summary_full = pitch_summary_full.replace([np.inf, -np.inf], np.nan).dropna(
    subset=FEATURES + [TARGET]
)

print(f"Rows available across all candidate cutoffs: {len(pitch_summary_full):,}")


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

def run_candidate(param_name, param_value, min_pitches, min_swings):
    ps = pitch_summary_full[
        (pitch_summary_full["pitches"] >= min_pitches)
        & (pitch_summary_full["swings"] >= min_swings)
    ].copy()

    n_rows = len(ps)
    if n_rows < 20:
        return None  # too few rows to fit anything meaningful

    X = ps[FEATURES]
    y = ps[TARGET]

    # internal regression CV R2 -- how well the 4 rate features
    # predict xwOBA on just the rows that clear THIS candidate's cutoff
    internal_model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 50)))
    ])
    internal_cv = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    internal_cv_scores = cross_val_score(internal_model, X, y, cv=internal_cv, scoring="r2")
    internal_r2 = internal_cv_scores.mean()

    internal_model.fit(X, y)
    ps["predicted_xwoba"] = internal_model.predict(X)
    ps["raw_effectiveness_index"] = -ps["predicted_xwoba"]
    ps["effectiveness_score"] = percentile_rank(ps["raw_effectiveness_index"])

    ps["confidence_score"] = (ps["swings"] / (ps["swings"] + CONFIDENCE_K)) * 100

    league_average = ps["effectiveness_score"].mean()
    ps["trusted_effectiveness"] = (
        ps["effectiveness_score"] * (ps["confidence_score"] / 100)
        + league_average * (1 - ps["confidence_score"] / 100)
    )

    ps["usage"] = ps["pitches"] / ps.groupby("player_name")["pitches"].transform("sum")

    n_extreme = (
        (ps["trusted_effectiveness"] <= DEGENERACY_EXTREME_THRESHOLD)
        | (ps["trusted_effectiveness"] >= 100 - DEGENERACY_EXTREME_THRESHOLD)
    ).sum()

    # RATE-INSTABILITY DIAGNOSTIC: a risk distinct from downstream R2.
    # hard_hit_rate/gb_rate are computed from balls_in_play, and
    # chase_rate from outside_zone -- both can be thin even when
    # pitches/swings clear the cutoff (a pitch type can accumulate
    # plenty of pitches and swings while producing few actual balls
    # in play). A rate computed from a handful of chances is
    # unstable as a number, independent of anything the downstream
    # confidence-weighted fit protects against. Track the 10th
    # percentile (not the bare minimum, which one outlier row could
    # distort) of each denominator to see how thin the population's
    # rate calculations are getting as cutoffs loosen.
    p10_balls_in_play = ps["balls_in_play"].quantile(0.10)
    p10_outside_zone = ps["outside_zone"].quantile(0.10)

    eff_pitcher = (
        ps.groupby("player_name")
        .apply(lambda g: np.average(g["trusted_effectiveness"], weights=g["usage"]))
        .reset_index(name="pitcher_effectiveness")
    )

    combined = eff_pitcher.merge(movement, on="player_name", how="inner")
    combined = combined.merge(cmd_pitcher, on="player_name", how="inner")
    combined = combined.merge(velocity, on="player_name", how="inner")

    n_pitchers = len(combined)

    corr_cols = [
        "pitcher_effectiveness", "pitcher_command",
        "trusted_velocity_score", "trusted_movement_score"
    ]

    for component in corr_cols:
        combined[component] = percentile_rank(combined[component])

    primary_df = combined.merge(primary_target_df, on="player_name", how="inner")
    secondary_df = combined.merge(secondary_target_df, on="player_name", how="inner")

    if len(primary_df) < 20 or len(secondary_df) < 20:
        return None

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
        "param": param_name,
        "value": param_value,
        "min_pitches": min_pitches,
        "min_swings": min_swings,
        "n_rows": n_rows,
        "n_pitchers": n_pitchers,
        "n_extreme": n_extreme,
        "p10_balls_in_play": p10_balls_in_play,
        "p10_outside_zone": p10_outside_zone,
        "internal_r2": internal_r2,
        "primary_cv_r2": primary_r2,
        "secondary_cv_r2": secondary_r2,
        "eff_coef_primary": eff_coef_primary,
        "eff_coef_secondary": eff_coef_secondary,
        "sign_agrees": np.sign(eff_coef_primary) == np.sign(eff_coef_secondary)
    }


results = []

print("\n==============================")
print(
    f"Sweeping full grid: MIN_PITCHES {MIN_PITCHES_GRID} x "
    f"MIN_SWINGS {MIN_SWINGS_GRID} ({len(MIN_PITCHES_GRID) * len(MIN_SWINGS_GRID)} combinations)"
)
print("==============================")

for min_pitches in MIN_PITCHES_GRID:
    for min_swings in MIN_SWINGS_GRID:
        label = f"pitches={min_pitches},swings={min_swings}"
        result = run_candidate(label, label, min_pitches, min_swings)
        if result is None:
            print(f"{label}: too few rows, skipped")
            continue
        results.append(result)
        print(
            f"{label}: rows={result['n_rows']}, "
            f"pitchers={result['n_pitchers']}, extreme={result['n_extreme']}, "
            f"internal_R2={result['internal_r2']:.3f}, "
            f"primary_R2={result['primary_cv_r2']:.3f}, "
            f"secondary_R2={result['secondary_cv_r2']:.3f}, "
            f"eff_coef={result['eff_coef_primary']:.4f}, "
            f"sign_agrees={result['sign_agrees']}, "
            f"p10_BIP={result['p10_balls_in_play']:.1f}, "
            f"p10_outzone={result['p10_outside_zone']:.1f}"
        )


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

print("\n\n==============================")
print("Sweep Summary (full table)")
print("==============================\n")
print(
    results_df[[
        "min_pitches", "min_swings", "n_rows", "n_pitchers", "n_extreme",
        "internal_r2", "primary_cv_r2", "secondary_cv_r2",
        "eff_coef_primary", "sign_agrees", "p10_balls_in_play", "p10_outside_zone"
    ]].to_string(index=False)
)

print("\n\n==============================")
print("Primary-target CV R2 (rows=MIN_PITCHES, cols=MIN_SWINGS)")
print("==============================\n")
print(
    results_df.pivot(index="min_pitches", columns="min_swings", values="primary_cv_r2")
    .round(3).to_string()
)

print("\n\n==============================")
print("Pitcher count (rows=MIN_PITCHES, cols=MIN_SWINGS)")
print("==============================\n")
print(
    results_df.pivot(index="min_pitches", columns="min_swings", values="n_pitchers")
    .to_string()
)

# CURRENT DEFAULT as of this sweep: MIN_PITCHES=60, MIN_SWINGS=50
# (07d_final_effectiveness.py's live settings after the prior sweep)
baseline = results_df[
    (results_df["min_pitches"] == 60) & (results_df["min_swings"] == 50)
]

if len(baseline) > 0:
    b = baseline.iloc[0]
    print(
        f"\nCurrent default (MIN_PITCHES=60, MIN_SWINGS=50): "
        f"rows={b['n_rows']}, pitchers={b['n_pitchers']}, "
        f"primary_R2={b['primary_cv_r2']:.3f}, "
        f"secondary_R2={b['secondary_cv_r2']:.3f}"
    )

best_primary = results_df.loc[results_df["primary_cv_r2"].idxmax()]
best_secondary = results_df.loc[results_df["secondary_cv_r2"].idxmax()]

print(
    f"\nBest primary-target CV R2: MIN_PITCHES={best_primary['min_pitches']}, "
    f"MIN_SWINGS={best_primary['min_swings']} "
    f"(R2={best_primary['primary_cv_r2']:.3f}, pitchers={best_primary['n_pitchers']}, "
    f"extreme={best_primary['n_extreme']})"
)
print(
    f"Best secondary-target CV R2: MIN_PITCHES={best_secondary['min_pitches']}, "
    f"MIN_SWINGS={best_secondary['min_swings']} "
    f"(R2={best_secondary['secondary_cv_r2']:.3f}, pitchers={best_secondary['n_pitchers']}, "
    f"extreme={best_secondary['n_extreme']})"
)

if len(baseline) > 0:
    b = baseline.iloc[0]
    if best_primary["n_pitchers"] < b["n_pitchers"] * 0.7:
        print(
            f"\nNOTE: the best primary-R2 candidate covers "
            f"{best_primary['n_pitchers']} pitchers, notably fewer than "
            f"the current default's {b['n_pitchers']} -- this may be an "
            f"R2 improvement bought by cutting the population down to "
            f"only the most reliable, easiest-to-predict pitchers "
            f"(survivorship bias), not a genuinely better cutoff. "
            f"Weigh the coverage loss against the accuracy gain before "
            f"adopting."
        )
    if best_primary["n_extreme"] > b["n_extreme"] * 1.5:
        print(
            f"\nWARNING: the best primary-R2 candidate shows "
            f"{best_primary['n_extreme']} pitch-types at the 0/100 "
            f"extremes, notably more than the current default's "
            f"{b['n_extreme']}."
        )
    RATE_INSTABILITY_FLOOR = 15  # a rate from fewer than this many
                                  # chances is unstable as a number,
                                  # independent of anything downstream
    if (
        best_primary["p10_balls_in_play"] < RATE_INSTABILITY_FLOOR
        or best_primary["p10_outside_zone"] < RATE_INSTABILITY_FLOOR
    ):
        print(
            f"\nWARNING: the best primary-R2 candidate's bottom 10% of "
            f"pitch-type rows have as few as "
            f"{best_primary['p10_balls_in_play']:.0f} balls in play "
            f"and {best_primary['p10_outside_zone']:.0f} out-of-zone "
            f"pitches -- hard_hit_rate/gb_rate/chase_rate computed "
            f"from that few chances are unstable numbers, a risk "
            f"distinct from (and not caught by) the R2 and degeneracy "
            f"checks above. Consider this even if R2 and coverage "
            f"both look favorable."
        )

# 2D edge check: is the best combination sitting on the boundary of
# EITHER grid dimension, not just one?
pitches_edge = best_primary["min_pitches"] in (min(MIN_PITCHES_GRID), max(MIN_PITCHES_GRID))
swings_edge = best_primary["min_swings"] in (min(MIN_SWINGS_GRID), max(MIN_SWINGS_GRID))
if pitches_edge or swings_edge:
    edge_dims = []
    if pitches_edge:
        edge_dims.append(f"MIN_PITCHES={best_primary['min_pitches']}")
    if swings_edge:
        edge_dims.append(f"MIN_SWINGS={best_primary['min_swings']}")
    print(
        f"\nNOTE: the best primary-R2 candidate sits at the EDGE of the "
        f"tested grid on {' and '.join(edge_dims)} -- consider extending "
        f"further in that direction."
    )

stable_sign = results_df[results_df["sign_agrees"]]
print(
    f"\n{len(stable_sign)} of {len(results_df)} candidates showed "
    f"sign-stable Effectiveness coefficients across both targets."
)
