"""
23_movement_dampening_sweep.py

Purpose:
--------
09_movement_component.py's DAMPENING_FACTOR=0.6 (how much of the
n_pitch_types trend gets removed from Movement's four confounded
features before PCA) was a reasonable-sounding round number, not a
data-driven choice. This sweeps a grid of candidate values and runs
EACH one through the same correlation-check + confidence-weighted
CV-Ridge fit used in 10_paom_score.py, so the choice is grounded in
which dampening level actually produces the most genuinely
predictive Movement score against real outcomes -- the same
philosophy already used everywhere else in PAOM (Ridge's own alpha
is chosen this way; every component's paom_score weight is learned,
not assigned).

Requires PAOM_movement_component.csv to already have the *_raw
columns (horizontal_coverage_raw, vertical_coverage_raw, hull_area_
raw, centroid_dispersion_raw) -- i.e. the CURRENT version of
09_movement_component.py must have been run at least once already.
Re-derives movement_score from those raw values at each candidate
dampening level (cheap -- no need to recompute geometry from
master_pitch_table_2025.csv, since the raw geometry doesn't change
with dampening, only the adjustment step does).

Output:
-------
Printed sweep table only -- no file saved. Once you've picked a
winner, set DAMPENING_FACTOR to that value in
09_movement_component.py and re-run the real pipeline.
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

MOVEMENT_FILE = "PAOM_movement_component.csv"
EFFECTIVENESS_FILE = "PAOM_effectiveness_component.csv"
COMMAND_FILE = "PAOM_command_component.csv"
VELOCITY_FILE = "PAOM_velocity_component.csv"
MASTER_FILE = "master_pitch_table_2025.csv"
EVENTS_FILE = "clean_statcast_2025.csv"

DAMPENING_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

CONFOUNDED_FEATURES = [
    "horizontal_coverage",
    "vertical_coverage",
    "hull_area",
    "centroid_dispersion"
]

MIN_PA_FOR_SECONDARY_TARGET = 50
N_CV_FOLDS = 5
RANDOM_SEED = 42


# ============================================================
# LOAD RAW MOVEMENT GEOMETRY (dampening-independent)
# ============================================================

print("Loading movement component (raw geometry)...")

movement_raw = pd.read_csv(MOVEMENT_FILE)

required_raw_cols = [f"{f}_raw" for f in CONFOUNDED_FEATURES] + [
    "avg_nn_distance", "n_pitch_types", "confidence_score", "player_name"
]

missing = [c for c in required_raw_cols if c not in movement_raw.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}. This script needs "
        f"the CURRENT 09_movement_component.py's *_raw columns -- "
        f"re-run it at least once first if PAOM_movement_component.csv "
        f"predates the dampening adjustment."
    )

print(f"Loaded {len(movement_raw):,} pitchers")


# ============================================================
# LOAD THE OTHER SCORED COMPONENTS (unaffected by dampening)
# ============================================================

print("\nLoading other components...")

effectiveness = pd.read_csv(EFFECTIVENESS_FILE)

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

command = pd.read_csv(COMMAND_FILE)
command["command_usage"] = (
    command["pitches"] / command.groupby("player_name")["pitches"].transform("sum")
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

velocity = pd.read_csv(VELOCITY_FILE)[
    ["player_name", "trusted_velocity_score", "confidence_score"]
].rename(columns={"confidence_score": "velocity_confidence"})

print(f"Effectiveness: {len(eff_pitcher):,}, Command: {len(cmd_pitcher):,}, Velocity: {len(velocity):,}")


# ============================================================
# BUILD VALIDATION TARGETS (same as 10_paom_score.py)
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

print("\n==============================")
print(f"Sweeping DAMPENING_FACTOR: {DAMPENING_GRID}")
print("==============================")

results = []

for dampening in DAMPENING_GRID:

    movement_df = movement_raw.copy()

    for feature in CONFOUNDED_FEATURES:
        x = movement_df["n_pitch_types"].values.astype(float)
        y = movement_df[f"{feature}_raw"].values.astype(float)

        slope, intercept = np.polyfit(x, y, 1)
        predicted_trend = intercept + slope * x
        feature_mean = y.mean()

        movement_df[feature] = y - dampening * (predicted_trend - feature_mean)

    pca_features = CONFOUNDED_FEATURES + ["avg_nn_distance"]

    X = movement_df[pca_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=len(pca_features))
    components = pca.fit_transform(X_scaled)

    pc1_explained = pca.explained_variance_ratio_[0]

    movement_df["raw_movement_score"] = components[:, 0]

    if movement_df["raw_movement_score"].corr(movement_df["horizontal_coverage"]) < 0:
        movement_df["raw_movement_score"] *= -1

    movement_df["movement_score"] = percentile_rank(movement_df["raw_movement_score"])

    league_avg = movement_df["movement_score"].mean()
    movement_df["trusted_movement_score"] = (
        movement_df["movement_score"] * (movement_df["confidence_score"] / 100)
        + league_avg * (1 - movement_df["confidence_score"] / 100)
    )

    movement_score_corr_with_pitch_count = movement_df["trusted_movement_score"].corr(
        movement_df["n_pitch_types"]
    )

    # merge into the full candidate set
    combined = eff_pitcher.merge(
        movement_df[["player_name", "trusted_movement_score", "confidence_score"]]
        .rename(columns={"confidence_score": "movement_confidence"}),
        on="player_name", how="inner"
    )
    combined = combined.merge(cmd_pitcher, on="player_name", how="inner")
    combined = combined.merge(velocity, on="player_name", how="inner")

    velocity_movement_corr = combined["trusted_movement_score"].corr(
        combined["trusted_velocity_score"]
    )

    corr_cols = [
        "pitcher_effectiveness", "trusted_movement_score",
        "pitcher_command", "trusted_velocity_score"
    ]

    # re-percentile within this candidate's population (same fix as 10_paom_score.py)
    for component in corr_cols:
        combined[component] = combined[component].rank(pct=True) * 100

    primary_df = combined.merge(primary_target_df, on="player_name", how="inner")
    secondary_df = combined.merge(secondary_target_df, on="player_name", how="inner")

    primary_weight = primary_df[
        ["effectiveness_confidence", "movement_confidence", "command_confidence", "velocity_confidence"]
    ].min(axis=1) / 100
    secondary_weight = secondary_df[
        ["effectiveness_confidence", "movement_confidence", "command_confidence", "velocity_confidence"]
    ].min(axis=1) / 100

    primary_r2, primary_std, primary_coefs = cv_ridge_fit(
        primary_df[corr_cols], primary_df["primary_target"], sample_weight=primary_weight
    )
    secondary_r2, secondary_std, secondary_coefs = cv_ridge_fit(
        secondary_df[corr_cols], secondary_df["secondary_target"], sample_weight=secondary_weight
    )

    movement_coef_primary = primary_coefs.set_index("component").loc[
        "trusted_movement_score", "coefficient"
    ]
    movement_coef_secondary = secondary_coefs.set_index("component").loc[
        "trusted_movement_score", "coefficient"
    ]

    results.append({
        "dampening_factor": dampening,
        "pc1_explained_variance": pc1_explained,
        "movement_score_corr_n_pitch_types": movement_score_corr_with_pitch_count,
        "movement_velocity_corr": velocity_movement_corr,
        "primary_cv_r2": primary_r2,
        "secondary_cv_r2": secondary_r2,
        "movement_coef_primary": movement_coef_primary,
        "movement_coef_secondary": movement_coef_secondary,
        "movement_sign_agrees": np.sign(movement_coef_primary) == np.sign(movement_coef_secondary)
    })

    print(
        f"\nDampening={dampening:.1f}: "
        f"corr(movement, n_pitch_types)={movement_score_corr_with_pitch_count:.3f}, "
        f"corr(movement, velocity)={velocity_movement_corr:.3f}, "
        f"primary R2={primary_r2:.3f}, secondary R2={secondary_r2:.3f}, "
        f"movement coef (primary/secondary)={movement_coef_primary:.4f}/{movement_coef_secondary:.4f}, "
        f"sign agrees={np.sign(movement_coef_primary) == np.sign(movement_coef_secondary)}"
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
    f"\nBest primary-target CV R2: dampening={best_primary['dampening_factor']:.1f} "
    f"(R2={best_primary['primary_cv_r2']:.3f})"
)
print(
    f"Best secondary-target CV R2: dampening={best_secondary['dampening_factor']:.1f} "
    f"(R2={best_secondary['secondary_cv_r2']:.3f})"
)

stable_sign = results_df[results_df["movement_sign_agrees"]]

if len(stable_sign) > 0:
    print(
        f"\nDampening values where Movement's coefficient sign agrees "
        f"across both targets: {stable_sign['dampening_factor'].tolist()}"
    )
else:
    print(
        "\nNo dampening value in this grid produced a stable "
        "(sign-agreeing) Movement coefficient across both targets."
    )
