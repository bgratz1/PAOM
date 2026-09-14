"""
10_paom_score.py

Purpose:
--------
Combine PAOM's components into a single pitcher-level PAOM score.

Method:
-------
1. Aggregate Effectiveness and Command (both pitch-type level) to
   pitcher level, usage-weighted -- unlike Movement, overall
   arsenal quality should reflect what actually gets thrown, not
   just what's structurally available.

2. Merge with Movement's single pitcher-level score (see
   09_movement_component.py -- all five raw geometric features
   combined via one PCA, no grouping, no orthogonalization).

3. Build TWO independent validation targets (see "Why two targets"
   below), both sign-oriented so higher = better for the pitcher.

4. Check Effectiveness, Command, Velocity, and Movement aren't
   secretly redundant with each other before combining them.

5. Learn combination weights via Ridge, evaluated with K-FOLD
   CROSS-VALIDATION rather than a single in-sample fit -- an
   in-sample R2 only tells you how well the model fits the data
   it was fit on, not how well it'd generalize. This mirrors the
   CV pattern already used in 07d and 08.

6. Compare the weights/R2 learned from BOTH validation targets.
   If they tell a similar story, that's real evidence the primary
   result isn't just circularity (see below). If they diverge,
   that's reported honestly, not hidden.

7. Normalize the Ridge-learned weights to sum to 1, then apply
   them as a literal WEIGHTED AVERAGE of the three component
   percentile scores -> paom_score, directly bounded 0-100 and
   reconstructible by hand from the three visible component
   scores (see Step 7's inline docstring for why this replaced an
   earlier Ridge-projection-then-percentile-rank approach).

Why two validation targets (addressing a real methodology gap):
------------------------------------------------------------------
Effectiveness's own regression (07d) predicts xwOBA from csw_rate,
chase_rate, hard_hit_rate, and gb_rate. The primary validation
target below (run value per 100 pitches) is a DIFFERENT quantity --
it captures full plate-appearance outcomes (walks, strikeouts,
extra bases) that xwOBA alone doesn't -- but it's still downstream
of much of the same underlying pitch-quality signal Effectiveness
was fit on. That means some of Effectiveness's apparent predictive
power here is expected overlap with what it already "knows," not
fully independent confirmation.

Rather than assume how much of that concern is real, this script
adds a SECOND validation target -- K% - BB% (strikeout rate minus
walk rate), computed directly from raw plate-appearance outcomes.
This is a standard, independent pitcher-quality metric that
Effectiveness's regression never targeted. If the combination's
weights and cross-validated R2 look similar under both targets,
that's genuine evidence the primary result holds up. If they
diverge, that's useful information about how much of the original
result was target-specific.

Why Tunneling isn't part of this score:
-----------------------------------------
Tunneling went through four independent construction attempts
(raw CSW/chase-based, physically-predicted, residualized against
individual pitch quality plus a release-similarity x movement
interaction term, and a trajectory-based decision-point tunnel
differential -- both raw and log-transformed to rule out outlier
distortion). All four showed essentially zero marginal
relationship with season-level run prevention once Effectiveness
and Movement were already in the model. That's a real, converged
empirical finding -- not a missing fix.

Rather than force a near-zero-signal component into the aggregate
score (which would also cut the scored population from ~700+ down
to ~250, since Tunneling requires >=2 pitch types), Tunneling is
repositioned as a MATCHING CRITERION used directly inside the
recommendation engine -- e.g. "does this candidate pitch to add
tunnel well with the pitcher's existing arsenal" -- rather than a
predictive input to an overall rating. It's still merged in below
(left join, so it's NaN for single-pitch-type pitchers) purely as
reference data for that downstream use, not used in the regression.

Why Release Consistency isn't part of this score either:
------------------------------------------------------------
18_release_consistency_component.py was tested as a fourth
component: it showed a small but stable, positive coefficient that
agreed in sign across both validation targets (a genuinely
different result from Tunneling's unstable/negative story) -- but
its marginal contribution to predicting either target was close to
negligible, and splitting it into its two raw sub-features
(within-type vs across-type dispersion) didn't meaningfully improve
CV R2 either. Given it isn't earning its place in the aggregate
score, it's kept OUT of paom_score and instead surfaced as its own
dashboard visualization (the release point map) -- descriptive
context about a pitcher's mechanics, not a rated component.

Output:
-------
PAOM_final_scores.csv
paom_combination_weights.csv
paom_validation_comparison.csv   (primary vs secondary target)
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
TUNNELING_FILE = "PAOM_tunneling_component.csv"
MOVEMENT_FILE = "PAOM_movement_component.csv"
COMMAND_FILE = "PAOM_command_component.csv"
VELOCITY_FILE = "PAOM_velocity_component.csv"
MASTER_FILE = "master_pitch_table_2025.csv"
EVENTS_FILE = "clean_statcast_2025.csv"

OUTPUT_FILE = "PAOM_final_scores.csv"
WEIGHTS_FILE = "paom_combination_weights.csv"
VALIDATION_COMPARISON_FILE = "paom_validation_comparison.csv"

MIN_PA_FOR_SECONDARY_TARGET = 50
N_CV_FOLDS = 5
RANDOM_SEED = 42


# ============================================================
# STEP 1: PITCHER-LEVEL EFFECTIVENESS (USAGE-WEIGHTED)
# ============================================================

print("Loading component files...")

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

"""
Velocity (21_velocity_component.py) is a CANDIDATE for paom_score,
not yet a confirmed member -- same "build it, validate it, then
decide" pattern used for every other component. It's already
pitcher-level, no rollup needed. Included in the correlation check
and CV-Ridge fit below; if it doesn't earn real marginal weight
(same standard applied to Release Consistency, which didn't and
was removed), it gets pulled back out.
"""

velocity = pd.read_csv(VELOCITY_FILE)[
    ["player_name", "trusted_velocity_score", "confidence_score"]
].rename(columns={"confidence_score": "velocity_confidence"})

print(f"Pitchers with velocity score: {len(velocity):,}")


# ============================================================
# STEP 2: LOAD MOVEMENT (SINGLE SCORE) + TUNNELING (REFERENCE ONLY)
# ============================================================

"""
SIXTH design. Movement is back to producing ONE single score
(09_movement_component.py: all five raw features combined via one
PCA, no grouping, no orthogonalization -- see that script's
docstring for the full history of why prior designs, including the
COVERAGE/DISTINCTNESS group split this replaces, kept hitting the
same "correlated features steal credit from each other" pattern no
matter how the feature set was structured).

Movement is now merged in as ONE direct input, the same pattern as
Effectiveness/Command/Velocity -- no multi-candidate merging, no
composite-display logic needed, since there's only one Movement
number to begin with.
"""

movement = pd.read_csv(MOVEMENT_FILE)[
    ["player_name", "trusted_movement_score", "confidence_score"]
].rename(columns={"confidence_score": "movement_confidence"})

print(f"Pitchers with movement score: {len(movement):,}")

tunneling = pd.read_csv(TUNNELING_FILE)[
    ["player_name", "trusted_tunneling_score", "raw_tunneling_score"]
]

print(
    f"Pitchers with tunneling score: {len(tunneling):,} "
    f"(reference only -- not used in paom_score; see docstring)"
)


# ============================================================
# STEP 3: BUILD PRIMARY VALIDATION TARGET (RUN VALUE)
# ============================================================

"""
Pitcher-level run value per 100 pitches, aggregated from raw
pitch-level totals (not averaged across pitch types) so pitch
volume drives the target the same way it drives real performance.

Statcast's delta_run_exp is credited from the batting team's
perspective -- a positive value means run expectancy went UP
on that pitch, which is BAD for the pitcher. We flip the sign
so higher = better for the pitcher, matching every other PAOM
score's convention.
"""

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

"""
K% - BB%, computed directly from plate-appearance outcomes --
independent of anything Effectiveness's own regression targeted.
Statcast's "events" column is populated only on the pitch that
ends a plate appearance, so counting non-null events rows per
pitcher approximates plate appearances faced.
"""

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

# scored components: inner join (candidate population -- Velocity
# included here to TEST it, per the docstring in Step 1c)
combined = eff_pitcher.merge(movement, on="player_name", how="inner")
combined = combined.merge(cmd_pitcher, on="player_name", how="inner")
combined = combined.merge(velocity, on="player_name", how="inner")

print(
    f"Pitchers with all four candidate components (Effectiveness + "
    f"Movement + Command + Velocity): {len(combined):,} "
    f"(dropped {len(eff_pitcher) - len(combined):,})"
)

# tunneling merged in as reference only -- left join, so it's
# NaN for single-pitch-type pitchers rather than dropping them
combined = combined.merge(tunneling, on="player_name", how="left")

print(
    f"Of those, {combined['trusted_tunneling_score'].notna().sum():,} "
    f"also have a tunneling score available for recommendation-time "
    f"matching."
)


# ============================================================
# STEP 4a: PAOM CONFIDENCE (MINIMUM OF THE CANDIDATE COMPONENTS)
# ============================================================

"""
Each component already computes its own confidence_score based on
sample size, used internally to shrink that component's own score
toward league average -- but that information previously died once
it fed into the combination, leaving no way to tell whether a given
paom_score is backed by well-established components or barely-
qualifying ones.

MINIMUM (not average): paom_score is only as trustworthy as its
weakest input. A pitcher with a rock-solid Effectiveness sample
(95% confidence) but a barely-qualifying Command sample (20%
confidence) should show LOW overall confidence, not a misleadingly
moderate average -- the minimum directly answers "what's the
shakiest thing this score rests on." Velocity is included here
since it's currently a candidate in the same fit below.

This does NOT gate who appears on the leaderboard -- low-confidence
pitchers are still scored and shown, just flagged.
"""

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

"""
pitcher_effectiveness, pitcher_command, trusted_velocity_score, and
trusted_movement_score were each percentile-ranked
within their OWN component's population (Effectiveness: 394
pitchers who clear its threshold; Movement/Command/Velocity: 722
who clear theirs). paom_score only covers the 394 who clear ALL of
them -- a subset of Movement's, Command's, and Velocity's
populations, not the full population those percentiles were
computed against.

A percentile computed against one population isn't guaranteed to
spread evenly across 0-100 when restricted to a subset of it -- if
the 394 pitchers who clear Effectiveness's stricter threshold are a
specific slice of the broader 722 (e.g. higher-workload, more
established pitchers), their Movement/Command percentiles could
bunch into a narrower band within just that 394 rather than
spanning the full range. This compresses the final paom_score
leaderboard (observed: top-20 spread went from ~22 points under
the old method to ~8 points before this fix) and distorts each
component's actual relative influence on the final score.

Effectiveness itself doesn't have this problem -- its percentile
was already computed over exactly this same 394-pitcher population.
Re-ranking it here is a near no-op (safe to include for uniformity,
changes nothing meaningful).

Re-percentiling here (rather than min-max, or leaving it as some
other scale) keeps every downstream number -- the correlation
check, the CV-Ridge fit, and the final weighted-average
contributions -- built on percentiles that genuinely span 0-100
across the SAME population paom_score is about, closing the
mismatch at its source rather than patching around it later.
"""

corr_cols = [
    "pitcher_effectiveness",
    "pitcher_command",
    "trusted_velocity_score",
    "trusted_movement_score"
]

for component in corr_cols:
    combined[component] = combined[component].rank(pct=True) * 100

print(
    "\nRe-percentiled Effectiveness/Command/Velocity/Movement's "
    "features within the paom-eligible population (fixes compression "
    "from each component's percentile originally being computed "
    "against a different, broader population)."
)

# A Movement x Velocity interaction term was tested here as a
# candidate (real, stable weight -- ~8% of the composite -- but it
# absorbed almost all of standalone Movement's own contribution,
# which collapsed to near-zero once the interaction was present).
# Read as "the interaction is a better-fitting repackaging of
# Movement's own signal" rather than clear evidence of NEW synergy
# information beyond what Movement and Velocity already capture
# separately -- removed pending a clearer test of that distinction.


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
    """
    Standardizes, cross-validates a Ridge fit (reporting HONEST
    out-of-sample R2 via K-fold, not in-sample), then refits on
    the FULL data to get the coefficients used for weights/scoring.

    Alpha is selected once via RidgeCV on the full data (consistent
    with how alpha selection works elsewhere in PAOM), then that
    SAME alpha is used inside the cross-validation loop -- so the
    reported CV R2 reflects genuine held-out performance of a fixed
    model, not a model whose hyperparameter was itself tuned on
    each fold's test data.

    CONFIDENCE-WEIGHTED FIT: if sample_weight is provided (paom_
    confidence, normalized to 0-1), it's passed into every fit AND
    every fold's scoring below -- not just applied to each
    pitcher's own final score the way component-level shrinkage
    already does. Previously, a pitcher with a barely-qualifying,
    noisy sample counted exactly as much as a full-season pitcher
    when Ridge decided what weight each COMPONENT should get. This
    protects the combination weights themselves, not just individual
    outputs.

    Uses a manual K-fold loop rather than cross_val_score's built-in
    CV, specifically to pass sample_weight through both fit() and
    score() on each fold -- cross_val_score's API for this
    (fit_params vs params) has changed across recent sklearn
    versions, while Ridge.fit()/.score()'s own sample_weight
    argument has been stable for a long time. This sidesteps that
    version risk entirely.
    """

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
        f"{in_sample_r2:.3f} -- expect this to look better than "
        f"the CV number; that gap is the honest cost of overfitting "
        f"to this specific sample.)"
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

# --- primary target (run value) ---
primary_model_df = combined.merge(primary_target_df, on="player_name", how="inner")
print(f"\nPitchers used for primary-target fit: {len(primary_model_df):,}")

primary_weights = primary_model_df["paom_confidence"] / 100

primary_result = cv_ridge_fit(
    primary_model_df[corr_cols],
    primary_model_df["primary_target"],
    "PRIMARY TARGET: Run value per 100 pitches",
    sample_weight=primary_weights
)

# --- secondary target (K% - BB%) ---
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
        f"targets -- treat that component's weight in the primary "
        f"result with caution; it may be less robust than it looks."
    )

comparison.to_csv(VALIDATION_COMPARISON_FILE, index=False)


# ============================================================
# STEP 7: FINALIZE WEIGHTS (FROM PRIMARY TARGET, FULL POPULATION)
# ============================================================

"""
The deployed paom_score uses the primary target's weights (fit on
the full population that has both components and the primary
target) -- the secondary target above is a diagnostic check on
that result, not a second scoring system.

Weights are NORMALIZED TO SUM TO 1 here and applied as a literal
weighted average of the three component PERCENTILE scores (Step 8),
rather than the previous approach (project the Ridge model's raw
regression output, then percentile-rank that result a second time).
The old approach produced a "percentile of a percentile" -- an
abstract number two steps removed from anything on the dashboard,
impossible to reconstruct by looking at the three visible component
scores. A weighted average is directly reconstructible by hand
("74 = 75% Effectiveness's 88th percentile + 19% Command's 52nd +
6% Movement's 61st") using the exact same Ridge-learned weights,
just applied differently.

Honest tradeoff: paom_score itself no longer carries the strict
"50 = exact population median" guarantee the individual components
have -- a weighted average of several roughly-uniform percentile
distributions tends to bunch toward the middle rather than staying
perfectly flat (a basic property of summing quasi-independent
variables). "50" on paom_score will be close to the median, not
guaranteed to be exactly it. That's an accepted cost of making the
score reconstructible from parts you can actually see.
"""

weights_df = primary_result["coefficients"].copy()
weights_df["abs_coefficient"] = weights_df["coefficient"].abs()
weights_df = weights_df.sort_values("abs_coefficient", ascending=False)

negative = weights_df[weights_df["coefficient"] < 0]

if len(negative) > 0:
    print(
        "\nNOTE: the following component(s) came back with a "
        "NEGATIVE coefficient in the primary (deployed) fit -- "
        "a negative weight in a literal weighted average can push "
        "paom_score outside the normal 0-100 range for pitchers "
        "extreme on that component (clipped below). Inspect before "
        "trusting the combination:"
    )
    print(negative)

coefficient_sum = weights_df["coefficient"].sum()

if coefficient_sum <= 0:
    raise ValueError(
        f"Sum of coefficients is {coefficient_sum:.4f} (<=0) -- "
        f"cannot normalize into meaningful weights. Inspect the "
        f"primary-target fit before proceeding."
    )

weights_df["normalized_weight"] = (
    weights_df["coefficient"] / coefficient_sum
)

print("\nNormalized weights (used as literal blend weights):")
print(weights_df[["component", "coefficient", "normalized_weight"]])

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
    print(
        f"\nNOTE: {out_of_range} pitcher(s) landed outside 0-100 "
        f"before clipping (expected only if a component has a "
        f"negative weight, per the warning above)."
    )

combined["paom_score"] = combined["paom_score"].clip(0, 100)


# ============================================================
# SAVE OUTPUT
# ============================================================

output = combined[
    [
        "player_name",
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
print("PAOM Final Score Saved")
print("==============================")

print(
    f"""
Pitchers scored:
{len(output):,}

Average PAOM score:
{output['paom_score'].mean():.2f}

Primary-target cross-validated R2 (honest, out-of-sample):
{primary_result['cv_r2_mean']:.3f} (+/- {primary_result['cv_r2_std']:.3f})

Pitchers also carrying a tunneling score (for recommendation-time
use, not part of paom_score):
{output['trusted_tunneling_score'].notna().sum():,}
"""
)

print("\nTop 20 PAOM scores")
print("-------------------")
print(output.head(20))

print("\nSaved:")
print(f"- {OUTPUT_FILE}")
print(f"- {WEIGHTS_FILE}")
print(f"- {VALIDATION_COMPARISON_FILE}")
