"""
62_drop_usage_backtest.py

Purpose:
--------
60_stage1_all_change_types.py extended the Stage 1 concept to DROP/
USAGE_INCREASE/USAGE_DECREASE, but its demonstration showed a much
more modest existing_quality_coef than ADD's dramatic result. Before
wiring this into 45 as the new default for these change types (the
way ADD's Stage 1 integration was, after clearing 55/56/58), this
runs the SAME two-part validation already established: point
accuracy (58's methodology) AND ranking concordance (61's
methodology) -- both scoped specifically to DROP/USAGE_INCREASE/
USAGE_DECREASE events, comparing BASE vs. the new all-types
integration.

Output:
-------
Printed point-accuracy comparison (correlation, MAE) and pairwise
concordance comparison, BASE vs. INTEGRATED, on the same real
DROP/USAGE_INCREASE/USAGE_DECREASE events. No file saved -- this is
a validation gate, not a deliverable.
"""

import importlib.util
import sys
import itertools

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 60 (cascades through 54, 50, 45, 38)
# ============================================================

print("Loading base + all-types-integrated prediction functions from 60...")

spec = importlib.util.spec_from_file_location(
    "stage1_all_change_types", "60_stage1_all_change_types.py"
)
stage60 = importlib.util.module_from_spec(spec)
sys.modules["stage1_all_change_types"] = stage60
spec.loader.exec_module(stage60)

training = stage60.training
similarity_features = stage60.similarity_features
predict_arsenal_change_effect = stage60.predict_arsenal_change_effect
predict_arsenal_change_with_stage1_all_types = stage60.predict_arsenal_change_with_stage1_all_types
estimate_realistic_usage_delta = stage60.estimate_realistic_usage_delta
SIMILARITY_FEATURE_COLS = stage60.SIMILARITY_FEATURE_COLS


# ============================================================
# SETTINGS
# ============================================================

CHANGE_TYPES = ["DROP", "USAGE_INCREASE", "USAGE_DECREASE"]
N_SAMPLE_PER_GROUP = 6  # per (change_type, pitch_type) combination --
                          # kept modest given 3 change types x up to
                          # 8 pitch types already gives a substantial
                          # combined sample
OUTCOME_METRIC = "xwoba_against"
RANDOM_SEED = 42


# ============================================================
# SAMPLE REAL DROP/USAGE_INCREASE/USAGE_DECREASE EVENTS
# ============================================================

print(f"\nSampling real {CHANGE_TYPES} events...")

candidate_events = training[training["change_type"].isin(CHANGE_TYPES)].copy()
candidate_events = candidate_events.dropna(subset=[f"{OUTCOME_METRIC}_after", f"{OUTCOME_METRIC}_before"])

sample_frames = []
for (ct, pt), group in candidate_events.groupby(["change_type", "pitch_type"]):
    n = min(N_SAMPLE_PER_GROUP, len(group))
    sample_frames.append(group.sample(n=n, random_state=RANDOM_SEED))
sample = pd.concat(sample_frames, ignore_index=True)

print(f"Sampled {len(sample):,} events across {sample.groupby('change_type').size().to_dict()}")


# ============================================================
# LEAVE-ONE-OUT PREDICTIONS, BOTH APPROACHES
# ============================================================

def compute_predictions(predict_fn, is_integrated, label):
    print(f"\nComputing leave-one-out predictions ({label})...")
    rows = []
    for _, event in sample.iterrows():
        player_id = event["player_id"]
        season_from = event["season_from"]
        season_to = event["season_to"]
        pitch_type = event["pitch_type"]
        change_type = event["change_type"]

        actual = event[f"{OUTCOME_METRIC}_after"]
        actual_change = actual - event[f"{OUTCOME_METRIC}_before"]
        target_outcome_before = event[f"{OUTCOME_METRIC}_before"]

        target_sim_rows = similarity_features[
            (similarity_features["player_id"] == player_id)
            & (similarity_features["season"] == season_from)
        ]
        if len(target_sim_rows) == 0:
            continue
        target_features = {c: target_sim_rows.iloc[0][c] for c in SIMILARITY_FEATURE_COLS}

        loo_training = training[
            ~(
                (training["player_id"] == player_id)
                & (training["season_from"] == season_from)
                & (training["season_to"] == season_to)
                & (training["pitch_type"] == pitch_type)
                & (training["change_type"] == change_type)
            )
        ]

        realistic_delta, _ = estimate_realistic_usage_delta(
            target_features, pitch_type, change_type, player_id,
            season_from, target_outcome_before, OUTCOME_METRIC
        )
        if realistic_delta is None:
            continue

        final_features = dict(target_features)
        final_features["usage_pct_delta"] = realistic_delta

        result = predict_fn(
            final_features, pitch_type, change_type, target_outcome_before,
            OUTCOME_METRIC, player_id, season_from, training_df=loo_training
        )

        if is_integrated:
            success = result.get("stage1_integration_status") == "success"
        else:
            success = result["predicted_outcome"] is not None

        if success:
            rows.append({
                "event_id": f"{player_id}_{season_to}_{change_type}_{pitch_type}",
                "predicted": result["predicted_outcome"],
                "predicted_change": result["predicted_change"],
                "actual": actual,
                "actual_change": actual_change,
            })

    return pd.DataFrame(rows)


base_predictions = compute_predictions(predict_arsenal_change_effect, False, "BASE")
integrated_predictions = compute_predictions(predict_arsenal_change_with_stage1_all_types, True, "INTEGRATED (all-types)")

print(f"\n{len(base_predictions):,} usable BASE predictions")
print(f"{len(integrated_predictions):,} usable INTEGRATED predictions")


# ============================================================
# PART 1: POINT ACCURACY (58's methodology)
# ============================================================

print("\n\n==============================")
print("PART 1: Point accuracy -- predicted vs. actual real outcomes")
print("==============================")

point_results = {}
for label, df in [("BASE", base_predictions), ("INTEGRATED", integrated_predictions)]:
    if len(df) < 5:
        print(f"\n{label}: too few predictions ({len(df)}) to evaluate")
        continue
    corr = df["predicted"].corr(df["actual"])
    mae = (df["predicted"] - df["actual"]).abs().mean()
    print(f"\n{label} (n={len(df)}):")
    print(f"  Correlation: {corr:.3f}")
    print(f"  MAE: {mae:.4f}")
    point_results[label] = {"corr": corr, "mae": mae}


# ============================================================
# PART 2: PAIRWISE CONCORDANCE (61's methodology)
# ============================================================

print("\n\n==============================")
print("PART 2: Pairwise concordance -- does the ranking carry real information?")
print("==============================")

def compute_concordance(df, label):
    if len(df) < 5:
        print(f"\n{label}: too few predictions for meaningful pairwise comparison")
        return None

    n_concordant = 0
    n_discordant = 0
    n_tied = 0

    for (i, row_a), (j, row_b) in itertools.combinations(df.iterrows(), 2):
        pred_diff = row_a["predicted_change"] - row_b["predicted_change"]
        actual_diff = row_a["actual_change"] - row_b["actual_change"]
        if pred_diff == 0 or actual_diff == 0:
            n_tied += 1
            continue
        if np.sign(pred_diff) == np.sign(actual_diff):
            n_concordant += 1
        else:
            n_discordant += 1

    total = n_concordant + n_discordant
    rate = n_concordant / total if total > 0 else np.nan
    print(f"\n{label} (n={len(df)} events, {total:,} comparable pairs):")
    print(f"  Concordance rate: {rate:.1%} (50% = pure chance)")
    return rate


base_concordance = compute_concordance(base_predictions, "BASE")
integrated_concordance = compute_concordance(integrated_predictions, "INTEGRATED (all-types)")


# ============================================================
# VERDICT
# ============================================================

print("\n\n==============================")
print("Verdict: wire the all-types Stage 1 extension into 45 as the new default?")
print("==============================")

if "BASE" in point_results and "INTEGRATED" in point_results:
    mae_improved = point_results["INTEGRATED"]["mae"] < point_results["BASE"]["mae"]
    concordance_improved = (
        integrated_concordance is not None and base_concordance is not None
        and integrated_concordance > base_concordance + 0.02
    )

    print(f"\nPoint accuracy: BASE MAE={point_results['BASE']['mae']:.4f}, "
          f"INTEGRATED MAE={point_results['INTEGRATED']['mae']:.4f} "
          f"({'IMPROVED' if mae_improved else 'NOT improved'})")
    if base_concordance is not None and integrated_concordance is not None:
        print(f"Ranking accuracy: BASE={base_concordance:.1%}, "
              f"INTEGRATED={integrated_concordance:.1%} "
              f"({'IMPROVED' if concordance_improved else 'NOT clearly improved'})")

    if mae_improved and concordance_improved:
        print(
            "\nBoth point accuracy AND ranking accuracy improve -- "
            "worth wiring into 45 as the new default for DROP/"
            "USAGE_INCREASE/USAGE_DECREASE, the same way ADD's Stage "
            "1 integration was after clearing this same bar."
        )
    elif mae_improved or concordance_improved:
        print(
            "\nMixed result -- improves on ONE dimension but not "
            "clearly on the other. Worth a judgment call rather than "
            "an automatic yes: the extension may still be worth "
            "adopting, but the case is weaker than ADD's was."
        )
    else:
        print(
            "\nNeither point accuracy nor ranking accuracy show a "
            "clear improvement -- consistent with the more modest "
            "existing_quality_coef seen in 60's demonstration. Not "
            "recommended to replace the current default for these "
            "change types without further investigation."
        )
