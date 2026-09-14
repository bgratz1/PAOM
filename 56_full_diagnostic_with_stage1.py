"""
56_full_diagnostic_with_stage1.py

Purpose:
--------
55_stage1_integration_validation.py confirmed stage1_quality_coef
carries real signal (4.8x the placebo magnitude, 100% vs. 58.3%
sign consistency) on a set of 6 stratified target pitchers. Before
treating this as validated enough to wire into 45 as the default,
this runs a broader check: a genuinely DIFFERENT, larger sample of
target pitchers (not the same 6, to rule out any residual sampling
coincidence), and the SAME Check 1/2/4-style diagnostics already
used throughout this project -- but run SIDE BY SIDE for the base
function (no Stage 1) and the Stage-1-integrated function, so the
comparison is direct rather than inferred from separate runs.

Output:
-------
Printed side-by-side comparison across Check 1 (within-target
variance), Check 2 (between-target variance), and Check 4 (mean-
reversion vs. change-specific magnitude/sign-consistency, now
including stage1_quality_coef explicitly), for BASE vs. INTEGRATED.
No file saved -- this is a validation gate for wiring into 45, not
a deliverable itself.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 54 (exposes both base and integrated functions)
# ============================================================

print("Loading Stage 1 + Stage 2 integration from 54...")

spec = importlib.util.spec_from_file_location(
    "stage2_integration", "54_stage2_integration.py"
)
stage2 = importlib.util.module_from_spec(spec)
sys.modules["stage2_integration"] = stage2
spec.loader.exec_module(stage2)

training = stage2.training
similarity_features = stage2.similarity_features
outcomes = stage2.outcomes
predict_arsenal_change_with_stage1 = stage2.predict_arsenal_change_with_stage1
predict_arsenal_change_effect = stage2.predict_arsenal_change_effect  # the BASE function
estimate_realistic_usage_delta = stage2.estimate_realistic_usage_delta
SIMILARITY_FEATURE_COLS = stage2.SIMILARITY_FEATURE_COLS


# ============================================================
# SETTINGS
# ============================================================

N_TARGET_PITCHERS = 12  # broader AND different from 55's 6 --
                          # random, not stratified, with a distinct
                          # seed, to genuinely rule out sampling
                          # coincidence rather than just re-running
                          # the same set with more decimal places
VIABLE_PITCH_TYPES = ["SI", "SL", "ST", "FS", "FC", "CH", "CU", "FF"]
OUTCOME_METRIC = "xwoba_against"

RANDOM_SEED = 99  # deliberately different from 42, used everywhere
                    # else in this project -- a genuinely different
                    # sample, not a relabeled version of the same one


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
# SELECT A BROADER, GENUINELY DIFFERENT TARGET SAMPLE
# ============================================================

print(f"\nSelecting {N_TARGET_PITCHERS} target pitchers (random sample, seed={RANDOM_SEED} -- deliberately different from 55's stratified 6)...")

complete_profiles = similarity_features.dropna(subset=SIMILARITY_FEATURE_COLS).copy()
target_rows = complete_profiles.sample(n=min(N_TARGET_PITCHERS, len(complete_profiles)), random_state=RANDOM_SEED)

for _, r in target_rows.iterrows():
    print(f"  {r['player_name']}, {r['season']} (max_velo_z={r['max_velo_z']:.2f})")


# ============================================================
# RUN BOTH VERSIONS ACROSS THE SAME GRID
# ============================================================

def run_grid(predict_fn, label, is_integrated):
    print(f"\nRunning grid ({label})...")
    rows = []
    for _, target_row in target_rows.iterrows():
        target_key = f"{target_row['player_name']}_{target_row['season']}"
        target_features = {c: target_row[c] for c in SIMILARITY_FEATURE_COLS}

        target_baseline_rows = outcomes[outcomes["player_id"] == target_row["player_id"]]
        if len(target_baseline_rows) == 0 or OUTCOME_METRIC not in target_baseline_rows.columns:
            continue
        target_outcome_before = target_baseline_rows.sort_values("season")[OUTCOME_METRIC].iloc[-1]

        for pitch_type in VIABLE_PITCH_TYPES:
            realistic_delta, _ = estimate_realistic_usage_delta(
                target_features, pitch_type, "ADD", target_row["player_id"],
                target_row["season"], target_outcome_before, OUTCOME_METRIC
            )
            if realistic_delta is None:
                continue

            final_features = dict(target_features)
            final_features["usage_pct_delta"] = realistic_delta

            if is_integrated:
                result = predict_fn(
                    final_features, pitch_type, "ADD", target_outcome_before,
                    OUTCOME_METRIC, target_row["player_id"], target_row["season"]
                )
                success = result.get("stage1_integration_status") == "success"
            else:
                result = predict_fn(
                    final_features, pitch_type, "ADD", target_outcome_before,
                    OUTCOME_METRIC, target_row["player_id"], target_row["season"]
                )
                success = result["predicted_outcome"] is not None

            if success:
                rows.append({
                    "target": target_key,
                    "pitch_type": pitch_type,
                    "predicted_change": result["predicted_change"],
                    "mean_reversion_coef": result.get("mean_reversion_coef"),
                    "change_specific_coef": result.get("change_specific_coef"),
                    "stage1_quality_coef": result.get("stage1_quality_coef"),
                })

    return pd.DataFrame(rows)


base_results = run_grid(predict_arsenal_change_effect, "BASE (no Stage 1)", is_integrated=False)
integrated_results = run_grid(predict_arsenal_change_with_stage1, "INTEGRATED (with Stage 1)", is_integrated=True)

print(f"\n{len(base_results):,} successful BASE predictions")
print(f"{len(integrated_results):,} successful INTEGRATED predictions")


# ============================================================
# CHECK 1 & 2: WITHIN VS. BETWEEN VARIANCE, SIDE BY SIDE
# ============================================================

print("\n\n==============================")
print("CHECK 1 & 2: Within-target vs. between-target variance -- BASE vs. INTEGRATED")
print("==============================")

for label, df in [("BASE", base_results), ("INTEGRATED", integrated_results)]:
    if len(df) == 0:
        print(f"\n{label}: no successful predictions -- skipping")
        continue
    within = df.groupby("target")["predicted_change"].std().mean()
    between = df.groupby("pitch_type")["predicted_change"].std().mean()
    print(f"\n{label}:")
    print(f"  Average within-target std (by pitch type): {within:.4f}")
    print(f"  Average between-target std (by pitcher):   {between:.4f}")
    print(f"  Ratio (between/within): {between/within:.3f}" if within > 0 else "  Ratio: n/a")


# ============================================================
# CHECK 4: MAGNITUDE + SIGN CONSISTENCY, SIDE BY SIDE
# ============================================================

print("\n\n==============================")
print("CHECK 4: Coefficient magnitude + sign consistency -- BASE vs. INTEGRATED")
print("==============================")

for label, df in [("BASE", base_results), ("INTEGRATED", integrated_results)]:
    if len(df) == 0:
        continue
    print(f"\n--- {label} ---")
    avg_mean_reversion = df["mean_reversion_coef"].abs().mean()
    avg_change_specific = df["change_specific_coef"].abs().mean()
    print(f"Average |mean_reversion_coef|:  {avg_mean_reversion:.4f}")
    print(f"Average |change_specific_coef|: {avg_change_specific:.4f}")
    change_specific_consistency = df.groupby("target")["change_specific_coef"].apply(sign_consistency).mean()
    print(f"change_specific_coef sign consistency: {change_specific_consistency:.1%}")

    if "stage1_quality_coef" in df.columns and df["stage1_quality_coef"].notna().sum() > 0:
        avg_stage1 = df["stage1_quality_coef"].abs().mean()
        stage1_consistency = df.groupby("target")["stage1_quality_coef"].apply(sign_consistency).mean()
        print(f"Average |stage1_quality_coef|:  {avg_stage1:.4f}")
        print(f"stage1_quality_coef sign consistency: {stage1_consistency:.1%}")
        print(
            f"stage1_quality_coef vs. mean_reversion_coef ratio: "
            f"{avg_stage1/avg_mean_reversion:.2f}x"
        )


# ============================================================
# VERDICT
# ============================================================

print("\n\n==============================")
print("Verdict: does the broader sample confirm 55's result?")
print("==============================")

if len(integrated_results) > 0 and "stage1_quality_coef" in integrated_results.columns:
    avg_stage1_broad = integrated_results["stage1_quality_coef"].abs().mean()
    avg_mean_reversion_broad = integrated_results["mean_reversion_coef"].abs().mean()
    stage1_consistency_broad = integrated_results.groupby("target")["stage1_quality_coef"].apply(sign_consistency).mean()

    if avg_stage1_broad > avg_mean_reversion_broad * 0.8 and stage1_consistency_broad > 0.7:
        print(
            f"\nCONFIRMED on a broader, different sample: "
            f"|stage1_quality_coef|={avg_stage1_broad:.4f} vs. "
            f"|mean_reversion_coef|={avg_mean_reversion_broad:.4f}, "
            f"sign consistency={stage1_consistency_broad:.1%}. "
            f"55's result was NOT specific to that particular sample -- "
            f"this looks like a genuine, robust finding, worth wiring "
            f"into 45 as the default for ADD events."
        )
    else:
        print(
            f"\nNOT confirmed as strongly on this broader sample: "
            f"|stage1_quality_coef|={avg_stage1_broad:.4f} vs. "
            f"|mean_reversion_coef|={avg_mean_reversion_broad:.4f}, "
            f"sign consistency={stage1_consistency_broad:.1%}. "
            f"Worth investigating why this differs from 55's result "
            f"before wiring into 45 as the default."
        )
