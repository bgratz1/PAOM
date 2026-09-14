"""
67_synergy_persistence_placebo_test.py

Purpose:
--------
66_synergy_persistence_test.py's raw, unweighted correlation check
was too crude to trust -- both versions came back near-zero,
nowhere near the clean signal the ORIGINAL synergy feature showed
under proper testing (43_synergy_placebo_test.py: 100% sign
consistency, 4x the placebo magnitude). This reruns the SAME
rigorous methodology -- 38's weighted comparison groups, 43's
placebo shuffle test -- with persistent_synergy (measured a full
season AFTER the change, season_from -> season_to+1) substituted in
place of the original.

REUSES 38's machinery WITHOUT modifying it: predict_arsenal_change_
effect already knows how to use a column literally named
arsenal_synergy_whiff_delta. Building a custom training_df where
THAT column holds persistent values instead of original ones lets
this reuse the entire validated weighted-regression and placebo
pipeline unchanged.

If persistent_synergy ALSO clears its own placebo test with
comparable strength, that's real, properly-controlled evidence for
the causal (tunneling) story. If it collapses under the same
rigorous test that validated the original, that favors the shared-
cause explanation.

Output:
-------
Printed magnitude and sign-consistency comparison, persistent
synergy vs. its own placebo, directly comparable to 43's original
results. No file saved -- this is a validation check, not a
deliverable.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 45 (cascades through 38)
# ============================================================

print("Loading training data and prediction function from 45...")

spec = importlib.util.spec_from_file_location(
    "similarity_recommendation_engine", "45_similarity_recommendation_engine.py"
)
rec_engine = importlib.util.module_from_spec(spec)
sys.modules["similarity_recommendation_engine"] = rec_engine
spec.loader.exec_module(rec_engine)

training = rec_engine.training
similarity_features = rec_engine.similarity_features
predict_arsenal_change_effect = rec_engine.predict_arsenal_change_effect
SIMILARITY_FEATURE_COLS = rec_engine.SIMILARITY_FEATURE_COLS
MIN_GROUP_SIZE_FOR_REGRESSION = rec_engine.MIN_GROUP_SIZE_FOR_REGRESSION


# ============================================================
# COMPUTE PERSISTENT SYNERGY (same logic as 66)
# ============================================================

print("\nComputing persistent synergy (season_from -> season_to+1)...")

ARSENAL_FILE = "pitcher_arsenal_evolution_2020_2025.csv"
PITCH_LEVEL_OUTCOMES_FILE = "pitcher_pitch_level_outcomes_2020_2025.csv"
MEANINGFUL_USAGE_FLOOR = 5.0
PITCH_TYPES = ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS"]

arsenal = pd.read_csv(ARSENAL_FILE)
pitch_level = pd.read_csv(PITCH_LEVEL_OUTCOMES_FILE)
arsenal_indexed = arsenal.set_index(["player_id", "season"])
pitch_level_indexed_lookup = pitch_level.set_index(["player_id", "season", "pitch_type"])["whiff_pct"]

add_events_only = training[training["change_type"] == "ADD"].copy()

persistent_records = []
for _, event in add_events_only.iterrows():
    player_id = event["player_id"]
    season_from = event["season_from"]
    season_to = event["season_to"]
    season_to_plus_1 = season_to + 1
    new_pitch_type = event["pitch_type"]

    key_from = (player_id, season_from)
    key_to = (player_id, season_to)
    if key_from not in arsenal_indexed.index or key_to not in arsenal_indexed.index:
        continue
    row_from = arsenal_indexed.loc[key_from]
    row_to = arsenal_indexed.loc[key_to]
    if isinstance(row_from, pd.DataFrame):
        row_from = row_from.iloc[0]
    if isinstance(row_to, pd.DataFrame):
        row_to = row_to.iloc[0]

    other_pitch_types = []
    for pt in PITCH_TYPES:
        if pt == new_pitch_type:
            continue
        usage_from = row_from.get(f"{pt}_usage_pct", np.nan)
        usage_to = row_to.get(f"{pt}_usage_pct", np.nan)
        if (
            pd.notna(usage_from) and usage_from >= MEANINGFUL_USAGE_FLOOR
            and pd.notna(usage_to) and usage_to >= MEANINGFUL_USAGE_FLOOR
        ):
            other_pitch_types.append(pt)
    if len(other_pitch_types) == 0:
        continue

    persist_deltas, persist_weights = [], []
    for pt in other_pitch_types:
        w_from = pitch_level_indexed_lookup.get((player_id, season_from, pt), np.nan)
        w_to_plus_1 = pitch_level_indexed_lookup.get((player_id, season_to_plus_1, pt), np.nan)
        usage_from = row_from.get(f"{pt}_usage_pct", np.nan)
        if pd.notna(w_from) and pd.notna(w_to_plus_1):
            persist_deltas.append((w_to_plus_1 - w_from) / 100.0)
            persist_weights.append(usage_from)

    if len(persist_deltas) == 0:
        continue

    persistent_records.append({
        "player_id": player_id,
        "season_from": season_from,
        "season_to": season_to,
        "pitch_type": new_pitch_type,
        "persistent_synergy_value": np.average(persist_deltas, weights=persist_weights),
    })

persistent_df = pd.DataFrame(persistent_records)
print(f"Computed persistent synergy for {len(persistent_df):,} ADD events "
      f"(original covered {len(training[training['change_type']=='ADD']):,})")


# ============================================================
# BUILD A CUSTOM training_df: SAME data, but arsenal_synergy_
# whiff_delta now holds PERSISTENT values instead of original ones
# ============================================================

print("\nBuilding custom training_df with persistent synergy substituted in...")

training_persistent = training.copy()
training_persistent["arsenal_synergy_whiff_delta"] = np.nan  # clear original values first

training_persistent = training_persistent.merge(
    persistent_df.rename(columns={"persistent_synergy_value": "_persistent_temp"}),
    on=["player_id", "season_from", "season_to", "pitch_type"],
    how="left"
)
training_persistent["arsenal_synergy_whiff_delta"] = training_persistent["_persistent_temp"]
training_persistent = training_persistent.drop(columns=["_persistent_temp"])

n_with_persistent = training_persistent["arsenal_synergy_whiff_delta"].notna().sum()
print(f"{n_with_persistent:,} training rows now carry a persistent synergy value")


# ============================================================
# SAME SIGN-CONSISTENCY LOGIC AS 39/43 (corrected, count-based majority)
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
# SELECT DIVERSE TARGET PITCHERS (same approach as 43)
# ============================================================

N_TARGET_PITCHERS = 6
CHANGE_TYPE = "ADD"
OUTCOME_METRIC = "xwoba_against"
RANDOM_SEED = 42

print(f"\nSelecting {N_TARGET_PITCHERS} diverse target pitchers (stratified by max_velo_z)...")

complete_profiles = similarity_features.dropna(subset=SIMILARITY_FEATURE_COLS).copy()
complete_profiles = complete_profiles.sort_values("max_velo_z").reset_index(drop=True)
strat_indices = np.linspace(0, len(complete_profiles) - 1, N_TARGET_PITCHERS).astype(int)
target_rows = complete_profiles.iloc[strat_indices]

for _, r in target_rows.iterrows():
    print(f"  {r['player_name']}, {r['season']}")

add_counts = training_persistent[training_persistent["change_type"] == CHANGE_TYPE]["pitch_type"].value_counts()
viable_pitch_types = add_counts[add_counts >= MIN_GROUP_SIZE_FOR_REGRESSION].index.tolist()


# ============================================================
# RUN THE GRID AGAINST REAL AND PLACEBO PERSISTENT-SYNERGY DATA
# ============================================================

def run_grid(training_data, label):
    print(f"\nRunning grid against {label} persistent-synergy data...")
    rows = []
    for _, target_row in target_rows.iterrows():
        target_features = {c: target_row[c] for c in SIMILARITY_FEATURE_COLS}
        target_features["usage_pct_delta"] = 10.0

        target_baseline_rows = training_data[training_data["player_id"] == target_row["player_id"]]
        target_outcome_before = (
            target_baseline_rows[f"{OUTCOME_METRIC}_before"].iloc[0]
            if len(target_baseline_rows) > 0
            else training_data[f"{OUTCOME_METRIC}_before"].median()
        )

        for pitch_type in viable_pitch_types:
            result = predict_arsenal_change_effect(
                target_features, pitch_type, CHANGE_TYPE,
                target_outcome_before, OUTCOME_METRIC,
                target_row["player_id"], target_row["season"],
                training_df=training_data
            )
            if result.get("used_synergy_feature"):
                rows.append({
                    "target": f"{target_row['player_name']}_{target_row['season']}",
                    "pitch_type": pitch_type,
                    "synergy_coef": result.get("synergy_coef"),
                })
    return pd.DataFrame(rows)


real_results = run_grid(training_persistent, "REAL persistent")

rng = np.random.default_rng(RANDOM_SEED)
training_placebo = training_persistent.copy()
valid_mask = training_placebo["arsenal_synergy_whiff_delta"].notna()
real_values = training_placebo.loc[valid_mask, "arsenal_synergy_whiff_delta"].values.copy()
shuffled_values = rng.permutation(real_values)
training_placebo.loc[valid_mask, "arsenal_synergy_whiff_delta"] = shuffled_values

placebo_results = run_grid(training_placebo, "PLACEBO (shuffled persistent)")


# ============================================================
# COMPARE
# ============================================================

print("\n\n==============================")
print("Persistent synergy: real vs. placebo (directly comparable to 43's original results)")
print("==============================\n")

for label, df in [("REAL (persistent)", real_results), ("PLACEBO (shuffled persistent)", placebo_results)]:
    if len(df) == 0:
        print(f"{label}: no rows produced")
        continue
    avg_mag = df["synergy_coef"].abs().mean()
    per_target_consistency = df.groupby("target")["synergy_coef"].apply(sign_consistency)
    avg_consistency = per_target_consistency.mean()
    print(f"{label}:")
    print(f"  n_predictions: {len(df)}")
    print(f"  avg |synergy_coef|: {avg_mag:.4f}")
    print(f"  avg sign consistency: {avg_consistency:.1%}\n")

if len(real_results) > 0 and len(placebo_results) > 0:
    real_mag = real_results["synergy_coef"].abs().mean()
    placebo_mag = placebo_results["synergy_coef"].abs().mean()
    real_consistency = real_results.groupby("target")["synergy_coef"].apply(sign_consistency).mean()
    placebo_consistency = placebo_results.groupby("target")["synergy_coef"].apply(sign_consistency).mean()

    print("==============================")
    print("Verdict")
    print("==============================\n")
    print("For direct comparison, the ORIGINAL synergy feature (43) scored:")
    print("  Real: magnitude=0.0071, consistency=100.0% | Placebo: magnitude=0.0017, consistency=58.3%\n")

    if placebo_consistency >= real_consistency * 0.8 and placebo_mag >= real_mag * 0.5:
        print(
            "WARNING: persistent synergy's placebo performs comparably "
            "to its real version -- weak/no evidence of genuine signal "
            "once measured a season apart from the outcome. This "
            "favors the SHARED-CAUSE explanation for the ORIGINAL "
            "feature's strength: it likely reflected same-season "
            "contamination rather than a lasting causal effect."
        )
    else:
        print(
            f"Persistent synergy clears its OWN placebo test (real "
            f"magnitude={real_mag:.4f} vs. placebo={placebo_mag:.4f}, "
            f"real consistency={real_consistency:.1%} vs. placebo="
            f"{placebo_consistency:.1%}) -- real, properly-controlled "
            f"evidence the improvement in other pitches PERSISTS a "
            f"full season later. This favors the CAUSAL story (a real, "
            f"lasting effect from the new pitch) over the shared-cause "
            f"explanation, though likely with a smaller effect size "
            f"than the original given the additional time gap."
        )
