"""
49_arsenal_synergy_features_v2.py

Purpose:
--------
42_arsenal_synergy_features.py computed the arsenal-synergy feature
using the Kaggle file's own whiff_rate column -- a value with NO
KNOWN reliability floor behind it (unlike everything else built
since then in this project). Now that 47_pull_pitch_level_outcomes
.py exists, with a real, evidence-based MIN_PITCHES=50 floor (see
48_pitch_level_outcome_stability_diagnostic.py), this recomputes the
SAME synergy concept using the BETTER-GROUNDED whiff source, and
directly cross-checks it against the original.

KEPT THE SAME: which pitch types count as the pitcher's "existing
arsenal" for a given ADD event -- still the Kaggle file's usage_pct
>= MEANINGFUL_USAGE_FLOOR (5%) in BOTH season_from and season_to,
identical to 42's logic. This part isn't being upgraded; only the
underlying WHIFF VALUES themselves are swapped for the more reliable
source.

SCALE NOTE: the Kaggle file's whiff_rate is 0-1; the new Savant pull
's whiff_pct is 0-100 (confirmed directly against real saved data
before building this). The new deltas are divided by 100 so both
versions are directly comparable on the same scale, not just
correlated.

A real, expected consequence of using the STRICTER MIN_PITCHES=50
floor: some pitch types that qualify by the Kaggle file's usage_pct
criterion won't have a valid (>=50 raw pitch) row in the new pull --
usage percentage and raw pitch count aren't the same thing (a 6%-
usage pitch on a pitcher who threw few total pitches that year could
still be under 50 raw pitches). These are skipped, same "require
both season's values present" handling as the original script.

Output:
-------
arsenal_synergy_features_v2.csv -- the recomputed feature.
Console output directly comparing v1 vs. v2 on the same events
(correlation, mean absolute difference, and where they most
disagree) -- the real test of whether the original feature's
computation was already sound, or whether it was quietly distorted
by unreliable underlying whiff values.
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

CHANGE_EVENTS_FILE = "arsenal_change_events_pitch_level.csv"
ARSENAL_FILE = "pitcher_arsenal_evolution_2020_2025.csv"
PITCH_LEVEL_OUTCOMES_FILE = "pitcher_pitch_level_outcomes_2020_2025.csv"
ORIGINAL_SYNERGY_FILE = "arsenal_synergy_features.csv"

OUTPUT_FILE = "arsenal_synergy_features_v2.csv"

MEANINGFUL_USAGE_FLOOR = 5.0  # SAME threshold used throughout this
                                # project -- unchanged, only the
                                # whiff VALUES are being upgraded

PITCH_TYPES = ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS"]


# ============================================================
# LOAD
# ============================================================

print("Loading change events, arsenal data, and NEW pitch-level outcomes...")

change_events = pd.read_csv(CHANGE_EVENTS_FILE)
arsenal = pd.read_csv(ARSENAL_FILE)
pitch_level = pd.read_csv(PITCH_LEVEL_OUTCOMES_FILE)

add_events = change_events[change_events["change_type"] == "ADD"].copy()

print(f"Loaded {len(add_events):,} ADD events")
print(f"New pitch-level outcomes (already MIN_PITCHES-filtered): {len(pitch_level):,} rows")

arsenal_indexed = arsenal.set_index(["player_id", "season"])
pitch_level_indexed = pitch_level.set_index(["player_id", "season", "pitch_type"])["whiff_pct"]


# ============================================================
# RECOMPUTE SYNERGY PER ADD EVENT, USING THE NEW WHIFF SOURCE
# ============================================================

print("\nRecomputing arsenal synergy using the new, filtered whiff source...")

records = []
n_no_other_pitches = 0
n_no_valid_whiff_data = 0

for _, event in add_events.iterrows():
    player_id = event["player_id"]
    season_from = event["season_from"]
    season_to = event["season_to"]
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

    # SAME qualification logic as 42 -- unchanged
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
        n_no_other_pitches += 1
        continue

    # NEW: whiff values come from the filtered Savant pull, not the
    # Kaggle file -- a pitch type can qualify by USAGE here but still
    # be missing from the new pull if its RAW pitch count fell below
    # MIN_PITCHES=50 (usage share and raw pitch count aren't the
    # same thing)
    whiff_deltas = []
    weights = []
    for pt in other_pitch_types:
        key_whiff_from = (player_id, season_from, pt)
        key_whiff_to = (player_id, season_to, pt)

        whiff_from = pitch_level_indexed.get(key_whiff_from, np.nan)
        whiff_to = pitch_level_indexed.get(key_whiff_to, np.nan)
        usage_from = row_from.get(f"{pt}_usage_pct", np.nan)

        if pd.notna(whiff_from) and pd.notna(whiff_to):
            whiff_deltas.append((whiff_to - whiff_from) / 100.0)  # rescale 0-100 -> 0-1
            weights.append(usage_from)

    if len(whiff_deltas) == 0:
        n_no_valid_whiff_data += 1
        continue

    whiff_deltas = np.array(whiff_deltas)
    weights = np.array(weights)

    synergy_score = np.average(whiff_deltas, weights=weights)

    records.append({
        "player_id": player_id,
        "season_from": season_from,
        "season_to": season_to,
        "pitch_type": new_pitch_type,
        "n_other_pitches_measured_v2": len(whiff_deltas),
        "arsenal_synergy_whiff_delta_v2": synergy_score,
    })

result_df = pd.DataFrame(records)

print(f"\nComputed v2 synergy for {len(result_df):,} ADD events "
      f"(original v1 covered {len(pd.read_csv(ORIGINAL_SYNERGY_FILE)):,})")
print(f"{n_no_other_pitches:,} events skipped -- no OTHER qualifying pitch")
print(f"{n_no_valid_whiff_data:,} events skipped -- other pitches existed but "
      f"lacked valid whiff data in the NEW, stricter pull (this is the real "
      f"coverage cost of using better-grounded data)")

result_df.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# CROSS-CHECK AGAINST THE ORIGINAL (v1)
# ============================================================

print(f"\n\n==============================")
print("Cross-check: v1 (Kaggle whiff_rate) vs. v2 (filtered Savant pull)")
print("==============================")

original = pd.read_csv(ORIGINAL_SYNERGY_FILE)

compare = original.merge(
    result_df, on=["player_id", "season_from", "season_to", "pitch_type"], how="inner"
)

print(f"\n{len(compare):,} events have both a v1 and v2 score to compare")

if len(compare) > 0:
    corr = compare["arsenal_synergy_whiff_delta"].corr(compare["arsenal_synergy_whiff_delta_v2"])
    mean_abs_diff = (compare["arsenal_synergy_whiff_delta"] - compare["arsenal_synergy_whiff_delta_v2"]).abs().mean()

    print(f"\nCorrelation between v1 and v2: {corr:.3f}")
    print(f"Mean absolute difference: {mean_abs_diff:.4f}")

    if corr > 0.7:
        print(
            "\nHigh correlation -- the original feature's computation "
            "was already sound; the Kaggle whiff_rate values weren't "
            "meaningfully distorting it, even without a formal "
            "reliability floor behind them."
        )
    elif corr > 0.4:
        print(
            "\nModerate correlation -- the two sources broadly agree "
            "but diverge meaningfully in some cases. Worth inspecting "
            "which events disagree most before treating either version "
            "as clearly superior."
        )
    else:
        print(
            "\nLow correlation -- the original feature may have been "
            "meaningfully distorted by unreliable Kaggle whiff_rate "
            "values. v2 (built on the evidence-based MIN_PITCHES "
            "floor) is likely the more trustworthy version going "
            "forward."
        )

    compare["abs_diff"] = (
        compare["arsenal_synergy_whiff_delta"] - compare["arsenal_synergy_whiff_delta_v2"]
    ).abs()

    print(f"\nEvents where v1 and v2 disagree most (ALL cases, including single-pitch-driven v2 values):")
    print(
        compare.nlargest(10, "abs_diff")[
            ["player_id", "season_from", "season_to", "pitch_type",
             "arsenal_synergy_whiff_delta", "arsenal_synergy_whiff_delta_v2",
             "n_other_pitches_measured_v2", "abs_diff"]
        ].to_string(index=False)
    )

    # FILTERED disagreement check -- a v2 score built from only ONE
    # comparison pitch is itself a thin-evidence case (the weighted
    # average collapses to that single pitch's own raw delta) --
    # excluding these separates genuine v1/v2 disagreement from
    # cases where v2 is ALSO on shaky ground, not necessarily more
    # trustworthy just because it came from the better-filtered
    # source
    multi_pitch_compare = compare[compare["n_other_pitches_measured_v2"] >= 2].copy()

    print(
        f"\n\n{len(multi_pitch_compare):,} of {len(compare):,} comparable events "
        f"have n_other_pitches_measured_v2 >= 2 (excluding single-pitch-driven v2 values)"
    )

    if len(multi_pitch_compare) > 0:
        multi_pitch_corr = multi_pitch_compare["arsenal_synergy_whiff_delta"].corr(
            multi_pitch_compare["arsenal_synergy_whiff_delta_v2"]
        )
        multi_pitch_mean_abs_diff = multi_pitch_compare["abs_diff"].mean()

        print(f"Correlation (n_other_pitches_measured >= 2 only): {multi_pitch_corr:.3f}")
        print(f"Mean absolute difference (same subset): {multi_pitch_mean_abs_diff:.4f}")

        print(f"\nEvents where v1 and v2 disagree most (n_other_pitches_measured_v2 only):")
        print(
            multi_pitch_compare.nlargest(10, "abs_diff")[
                ["player_id", "season_from", "season_to", "pitch_type",
                 "arsenal_synergy_whiff_delta", "arsenal_synergy_whiff_delta_v2",
                 "n_other_pitches_measured_v2", "abs_diff"]
            ].to_string(index=False)
        )

        if abs(multi_pitch_corr - corr) > 0.1:
            print(
                f"\nNOTE: correlation shifts meaningfully once single-pitch-"
                f"driven v2 cases are excluded ({corr:.3f} -> "
                f"{multi_pitch_corr:.3f}) -- a real sign that thin-evidence "
                f"v2 values were distorting the overall comparison, not just "
                f"a coincidence."
            )
        else:
            print(
                f"\nCorrelation barely changes once single-pitch-driven v2 "
                f"cases are excluded ({corr:.3f} -> {multi_pitch_corr:.3f}) "
                f"-- the overall agreement isn't being driven by thin-"
                f"evidence cases specifically."
            )
    else:
        print("No events remain with n_other_pitches_measured_v2 >= 2 -- cannot run the filtered comparison.")
