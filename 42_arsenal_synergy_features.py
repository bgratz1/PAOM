"""
42_arsenal_synergy_features.py

Purpose:
--------
Tests a specific baseball mechanism directly: a new pitch can help a
pitcher's SEASON even if the pitch itself looks unremarkable in
isolation -- tunneling (a new slider that looks identical to the
fastball out of the hand can make the FASTBALL better, since hitters
can't cheat on one pitch without getting fooled by the other) and
setup value (a show-me changeup thrown 8% of the time specifically
to keep hitters honest elsewhere) are both real, well-understood
mechanisms that a "how good was the new pitch on its own" measure
can't capture at all.

This measures it directly: for each ADD event, did the pitcher's
OTHER, already-established pitches get BETTER (higher whiff_rate)
or WORSE after the new pitch was added? A positive synergy score is
consistent with the new pitch making the rest of the arsenal play
up; a negative one could mean the opposite (e.g. reps/development
time diverted from existing pitches), or nothing at all if the
underlying mechanism isn't tunneling-related for that pitcher.

RESOLVED (66/67_synergy_persistence_test.py): whether this reflects
a genuine tunneling effect or same-season "good year" contamination
was tested directly by checking whether the signal persists a full
season after the change. It doesn't -- a placebo test on the
persistent version came back indistinguishable from chance. This
feature remains a real, validated PREDICTOR (its own placebo test
proved that), but the mechanism behind it is most likely NOT
tunneling -- more likely a shared underlying cause (health,
mechanics, form) showing up in both this feature and the outcome
being predicted, not a causal link between them.

SCOPE: only pitch types that were qualifying (>=5% usage, the same
MEANINGFUL_USAGE_FLOOR used throughout this project) in BOTH
season_from and season_to are included in a given event's synergy
score -- a pitch that was ALSO dropped in the same transition has no
valid "after" whiff_rate to compare, so it's excluded rather than
producing a misleading delta.

WEIGHTING: each existing pitch's whiff_rate delta is weighted by its
OWN season_from usage share -- a bigger, more prominent existing
pitch improving matters more to the synergy story than a rarely-
thrown one, the same "usage matters" principle already used
throughout this project's other usage-weighted rollups.

Output:
-------
arsenal_synergy_features.csv -- one row per (player_id, season_from,
season_to, pitch_type) for ADD events, with a usage-weighted average
whiff_rate delta across the pitcher's other qualifying pitches, plus
the number of pitches that delta is based on (for reliability
weighting downstream, same as every other feature in this project).
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

OUTPUT_FILE = "arsenal_synergy_features.csv"

MEANINGFUL_USAGE_FLOOR = 5.0  # SAME threshold used throughout this
                                # project (09/34/37/40)

PITCH_TYPES = ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS"]


# ============================================================
# LOAD
# ============================================================

print("Loading change events and arsenal data...")

change_events = pd.read_csv(CHANGE_EVENTS_FILE)
arsenal = pd.read_csv(ARSENAL_FILE)

add_events = change_events[change_events["change_type"] == "ADD"].copy()

print(f"Loaded {len(add_events):,} ADD events")

arsenal_indexed = arsenal.set_index(["player_id", "season"])


# ============================================================
# COMPUTE SYNERGY PER ADD EVENT
# ============================================================

print("\nComputing arsenal synergy features...")

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

    # "other" pitches: qualifying in season_from, EXCLUDING the newly
    # added pitch type itself, AND still qualifying in season_to --
    # a pitch dropped in the same transition has no valid "after"
    # whiff_rate to compare, so it's excluded rather than producing
    # a misleading delta
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

    whiff_deltas = []
    weights = []
    for pt in other_pitch_types:
        whiff_from = row_from.get(f"{pt}_whiff_rate", np.nan)
        whiff_to = row_to.get(f"{pt}_whiff_rate", np.nan)
        usage_from = row_from.get(f"{pt}_usage_pct", np.nan)
        if pd.notna(whiff_from) and pd.notna(whiff_to):
            whiff_deltas.append(whiff_to - whiff_from)
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
        "n_other_pitches_measured": len(whiff_deltas),
        "arsenal_synergy_whiff_delta": synergy_score,
    })

result_df = pd.DataFrame(records)

print(f"\nComputed synergy features for {len(result_df):,} ADD events")
print(f"{n_no_other_pitches:,} events skipped -- no OTHER qualifying pitch in both seasons")
print(f"{n_no_valid_whiff_data:,} events skipped -- other pitches existed but had no valid whiff_rate data")


# ============================================================
# SAVE
# ============================================================

result_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n==============================")
print(f"Saved: {OUTPUT_FILE}")
print(f"==============================")

print(f"\nDistribution of arsenal_synergy_whiff_delta:")
print(result_df["arsenal_synergy_whiff_delta"].describe())

print(f"\nSample rows -- BIGGEST synergy gains (other pitches improved most):")
print(
    result_df.nlargest(10, "arsenal_synergy_whiff_delta")
    [["player_id", "season_from", "season_to", "pitch_type",
      "n_other_pitches_measured", "arsenal_synergy_whiff_delta"]]
    .to_string(index=False)
)

print(f"\nSample rows -- BIGGEST synergy losses (other pitches declined most):")
print(
    result_df.nsmallest(10, "arsenal_synergy_whiff_delta")
    [["player_id", "season_from", "season_to", "pitch_type",
      "n_other_pitches_measured", "arsenal_synergy_whiff_delta"]]
    .to_string(index=False)
)
