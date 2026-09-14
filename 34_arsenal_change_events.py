"""
34_arsenal_change_events.py

Purpose:
--------
Extracts ARSENAL CHANGE EVENTS from pitcher_arsenal_evolution_2020_
2025.csv (the Kaggle dataset) -- pitch adds, drops, and meaningful
usage-share shifts between consecutive seasons for each pitcher.
This is the foundational dataset everything downstream depends on:
the similarity-weighted regression trains on these events, and the
"most similar historical pitchers who made this change" comps are
drawn directly from them.

WHAT COUNTS AS A CHANGE:
- ADD: a pitch type below MEANINGFUL_USAGE_FLOOR (or absent) in
  season t, at or above it in season t+1
- DROP: the mirror image -- at or above the floor in season t,
  below it (or absent) in season t+1
- USAGE_SHIFT: present at a meaningful level in BOTH seasons, but
  usage share moved by at least USAGE_SHIFT_THRESHOLD percentage
  points without crossing the floor either direction (a real,
  intentional re-weighting of an existing pitch, not a full add/drop)

MEANINGFUL_USAGE_FLOOR reuses the SAME 5% threshold already
established in 09_movement_component.py's MIN_PITCH_TYPE_USAGE --
deliberate reuse for consistency ("is this pitch really part of the
arsenal" should mean the same thing everywhere in this project), not
a new, independently-chosen number.

USAGE_SHIFT_THRESHOLD (8 percentage points) is a reasonable starting
guess, NOT yet validated -- same honest caveat every other threshold
in this project started with before being swept against real
downstream results (see 26-31_*_sweep.py). Worth the same treatment
later once the regression exists to validate against.

ONLY TRUE CONSECUTIVE SEASONS (t, t+1 where t+1 == t+1, no gap year)
are compared. A pitcher with a skipped season (injury, minors, etc.)
gets NO change event across that gap -- attributing an outcome
change to "arsenal change" when a full year of other unknown things
also happened in between makes the causal story meaningfully murkier,
so this is deliberately excluded rather than silently treated the
same as a genuine adjacent-year comparison.

Output:
-------
1. arsenal_change_events_pitch_level.csv -- one row per (player_id,
   season_from, season_to, pitch_type, change_type), with the
   changed pitch's own characteristics (velo, spin, movement, whiff
   rate) captured at the relevant season.
2. arsenal_change_events_summary.csv -- one row per (player_id,
   season_from, season_to), summarizing ALL changes in that
   transition (a pitcher can add one pitch and drop another in the
   same offseason) -- this is the natural unit for the eventual
   regression's independent variable, and lets "stable arsenal"
   pitcher-seasons (zero changes) serve as an implicit control group.
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "pitcher_arsenal_evolution_2020_2025.csv"

PITCH_LEVEL_OUTPUT = "arsenal_change_events_pitch_level.csv"
SUMMARY_OUTPUT = "arsenal_change_events_summary.csv"

MEANINGFUL_USAGE_FLOOR = 5.0  # percentage points -- SAME threshold as
                                # 09_movement_component.py's
                                # MIN_PITCH_TYPE_USAGE=0.05, reused
                                # deliberately for consistency

USAGE_SHIFT_THRESHOLD = 8.0  # percentage points -- starting guess,
                               # not yet validated, see docstring above

# common pitch type codes -- excludes rare/obsolete classifications
# (SC, PO, UN, FO, EP) that were >99% missing when this file was
# first inspected, not meaningful enough to track change events for
PITCH_TYPES = ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS"]


# ============================================================
# LOAD
# ============================================================

print("Loading arsenal evolution data...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} pitcher-season rows, {df['player_id'].nunique():,} unique pitchers")

df = df.sort_values(["player_id", "season"]).reset_index(drop=True)


# ============================================================
# DETECT CHANGE EVENTS BETWEEN CONSECUTIVE SEASONS
# ============================================================

print("\nDetecting change events between consecutive seasons...")

pitch_level_events = []
summary_rows = []

n_pitchers_with_multiple_seasons = 0
n_true_consecutive_pairs = 0
n_gap_pairs_skipped = 0

for player_id, group in df.groupby("player_id"):
    group = group.sort_values("season")
    seasons = group["season"].tolist()

    if len(seasons) < 2:
        continue  # can't detect a change with only one season

    n_pitchers_with_multiple_seasons += 1

    for i in range(len(seasons) - 1):
        season_from = seasons[i]
        season_to = seasons[i + 1]

        if season_to - season_from != 1:
            n_gap_pairs_skipped += 1
            continue  # gap year -- deliberately excluded, see docstring

        n_true_consecutive_pairs += 1

        row_from = group[group["season"] == season_from].iloc[0]
        row_to = group[group["season"] == season_to].iloc[0]

        transition_events = []

        for pt in PITCH_TYPES:
            usage_before = row_from.get(f"{pt}_usage_pct", np.nan)
            usage_after = row_to.get(f"{pt}_usage_pct", np.nan)

            usage_before = 0.0 if pd.isna(usage_before) else usage_before
            usage_after = 0.0 if pd.isna(usage_after) else usage_after

            was_present = usage_before >= MEANINGFUL_USAGE_FLOOR
            is_present = usage_after >= MEANINGFUL_USAGE_FLOOR

            change_type = None

            if not was_present and is_present:
                change_type = "ADD"
                char_season = row_to
            elif was_present and not is_present:
                change_type = "DROP"
                char_season = row_from
            elif was_present and is_present:
                delta = usage_after - usage_before
                if abs(delta) >= USAGE_SHIFT_THRESHOLD:
                    change_type = "USAGE_INCREASE" if delta > 0 else "USAGE_DECREASE"
                    char_season = row_to

            if change_type is not None:
                event = {
                    "player_id": player_id,
                    "player_name": row_to["player_name"],
                    "season_from": season_from,
                    "season_to": season_to,
                    "pitch_type": pt,
                    "change_type": change_type,
                    "usage_pct_before": usage_before,
                    "usage_pct_after": usage_after,
                    "usage_pct_delta": usage_after - usage_before,
                    "avg_speed": char_season.get(f"{pt}_avg_speed", np.nan),
                    "avg_spin": char_season.get(f"{pt}_avg_spin", np.nan),
                    "whiff_rate": char_season.get(f"{pt}_whiff_rate", np.nan),
                    "avg_pfx_x": char_season.get(f"{pt}_avg_pfx_x", np.nan),
                    "avg_pfx_z": char_season.get(f"{pt}_avg_pfx_z", np.nan),
                }
                pitch_level_events.append(event)
                transition_events.append(event)

        n_adds = sum(1 for e in transition_events if e["change_type"] == "ADD")
        n_drops = sum(1 for e in transition_events if e["change_type"] == "DROP")
        n_increases = sum(1 for e in transition_events if e["change_type"] == "USAGE_INCREASE")
        n_decreases = sum(1 for e in transition_events if e["change_type"] == "USAGE_DECREASE")

        summary_rows.append({
            "player_id": player_id,
            "player_name": row_to["player_name"],
            "season_from": season_from,
            "season_to": season_to,
            "n_pitches_added": n_adds,
            "n_pitches_dropped": n_drops,
            "n_pitches_usage_increased": n_increases,
            "n_pitches_usage_decreased": n_decreases,
            "total_changes": len(transition_events),
            "pitches_added": ",".join(e["pitch_type"] for e in transition_events if e["change_type"] == "ADD"),
            "pitches_dropped": ",".join(e["pitch_type"] for e in transition_events if e["change_type"] == "DROP"),
            "had_any_change": len(transition_events) > 0
        })

print(f"\n{n_pitchers_with_multiple_seasons:,} pitchers had 2+ seasons in the data")
print(f"{n_true_consecutive_pairs:,} true consecutive-season pairs compared")
print(f"{n_gap_pairs_skipped:,} gap-year pairs excluded (season_to - season_from != 1)")


# ============================================================
# SAVE
# ============================================================

pitch_level_df = pd.DataFrame(pitch_level_events)
summary_df = pd.DataFrame(summary_rows)

pitch_level_df.to_csv(PITCH_LEVEL_OUTPUT, index=False)
summary_df.to_csv(SUMMARY_OUTPUT, index=False)

print(f"\n==============================")
print(f"Saved: {PITCH_LEVEL_OUTPUT} ({len(pitch_level_df):,} events)")
print(f"Saved: {SUMMARY_OUTPUT} ({len(summary_df):,} pitcher-season-transitions)")
print(f"==============================")

print(f"\nChange type breakdown (pitch-level events):")
print(pitch_level_df["change_type"].value_counts())

print(f"\nTransitions with at least one change: {summary_df['had_any_change'].sum():,} of {len(summary_df):,}")
print(f"Transitions with a STABLE arsenal (zero changes): {(~summary_df['had_any_change']).sum():,}")

print(f"\nMost commonly added pitch types:")
print(pitch_level_df[pitch_level_df["change_type"] == "ADD"]["pitch_type"].value_counts())

print(f"\nMost commonly dropped pitch types:")
print(pitch_level_df[pitch_level_df["change_type"] == "DROP"]["pitch_type"].value_counts())

print(f"\nSample ADD events:")
print(
    pitch_level_df[pitch_level_df["change_type"] == "ADD"]
    [["player_name", "season_from", "season_to", "pitch_type", "usage_pct_after", "avg_speed"]]
    .head(10)
    .to_string(index=False)
)
