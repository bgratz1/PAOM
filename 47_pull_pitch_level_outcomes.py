"""
47_pull_pitch_level_outcomes.py

Purpose:
--------
Every outcome metric used so far (32_pull_pitcher_season_metrics.py)
is SEASON-WIDE -- one xwOBA-against number per pitcher per year,
blending every pitch they threw into one aggregate. This is the
root cause of the mean-reversion-dominance problem this whole
thread kept running into: asking one arsenal change to explain a
whole season's contact quality means the change-specific signal is
competing against everything else that happened that season.

This pulls PITCH-TYPE-LEVEL outcome data instead -- how well did
hitters make contact specifically against THIS pitch type, not the
pitcher's whole arsenal blended together. Source: Baseball Savant's
pitch-arsenal-stats leaderboard (via pybaseball.
statcast_pitcher_arsenal_stats), the same general Savant leaderboard
mechanism already validated for the season-wide pull in
32_pull_pitcher_season_metrics.py.

BUILT DEFENSIVELY: the exact column names/row structure returned by
this specific endpoint were not independently verified live (no
network access to Savant from this environment) -- pybaseball's own
source code has a telling comment ("test to see if pitch types needs
to be implemented") suggesting even the library's maintainers were
uncertain whether pitch-type breakdown comes back automatically.
Column detection and row-structure checks below are explicit and
printed clearly, so a real run reveals the truth rather than this
script silently assuming a structure that turns out wrong.

MIN_PA is set low (not the function's own default of 25) --
consistent with this project's established "more pitchers, not
fewer" philosophy (see Effectiveness's cutoff work), rather than
silently excluding thin-sample pitch types before we've even looked
at the real coverage tradeoff.

Output:
-------
pitcher_pitch_level_outcomes_2020_2025.csv -- one row per (player_id,
season, pitch_type), with xwOBA-against and other outcome columns
found for that SPECIFIC pitch type (not season-wide).
"""

import pandas as pd
import numpy as np
import time

import pybaseball

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

START_SEASON = 2020
END_SEASON = 2025

OUTPUT_FILE = "pitcher_pitch_level_outcomes_2020_2025.csv"

MIN_PA = 1  # SAME "include everyone, don't silently exclude thin
             # samples before looking at the real tradeoff"
             # philosophy already established for Effectiveness's
             # cutoffs -- NOT the function's own default of 25.
             # This is Savant's OWN plate-appearances-against filter
             # at the SOURCE PULL level -- separate from MIN_PITCHES
             # below, which filters the pitch-type-level rows this
             # script actually produces.

MIN_PITCHES = 50  # Added based on real evidence from 48_pitch_
                    # level_outcome_stability_diagnostic.py's actual
                    # run: xwoba_against's std dropped from 0.361 at
                    # the smallest bucket to 0.056 at the largest (a
                    # 6.4x difference); run_value_per_100 was far
                    # worse, with values like -271.1 and -256.5 built
                    # from just 1-2 pitches each. Every extreme value
                    # across all three metrics had a median pitch
                    # count of 1-16, against a population median of
                    # 106 -- unambiguous small-sample instability, not
                    # a guess. 50 sits where the sharpest gains in
                    # stability (Check 3's std/max_abs_from_median
                    # columns) had mostly already happened, retaining
                    # 67% of rows and 85% of pitchers. NOTE:
                    # run_value_per_100 remains noisier than the other
                    # two metrics even at this floor (max_abs_from_
                    # median was still 12.6 at 50) -- a real, disclosed
                    # limitation, not silently fixed beyond this floor.


# ============================================================
# PULL, YEAR BY YEAR
# ============================================================

print(f"Pulling pitch-arsenal-level outcome stats, {START_SEASON}-{END_SEASON}...")

frames = []

for season in range(START_SEASON, END_SEASON + 1):
    print(f"  {season}...")
    try:
        season_df = pybaseball.statcast_pitcher_arsenal_stats(season, minPA=MIN_PA)
        season_df["season"] = season
        frames.append(season_df)
    except Exception as e:
        print(f"    FAILED for {season}: {e}")
    time.sleep(1)  # be polite to Savant's leaderboard endpoint

if not frames:
    raise ValueError("No seasons pulled successfully -- nothing to process.")

raw = pd.concat(frames, ignore_index=True)

print(f"\nPulled {len(raw):,} total rows across all seasons")
print(f"Columns returned: {raw.columns.tolist()}")


# ============================================================
# CONFIRM ROW STRUCTURE: IS THIS ACTUALLY PITCH-TYPE-LEVEL?
# ============================================================

# defensive detection -- exact column names not independently
# verified live, see docstring above
player_id_col = next(
    (c for c in raw.columns if c.lower() in ("player_id", "pitcher", "pitcher_id")),
    None
)
pitch_type_col = next(
    (c for c in raw.columns if c.lower() in ("pitch_type", "pitch_name", "pitchtype")),
    None
)
xwoba_col = next(
    (c for c in raw.columns if "woba" in c.lower() and ("est" in c.lower() or c.lower().startswith("x"))),
    None
)
whiff_col = next(
    (c for c in raw.columns if "whiff" in c.lower()),
    None
)
run_value_col = next(
    (c for c in raw.columns if "run_value" in c.lower()),
    None
)
pitches_col = next(
    (c for c in raw.columns if c.lower() in ("pitches", "n_pitches")),
    None
)

print(
    f"\nDetected columns -- player_id: '{player_id_col}', "
    f"pitch_type: '{pitch_type_col}', xwoba: '{xwoba_col}', "
    f"whiff: '{whiff_col}', run_value: '{run_value_col}', "
    f"pitches: '{pitches_col}'"
)

if player_id_col is None or pitch_type_col is None:
    raise ValueError(
        f"Could not identify player_id and/or pitch_type columns in "
        f"{raw.columns.tolist()}. This is the critical check -- if "
        f"pitch_type isn't a real column here, this endpoint is NOT "
        f"returning pitch-type-level rows and a different approach "
        f"is needed. Inspect the printed column list manually."
    )

# THE key confirmation: does a single player_id+season have MULTIPLE
# rows (one per pitch type), or just one row per pitcher-season (an
# aggregate, same as what 32 already pulls, in which case this
# script hasn't actually gained anything new)?
rows_per_pitcher_season = raw.groupby([player_id_col, "season"]).size()
print(f"\nRows per (player_id, season): min={rows_per_pitcher_season.min()}, "
      f"median={rows_per_pitcher_season.median()}, max={rows_per_pitcher_season.max()}")

if rows_per_pitcher_season.median() <= 1:
    print(
        "\nWARNING: most pitcher-seasons have only ONE row -- this "
        "does NOT appear to be genuinely pitch-type-level data. "
        "Check whether the pitchType URL parameter needs to be set "
        "explicitly (looped per pitch type) rather than left blank."
    )
else:
    print(
        "\nCONFIRMED: multiple rows per pitcher-season -- this IS "
        "genuinely broken out by pitch type."
    )


# ============================================================
# BUILD AND SAVE OUTPUT
# ============================================================

keep_cols = {player_id_col: "player_id", "season": "season", pitch_type_col: "pitch_type"}
if xwoba_col:
    keep_cols[xwoba_col] = "xwoba_against"
if whiff_col:
    keep_cols[whiff_col] = "whiff_pct"
if run_value_col:
    keep_cols[run_value_col] = "run_value_per_100"
if pitches_col:
    keep_cols[pitches_col] = "pitches"

available_cols = {k: v for k, v in keep_cols.items() if k in raw.columns}
final = raw[list(available_cols.keys())].rename(columns=available_cols)

final["player_id"] = pd.to_numeric(final["player_id"], errors="coerce")
final = final[final["player_id"].notna()].copy()
final["player_id"] = final["player_id"].astype(int)

if "pitches" in final.columns:
    n_before_filter = len(final)
    final = final[final["pitches"] >= MIN_PITCHES].copy()
    print(
        f"\nApplied MIN_PITCHES={MIN_PITCHES} filter: {n_before_filter:,} -> "
        f"{len(final):,} rows ({len(final)/n_before_filter:.1%} retained)"
    )
else:
    print(
        f"\nWARNING: 'pitches' column not found -- could not apply "
        f"MIN_PITCHES={MIN_PITCHES} filter. Output is UNFILTERED and "
        f"will contain the same small-sample instability "
        f"48_pitch_level_outcome_stability_diagnostic.py found."
    )

final.to_csv(OUTPUT_FILE, index=False)

print(f"\n==============================")
print(f"Saved: {OUTPUT_FILE}")
print(f"==============================")
print(f"\nRows: {len(final):,}")
print(f"Unique pitchers: {final['player_id'].nunique():,}")
print(f"Unique pitch types: {sorted(final['pitch_type'].unique().tolist()) if pitch_type_col else 'N/A'}")
print(f"Seasons: {sorted(final['season'].unique())}")
print(f"\nColumns: {final.columns.tolist()}")

if "xwoba_against" in final.columns:
    print(f"\nDistribution of xwoba_against (pitch-type level):")
    print(final["xwoba_against"].describe())

print(f"\nSample rows:")
print(final.head(15).to_string(index=False))
