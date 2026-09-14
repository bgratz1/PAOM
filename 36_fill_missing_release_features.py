"""
36_fill_missing_release_features.py

Purpose:
--------
35_pull_release_point_features.py's arm-angle leaderboard only
covers ~280 pitchers per year -- a real, substantial gap against the
~1,400-1,500 pitchers covered by pitcher_arsenal_evolution_2020_2025
.csv and pitcher_season_metrics_2020_2025.csv. Rather than dropping
every pitcher outside that smaller leaderboard from the similarity
metric (explicitly not wanted -- the whole point of this thread has
been not losing smaller-sample/less-prominent pitchers), this fills
the gap: for pitchers ENTIRELY MISSING from the arm-angle leaderboard,
pull raw release position (release_pos_x, release_pos_z) directly
from Statcast pitch-level data instead.

WHY release_x/release_z, NOT arm_angle, for the fallback: arm angle
is a COMPUTED metric specific to Savant's leaderboard, not a raw
Statcast column -- there's no way to derive it from raw pitch-level
data without reimplementing Savant's own (undocumented) calculation.
Raw release position is available directly, so that's what the
fallback provides. Every pitcher-season in the final combined output
will have release_x/release_z; only the ~280/year leaderboard-covered
subset will additionally have arm_angle.

WHY THIS IS LIGHTER THAN IT SOUNDS: pybaseball.statcast_pitcher()
takes a specific MLBAM player_id and pulls ONLY that pitcher's
pitch-level data for a date range -- a targeted, bounded pull, NOT a
full league-wide season dump. One call per MISSING PITCHER (not per
missing pitcher-SEASON -- each pitcher's full 2020-2025 range is
pulled in a single call, then aggregated by season locally), rather
than downloading everyone's data and filtering afterward.

STILL A REAL TIME COST: if ~1,000+ pitchers are missing from the
leaderboard, this means ~1,000+ individual API calls with a polite
delay between each -- expect this to take a while to run, likely
much longer than any other script in this project so far. This is
the honest tradeoff for filling the coverage gap without a full,
even heavier league-wide multi-year pull.

Output:
-------
pitcher_release_features_combined_2020_2025.csv -- one row per
(player_id, season), release_x and release_z for EVERY pitcher in
the broader population, arm_angle where available, and a
release_feature_source column ("arm_angle_leaderboard" or
"raw_statcast_fallback") so it's always clear which pitchers got
which kind of data.
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

ARM_ANGLE_FILE = "pitcher_release_features_2020_2025.csv"
ARSENAL_FILE = "pitcher_arsenal_evolution_2020_2025.csv"

OUTPUT_FILE = "pitcher_release_features_combined_2020_2025.csv"

START_SEASON = 2020
END_SEASON = 2025

# a wide date range covering all seasons -- pulled ONCE per missing
# pitcher, then split into seasons locally using game_date, rather
# than one call per pitcher-season
FULL_RANGE_START = f"{START_SEASON}-03-01"
FULL_RANGE_END = f"{END_SEASON}-11-30"

SLEEP_BETWEEN_CALLS = 1.5  # be polite to Savant's per-pitcher endpoint


# ============================================================
# LOAD PRIMARY SOURCE (ARM ANGLE LEADERBOARD)
# ============================================================

print("Loading arm-angle leaderboard data (primary source)...")

primary = pd.read_csv(ARM_ANGLE_FILE)

required_primary_cols = ["player_id", "season", "release_x", "release_z"]
missing_primary = [c for c in required_primary_cols if c not in primary.columns]
if missing_primary:
    raise ValueError(
        f"Missing required columns in {ARM_ANGLE_FILE}: {missing_primary}. "
        f"This script needs 35_pull_release_point_features.py's output."
    )

primary["release_feature_source"] = "arm_angle_leaderboard"

print(f"Primary source: {len(primary):,} rows, {primary['player_id'].nunique():,} unique pitchers")


# ============================================================
# DETERMINE WHO'S MISSING
# ============================================================

print("\nLoading broader population to determine coverage gap...")

arsenal = pd.read_csv(ARSENAL_FILE)

broader_population = arsenal[["player_id", "season"]].drop_duplicates()

covered = primary[["player_id", "season"]].drop_duplicates()
covered_pairs = set(zip(covered["player_id"], covered["season"]))

missing_pairs = broader_population[
    ~broader_population.apply(lambda r: (r["player_id"], r["season"]) in covered_pairs, axis=1)
]

missing_player_ids = sorted(missing_pairs["player_id"].unique())

print(
    f"{len(missing_pairs):,} pitcher-season pairs missing from the "
    f"arm-angle leaderboard, across {len(missing_player_ids):,} "
    f"unique pitchers"
)
print(
    f"\nExpected time: roughly {len(missing_player_ids) * SLEEP_BETWEEN_CALLS / 60:.0f}+ "
    f"minutes just for the polite delays between calls, before "
    f"accounting for actual download time -- this WILL take a while."
)


# ============================================================
# FILL THE GAP: TARGETED PER-PITCHER STATCAST PULLS
# ============================================================

print(f"\nPulling raw release position for {len(missing_player_ids):,} missing pitchers...")

fallback_rows = []
n_failed = 0

for i, player_id in enumerate(missing_player_ids):
    if i % 50 == 0:
        print(f"  {i}/{len(missing_player_ids)}...")

    try:
        pitch_data = pybaseball.statcast_pitcher(
            FULL_RANGE_START, FULL_RANGE_END, player_id
        )

        if pitch_data.empty or "release_pos_x" not in pitch_data.columns:
            n_failed += 1
            continue

        pitch_data["season"] = pd.to_datetime(pitch_data["game_date"]).dt.year

        season_agg = (
            pitch_data.groupby("season")
            .agg(release_x=("release_pos_x", "mean"), release_z=("release_pos_z", "mean"))
            .reset_index()
        )
        season_agg["player_id"] = player_id

        fallback_rows.append(season_agg)

    except Exception as e:
        n_failed += 1

    time.sleep(SLEEP_BETWEEN_CALLS)

print(f"\nCompleted. {n_failed:,} pitchers failed or returned no data.")

if fallback_rows:
    fallback = pd.concat(fallback_rows, ignore_index=True)
    fallback["release_feature_source"] = "raw_statcast_fallback"
    fallback["arm_angle"] = np.nan  # not derivable from raw Statcast

    # only keep the seasons actually needed (a pitcher's full-range
    # pull may include seasons outside what was actually missing,
    # e.g. if they were already covered by the primary source for
    # SOME of their seasons but not others)
    fallback = fallback.merge(
        missing_pairs[["player_id", "season"]], on=["player_id", "season"], how="inner"
    )

    print(f"Filled {len(fallback):,} pitcher-season rows via raw Statcast fallback")
else:
    fallback = pd.DataFrame(columns=["player_id", "season", "release_x", "release_z", "arm_angle", "release_feature_source"])
    print("No fallback rows produced -- all missing pitchers failed or had no data")


# ============================================================
# COMBINE AND SAVE
# ============================================================

print("\nCombining primary and fallback sources...")

combine_cols = ["player_id", "season", "arm_angle", "release_x", "release_z", "release_feature_source"]
primary_subset = primary[[c for c in combine_cols if c in primary.columns]]
fallback_subset = fallback[[c for c in combine_cols if c in fallback.columns]]

final = pd.concat([primary_subset, fallback_subset], ignore_index=True)
final = final.drop_duplicates(subset=["player_id", "season"], keep="first")

final.to_csv(OUTPUT_FILE, index=False)

print(f"\n==============================")
print(f"Saved: {OUTPUT_FILE}")
print(f"==============================")
print(f"\nTotal rows: {len(final):,}")
print(f"Unique pitchers: {final['player_id'].nunique():,}")
print(f"\nSource breakdown:")
print(final["release_feature_source"].value_counts())
print(
    f"\nCoverage vs. broader population: {len(final):,} of "
    f"{len(broader_population):,} pitcher-season pairs "
    f"({len(final)/len(broader_population):.1%})"
)
