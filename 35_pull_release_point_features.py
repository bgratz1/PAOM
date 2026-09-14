"""
35_pull_release_point_features.py

Purpose:
--------
Pulls per-pitcher release-point features (arm angle, release position)
for 2020-2025, closing the gap needed for the similarity-weighted
regression's distance metric -- Movement and Velocity already have
validated per-pitcher features from this project's own scoring
components, but release point was never turned into scoring features
despite existing elsewhere in this project (19_release_map_export.py,
the dashboard's Release Point Map, an earlier tabled "Release
Consistency" component).

DATA SOURCE: savant_extras.pitcher_arm_angle_range(), a third-party
package -- but inspected directly before use (not taken on faith):
it's a thin, fully transparent wrapper that hits Baseball Savant's
own public leaderboard CSV export
(baseballsavant.mlb.com/leaderboard/pitcher-arm-angles) directly --
the SAME underlying mechanism already trusted for the xwOBA-against
pull in 32_pull_pitcher_season_metrics.py, just a different
leaderboard path. The "new dependency" risk is really just "a few
lines building a URL," not an opaque new data pipeline.

Chosen over pulling full multi-year raw Statcast pitch-level data
and computing release_pos_x/release_pos_z/release_extension
ourselves -- this leaderboard is dramatically lighter (one CSV per
season vs. hundreds of thousands of pitch-level rows), and arm angle
specifically may be a MORE meaningful similarity feature than raw
release coordinates anyway: raw release x/z can shift somewhat with
mound positioning or stride variation even without a real mechanical
change, while arm angle is a more direct read on delivery style.

Column names are NOT verified live (no network access from this
environment to the real endpoint) -- handled defensively below,
same pattern as every other external pull in this project since the
FanGraphs endpoint turned out to be blocked.

Output:
-------
pitcher_release_features_2020_2025.csv -- one row per (player_id,
season), with arm angle and release position features. Joins
directly to pitcher_arsenal_evolution_2020_2025.csv and
pitcher_season_metrics_2020_2025.csv on player_id + season.
"""

import pandas as pd
import numpy as np

from savant_extras import pitcher_arm_angle_range

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

START_SEASON = 2020
END_SEASON = 2025

OUTPUT_FILE = "pitcher_release_features_2020_2025.csv"


# ============================================================
# PULL
# ============================================================

print(f"Pulling pitcher arm angle / release point data, {START_SEASON}-{END_SEASON}...")
print("(one Savant leaderboard CSV per season, much lighter than a full pitch-level pull)")

raw = pitcher_arm_angle_range(START_SEASON, END_SEASON)

if raw.empty:
    raise ValueError(
        "pitcher_arm_angle_range returned no data at all -- either "
        "the Savant endpoint is unreachable from this network, or "
        "the leaderboard URL/parameters have changed since this "
        "script was written. Check https://baseballsavant.mlb.com/"
        "leaderboard/pitcher-arm-angles manually to confirm the "
        "leaderboard still exists in this form."
    )

print(f"\nPulled {len(raw):,} rows across all seasons")
print(f"Columns returned: {raw.columns.tolist()}")


# ============================================================
# DEFENSIVE COLUMN DETECTION
# ============================================================

# player ID -- Savant's own leaderboards have consistently used the
# MLBAM-native "player_id" or "pitcher" naming elsewhere in this
# project's pulls (expected_stats used "player_id"); check the
# plausible variants
player_id_col = next(
    (c for c in raw.columns if c.lower() in ("player_id", "pitcher", "pitcher_id", "mlbam_id", "playerid")),
    None
)

# release/arm-angle columns -- documented as present but exact
# casing/naming not independently confirmed live
angle_col = next((c for c in raw.columns if "angle" in c.lower()), None)
release_x_col = next(
    (c for c in raw.columns if "release" in c.lower() and "x" in c.lower()),
    None
)
release_z_col = next(
    (c for c in raw.columns if ("release" in c.lower() or "shoulder" in c.lower()) and "z" in c.lower()),
    None
)

name_col = next(
    (c for c in raw.columns if c.lower() in ("name", "player_name", "pitcher_name")),
    None
)

print(f"\nDetected columns -- player_id: '{player_id_col}', name: '{name_col}', "
      f"angle: '{angle_col}', release_x: '{release_x_col}', release_z: '{release_z_col}'")

if player_id_col is None:
    print(
        "\nWARNING: no player ID column detected. This data cannot be "
        "joined to the rest of the project without one -- inspect "
        f"{raw.columns.tolist()} manually and adjust the detection "
        "logic above."
    )

if angle_col is None and release_x_col is None and release_z_col is None:
    print(
        "\nWARNING: no release-angle or release-position columns "
        "detected at all -- this leaderboard's schema may have "
        "changed since this script was written. Inspect the printed "
        "column list above."
    )


# ============================================================
# BUILD OUTPUT
# ============================================================

keep_cols = {}
if player_id_col:
    keep_cols[player_id_col] = "player_id"
if "year" in raw.columns:
    keep_cols["year"] = "season"
if name_col:
    keep_cols[name_col] = "name"
if angle_col:
    keep_cols[angle_col] = "arm_angle"
if release_x_col:
    keep_cols[release_x_col] = "release_x"
if release_z_col:
    keep_cols[release_z_col] = "release_z"

available_cols = {k: v for k, v in keep_cols.items() if k in raw.columns}
final = raw[list(available_cols.keys())].rename(columns=available_cols)

if "player_id" in final.columns:
    final["player_id"] = pd.to_numeric(final["player_id"], errors="coerce")
    n_before = len(final)
    final = final[final["player_id"].notna()].copy()
    final["player_id"] = final["player_id"].astype(int)
    if len(final) < n_before:
        print(f"\nDropped {n_before - len(final):,} rows with no valid player_id")

final.to_csv(OUTPUT_FILE, index=False)

print(f"\n==============================")
print(f"Saved: {OUTPUT_FILE}")
print(f"==============================")
print(f"\nRows: {len(final):,}")
if "player_id" in final.columns:
    print(f"Unique pitchers: {final['player_id'].nunique():,}")
if "season" in final.columns:
    print(f"Seasons: {sorted(final['season'].unique())}")
print(f"\nColumns: {final.columns.tolist()}")
print(f"\nSample rows:")
print(final.head(10).to_string(index=False))
