"""
32_pull_pitcher_season_metrics.py

Purpose:
--------
Pulls season-level pitcher EVALUATIVE metrics (WAR, ERA+, xwOBA-
against) for 2020-2025, to eventually regress arsenal CHANGES (from
pitcher_arsenal_evolution_2020_2025.csv) against changes in these
outcomes -- "if pitcher X adds pitch Y, predicted xwOBA/WAR change
is Z."

REVISED after a real-data run: this originally pulled WAR/FIP/xFIP
from FanGraphs via pybaseball.pitching_stats(), which HTML-scrapes
fangraphs.com/leaders-legacy.aspx. That endpoint is currently
blocking scraping traffic at the site level -- a real, widely-
reported, currently-open issue (pybaseball GitHub #479, explicitly
acknowledged by maintainers as "cannot be fixed on the code side"),
not a bug in this script or anything specific to one machine.

FIX: WAR now comes from pybaseball.bwar_pitch(), which pulls a
STATIC, publicly-hosted CSV file directly from baseball-reference.com
(not a scraped HTML page) -- much less likely to hit the same
anti-bot blocking. This ALSO eliminates the ID-crosswalk step
entirely for WAR: the file's own mlb_ID column IS the MLBAM ID
already, so no playerid_reverse_lookup is needed for this part.

WAR CHOICE, REVISED: this is now Baseball-Reference WAR (bWAR), not
FanGraphs WAR (fWAR) -- a real, different methodology, switched
specifically because bWAR's data source is currently reachable and
fWAR's currently is not. If FanGraphs' block lifts later and fWAR
specifically matters, that's a separate, reversible choice.

OPEN QUESTION, NOT YET RESOLVED: bwar_pitch() does not include
FIP/xFIP (only WAR, its components, and ERA+). Getting FIP requires
either (a) trying pybaseball.pitching_stats_bref() -- a DIFFERENT
scrape target than FanGraphs, untested here, may or may not also be
blocked -- or (b) computing FIP directly from raw Statcast counting
stats (HR, BB, HBP, K, IP) already available in this project's own
pipeline, which is more robust (no external scraper dependency at
all) but requires pulling full pitch-level Statcast data across all
six seasons, a much heavier operation than anything else in this
script. Neither is built here yet -- a real decision point, not
something to silently pick.

Output:
-------
pitcher_season_metrics_2020_2025.csv -- one row per (player_id
[MLBAM], season), with war, era_plus, and xwoba_against where
available. Joins directly to pitcher_arsenal_evolution_2020_2025.csv
on player_id + season. FIP/xFIP are NOT included pending the
decision above.
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

OUTPUT_FILE = "pitcher_season_metrics_2020_2025.csv"


# ============================================================
# STEP 1: PULL WAR FROM BASEBALL-REFERENCE'S STATIC DATA FILE
# ============================================================

print("Pulling WAR data from Baseball-Reference (static file, not scraped HTML)...")

bwar = pybaseball.bwar_pitch()

print(f"\nPulled {len(bwar):,} total player-season-stint rows (all-time)")
print(f"Columns returned: {bwar.columns.tolist()}")

expected_bwar_cols = ["mlb_ID", "year_ID", "name_common", "WAR", "ERA_plus"]
missing_bwar = [c for c in expected_bwar_cols if c not in bwar.columns]
if missing_bwar:
    print(
        f"\nWARNING: expected columns not found: {missing_bwar}. Check "
        f"the printed column list above -- baseball-reference.com may "
        f"have changed this file's structure since this script was "
        f"written."
    )

bwar = bwar[
    (bwar["year_ID"] >= START_SEASON) & (bwar["year_ID"] <= END_SEASON)
].copy()

print(f"\n{len(bwar):,} rows remain after filtering to {START_SEASON}-{END_SEASON}")

# a pitcher traded mid-season gets multiple STINT rows in one year --
# sum WAR (and BIP, for the filter below) across stints to get one
# true season total per pitcher
bwar_season = (
    bwar.groupby(["mlb_ID", "year_ID", "name_common"])
    .agg(war=("WAR", "sum"), era_plus=("ERA_plus", "mean"), bip=("BIP", "sum"))
    .reset_index()
)

# MINIMUM VOLUME FILTER: Baseball-Reference's WAR file includes
# ANYONE who recorded a pitching appearance, including position
# players pitching mop-up innings in blowouts (a real, semi-common
# modern phenomenon) -- these aren't small, noisy samples of a real
# pitcher's performance, they're a fundamentally different KIND of
# event and shouldn't be in this dataset at all. Confirmed directly:
# a real-data run showed Albert Pujols with ERA_plus=15 (meaning his
# effective ERA was ~7x league average -- no real pitcher, however
# bad, posts that).
#
# MIN_BIP is a reasonable, CONSERVATIVE starting floor, not a
# validated number -- same honest caveat every other cutoff in this
# project started with before being swept against real downstream
# validation (see 30/31_effectiveness_cutoff_sweep.py). Worth the
# same treatment later if this filter's exact value ends up mattering
# to the eventual regression.
MIN_BIP = 20

n_before_filter = len(bwar_season)
bwar_season = bwar_season[bwar_season["bip"] >= MIN_BIP].copy()
n_excluded = n_before_filter - len(bwar_season)

print(
    f"\nExcluded {n_excluded:,} pitcher-season rows with fewer than "
    f"{MIN_BIP} balls in play against (position-player mop-up "
    f"appearances and similarly trivial samples)"
)

bwar_season = bwar_season.rename(
    columns={"mlb_ID": "player_id", "year_ID": "season", "name_common": "name"}
)
bwar_season["player_id"] = bwar_season["player_id"].astype(int)

print(f"{len(bwar_season):,} pitcher-season rows after combining multi-team stints")


# ============================================================
# STEP 3: PULL STATCAST EXPECTED STATS (xwOBA-AGAINST) PER SEASON
# ============================================================

print(f"\nPulling Statcast expected stats (xwOBA-against) per season...")

expected_stats_frames = []

for season in range(START_SEASON, END_SEASON + 1):
    print(f"  {season}...")
    try:
        season_df = pybaseball.statcast_pitcher_expected_stats(season, minPA=1)
        season_df["season"] = season
        expected_stats_frames.append(season_df)
    except Exception as e:
        print(f"    FAILED for {season}: {e}")
    time.sleep(1)  # be polite to Savant's leaderboard endpoint

expected_stats = pd.concat(expected_stats_frames, ignore_index=True)

print(f"\nPulled {len(expected_stats):,} rows across all seasons")
print(f"Expected-stats columns: {expected_stats.columns.tolist()}")

# column naming for player_id and xwoba can vary -- find them
# defensively rather than assuming an exact name
player_id_col = next(
    (c for c in expected_stats.columns if c.lower() in ("player_id", "playerid")),
    None
)
xwoba_candidates = [c for c in expected_stats.columns if "woba" in c.lower() and "est" in c.lower()]
if not xwoba_candidates:
    xwoba_candidates = [c for c in expected_stats.columns if c.lower() == "xwoba"]
xwoba_col = xwoba_candidates[0] if xwoba_candidates else None

if player_id_col is None or xwoba_col is None:
    print(
        f"\nWARNING: could not automatically identify the player_id "
        f"and/or xwOBA columns from {expected_stats.columns.tolist()}. "
        f"Inspect these manually and adjust player_id_col/xwoba_col "
        f"above before trusting the merged output."
    )

if player_id_col and xwoba_col:
    expected_stats = expected_stats[[player_id_col, "season", xwoba_col]].rename(
        columns={player_id_col: "player_id", xwoba_col: "xwoba_against"}
    )
    expected_stats["player_id"] = expected_stats["player_id"].astype(int)


# ============================================================
# STEP 3b: PULL FIP COMPONENTS FROM BASEBALL-REFERENCE, COMPUTE FIP
# ============================================================

"""
pybaseball.pitching_stats_bref() does NOT return a pre-made FIP
column at all -- Baseball-Reference's basic pitching table only has
the raw ingredients (HR, BB, HBP, SO, IP, ER). So "try to pull it,
if not compute it" collapses into one step here: pull this table,
then compute FIP from its own components, since a ready-made column
isn't available regardless of whether the pull itself succeeds.

FIP = ((13*HR) + (3*(BB+HBP)) - (2*SO)) / IP + constant

The constant is DERIVED per season from this SAME pulled data (league
totals across every pitcher returned that season), not hardcoded --
FanGraphs' own "Guts!" constants page is JS-rendered and couldn't be
fetched directly to verify exact published values, so deriving it
from the identical underlying data this script already has is more
defensible than risking a stale or mismatched external number:

constant = lgERA - ((13*lgHR + 3*(lgBB+lgHBP) - 2*lgSO) / lgIP)
lgERA = 9 * sum(ER) / sum(IP)

NOTE: this scrapes baseball-reference.com via HTML (same general
mechanism as the FanGraphs pull that's currently blocked, just a
DIFFERENT site) -- untested whether Baseball-Reference is currently
blocking this kind of traffic. If every season below fails, that's
the likely explanation; the remaining fallback would be computing
FIP from full pitch-level Statcast data instead, a much heavier
pull not built here since it may not end up being necessary.

Join key: this table's own player identifier is NOT confirmed to be
the MLBAM ID (unlike bwar_pitch's mlb_ID) -- detected defensively
below, falling back to a NAME-based join against bwar_season's own
name column (both Baseball-Reference-sourced, so formatting should
at least be internally consistent) if no ID-like column is found.
"""

print(f"\nAttempting to pull FIP components from Baseball-Reference, {START_SEASON}-{END_SEASON}...")

fip_frames = []
fip_pull_failed_seasons = []

for season in range(START_SEASON, END_SEASON + 1):
    print(f"  {season}...")
    try:
        season_df = pybaseball.pitching_stats_bref(season)
        season_df["season"] = season
        fip_frames.append(season_df)
    except Exception as e:
        print(f"    FAILED for {season}: {e}")
        fip_pull_failed_seasons.append(season)
    time.sleep(1)  # be polite to Baseball-Reference

if fip_pull_failed_seasons:
    print(
        f"\nWARNING: {len(fip_pull_failed_seasons)} of "
        f"{END_SEASON - START_SEASON + 1} seasons failed to pull "
        f"({fip_pull_failed_seasons}) -- FIP will be missing for "
        f"these seasons. If ALL seasons failed, Baseball-Reference is "
        f"likely blocking this endpoint the same way FanGraphs is; "
        f"the remaining option is computing FIP from full pitch-level "
        f"Statcast data instead (heavier, not built here yet)."
    )

fip_computed = None

if fip_frames:
    fip_raw = pd.concat(fip_frames, ignore_index=True)
    print(f"\nPulled {len(fip_raw):,} pitcher-season rows with FIP components")
    print(f"Columns returned: {fip_raw.columns.tolist()}")

    fip_required = ["HR", "BB", "HBP", "SO", "IP", "ER"]
    fip_missing = [c for c in fip_required if c not in fip_raw.columns]

    if fip_missing:
        print(
            f"\nWARNING: expected FIP-component columns not found: "
            f"{fip_missing}. Cannot compute FIP this run -- check the "
            f"printed column list above."
        )
    else:
        # derive the FIP constant PER SEASON from this same pulled
        # data's league totals, rather than a hardcoded external value
        league_totals = fip_raw.groupby("season").agg(
            lg_hr=("HR", "sum"), lg_bb=("BB", "sum"), lg_hbp=("HBP", "sum"),
            lg_so=("SO", "sum"), lg_ip=("IP", "sum"), lg_er=("ER", "sum")
        )
        league_totals["lg_era"] = 9 * league_totals["lg_er"] / league_totals["lg_ip"]
        league_totals["fip_constant"] = league_totals["lg_era"] - (
            (13 * league_totals["lg_hr"] + 3 * (league_totals["lg_bb"] + league_totals["lg_hbp"])
             - 2 * league_totals["lg_so"]) / league_totals["lg_ip"]
        )

        print("\nDerived per-season FIP constants:")
        print(league_totals["fip_constant"].round(3))

        fip_raw = fip_raw.merge(
            league_totals[["fip_constant"]], on="season", how="left"
        )
        fip_raw["fip"] = (
            (13 * fip_raw["HR"] + 3 * (fip_raw["BB"] + fip_raw["HBP"]) - 2 * fip_raw["SO"])
            / fip_raw["IP"]
        ) + fip_raw["fip_constant"]

        # defensively find a join key -- an ID-like column if present,
        # otherwise fall back to name
        fip_id_col = next(
            (c for c in fip_raw.columns if c.lower() in ("mlbid", "mlb_id", "player_id", "playerid")),
            None
        )
        fip_name_col = next(
            (c for c in fip_raw.columns if c.lower() == "name"),
            None
        )

        if fip_id_col:
            print(f"\nJoining FIP by detected ID column: '{fip_id_col}'")
            fip_computed = fip_raw[[fip_id_col, "season", "fip"]].rename(
                columns={fip_id_col: "player_id"}
            )
            fip_computed["player_id"] = fip_computed["player_id"].astype(int)
        elif fip_name_col:
            print(
                f"\nNo ID column found -- joining FIP by name column "
                f"'{fip_name_col}' against bwar_season's own name field "
                f"instead (less reliable than an ID join, but both "
                f"sides are Baseball-Reference-sourced so formatting "
                f"should be reasonably consistent)."
            )
            fip_computed = fip_raw[[fip_name_col, "season", "fip"]].rename(
                columns={fip_name_col: "name"}
            )
        else:
            print(
                f"\nWARNING: no ID or name column found to join FIP "
                f"back to the rest of this output -- inspect "
                f"{fip_raw.columns.tolist()} manually."
            )


# ============================================================
# STEP 4: MERGE AND SAVE
# ============================================================

print("\nMerging Baseball-Reference WAR data with Statcast expected stats and FIP...")

final = bwar_season.copy()

if player_id_col and xwoba_col:
    final = final.merge(expected_stats, on=["player_id", "season"], how="left")
    n_with_xwoba = final["xwoba_against"].notna().sum()
    print(f"{n_with_xwoba:,} of {len(final):,} rows matched to an xwOBA-against value")
else:
    print("Skipping xwOBA merge -- expected-stats columns weren't identified above")

if fip_computed is not None:
    if "player_id" in fip_computed.columns:
        final = final.merge(fip_computed, on=["player_id", "season"], how="left")
    else:
        final = final.merge(fip_computed, on=["name", "season"], how="left")
    n_with_fip = final["fip"].notna().sum()
    print(f"{n_with_fip:,} of {len(final):,} rows matched to a FIP value")
else:
    print("Skipping FIP merge -- FIP could not be computed this run (see warnings above)")

final.to_csv(OUTPUT_FILE, index=False)

print(f"\n==============================")
print(f"Saved: {OUTPUT_FILE}")
print(f"==============================")
print(f"\nRows: {len(final):,}")
print(f"Unique pitchers: {final['player_id'].nunique():,}")
print(f"Seasons: {sorted(final['season'].unique())}")
print(f"\nColumns: {final.columns.tolist()}")
print(f"\nSample rows:")
print(final.head(10).to_string(index=False))
