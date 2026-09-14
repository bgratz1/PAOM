"""
77_master_pitch_table_by_year.py

Purpose:
--------
Direct extension of the real, confirmed master_pitch_table_2025.csv
generator script (shared directly by the user) to any other season
-- same aggregation logic, unchanged, just parameterized by year.
Feeds directly into Movement (09) and Velocity (21), both of which
depend on this file's exact schema.

Requires clean_statcast_{YEAR}.csv (from 75_clean_statcast_by_year.py,
which was updated specifically to produce the is_contact, is_strike,
is_sweet_spot flags and HB/IVB columns this script depends on --
discovered only after this real script was shared).

Output:
-------
master_pitch_table_{YEAR}.csv -- one row per (player_name, pitcher,
pitch_type, pitch_name, p_throws), matching the original's exact
column set and aggregation logic.
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

import sys
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
INPUT_FILE = f"clean_statcast_{YEAR}.csv"
OUTPUT_FILE = f"master_pitch_table_{YEAR}.csv"


# ============================================================
# LOAD
# ============================================================

print(f"Loading clean Statcast data for {YEAR}...")

df = pd.read_csv(INPUT_FILE)

print(f"Total pitches: {len(df)}")


# ============================================================
# DERIVED FLAGS -- UNCHANGED from the original
# ============================================================

df["is_contact"] = df["is_contact"].astype(int)
df["is_in_play"] = df["is_in_play"].astype(int)


# ============================================================
# GROUP PITCHER + PITCH TYPE -- UNCHANGED from the original
# ============================================================

summary = (
    df
    .groupby(
        [
            "player_name",
            "pitcher",
            "pitch_type",
            "pitch_name",
            "p_throws"
        ]
    )
    .agg(
        pitches=("pitch_type", "count"),
        balls=("balls", "count"),

        velo=("release_speed", "mean"),
        spin=("release_spin_rate", "mean"),
        spin_axis=("spin_axis", "mean"),
        HB=("HB", "mean"),
        IVB=("IVB", "mean"),
        extension=("release_extension", "mean"),
        release_x=("release_pos_x", "mean"),
        release_z=("release_pos_z", "mean"),

        swing_rate=("is_swing", "mean"),
        whiff_rate=("is_whiff", "mean"),
        called_strike_rate=("is_called_strike", "mean"),
        chase_rate=("is_chase", "mean"),
        strike_rate=("is_strike", "mean"),

        xwoba=("estimated_woba_using_speedangle", "mean"),
        xba=("estimated_ba_using_speedangle", "mean"),
        avg_exit_velocity=("launch_speed", "mean"),
        avg_launch_angle=("launch_angle", "mean"),

        hard_hit_rate=("is_hard_hit", "mean"),
        sweet_spot_rate=("is_sweet_spot", "mean"),

        run_value=("delta_run_exp", "sum"),
        avg_run_value=("delta_run_exp", "mean"),

        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        contacts=("is_contact", "sum"),
        balls_in_play=("is_in_play", "sum")
    )
    .reset_index()
)


# ============================================================
# CSW -- UNCHANGED
# ============================================================

summary["csw_rate"] = summary["whiff_rate"] + summary["called_strike_rate"]


# ============================================================
# RUN VALUE PER 100 PITCHES -- UNCHANGED
# ============================================================

summary["run_value_per_100"] = summary["run_value"] / summary["pitches"] * 100


# ============================================================
# USAGE -- UNCHANGED
# ============================================================

total_pitcher_pitches = summary.groupby("player_name")["pitches"].transform("sum")
summary["usage"] = summary["pitches"] / total_pitcher_pitches


# ============================================================
# SAVE
# ============================================================

summary = summary.round(4)

summary.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved {OUTPUT_FILE}")
print(f"\nRows: {len(summary)}")
print(f"\nColumns: {summary.columns.tolist()}")
print(f"\nSample:")
print(summary.head())
