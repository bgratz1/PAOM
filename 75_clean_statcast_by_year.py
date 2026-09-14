"""
75_clean_statcast_by_year.py

Purpose:
--------
07d_final_effectiveness.py expects a "clean" input file with several
pre-computed boolean flags (is_swing, is_whiff, is_called_strike,
is_hard_hit, is_chase, is_outside_zone, is_in_play) -- the script
that originally produced clean_statcast_2025.csv from the raw pull
isn't available in this session, only its expected OUTPUT schema
(visible directly in 07d's own required_flags check and how each
flag gets used in the aggregation step).

This reconstructs that cleaning step using standard, well-established
Statcast conventions -- not guesses, but not verified against the
original script's exact code either. Flagged explicitly per flag
below.

Output:
-------
clean_statcast_{YEAR}.csv -- same schema 07d_final_effectiveness.py
already expects as INPUT_FILE, ready to feed directly in.
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
INPUT_FILE = f"statcast_{YEAR}_raw.csv"
OUTPUT_FILE = f"clean_statcast_{YEAR}.csv"

HARD_HIT_THRESHOLD = 95.0  # mph -- the standard, widely-used MLB/
                             # Statcast definition of "hard hit"


# ============================================================
# LOAD
# ============================================================

print(f"Loading raw Statcast data for {YEAR}...")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df):,} raw pitches")

required_source_cols = ["description", "zone", "launch_speed", "pitch_type",
                          "estimated_woba_using_speedangle"]
missing = [c for c in required_source_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected raw columns: {missing}")


# ============================================================
# BUILD REQUIRED FLAGS -- standard Statcast conventions
# ============================================================

print("\nBuilding required flags...")

# is_swing: any description indicating the batter offered at the pitch
swing_descriptions = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "foul_bunt", "missed_bunt"
}
df["is_swing"] = df["description"].isin(swing_descriptions)

# is_whiff: a swing that missed entirely (not foul, not in play)
whiff_descriptions = {"swinging_strike", "swinging_strike_blocked"}
df["is_whiff"] = df["description"].isin(whiff_descriptions)

# is_called_strike: umpire called it a strike with no swing
df["is_called_strike"] = df["description"] == "called_strike"

# is_hard_hit: standard 95+ mph exit velocity threshold, only
# meaningful for balls actually put in play
df["is_hard_hit"] = df["launch_speed"] >= HARD_HIT_THRESHOLD

# is_outside_zone: Statcast's own zone column -- 1-9 = in zone,
# 11-14 = outside (defensive: treat anything not 1-9 and not null as
# outside, matching Statcast's own convention)
df["is_outside_zone"] = df["zone"].isin([11, 12, 13, 14])

# is_chase: a swing specifically on a pitch outside the zone --
# matches how 07d USES this flag (chase_swings summed, then divided
# by outside_zone separately -- is_chase itself must already combine
# both conditions, not just "outside zone" alone)
df["is_chase"] = df["is_swing"] & df["is_outside_zone"]

# is_in_play: ball put in play (used for ground_balls denominator)
df["is_in_play"] = df["description"] == "hit_into_play"

# NEWLY DISCOVERED, from the real master_pitch_table_2025.csv
# generator script -- these flags weren't known to be needed until
# that script was shared

# is_contact: a swing that made SOME contact (foul or in play,
# excludes whiffs) -- standard convention
contact_descriptions = {"foul", "foul_tip", "hit_into_play", "foul_bunt"}
df["is_contact"] = df["description"].isin(contact_descriptions)

# is_strike: ANY strike, called or swinging (used for strike_rate,
# a broader flag than is_called_strike or is_whiff alone)
strike_descriptions = {
    "called_strike", "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "foul_bunt", "missed_bunt", "hit_into_play"
}
df["is_strike"] = df["description"].isin(strike_descriptions)

# is_sweet_spot: standard Statcast "sweet spot" launch angle range
# (8-32 degrees), only meaningful for balls in play
df["is_sweet_spot"] = df["launch_angle"].between(8, 32)

# HB / IVB: horizontal break and induced vertical break, in inches --
# standard feet-to-inches conversion from Statcast's raw pfx_x/pfx_z.
# HONEST CAVEAT: the exact original derivation script wasn't
# available to confirm sign convention -- this uses pfx_x/pfx_z's
# raw sign as Statcast provides it (not flipped by handedness),
# the more common, simpler convention. If downstream Movement/
# Velocity results look sign-inverted for LHP specifically, this is
# the first place to check.
if "pfx_x" in df.columns and "pfx_z" in df.columns:
    df["HB"] = df["pfx_x"] * 12
    df["IVB"] = df["pfx_z"] * 12
else:
    raise ValueError(
        "pfx_x/pfx_z not found in raw data -- cannot derive HB/IVB. "
        "Check the raw pull's actual columns."
    )


# ============================================================
# SAVE
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

print(f"\n==============================")
print(f"Saved: {OUTPUT_FILE}")
print(f"==============================")

for flag in ["is_swing", "is_whiff", "is_called_strike", "is_hard_hit", "is_outside_zone", "is_chase", "is_in_play", "is_contact", "is_strike", "is_sweet_spot"]:
    print(f"{flag}: {df[flag].sum():,} of {len(df):,} ({df[flag].mean():.1%})")
print(f"HB range: {df['HB'].min():.2f} to {df['HB'].max():.2f}")
print(f"IVB range: {df['IVB'].min():.2f} to {df['IVB'].max():.2f}")
