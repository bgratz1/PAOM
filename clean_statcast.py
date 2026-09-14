# ============================================================
# 02_clean_statcast.py
#
# Cleans raw Statcast data for Pitch Arsenal Optimization Model
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# Load raw Statcast data
# ------------------------------------------------------------

print("Loading raw Statcast data...")

df = pd.read_csv("statcast_2025_raw.csv", low_memory=False)

print(f"Loaded {len(df):,} pitches.")

# ------------------------------------------------------------
# Columns to keep
# ------------------------------------------------------------

cols = [
    # Player information
    "player_name",
    "pitcher",
    "pitch_type",
    "pitch_name",
    "game_date",

    # Handedness
    "p_throws",
    "stand",

    # Count / situation
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",

    # Pitch physics
    "release_speed",
    "release_spin_rate",
    "spin_axis",

    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",

    "pfx_x",
    "pfx_z",

    "plate_x",
    "plate_z",

    "zone",

    # Outcome
    "description",
    "events",
    "type",

    # Batted ball
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle",

    # Run value
    "delta_run_exp",
    "woba_value"
]

# Keep only columns that exist
cols = [c for c in cols if c in df.columns]

df = df[cols].copy()

print(f"Keeping {len(cols)} columns.")

# ------------------------------------------------------------
# Remove rows with unknown pitch type
# ------------------------------------------------------------

df = df[df["pitch_type"].notna()].copy()

# ------------------------------------------------------------
# Convert movement to inches
# ------------------------------------------------------------

df["HB"] = df["pfx_x"] * 12
df["IVB"] = df["pfx_z"] * 12


# Season
df["season"] = 2025
# ------------------------------------------------------------
# Swing events
# ------------------------------------------------------------

swing_events = [
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score"
]

df["is_swing"] = df["description"].isin(swing_events)

# ------------------------------------------------------------
# Whiffs
# ------------------------------------------------------------

whiff_events = [
    "swinging_strike",
    "swinging_strike_blocked"
]

df["is_whiff"] = df["description"].isin(whiff_events)

# ------------------------------------------------------------
# Called strikes
# ------------------------------------------------------------

df["is_called_strike"] = df["description"] == "called_strike"

# ------------------------------------------------------------
# Foul balls
# ------------------------------------------------------------

foul_events = [
    "foul",
    "foul_tip"
]

df["is_foul"] = df["description"].isin(foul_events)

# ------------------------------------------------------------
# Ball in play
# ------------------------------------------------------------

df["is_in_play"] = df["description"].str.contains(
    "hit_into_play",
    na=False
)

# ------------------------------------------------------------
# Contact
# ------------------------------------------------------------

df["is_contact"] = df["launch_speed"].notna()

# ------------------------------------------------------------
# Hard-hit balls
# ------------------------------------------------------------

df["is_hard_hit"] = df["launch_speed"] >= 95

# ------------------------------------------------------------
# Sweet spot contact
# ------------------------------------------------------------

df["is_sweet_spot"] = (
    (df["launch_angle"] >= 8) &
    (df["launch_angle"] <= 32)
)

# ------------------------------------------------------------
# Chase
#
# Statcast zone numbers:
#
# 1-9 = strike zone
# 11-14 = outside zone
# ------------------------------------------------------------

df["is_outside_zone"] = df["zone"].isin([11, 12, 13, 14])

df["is_chase"] = (
    df["is_outside_zone"] &
    df["is_swing"]
)

# ------------------------------------------------------------
# Strike indicator
# ------------------------------------------------------------

strike_descriptions = [
    "called_strike",
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score"
]

df["is_strike"] = df["description"].isin(strike_descriptions)

# Pitch number for each pitcher
df["pitch_num"] = df.groupby("pitcher").cumcount() + 1

# ------------------------------------------------------------
# Two-strike count
# ------------------------------------------------------------

df["two_strike_count"] = df["strikes"] == 2

# ------------------------------------------------------------
# Ahead / behind in count
# ------------------------------------------------------------

df["pitcher_ahead"] = df["balls"] < df["strikes"]

df["pitcher_behind"] = df["balls"] > df["strikes"]

# ------------------------------------------------------------
# Remove pitches with missing velocity
# ------------------------------------------------------------

df = df[df["release_speed"].notna()].copy()

# ------------------------------------------------------------
# Reset index
# ------------------------------------------------------------

df.reset_index(drop=True, inplace=True)

# ------------------------------------------------------------
# Save cleaned data
# ------------------------------------------------------------

output_file = "clean_statcast_2025.csv"

df.to_csv(output_file, index=False)

print("\nCleaning complete!")
print(f"Saved to: {output_file}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nSample:")
print(df.head())