# ============================================================
# 03_pitch_summary.py
#
# Creates pitcher + pitch type level summary
# ============================================================

import pandas as pd


# ------------------------------------------------------------
# Load cleaned Statcast
# ------------------------------------------------------------

print("Loading cleaned Statcast...")

df = pd.read_csv(
    "clean_statcast_2025.csv",
    low_memory=False
)

print(f"Loaded {len(df):,} pitches")


# ------------------------------------------------------------
# Calculate pitcher pitch totals
# Used for usage percentage
# ------------------------------------------------------------

pitcher_totals = (
    df
    .groupby("player_name")
    .size()
)


# ------------------------------------------------------------
# Contact-only dataset
# Needed because xwOBA only exists on contact
# ------------------------------------------------------------

contact = df[
    df["launch_speed"].notna()
].copy()



# ------------------------------------------------------------
# Aggregate pitch characteristics
# ------------------------------------------------------------

summary = (

df
.groupby(
    [
        "player_name",
        "pitch_type",
        "pitch_name"
    ]
)

.agg(

    # Volume
    pitches=("pitch_type","count"),

    # Physics
    velo=("release_speed","mean"),
    spin=("release_spin_rate","mean"),

    HB=("HB","mean"),
    IVB=("IVB","mean"),

    extension=("release_extension","mean"),

    release_x=("release_pos_x","mean"),
    release_z=("release_pos_z","mean"),


    # Results

    whiff_rate=("is_whiff","mean"),

    swing_rate=("is_swing","mean"),

    called_strike_rate=(
        "is_called_strike",
        "mean"
    ),

    chase_rate=(
        "is_chase",
        "mean"
    ),

    hard_hit_rate=(
        "is_hard_hit",
        "mean"
    ),

    strike_rate=(
        "is_strike",
        "mean"
    ),


)

.reset_index()

)


# ------------------------------------------------------------
# Add contact results separately
# ------------------------------------------------------------

contact_summary = (

contact

.groupby(
    [
        "player_name",
        "pitch_type"
    ]
)

.agg(

    xwoba_contact=(
        "estimated_woba_using_speedangle",
        "mean"
    ),

    avg_exit_velocity=(
        "launch_speed",
        "mean"
    ),

    avg_launch_angle=(
        "launch_angle",
        "mean"
    )

)

.reset_index()

)


# Merge contact results

summary = summary.merge(
    contact_summary,
    on=[
        "player_name",
        "pitch_type"
    ],
    how="left"
)



# ------------------------------------------------------------
# Add usage
# ------------------------------------------------------------

summary["usage"] = (

summary["pitches"]

/

summary["player_name"]
.map(pitcher_totals)

)



# ------------------------------------------------------------
# Remove extremely small samples
# ------------------------------------------------------------

summary = summary[
    summary["pitches"] >= 100
].copy()



# ------------------------------------------------------------
# Round numbers
# ------------------------------------------------------------

summary = summary.round(3)



# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

summary.to_csv(
    "pitch_summary.csv",
    index=False
)


print("\nComplete!")
print(
    f"Pitcher-pitch combinations: {len(summary)}"
)


print("\nSample:")
print(summary.head())