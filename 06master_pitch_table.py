import pandas as pd
import numpy as np


print("Loading clean Statcast data...")

df = pd.read_csv(
    "clean_statcast_2025.csv"
)


###############################################################
# BASIC COUNTS
###############################################################

print(f"Total pitches: {len(df)}")


###############################################################
# CREATE DERIVED FLAGS
###############################################################

# contact
df["is_contact"] = (
    df["is_contact"]
    .astype(int)
)


# balls in play
df["is_in_play"] = (
    df["is_in_play"]
    .astype(int)
)


###############################################################
# AGGREGATION FUNCTIONS
###############################################################

def safe_mean(x):
    return x.mean()


def safe_sum(x):
    return x.sum()



###############################################################
# GROUP PITCHER + PITCH TYPE
###############################################################

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

        pitches=("pitch_type","count"),

        # usage inputs

        balls=("balls","count"),


        ###################################################
        # PHYSICS
        ###################################################

        velo=("release_speed","mean"),

        spin=("release_spin_rate","mean"),

        spin_axis=("spin_axis","mean"),

        HB=("HB","mean"),

        IVB=("IVB","mean"),

        extension=("release_extension","mean"),

        release_x=("release_pos_x","mean"),

        release_z=("release_pos_z","mean"),



        ###################################################
        # SWING METRICS
        ###################################################

        swing_rate=("is_swing","mean"),

        whiff_rate=("is_whiff","mean"),

        called_strike_rate=("is_called_strike","mean"),

        chase_rate=("is_chase","mean"),

        strike_rate=("is_strike","mean"),



        ###################################################
        # CONTACT QUALITY
        ###################################################

        xwoba=("estimated_woba_using_speedangle","mean"),

        xba=("estimated_ba_using_speedangle","mean"),

        avg_exit_velocity=("launch_speed","mean"),

        avg_launch_angle=("launch_angle","mean"),


        hard_hit_rate=("is_hard_hit","mean"),

        sweet_spot_rate=("is_sweet_spot","mean"),



        ###################################################
        # RUN VALUE
        ###################################################

        run_value=("delta_run_exp","sum"),
        avg_run_value=("delta_run_exp","mean"),

        ###################################################
        # SAMPLE COUNTS
        ###################################################

        swings=("is_swing","sum"),

        whiffs=("is_whiff","sum"),

        contacts=("is_contact","sum"),

        balls_in_play=("is_in_play","sum")

    )

    .reset_index()

)



###############################################################
# CSW
###############################################################

summary["csw_rate"] = (

    summary["whiff_rate"]

    +

    summary["called_strike_rate"]

)



###############################################################
# RUN VALUE PER 100 PITCHES
###############################################################

summary["run_value_per_100"] = (

    summary["run_value"]

    /

    summary["pitches"]

    *

    100

)

summary["run_value_per_100"] = (
    summary["run_value"]
    /
    summary["pitches"]
    *
    100
)

###############################################################
# USAGE
###############################################################

total_pitcher_pitches = (

    summary

    .groupby("player_name")["pitches"]

    .transform("sum")

)


summary["usage"] = (

    summary["pitches"]

    /

    total_pitcher_pitches

)



###############################################################
# SAVE
###############################################################

summary = summary.round(4)


summary.to_csv(

    "master_pitch_table_2025.csv",

    index=False

)



print("\nSaved master_pitch_table_2025.csv")


print("\nRows:")

print(len(summary))


print("\nColumns:")

print(summary.columns.tolist())


print("\nSample:")

print(summary.head())
