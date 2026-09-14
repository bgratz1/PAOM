# ============================================================
# 06_pairwise_outcomes.py
#
# Adds outcome metrics to pitch pair relationships
# ============================================================

import pandas as pd
from itertools import combinations


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(
    "clean_statcast_2025.csv"
)


pairwise = pd.read_csv(
    "pitch_relationships.csv"
)


# ------------------------------------------------------------
# Calculate pitch pair outcomes
# ------------------------------------------------------------

results = []


for pitcher, group in df.groupby("player_name"):


    # Available pitches

    pitch_types = (
        group["pitch_name"]
        .dropna()
        .unique()
    )


    if len(pitch_types) < 2:
        continue


    for pitch1, pitch2 in combinations(
        pitch_types,
        2
    ):


        subset = group[
            group["pitch_name"]
            .isin(
                [
                    pitch1,
                    pitch2
                ]
            )
        ]


        # Need enough pitches

        if len(subset) < 100:
            continue



        swings = subset[
            subset["is_swing"]
        ]


        whiffs = subset[
            subset["is_whiff"]
        ]


        # -----------------------------
        # Outcomes
        # -----------------------------


        whiff_rate = (

            len(whiffs)

            /

            len(swings)

            if len(swings) > 0

            else None

        )


        csw_rate = (

            (
                subset["is_called_strike"]
                |
                subset["is_whiff"]
            )

            .mean()

        )


        xwoba = (

            subset[
                "estimated_woba_using_speedangle"
            ]

            .mean()

        )


        run_value = (

            subset[
                "delta_run_exp"
            ]

            .mean()

        )


        results.append(

            {

            "player_name": pitcher,

            "pitch_1": pitch1,

            "pitch_2": pitch2,

            "pair_pitches": len(subset),

            "whiff_rate": whiff_rate,

            "csw_rate": csw_rate,

            "xwoba": xwoba,

            "run_value": run_value

            }

        )



outcomes = pd.DataFrame(results)



# ------------------------------------------------------------
# Merge with tunneling metrics
# ------------------------------------------------------------


final = pairwise.merge(

    outcomes,

    on=[
        "player_name",
        "pitch_1",
        "pitch_2"
    ],

    how="inner"

)



final.to_csv(

    "tunneling_training_data.csv",

    index=False

)


print(final.head())

print(
    f"Rows: {len(final)}"
)