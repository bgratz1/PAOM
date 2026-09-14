# ============================================================
# 05_pairwise_metrics.py
#
# Calculates relationships between pitches in each arsenal
# ============================================================

import pandas as pd
from itertools import combinations


# ------------------------------------------------------------
# Load pitch summary
# ------------------------------------------------------------

print("Loading pitch summary...")

df = pd.read_csv(
    "pitch_summary.csv"
)


# ------------------------------------------------------------
# Store comparisons
# ------------------------------------------------------------

pairs = []


# ------------------------------------------------------------
# Loop through each pitcher
# ------------------------------------------------------------

for pitcher, group in df.groupby("player_name"):


    # Convert pitches to dictionaries

    pitches = group.to_dict(
        "records"
    )


    # Need at least 2 pitches

    if len(pitches) < 2:
        continue



    # Compare every pitch combination

    for p1, p2 in combinations(
        pitches,
        2
    ):


        comparison = {


            # Identity

            "player_name": pitcher,


            "pitch_1": p1["pitch_name"],

            "pitch_2": p2["pitch_name"],



            # Usage

            "usage_1": p1["usage"],

            "usage_2": p2["usage"],



            # ------------------------------------------------
            # Velocity separation
            # ------------------------------------------------

            "velo_diff":

            abs(
                p1["velo"]
                -
                p2["velo"]
            ),



            # ------------------------------------------------
            # Movement separation
            # ------------------------------------------------

            "HB_diff":

            abs(
                p1["HB"]
                -
                p2["HB"]
            ),



            "IVB_diff":

            abs(
                p1["IVB"]
                -
                p2["IVB"]
            ),



            # ------------------------------------------------
            # Spin separation
            # ------------------------------------------------

            "spin_diff":

            abs(
                p1["spin"]
                -
                p2["spin"]
            ),



            # ------------------------------------------------
            # Release separation
            # ------------------------------------------------

            "release_x_diff":

            abs(
                p1["release_x"]
                -
                p2["release_x"]
            ),



            "release_z_diff":

            abs(
                p1["release_z"]
                -
                p2["release_z"]
            ),



            # ------------------------------------------------
            # Difference in effectiveness
            # ------------------------------------------------

            "whiff_diff":

            abs(
                p1["whiff_rate"]
                -
                p2["whiff_rate"]
            ),



            "hard_hit_diff":

            abs(
                p1["hard_hit_rate"]
                -
                p2["hard_hit_rate"]
            )

        }


        pairs.append(
            comparison
        )



# ------------------------------------------------------------
# Create dataframe
# ------------------------------------------------------------

pairwise = pd.DataFrame(
    pairs
)



# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

pairwise.to_csv(
    "pitch_relationships.csv",
    index=False
)


print("\nComplete!")

print(
    f"Pitch relationships created: {len(pairwise):,}"
)


print("\nSample:")
print(pairwise.head())