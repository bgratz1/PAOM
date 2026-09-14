"""
86_pairwise_metrics_by_year.py

Purpose:
--------
The REAL 05_pairwise_metrics.py, parameterized by YEAR -- all
pairwise release/movement/velocity/spin logic UNCHANGED from the
original.

ASSUMPTION, disclosed explicitly: the original reads "pitch_summary
.csv" -- a file this project never obtained the generating script
for. Its required columns (player_name, pitch_type, pitches, usage,
velo, spin, HB, IVB, extension, release_x, release_z) exactly match
what's already in master_pitch_table's real, confirmed schema. Best
evidence-based read: pitch_summary.csv IS the same underlying data,
referenced under an earlier name before later scripts (07d onward)
renamed it to master_pitch_table. Using master_pitch_table_{YEAR}
.csv (from 77) as the substitute here. If real pairwise output ever
looks wrong, this substitution is the first place to check.

Output:
-------
pitch_relationships_{YEAR}.csv
"""

import pandas as pd
import numpy as np
from itertools import combinations


# ============================================================
# SETTINGS
# ============================================================

import sys
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

INPUT_FILE = f"master_pitch_table_{YEAR}.csv"  # substitute for
                                                  # pitch_summary.csv
                                                  # -- see docstring
OUTPUT_FILE = f"pitch_relationships_{YEAR}.csv"


print(f"Loading pitch data for {YEAR}...")

df = pd.read_csv(INPUT_FILE)


# ------------------------------------------------------------
# Validate columns
# ------------------------------------------------------------

required_cols = [
    "player_name",
    "pitch_type",
    "pitches",
    "usage",

    "velo",
    "spin",

    "HB",
    "IVB",

    "extension",
    "release_x",
    "release_z"
]


missing = [
    c for c in required_cols
    if c not in df.columns
]


if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ------------------------------------------------------------
# Function:
# Angle between two movement vectors
#
# Returns degrees
# ------------------------------------------------------------

def movement_angle(hb1, ivb1, hb2, ivb2):

    v1 = np.array([hb1, ivb1])
    v2 = np.array([hb2, ivb2])


    magnitude = (
        np.linalg.norm(v1)
        *
        np.linalg.norm(v2)
    )


    if magnitude == 0:
        return 0


    cosine = np.dot(v1, v2) / magnitude


    cosine = np.clip(
        cosine,
        -1,
        1
    )


    angle = np.degrees(
        np.arccos(cosine)
    )


    return angle



# ------------------------------------------------------------
# Create pairs
# ------------------------------------------------------------

pairs = []


print("Creating pitch pairs...")


for pitcher, group in df.groupby("player_name"):


    pitch_list = group.to_dict("records")


    if len(pitch_list) < 2:
        continue


    for p1, p2 in combinations(pitch_list, 2):


        # ====================================================
        # RELEASE SIMILARITY
        # ====================================================

        release_x_diff = abs(
            p1["release_x"]
            -
            p2["release_x"]
        )


        release_z_diff = abs(
            p1["release_z"]
            -
            p2["release_z"]
        )


        release_distance = np.sqrt(
            release_x_diff**2
            +
            release_z_diff**2
        )


        extension_diff = abs(
            p1["extension"]
            -
            p2["extension"]
        )


        release_similarity = np.exp(
            -release_distance
        )


        extension_similarity = np.exp(
            -extension_diff
        )



        # ====================================================
        # MOVEMENT SEPARATION
        # ====================================================

        HB_diff = abs(
            p1["HB"]
            -
            p2["HB"]
        )


        IVB_diff = abs(
            p1["IVB"]
            -
            p2["IVB"]
        )


        movement_distance = np.sqrt(
            HB_diff**2
            +
            IVB_diff**2
        )


        movement_angle_diff = movement_angle(

            p1["HB"],
            p1["IVB"],

            p2["HB"],
            p2["IVB"]

        )



        # ====================================================
        # VELOCITY / SPIN
        # ====================================================

        velo_diff = abs(
            p1["velo"]
            -
            p2["velo"]
        )


        spin_diff = abs(
            p1["spin"]
            -
            p2["spin"]
        )



        # ====================================================
        # USAGE
        # ====================================================

        usage_1 = p1["usage"]

        usage_2 = p2["usage"]


        pair_weight = np.sqrt(
            usage_1 *
            usage_2
        )



        pairs.append({

            "player_name":
                pitcher,

            "season":
                YEAR,

            "pitch_1":
                p1["pitch_type"],

            "pitch_2":
                p2["pitch_type"],



            "usage_1":
                usage_1,

            "usage_2":
                usage_2,

            "pair_weight":
                pair_weight,



            "release_x_diff":
                release_x_diff,

            "release_z_diff":
                release_z_diff,

            "release_distance":
                release_distance,

            "extension_diff":
                extension_diff,

            "release_similarity":
                release_similarity,

            "extension_similarity":
                extension_similarity,



            "HB_diff":
                HB_diff,

            "IVB_diff":
                IVB_diff,

            "movement_distance":
                movement_distance,

            "movement_angle_diff":
                movement_angle_diff,



            "velo_diff":
                velo_diff,

            "spin_diff":
                spin_diff

        })



# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

relationships = pd.DataFrame(pairs)


relationships = relationships.round(4)


relationships.to_csv(
    OUTPUT_FILE,
    index=False
)


print()

print(f"Pitch relationships created for {YEAR}!")

print(
    f"Total pitch pairs: {len(relationships)}"
)

print()

print(
    relationships.head()
)
