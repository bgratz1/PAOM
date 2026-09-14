# ============================================================
# 05b_validate_pair_metrics.py
#
# Validates pitch pair representation
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt


print("Loading pitch relationships...")

df = pd.read_csv(
    "pitch_relationships.csv"
)


print()

print(
    f"Total pitch pairs: {len(df)}"
)



# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

metrics = [

    "release_distance",
    "extension_diff",

    "HB_diff",
    "IVB_diff",

    "movement_distance",
    "movement_angle_diff",

    "velo_diff",
    "spin_diff"

]


print("\nMetric Summary")
print("----------------")

print(
    df[metrics].describe()
)



# ------------------------------------------------------------
# Best complementarity examples
#
# Similar release + large movement separation
# ------------------------------------------------------------

df["raw_complementarity"] = (

    df["release_similarity"]

    *

    (
        df["movement_distance"]
        /
        df["movement_distance"].max()
    )

    *

    (
        df["velo_diff"]
        /
        df["velo_diff"].max()
    )

)



print("\nTOP COMPLEMENTARY PAIRS")
print("-----------------------")

print(

    df.sort_values(
        "raw_complementarity",
        ascending=False
    )
    [
        [
            "player_name",
            "pitch_1",
            "pitch_2",

            "release_distance",

            "HB_diff",
            "IVB_diff",

            "movement_angle_diff",

            "movement_distance",

            "velo_diff"
        ]
    ]
    .head(20)

)



# ------------------------------------------------------------
# Most redundant pairs
#
# Similar movement + similar velocity
# ------------------------------------------------------------

print("\nMOST REDUNDANT PAIRS")
print("--------------------")


print(

    df.sort_values(
        [
            "movement_distance",
            "velo_diff"
        ],
        ascending=True
    )
    [
        [
            "player_name",
            "pitch_1",
            "pitch_2",

            "movement_distance",

            "HB_diff",
            "IVB_diff",

            "movement_angle_diff",

            "velo_diff"
        ]
    ]
    .head(20)

)



# ------------------------------------------------------------
# Plot 1
#
# Release similarity vs movement distance
# ------------------------------------------------------------

plt.figure(figsize=(8,6))


plt.scatter(

    df["release_distance"],

    df["movement_distance"],

    alpha=0.25

)


plt.xlabel(
    "Release Distance (lower = more similar)"
)


plt.ylabel(
    "Movement Distance (higher = more separation)"
)


plt.title(
    "Pitch Pair Complementarity Space"
)


plt.grid(True)

plt.tight_layout()

plt.savefig(
    "pair_release_vs_movement.png",
    dpi=300
)

plt.close()



# ------------------------------------------------------------
# Plot 2
#
# Horizontal vs vertical separation
# ------------------------------------------------------------

plt.figure(figsize=(8,6))


plt.scatter(

    df["HB_diff"],

    df["IVB_diff"],

    alpha=0.25

)


plt.xlabel(
    "Horizontal Separation"
)


plt.ylabel(
    "Vertical Separation"
)


plt.title(
    "Movement Separation Profile"
)


plt.grid(True)

plt.tight_layout()


plt.savefig(
    "movement_separation_profile.png",
    dpi=300
)


plt.close()



# ------------------------------------------------------------
# Plot 3
#
# Movement angle distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,6))


plt.hist(
    df["movement_angle_diff"],
    bins=40
)


plt.xlabel(
    "Movement Direction Difference (degrees)"
)


plt.ylabel(
    "Pitch Pairs"
)


plt.title(
    "Movement Direction Separation"
)


plt.grid(True)

plt.tight_layout()


plt.savefig(
    "movement_angle_distribution.png",
    dpi=300
)


plt.close()



print()

print("Validation complete.")

print("Saved plots:")
print("- pair_release_vs_movement.png")
print("- movement_separation_profile.png")
print("- movement_angle_distribution.png")