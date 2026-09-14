import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from scipy.spatial import ConvexHull


print("Loading files...")

pitch_summary = pd.read_csv("pitch_summary.csv")
relationships = pd.read_csv("pitch_relationships.csv")


# ============================================================
# PART 1: PITCH EFFECTIVENESS SCORE
# ============================================================

print("\nCalculating pitch effectiveness...")


# Metrics where higher = better
positive_metrics = [
    "whiff_rate",
    "called_strike_rate",
    "chase_rate",
    "strike_rate"
]


# Metrics where lower = better
negative_metrics = [
    "xwoba_contact",
    "hard_hit_rate",
    "avg_exit_velocity"
]


# Convert metrics into percentiles
for col in positive_metrics:
    pitch_summary[col + "_pct"] = (
        pitch_summary[col]
        .rank(pct=True)
    )


for col in negative_metrics:
    pitch_summary[col + "_pct"] = (
        1 -
        pitch_summary[col]
        .rank(pct=True)
    )


effectiveness_cols = [
    x + "_pct"
    for x in positive_metrics + negative_metrics
]


pitch_summary["pitch_effectiveness"] = (
    pitch_summary[effectiveness_cols]
    .mean(axis=1)
    * 100
)


# Aggregate to pitcher level
effectiveness = (
    pitch_summary
    .groupby("player_name")
    .agg(
        effectiveness_score=
            ("pitch_effectiveness","mean"),

        best_pitch_score=
            ("pitch_effectiveness","max"),

        num_effective_pitches=
            ("pitch_effectiveness",
             lambda x: (x >= 75).sum()),

        num_pitch_types=
            ("pitch_type","nunique")
    )
    .reset_index()
)



# ============================================================
# PART 2: PITCH INTERACTION SCORE
# ============================================================

print("Calculating pitch interaction...")


# Convert relationship metrics into percentiles

interaction_metrics = [

    "release_similarity",

    "HB_diff",

    "IVB_diff",

    "velo_diff",

    "extension_diff"

]


for col in interaction_metrics:

    relationships[col+"_pct"] = (
        relationships[col]
        .rank(pct=True)
    )


# For release similarity:
# higher similarity is good for tunneling

relationships["release_similarity_pct"] = (
    relationships["release_similarity"]
    .rank(pct=True)
)



interaction_cols = [
    x+"_pct"
    for x in interaction_metrics
]


relationships["pair_interaction_score"] = (

    relationships[interaction_cols]
    .mean(axis=1)

    *100
)


interaction = (

    relationships
    .groupby("player_name")
    .agg(

        interaction_score=
            ("pair_interaction_score","mean"),

        best_pitch_pair=
            ("pair_interaction_score","max"),

        avg_release_similarity=
            ("release_similarity","mean"),

        avg_movement_difference=
            ("movement_distance","mean")

    )

    .reset_index()

)



# ============================================================
# PART 3: MOVEMENT COVERAGE SCORE
# ============================================================

print("Calculating movement coverage...")


coverage_rows=[]


for pitcher, group in pitch_summary.groupby("player_name"):


    # Need at least 2 pitches to calculate shape
    if len(group) < 2:
        continue


    hb_range = (
        group["HB"].max()
        -
        group["HB"].min()
    )


    ivb_range = (
        group["IVB"].max()
        -
        group["IVB"].min()
    )


    velo_range = (
        group["velo"].max()
        -
        group["velo"].min()
    )


    # Convex hull area in movement space

    points = group[
        ["HB","IVB"]
    ].dropna().values


    if len(points) >= 3:

        try:
            hull = ConvexHull(points)
            hull_area = hull.volume

        except:
            hull_area = 0

    else:
        hull_area = 0



    coverage_rows.append({

        "player_name": pitcher,

        "horizontal_coverage": hb_range,

        "vertical_coverage": ivb_range,

        "velocity_range": velo_range,

        "movement_area": hull_area

    })



coverage = pd.DataFrame(coverage_rows)



# Convert coverage metrics to percentiles

coverage_metrics = [

    "horizontal_coverage",

    "vertical_coverage",

    "velocity_range",

    "movement_area"

]


for col in coverage_metrics:

    coverage[col+"_pct"] = (
        coverage[col]
        .rank(pct=True)
    )


coverage["coverage_score"] = (

    coverage[
        [
        x+"_pct"
        for x in coverage_metrics
        ]
    ]

    .mean(axis=1)

    *100
)



# ============================================================
# PART 4: COMBINE
# ============================================================


print("Combining arsenal features...")


arsenal = (

    effectiveness

    .merge(
        interaction,
        on="player_name",
        how="left"
    )

    .merge(
        coverage,
        on="player_name",
        how="left"
    )

)



arsenal["total_pitches"] = (
    pitch_summary
    .groupby("player_name")["pitches"]
    .sum()
    .values
)



arsenal = arsenal.round(3)



arsenal.to_csv(
    "pitcher_arsenal_features.csv",
    index=False
)



print("\nComplete!")

print(
    f"Pitchers created: {len(arsenal)}"
)


print("\nSample:")
print(
    arsenal.head(10)
)