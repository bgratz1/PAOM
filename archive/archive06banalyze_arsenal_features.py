import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD DATA
# ============================================================

print("Loading arsenal features...")

df = pd.read_csv("pitcher_arsenal_features.csv")


print("\nNumber of pitchers:")
print(len(df))


print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# BASIC SUMMARY
# ============================================================

print("\n\nSUMMARY STATISTICS")
print("------------------")

print(
    df[
        [
            "effectiveness_score",
            "interaction_score",
            "coverage_score"
        ]
    ]
    .describe()
)


# ============================================================
# TOP PITCHERS BY COMPONENT
# ============================================================

components = [
    "effectiveness_score",
    "interaction_score",
    "coverage_score"
]


for component in components:

    print("\n\nTOP 20 BY:", component)
    print("----------------------------")

    print(
        df[
            [
                "player_name",
                component,
                "num_pitch_types",
                "total_pitches"
            ]
        ]
        .sort_values(
            component,
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# LOW END CHECK
# ============================================================

for component in components:

    print("\n\nBOTTOM 10 BY:", component)
    print("----------------------------")

    print(
        df[
            [
                "player_name",
                component,
                "num_pitch_types",
                "total_pitches"
            ]
        ]
        .sort_values(
            component,
            ascending=True
        )
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# COMPONENT CORRELATIONS
# ============================================================

print("\n\nCOMPONENT CORRELATIONS")
print("---------------------")

print(
    df[
        components
    ]
    .corr()
    .round(3)
)


# ============================================================
# LOOK FOR PITCHER ARCHETYPES
# ============================================================


print("\n\nHIGH EFFECTIVENESS / LOW COVERAGE")
print("---------------------------------")

print(
    df[
        [
            "player_name",
            "effectiveness_score",
            "coverage_score"
        ]
    ]
    .query(
        "effectiveness_score > 80 and coverage_score < 40"
    )
    .sort_values(
        "effectiveness_score",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


print("\n\nHIGH COVERAGE / LOW EFFECTIVENESS")
print("---------------------------------")

print(
    df[
        [
            "player_name",
            "effectiveness_score",
            "coverage_score"
        ]
    ]
    .query(
        "coverage_score > 80 and effectiveness_score < 50"
    )
    .sort_values(
        "coverage_score",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)



# ============================================================
# VISUALIZATIONS
# ============================================================


plt.figure(figsize=(8,6))

plt.scatter(
    df["effectiveness_score"],
    df["coverage_score"],
    alpha=0.5
)

plt.xlabel(
    "Pitch Effectiveness Score"
)

plt.ylabel(
    "Movement Coverage Score"
)

plt.title(
    "Pitch Effectiveness vs Movement Coverage"
)

plt.grid(True)

plt.savefig(
    "effectiveness_vs_coverage.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



plt.figure(figsize=(8,6))

plt.scatter(
    df["effectiveness_score"],
    df["interaction_score"],
    alpha=0.5
)

plt.xlabel(
    "Pitch Effectiveness Score"
)

plt.ylabel(
    "Pitch Interaction Score"
)

plt.title(
    "Pitch Effectiveness vs Interaction"
)

plt.grid(True)

plt.savefig(
    "effectiveness_vs_interaction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



plt.figure(figsize=(8,6))

plt.scatter(
    df["interaction_score"],
    df["coverage_score"],
    alpha=0.5
)

plt.xlabel(
    "Pitch Interaction Score"
)

plt.ylabel(
    "Movement Coverage Score"
)

plt.title(
    "Pitch Interaction vs Coverage"
)

plt.grid(True)

plt.savefig(
    "interaction_vs_coverage.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



print("\n\nAnalysis complete.")
print("Saved plots:")
print("- effectiveness_vs_coverage.png")
print("- effectiveness_vs_interaction.png")
print("- interaction_vs_coverage.png")