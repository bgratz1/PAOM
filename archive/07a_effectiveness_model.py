import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


###############################################################
# LOAD DATA
###############################################################

print("Loading master pitch table...")

df = pd.read_csv(
    "master_pitch_table_2025.csv"
)


print(f"Initial rows: {len(df)}")


###############################################################
# MINIMUM SAMPLE FILTER
###############################################################

MIN_PITCHES = 100

df = df[
    df["pitches"] >= MIN_PITCHES
].copy()


print(
    f"After {MIN_PITCHES} pitch filter: {len(df)}"
)



###############################################################
# TARGET
###############################################################

target = "xwoba"



###############################################################
# FEATURES
###############################################################

# These describe WHY a pitch is effective

numeric_features = [

    # pitch results

    "whiff_rate",
    "csw_rate",
    "chase_rate",

    "hard_hit_rate",
    "sweet_spot_rate",

    "avg_exit_velocity",
    "avg_launch_angle",


    # pitch traits

    "velo",
    "spin",
    "spin_axis",

    "HB",
    "IVB",

    "extension",

    "release_x",
    "release_z"

]


categorical_features = [

    "pitch_type"

]


features = (
    numeric_features
    +
    categorical_features
)



###############################################################
# REMOVE MISSING VALUES
###############################################################

model_df = df[
    features + [target]
].dropna()



print(
    f"Rows used in model: {len(model_df)}"
)



###############################################################
# BUILD MODEL PIPELINE
###############################################################

X = model_df[features]

y = model_df[target]



preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",

            StandardScaler(),

            numeric_features
        ),


        (
            "pitch_type",

            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),

            categorical_features
        )

    ]

)



model = Ridge(
    alpha=10
)



pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )

    ]

)



###############################################################
# CROSS VALIDATION
###############################################################

scores = cross_val_score(

    pipeline,

    X,

    y,

    cv=5,

    scoring="r2"

)


print("\nCross Validation R2")

print(scores)

print(
    "Average:",
    scores.mean()
)



###############################################################
# TRAIN FINAL MODEL
###############################################################

pipeline.fit(
    X,
    y
)



###############################################################
# EXTRACT COEFFICIENTS
###############################################################

feature_names = (

    pipeline

    .named_steps["preprocessor"]

    .get_feature_names_out()

)


coefficients = (

    pipeline

    .named_steps["model"]

    .coef_

)



coef_df = pd.DataFrame(

    {

        "feature": feature_names,

        "coefficient": coefficients

    }

)


coef_df["abs_coefficient"] = (
    coef_df["coefficient"]
    .abs()
)



coef_df = (

    coef_df

    .sort_values(
        "abs_coefficient",
        ascending=False
    )

)



coef_df.to_csv(

    "effectiveness_model_coefficients.csv",

    index=False

)



print("\nTop Model Drivers")

print(
    coef_df.head(20)
)



###############################################################
# CREATE EFFECTIVENESS SCORE
###############################################################

score_features = [

    "whiff_rate",
    "csw_rate",
    "chase_rate",

    "hard_hit_rate",
    "sweet_spot_rate",

    "avg_exit_velocity",

]



score_df = df.copy()



# z-score components

for col in score_features:

    score_df[col+"_z"] = (

        score_df[col]

        -

        score_df[col].mean()

    ) / score_df[col].std()



# Higher is better

score_df["effectiveness_score"] = (

    0.30 * score_df["whiff_rate_z"]

    +

    0.25 * score_df["csw_rate_z"]

    +

    0.15 * score_df["chase_rate_z"]


    -

    0.15 * score_df["hard_hit_rate_z"]

    -

    0.10 * score_df["sweet_spot_rate_z"]


    -

    0.05 * score_df["avg_exit_velocity_z"]

)



###############################################################
# RESCALE 0-100
###############################################################

score_df["effectiveness_score"] = (

    100 *

    (

        score_df["effectiveness_score"]

        -

        score_df["effectiveness_score"].min()

    )

    /

    (

        score_df["effectiveness_score"].max()

        -

        score_df["effectiveness_score"].min()

    )

)



###############################################################
# SAVE OUTPUT
###############################################################

output_cols = [

    "player_name",

    "pitch_type",

    "pitch_name",

    "pitches",

    "xwoba",

    "effectiveness_score"

]


effectiveness_output = (

    score_df[output_cols]

    .sort_values(
        "effectiveness_score",
        ascending=False
    )

)



effectiveness_output.to_csv(

    "pitch_effectiveness_scores.csv",

    index=False

)



print("\nSaved:")
print("- effectiveness_model_coefficients.csv")
print("- pitch_effectiveness_scores.csv")


print("\nTop pitches")

print(
    effectiveness_output.head(20)
)