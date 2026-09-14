import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


###############################################################
# LOAD DATA
###############################################################

print("Loading pitch data...")

df = pd.read_csv(
    "clean_statcast_2025.csv"
)


print("Initial rows:", len(df))


###############################################################
# CREATE REQUIRED VARIABLES
###############################################################

# Whiff rate
df["is_whiff"] = (
    df["description"]
    .isin([
        "swinging_strike",
        "swinging_strike_blocked"
    ])
)


# Called strike
df["is_called_strike"] = (
    df["description"]
    == "called_strike"
)


# Swing
df["is_swing"] = (
    df["description"]
    .isin([
        "foul",
        "hit_into_play",
        "swinging_strike",
        "swinging_strike_blocked"
    ])
)


# Chase
df["is_outside_zone"] = (
    ~df["zone"].isin(
        [1,2,3,4,5,6,7,8,9]
    )
)


df["is_chase"] = (
    df["is_swing"]
    &
    df["is_outside_zone"]
)


# Hard hit
df["is_hard_hit"] = (
    df["launch_speed"] >= 95
)


###############################################################
# AGGREGATE PITCHER x PITCH TYPE
###############################################################

pitch_df = (

    df

    .groupby(
        [
            "player_name",
            "pitch_type",
            "pitch_name"
        ]
    )

    .agg(

        pitches=("pitch_type", "count"),

        xwoba=(
            "estimated_woba_using_speedangle",
            "mean"
        ),

        whiff_rate=(
            "is_whiff",
            "mean"
        ),

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
        )

    )

    .reset_index()

)


# CSW = Whiff + Called Strike

pitch_df["csw_rate"] = (
    pitch_df["whiff_rate"]
    +
    pitch_df["called_strike_rate"]
)

###############################################################
# SAMPLE FILTER
###############################################################

MIN_PITCHES = 100


pitch_df = pitch_df[
    pitch_df["pitches"] >= MIN_PITCHES
]


print(
    "Pitch types used:",
    len(pitch_df)
)



###############################################################
# STANDARDIZE FEATURES
###############################################################

features = [

    "xwoba",
    "whiff_rate",
    "csw_rate",
    "chase_rate",
    "hard_hit_rate"

]


X = pitch_df[features].copy()


# reverse variables where lower is better

X["xwoba"] *= -1
X["hard_hit_rate"] *= -1



scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)



###############################################################
# RIDGE MODEL FOR WEIGHTS
###############################################################

# target = standardized effectiveness outcome

y = X_scaled[:,0]


X_model = X_scaled[:,1:]


ridge = Ridge(
    alpha=10
)


ridge.fit(
    X_model,
    y
)



weights = pd.DataFrame(

    {

    "feature":
    [
    "whiff_rate",
    "csw_rate",
    "chase_rate",
    "hard_hit_rate"
    ],

    "coefficient":
    ridge.coef_

    }

)


weights["importance"] = (
    weights["coefficient"]
    .abs()
)


weights = weights.sort_values(
    "importance",
    ascending=False
)


print("\nEffectiveness weights")
print(weights)


weights.to_csv(
    "effectiveness_weights.csv",
    index=False
)



###############################################################
# CREATE EFFECTIVENESS SCORE
###############################################################

component_scores = X_scaled.copy()


# use learned coefficients

score = (

component_scores[:,0] * 0.5

+

component_scores[:,1]
* ridge.coef_[0]

+

component_scores[:,2]
* ridge.coef_[1]

+

component_scores[:,3]
* ridge.coef_[2]

+

component_scores[:,4]
* ridge.coef_[3]

)



# normalize 0-100

pitch_df["effectiveness_score"] = (

100 *

(score - score.min())

/

(score.max()-score.min())

)



###############################################################
# SAVE OUTPUT
###############################################################

output = pitch_df[
[
"player_name",
"pitch_type",
"pitch_name",
"pitches",
"xwoba",
"whiff_rate",
"csw_rate",
"chase_rate",
"hard_hit_rate",
"effectiveness_score"
]
]


output = output.sort_values(
    "effectiveness_score",
    ascending=False
)



output.to_csv(
    "final_pitch_effectiveness_scores.csv",
    index=False
)


print("\nTop pitches")

print(
    output.head(20)
)


print("\nSaved:")
print("- final_pitch_effectiveness_scores.csv")
print("- effectiveness_weights.csv")