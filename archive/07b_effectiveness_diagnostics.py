import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score


###############################################################
# LOAD DATA
###############################################################

print("Loading data...")

df = pd.read_csv(
    "master_pitch_table_2025.csv"
)


###############################################################
# FILTER SAMPLE SIZE
###############################################################

MIN_PITCHES = 100


df = df[
    df["pitches"] >= MIN_PITCHES
].copy()


print(
    "Rows:",
    len(df)
)



###############################################################
# EFFECTIVENESS VARIABLES ONLY
###############################################################

features = [

    # outcome quality

    "xwoba",

    # swing decisions

    "whiff_rate",
    "csw_rate",
    "chase_rate",

    # contact quality

    "hard_hit_rate",
    "sweet_spot_rate",
    "avg_exit_velocity"

]


model_df = df[
    features
].dropna()



###############################################################
# CORRELATION MATRIX
###############################################################

print("\nCorrelation Matrix")
print("------------------")


corr = (

    model_df

    .corr()

    .round(3)

)


print(corr)


corr.to_csv(
    "effectiveness_correlations.csv"
)



###############################################################
# VIF
###############################################################

print("\nVIF")
print("------------------")


X = model_df.drop(
    columns=["xwoba"]
)


scaler = StandardScaler()


X_scaled = pd.DataFrame(

    scaler.fit_transform(X),

    columns=X.columns

)



vif = pd.DataFrame()


vif["feature"] = X_scaled.columns


vif["VIF"] = [

    variance_inflation_factor(
        X_scaled.values,
        i
    )

    for i in range(len(X_scaled.columns))

]


vif = (

    vif

    .sort_values(
        "VIF",
        ascending=False
    )

)


print(vif)


vif.to_csv(
    "effectiveness_vif.csv",
    index=False
)



###############################################################
# REGRESSION COMPARISON
###############################################################

print("\nRegression Comparison")
print("---------------------")


target = model_df["xwoba"]


models = {


"Full":

[
"whiff_rate",
"csw_rate",
"chase_rate",
"hard_hit_rate",
"sweet_spot_rate",
"avg_exit_velocity"
],



"No whiff":

[
"csw_rate",
"chase_rate",
"hard_hit_rate",
"sweet_spot_rate",
"avg_exit_velocity"
],



"Core":

[
"csw_rate",
"chase_rate",
"hard_hit_rate"
]


}



results = []


for name, cols in models.items():


    X_model = model_df[cols]


    X_model = StandardScaler().fit_transform(
        X_model
    )


    ridge = Ridge(
        alpha=10
    )


    r2 = cross_val_score(

        ridge,

        X_model,

        target,

        cv=5,

        scoring="r2"

    ).mean()



    lr = LinearRegression()


    lr.fit(
        X_model,
        target
    )


    results.append(

        {

        "model": name,

        "variables": len(cols),

        "CV_R2": r2

        }

    )



results = pd.DataFrame(results)


print(results)


results.to_csv(

    "effectiveness_model_comparison.csv",

    index=False

)



###############################################################
# FINAL RECOMMENDED COEFFICIENT MODEL
###############################################################

print("\nFinal Ridge Coefficients")
print("------------------------")


final_features = models["Core"]


X_final = StandardScaler().fit_transform(

    model_df[final_features]

)


ridge = Ridge(
    alpha=10
)


ridge.fit(

    X_final,

    target

)



coef = pd.DataFrame(

    {

    "feature": final_features,

    "coefficient": ridge.coef_

    }

)



coef["importance"] = (

    coef["coefficient"]

    .abs()

)



coef = coef.sort_values(

    "importance",

    ascending=False

)



print(coef)


coef.to_csv(

    "effectiveness_final_coefficients.csv",

    index=False

)



print("\nDiagnostics complete.")