"""
76_effectiveness_by_year.py

Purpose:
--------
07d_final_effectiveness.py, parameterized by YEAR instead of
hardcoded to 2025 -- everything else (the RidgeCV regression, the
hyperbolic confidence formula, the percentile-rank scaling, the
shrinkage toward league average) is UNCHANGED from the original,
already-validated logic. Only the input/output file names and the
season label change.

Run this once per year (2020-2025) after 74 (raw pull) and 75
(cleaning) have produced that year's clean_statcast_{YEAR}.csv.
"""


import pandas as pd
import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

import sys
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

INPUT_FILE = f"clean_statcast_{YEAR}.csv"

OUTPUT_FILE = f"PAOM_effectiveness_component_{YEAR}.csv"
WEIGHTS_FILE = f"final_effectiveness_weights_{YEAR}.csv"


MIN_PITCHES = 60   # UNCHANGED from the original -- established via
                     # 30/31's cutoff sweeps for 2025 data. Not
                     # re-swept per year here; if per-year sweeps
                     # ever seem warranted, that's separate follow-up
                     # work, not assumed necessary by default.
MIN_SWINGS = 40


FEATURES = [
    "csw_rate",
    "chase_rate",
    "hard_hit_rate",
    "gb_rate"
]


TARGET = "xwoba"


# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading Statcast data for {YEAR}...")

df = pd.read_csv(INPUT_FILE)

print(f"Initial rows: {len(df):,}")


# ============================================================
# CREATE REQUIRED FLAGS
# ============================================================

required_flags = [
    "is_swing",
    "is_whiff",
    "is_called_strike",
    "is_hard_hit",
    "is_chase"
]


missing_flags = [
    x for x in required_flags
    if x not in df.columns
]


if missing_flags:
    raise ValueError(
        f"Missing required columns: {missing_flags}"
    )


# ============================================================
# FILTER VALID PITCHES
# ============================================================

df = df[
    df["pitch_type"].notna()
]


df = df[
    df["estimated_woba_using_speedangle"].notna()
]


print(f"Rows after cleaning: {len(df):,}")


# ============================================================
# AGGREGATE PITCHER x PITCH TYPE
# ============================================================


print("\nAggregating pitcher x pitch type...")


pitch_summary = (
    df
    .groupby(
        [
            "player_name",
            "pitcher",
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

        swings=(
            "is_swing",
            "sum"
        ),

        whiffs=(
            "is_whiff",
            "sum"
        ),

        called_strikes=(
            "is_called_strike",
            "sum"
        ),

        hard_hits=(
            "is_hard_hit",
            "sum"
        ),

        chase_swings=(
            "is_chase",
            "sum"
        ),

        outside_zone=(
            "is_outside_zone",
            "sum"
        ),

        balls_in_play=(
            "is_in_play",
            "sum"
        ),

        ground_balls=(
            "launch_angle",
            lambda x: (
                x < 10
            ).sum()
        )

    )
    .reset_index()
)

# ============================================================
# FILTER SMALL SAMPLES
# ============================================================


pitch_summary = pitch_summary[
    pitch_summary["pitches"] >= MIN_PITCHES
]


pitch_summary = pitch_summary[
    pitch_summary["swings"] >= MIN_SWINGS
]


print(
    f"Pitch types after sample filter: "
    f"{len(pitch_summary):,}"
)


# ============================================================
# CALCULATE RATES
# ============================================================


pitch_summary["whiff_rate"] = (
    pitch_summary["whiffs"]
    /
    pitch_summary["swings"]
)


pitch_summary["csw_rate"] = (
    pitch_summary["whiffs"]
    +
    pitch_summary["called_strikes"]
) / pitch_summary["pitches"]


pitch_summary["chase_rate"] = (
    pitch_summary["chase_swings"]
    /
    pitch_summary["outside_zone"]
)


pitch_summary["hard_hit_rate"] = (
    pitch_summary["hard_hits"]
    /
    pitch_summary["pitches"]
)

pitch_summary["gb_rate"] = (
    pitch_summary["ground_balls"]
    /
    pitch_summary["balls_in_play"]
)

pitch_summary = (
    pitch_summary
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .dropna(
        subset=
        FEATURES +
        [TARGET]
    )
)


print(
    f"Rows used in model: "
    f"{len(pitch_summary):,}"
)

# ============================================================
# MODEL EFFECTIVENESS
# ============================================================

print("\nRunning effectiveness regression...")


X = pitch_summary[FEATURES]
y = pitch_summary[TARGET]


model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            RidgeCV(
                alphas=np.logspace(
                    -3,
                    3,
                    50
                )
            )
        )
    ]
)


cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="r2"
)


print(f"\nCross Validation R2 ({YEAR})")
print("-------------------")
print(cv_scores)
print(
    f"Average: {cv_scores.mean():.3f}"
)


model.fit(
    X,
    y
)


ridge = model.named_steps["ridge"]


coefficients = pd.DataFrame(
    {
        "feature": FEATURES,
        "coefficient": ridge.coef_
    }
)


coefficients["importance"] = (
    coefficients["coefficient"]
    .abs()
)


coefficients = (
    coefficients
    .sort_values(
        "importance",
        ascending=False
    )
)


print(f"\nEffectiveness Weights ({YEAR})")
print("---------------------")
print(coefficients)


coefficients.to_csv(
    WEIGHTS_FILE,
    index=False
)


pitch_summary["predicted_xwoba"] = (
    model.predict(X)
)


pitch_summary["raw_effectiveness_index"] = (
    -pitch_summary["predicted_xwoba"]
)


def percentile_rank(series):
    return series.rank(pct=True) * 100


pitch_summary["effectiveness_score"] = (
    percentile_rank(
        pitch_summary[
            "raw_effectiveness_index"
        ]
    )
)


pitch_summary["confidence_score"] = (
    pitch_summary["swings"]
    / (pitch_summary["swings"] + 200)
) * 100


league_average = (
    pitch_summary[
        "effectiveness_score"
    ]
    .mean()
)


pitch_summary[
    "trusted_effectiveness"
] = (

    pitch_summary[
        "effectiveness_score"
    ]
    *
    (
        pitch_summary[
            "confidence_score"
        ]
        /
        100
    )

    +

    league_average
    *
    (
        1 -
        pitch_summary[
            "confidence_score"
        ]
        /
        100
    )

)


pitch_summary["usage"] = (
    pitch_summary["pitches"]
    /
    pitch_summary
    .groupby("player_name")["pitches"]
    .transform("sum")
)

pitch_summary["season"] = YEAR


final_columns = [

    "player_name",
    "pitcher",
    "pitch_type",
    "pitch_name",
    "season",

    "pitches",
    "swings",
    "usage",

    "xwoba",
    "whiffs",
    "called_strikes",
    "hard_hits",
    "chase_swings",
    "outside_zone",

    "whiff_rate",
    "csw_rate",
    "chase_rate",
    "hard_hit_rate",

    "predicted_xwoba",
    "raw_effectiveness_index",

    "effectiveness_score",
    "confidence_score",
    "trusted_effectiveness"

]


effectiveness_output = (
    pitch_summary[
        final_columns
    ]
    .sort_values(
        "trusted_effectiveness",
        ascending=False
    )
)


effectiveness_output.to_csv(
    OUTPUT_FILE,
    index=False
)


print(f"\n==============================")
print(f"Effectiveness Component Saved ({YEAR})")
print("==============================")


print(
    f"""
Pitcher x pitch rows:
{len(effectiveness_output):,}

Average effectiveness:
{effectiveness_output['trusted_effectiveness'].mean():.2f}

Average confidence:
{effectiveness_output['confidence_score'].mean():.2f}
"""
)

print("\nSaved:")
print(f"- {OUTPUT_FILE}")
print(f"- {WEIGHTS_FILE}")
