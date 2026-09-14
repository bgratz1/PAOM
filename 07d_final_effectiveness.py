"""
07d_final_effectiveness.py

Purpose:
--------
Create the PAOM Effectiveness Component.

Goal:
-----
Measure how effective each pitcher x pitch type is independent of usage.

Method:
-------
1. Aggregate Statcast data to pitcher x pitch type.
2. Model xwOBA using pitch performance metrics:
       - whiff_rate
       - csw_rate
       - chase_rate
       - hard_hit_rate

3. Convert model output into:
       - raw_effectiveness_index
       - effectiveness_score (0-100)
       - confidence_score
       - trusted_effectiveness

The raw index is saved because later PAOM components
should use continuous values rather than percentile rankings.
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

INPUT_FILE = "clean_statcast_2025.csv"

OUTPUT_FILE = "PAOM_effectiveness_component.csv"
WEIGHTS_FILE = "final_effectiveness_weights.csv"


MIN_PITCHES = 60  # Set via 30_effectiveness_cutoff_sweep.py (one-at-
                   # a-time), then CONFIRMED via 31_effectiveness_
                   # joint_cutoff_sweep.py's full 2D grid. Effectiveness
                   # is the limiting factor for the whole pipeline
                   # (every 10_paom_score.py run before this change
                   # showed exactly 394 pitchers vs. 722 for Command/
                   # Movement/Velocity, with "dropped 0" at the merge --
                   # this cutoff alone determines paom_score's final
                   # population).
MIN_SWINGS = 40    # LOWERED from 50, based on 31_effectiveness_joint_
                   # cutoff_sweep.py's full (MIN_PITCHES x MIN_SWINGS)
                   # grid -- the one-at-a-time sweep alone couldn't
                   # answer this, since it only tested MIN_SWINGS while
                   # holding MIN_PITCHES at the OLD default (75), where
                   # swings genuinely wasn't binding. The joint grid
                   # showed that's specific to pitches=75: at
                   # MIN_PITCHES=60, R2 stays flat (459 pitchers,
                   # primary R2=0.240, secondary R2=0.412) across
                   # swings=20/30/40, then starts dropping at 50
                   # (0.228) and further at 65 (0.201) -- swings DOES
                   # become binding at this lower pitch floor. 40 was
                   # chosen over 20 (the technical grid maximum) because
                   # both produced IDENTICAL results (459 pitchers, same
                   # R2 on both targets) -- 40 sits safely inside the
                   # tested range rather than at its edge, avoiding the
                   # "did the sweep just stop before finding a real
                   # peak" ambiguity a boundary value carries, with zero
                   # cost since the two values are numerically tied.
                   # This pairing (60/40) beat the previously-live
                   # 60/50 on every axis simultaneously: +11 pitchers
                   # (448->459), better primary R2 (0.228->0.240),
                   # better secondary R2 (0.388->0.412), no degeneracy,
                   # no rate-instability concern (p10 balls-in-play=42,
                   # p10 out-of-zone=18, both comfortably clear of the
                   # instability floor).


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

print("Loading Statcast data...")

df = pd.read_csv(INPUT_FILE)

print(f"Initial rows: {len(df):,}")


# ============================================================
# CREATE REQUIRED FLAGS
# ============================================================

# Make sure boolean columns exist

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


# remove pitches without xwOBA
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

        # volume
        pitches=("pitch_type", "count"),

        # outcomes
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

        # batted ball outcomes
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
# Replace bad divisions

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


# ============================================================
# RIDGE REGRESSION
# ============================================================

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


# ============================================================
# CROSS VALIDATION
# ============================================================


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


print("\nCross Validation R2")
print("-------------------")
print(cv_scores)
print(
    f"Average: {cv_scores.mean():.3f}"
)


# ============================================================
# FIT FINAL MODEL
# ============================================================


model.fit(
    X,
    y
)


ridge = model.named_steps["ridge"]


# ============================================================
# MODEL COEFFICIENTS
# ============================================================


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


print("\nEffectiveness Weights")
print("---------------------")
print(coefficients)


coefficients.to_csv(
    WEIGHTS_FILE,
    index=False
)


# ============================================================
# CREATE RAW EFFECTIVENESS INDEX
# ============================================================

"""
The regression predicts xwOBA.

Higher xwOBA = worse for pitcher.

Therefore we reverse the predicted value.

A lower predicted xwOBA means
a more effective pitch.
"""


pitch_summary["predicted_xwoba"] = (
    model.predict(X)
)


pitch_summary["raw_effectiveness_index"] = (
    -pitch_summary["predicted_xwoba"]
)


# ============================================================
# SCALE TO 0-100 (PERCENTILE RANK)
# ============================================================


def percentile_rank(series):
    """
    Percentile rank instead of min-max scaling -- consistent with
    the same fix applied to every other PAOM component. min-max
    makes every pitcher's score entirely dependent on wherever the
    single most extreme pitcher happens to sit, which means "50"
    doesn't reliably mean "average" and scores aren't comparable
    across components. Percentile rank fixes both: 50 = exact
    median, 90 = better than 90% of qualifying pitchers,
    consistently across every component.
    """

    return series.rank(pct=True) * 100


pitch_summary["effectiveness_score"] = (
    percentile_rank(
        pitch_summary[
            "raw_effectiveness_index"
        ]
    )
)


# ============================================================
# CONFIDENCE SCORE
# ============================================================


"""
Confidence should be based on swings, because effectiveness
metrics are swing outcomes.

HYPERBOLIC form (n / (n+k)), not a linear ramp with a hard cap.
This was swept against the original linear-cap formula
(26_effectiveness_confidence_sweep.py) across both forms' own
parameter grids, evaluated through the real downstream confidence-
weighted CV-Ridge fit against both validation targets -- not just
picked as a round number the way the original 500-swing saturation
was. Every hyperbolic candidate tested beat every linear-cap
candidate tested on secondary-target R2, a consistent form-level
separation, not a lucky single parameter. k=200 was chosen as the
best-balanced point within hyperbolic's own grid (primary R2 peaked
at k=200; secondary R2 peaked at k=100, but the whole k=100-300
neighborhood was within ~0.003 of each other on both targets, so
k=200 sits closest to optimal on both rather than being the single
best on just one). No degeneracy was found at this parameter
(zero pitchers landed near the exact 0/100 extremes after rollup,
the guardrail built into the sweep specifically to catch the
"R2 improved because shrinkage effectively vanished" failure mode
already seen with Velocity's confidence-removal experiment).

Unlike the old formula, this never fully caps at 100% confidence --
always leaves a small residual shrinkage even for very large
samples -- and ramps up faster early / more gradually later,
better matching how standard error actually shrinks with sample
size (~1/sqrt(n), not linearly). k is interpretable as "the swing
count at which confidence reaches 50%."
"""


pitch_summary["confidence_score"] = (
    pitch_summary["swings"]
    / (pitch_summary["swings"] + 200)
) * 100


# ============================================================
# TRUSTED EFFECTIVENESS
# ============================================================


"""
Shrink extreme small-sample performances
toward the league average.

Example:
A pitcher with a perfect splitter over
100 swings should not automatically rank
above an established elite pitch.

(Reverted after a shrinkage-removal experiment showed clear
evidence this protection was doing real work -- see paom.md /
conversation history for the comparison.)
"""


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


print(
    "\nTop pitches by trusted effectiveness"
)

print(
    pitch_summary
    .sort_values(
        "trusted_effectiveness",
        ascending=False
    )
    [
        [
            "player_name",
            "pitch_type",
            "pitch_name",
            "pitches",
            "swings",
            "xwoba",
            "effectiveness_score",
            "confidence_score",
            "trusted_effectiveness"
        ]
    ]
    .head(20)
)
# ============================================================
# CALCULATE PITCH USAGE
# ============================================================


pitch_summary["usage"] = (
    pitch_summary["pitches"]
    /
    pitch_summary
    .groupby("player_name")["pitches"]
    .transform("sum")
)


# ============================================================
# FINAL OUTPUT COLUMNS
# ============================================================


final_columns = [

    # identifiers
    "player_name",
    "pitcher",
    "pitch_type",
    "pitch_name",

    # volume
    "pitches",
    "swings",
    "usage",

    # raw outcomes
    "xwoba",
    "whiffs",
    "called_strikes",
    "hard_hits",
    "chase_swings",
    "outside_zone",

    # rates
    "whiff_rate",
    "csw_rate",
    "chase_rate",
    "hard_hit_rate",

    # model outputs
    "predicted_xwoba",
    "raw_effectiveness_index",

    # final scores
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


# ============================================================
# SAVE FILE
# ============================================================


effectiveness_output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================


print("\n==============================")
print("Effectiveness Component Saved")
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


print("\nTop 20 trusted pitches")
print("---------------------")


print(
    effectiveness_output
    [
        [
            "player_name",
            "pitch_type",
            "pitch_name",
            "pitches",
            "swings",
            "xwoba",
            "trusted_effectiveness"
        ]
    ]
    .head(20)
)


print("\nSaved:")
print(f"- {OUTPUT_FILE}")
print(f"- {WEIGHTS_FILE}")


# ============================================================
# OPTIONAL QUALITY CHECKS
# ============================================================


print("\nCorrelation Checks")
print("------------------")


print(

    effectiveness_output
    [
        [
            "trusted_effectiveness",
            "xwoba",
            "whiff_rate",
            "hard_hit_rate"
        ]
    ]
    .corr()

)
pitch_summary["predicted_xwoba"] = model.predict(X)

pitch_summary["effectiveness_residual"] = (
    pitch_summary["xwoba"]
    -
    pitch_summary["predicted_xwoba"]
)
pitch_summary.to_csv(
    "PAOM_effectiveness_component.csv",
    index=False
)
