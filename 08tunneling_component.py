import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score


# ============================================================
# LOAD DATA
# ============================================================

print("Loading effectiveness component...")

pitch_data = pd.read_csv(
    "PAOM_effectiveness_component.csv"
)


print("Loading pitch relationships...")

pairs = pd.read_csv(
    "pitch_relationships.csv"
)


print("Loading trajectory-based tunnel pairs...")

tunnel_pairs = pd.read_csv(
    "pitch_tunnel_pairs.csv"
)

pairs = pairs.merge(
    tunnel_pairs,
    on=["player_name", "pitch_1", "pitch_2"],
    how="left"
)

print(
    f"Pairs with trajectory-based tunnel data: "
    f"{pairs['tunnel_differential'].notna().sum():,} / {len(pairs):,}"
)


# ============================================================
# LOG-TRANSFORM TUNNEL DIFFERENTIAL
# ============================================================

"""
tunnel_differential = plate_distance / (decision_distance + epsilon)
is heavily right-skewed -- a handful of pairs with near-zero
decision_distance produce extreme ratio values dominated by the
epsilon constant rather than reflecting meaningfully "more tunneled"
pairs. Those few extreme points can disproportionately drive (or
mask) a regression coefficient. log1p compresses that tail while
preserving rank order, so the regression isn't dominated by a
handful of near-zero-decision-distance outliers.
"""

pairs["tunnel_differential_log"] = np.log1p(
    pairs["tunnel_differential"]
)


# ============================================================
# CREATE PITCH DECEPTION SCORES
# ============================================================

print("\nCreating pitch deception metrics...")


# deception does NOT use effectiveness
# CSW captures whiffs + called strikes
# Chase captures hitter expansion

pitch_data["csw_z"] = (
    pitch_data["csw_rate"]
    - pitch_data["csw_rate"].mean()
) / pitch_data["csw_rate"].std()


pitch_data["chase_z"] = (
    pitch_data["chase_rate"]
    - pitch_data["chase_rate"].mean()
) / pitch_data["chase_rate"].std()


# ============================================================
# LEARNED PITCH-LEVEL DECEPTION INDEX
# ============================================================

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


print("\nCreating learned deception index...")


# deception outcomes only
deception_features = [
    "csw_rate",
    "chase_rate"
]


# keep only available data
deception_df = pitch_data[
    deception_features
].dropna()


# standardize outcomes
scaler = StandardScaler()

deception_scaled = scaler.fit_transform(
    deception_df
)


# PCA learns weights
pca = PCA(
    n_components=1
)


deception_pc = pca.fit_transform(
    deception_scaled
)


# assign back
pitch_data.loc[
    deception_df.index,
    "deception_score"
] = deception_pc[:,0]


# make higher = better deception
if pitch_data["deception_score"].corr(
    pitch_data["csw_rate"]
) < 0:
    
    pitch_data["deception_score"] *= -1


print("\nDeception PCA weights")
print("--------------------")

print(
    pd.DataFrame(
        {
            "feature": deception_features,
            "weight": pca.components_[0]
        }
    )
)


# ============================================================
# MERGE PITCH PAIR OUTCOMES
# ============================================================


pitches = pitch_data[
    [
        "player_name",
        "pitch_type",
        "csw_rate",
        "chase_rate",
        "deception_score",
        "trusted_effectiveness"
    ]
]


pairs = pairs.merge(
    pitches,
    left_on=[
        "player_name",
        "pitch_1"
    ],
    right_on=[
        "player_name",
        "pitch_type"
    ],
    how="left"
)


pairs = pairs.rename(
    columns={
        "csw_rate":"csw_1",
        "chase_rate":"chase_1",
        "deception_score":"deception_1",
        "trusted_effectiveness":"effectiveness_1"
    }
)


pairs = pairs.drop(
    columns=["pitch_type"]
)


pairs = pairs.merge(
    pitches,
    left_on=[
        "player_name",
        "pitch_2"
    ],
    right_on=[
        "player_name",
        "pitch_type"
    ],
    how="left"
)


pairs = pairs.rename(
    columns={
        "csw_rate":"csw_2",
        "chase_rate":"chase_2",
        "deception_score":"deception_2",
        "trusted_effectiveness":"effectiveness_2"
    }
)


pairs = pairs.drop(
    columns=["pitch_type"]
)



# ============================================================
# PAIR LEVEL DECEPTION
# ============================================================


pairs["pair_weight"] = np.sqrt(
    pairs["usage_1"]
    *
    pairs["usage_2"]
)


pairs["pair_csw"] = (
    pairs["csw_1"] * pairs["usage_1"]
    +
    pairs["csw_2"] * pairs["usage_2"]
) / (
    pairs["usage_1"]
    +
    pairs["usage_2"]
)


pairs["pair_chase"] = (
    pairs["chase_1"] * pairs["usage_1"]
    +
    pairs["chase_2"] * pairs["usage_2"]
) / (
    pairs["usage_1"]
    +
    pairs["usage_2"]
)


# standardize pair outcomes

pairs["csw_z"] = (
    pairs["pair_csw"]
    -
    pairs["pair_csw"].mean()
) / pairs["pair_csw"].std()


pairs["chase_z"] = (
    pairs["pair_chase"]
    -
    pairs["pair_chase"].mean()
) / pairs["pair_chase"].std()



pairs["pair_deception_score"] = (
    0.7 * pairs["csw_z"]
    +
    0.3 * pairs["chase_z"]
)


# ============================================================
# RESIDUALIZE AGAINST INDIVIDUAL PITCH QUALITY
# ============================================================

"""
pair_deception_score is built from csw_rate/chase_rate -- the
same two pitches' own outcome rates. That makes it mostly a
restatement of "how good are these two pitches individually,"
not "do these two pitches deceive hitters together" -- which is
why it correlated so heavily (0.617) with Effectiveness.

trusted_effectiveness is a broader quality measure (built from
csw_rate, chase_rate, hard_hit_rate, and gb_rate together via
the 07d regression against xwoba) -- not the identical ingredient
set, so residualizing against it removes "these are just two good
pitches" and leaves the part of pair performance that individual
quality alone doesn't explain. This is a partial fix: a fully
rigorous tunneling measure would use actual pitch-sequence data
(outcomes on the pitch immediately following a given pitch type)
rather than an all-pairs combination -- worth flagging as future
work.
"""

pairs["pair_expected_effectiveness"] = (
    pairs["effectiveness_1"] * pairs["usage_1"]
    +
    pairs["effectiveness_2"] * pairs["usage_2"]
) / (
    pairs["usage_1"]
    +
    pairs["usage_2"]
)

pairs["expected_effectiveness_z"] = (
    pairs["pair_expected_effectiveness"]
    -
    pairs["pair_expected_effectiveness"].mean()
) / pairs["pair_expected_effectiveness"].std()

# drop rows missing either input before residualizing
_resid_inputs = pairs[
    ["pair_deception_score", "expected_effectiveness_z"]
].dropna()

from sklearn.linear_model import LinearRegression as _LinReg

_resid_model = _LinReg()
_resid_model.fit(
    _resid_inputs[["expected_effectiveness_z"]],
    _resid_inputs["pair_deception_score"]
)

pairs.loc[_resid_inputs.index, "residual_deception_score"] = (
    _resid_inputs["pair_deception_score"]
    -
    _resid_model.predict(_resid_inputs[["expected_effectiveness_z"]])
)

print(
    "\nVariance explained by individual-quality baseline "
    f"(R2 of pair_deception_score ~ expected_effectiveness_z): "
    f"{_resid_model.score(_resid_inputs[['expected_effectiveness_z']], _resid_inputs['pair_deception_score']):.3f}"
)


# ============================================================
# TUNNEL INTERACTION FEATURE
# ============================================================

"""
True tunneling requires similar release AND divergent late
movement together -- not just some release difference plus some
movement difference added separately. release_similarity already
decays with release_distance (exp(-release_distance)), so
multiplying it by movement_distance rewards pairs that are close
out of the hand AND far apart in movement -- the actual tunnel
shape -- rather than crediting release similarity and movement
separation as independent, additive effects.
"""

pairs["tunnel_interaction"] = (
    pairs["release_similarity"]
    *
    pairs["movement_distance"]
)



# remove bad pairs

pairs = pairs.dropna(
    subset=[
        # physical variables
        "release_distance",
        "movement_angle_diff",
        "extension_diff",
        "velo_diff",
        "spin_diff",
        "tunnel_interaction",
        "tunnel_differential",
        "tunnel_differential_log",
        "csw_1",
        "csw_2",
        "chase_1",
        "chase_2",
        "pair_deception_score",
        "residual_deception_score"
    ]
)


print(
    "Pairs used:",
    len(pairs)
)



# ============================================================
# REGRESSION
# ============================================================


features = [
    "release_distance",
    "movement_angle_diff",
    "extension_diff",
    "velo_diff",
    "spin_diff",
    "tunnel_interaction",
    "tunnel_differential"
]

print("\nMissing values before regression:")
print(
    pairs[
        features +
        ["residual_deception_score"]
    ]
    .isna()
    .sum()
)

X = pairs[features]

y = pairs["residual_deception_score"]



model = Pipeline(
    [
        (
            "scale",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=10)
        )
    ]
)



kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="r2"
)


print("\nTunneling Regression R2")
print("-----------------------")
print(scores)
print("Average:", scores.mean())



# fit final model

model.fit(
    X,
    y
)


coef = pd.DataFrame(
    {
        "feature":features,
        "coefficient":
            model.named_steps["ridge"].coef_
    }
)


coef["importance"] = (
    coef["coefficient"]
    .abs()
)


print("\nPhysical drivers")
print(coef.sort_values(
    "importance",
    ascending=False
))


# ============================================================
# DIAGNOSTIC: LOG-TRANSFORMED TUNNEL DIFFERENTIAL
# ============================================================

"""
tunnel_differential is heavily right-skewed (a handful of
near-zero decision_distance pairs produce extreme ratios
dominated by the epsilon constant). Refit the same regression
with tunnel_differential_log swapped in, to check whether that
skew was masking a real relationship or whether the near-zero
coefficient holds up once the outlier influence is tamed.
"""

print("\n==============================")
print("Diagnostic: Log-Transformed Tunnel Differential")
print("==============================")

features_log = [
    f if f != "tunnel_differential" else "tunnel_differential_log"
    for f in features
]

X_log = pairs[features_log]

model_log = Pipeline(
    [
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=10))
    ]
)

scores_log = cross_val_score(
    model_log,
    X_log,
    y,
    cv=kf,
    scoring="r2"
)

print(f"\nR2 with tunnel_differential (raw): {scores.mean():.4f}")
print(f"R2 with tunnel_differential_log:   {scores_log.mean():.4f}")

model_log.fit(X_log, y)

coef_log = pd.DataFrame(
    {
        "feature": features_log,
        "coefficient": model_log.named_steps["ridge"].coef_
    }
)

coef_log["importance"] = coef_log["coefficient"].abs()

coef_log = coef_log.sort_values("importance", ascending=False)

print("\nPhysical drivers (log-transformed version)")
print(coef_log)

raw_td_coef = coef.loc[
    coef["feature"] == "tunnel_differential", "coefficient"
].values[0]

log_td_coef = coef_log.loc[
    coef_log["feature"] == "tunnel_differential_log", "coefficient"
].values[0]

print(
    f"\ntunnel_differential coefficient -- raw: {raw_td_coef:.4f}, "
    f"log-transformed: {log_td_coef:.4f}"
)

if abs(log_td_coef) > abs(raw_td_coef) * 2 and log_td_coef > 0:
    print(
        "\nLog-transforming meaningfully increased the positive "
        "coefficient -- the raw version's near-zero result was "
        "likely distorted by outlier skew. The log-transformed "
        "feature may be the better one to use going forward."
    )
else:
    print(
        "\nLog-transforming did not meaningfully change the "
        "story -- tunnel_differential's weak relationship with "
        "residual deception does not appear to be an outlier-skew "
        "artifact."
    )

coef_log.to_csv("tunnel_differential_log_diagnostic.csv", index=False)


# ============================================================
# PREDICTED (PHYSICALLY-GROUNDED) DECEPTION SCORE
# ============================================================

"""
pair_deception_score (built directly from CSW/chase rate) is an
OUTCOME measure -- it shares raw ingredients with Effectiveness's
own regression target (xwoba is itself modeled from csw_rate and
chase_rate), which makes the two components correlate for a
structural reason rather than because they're independently
confirming real deception.

The fitted model above now predicts residual_deception_score --
pair_deception_score with the individual-quality baseline
(pair_expected_effectiveness) subtracted out -- using physical
traits PLUS a release-similarity x movement-distance interaction
term. model.predict(X) is the actual tunneling signal: "how much
deception, beyond what individual pitch quality already explains,
does this pitcher's physical release/movement profile predict."
This is what makes Tunneling a genuinely distinct third component
from Effectiveness, rather than restating CSW/chase under a
different name.
"""

pairs["predicted_deception_score"] = model.predict(X)


# ============================================================
# CREATE PAIR SCORES
# ============================================================


pair_output = pairs[
    [
        "player_name",
        "pitch_1",
        "pitch_2",
        "pair_weight",
        "pair_deception_score",
        "residual_deception_score",
        "predicted_deception_score",
        "pair_expected_effectiveness",
        "tunnel_interaction",
        "tunnel_differential",
        "decision_distance",
        "plate_distance",
        "pair_csw",
        "pair_chase",
        "release_distance",
        "extension_diff",
        "movement_angle_diff",
        "velo_diff",
        "spin_diff"
    ]
]


pair_output.to_csv(
    "PAOM_pair_deception_scores.csv",
    index=False
)



# ============================================================
# PITCHER LEVEL TUNNELING SCORE
# ============================================================


pitcher_scores = (
    pair_output
    .groupby("player_name")
    .apply(
        lambda x:
        np.average(
            x["predicted_deception_score"],
            weights=x["pair_weight"]
        )
    )
    .reset_index(
        name="raw_tunneling_score"
    )
)



counts = (
    pair_output
    .groupby("player_name")
    .size()
    .reset_index(
        name="num_pairs"
    )
)


pitcher_scores = pitcher_scores.merge(
    counts,
    on="player_name"
)



# confidence adjustment

pitcher_scores["confidence"] = (
    np.minimum(
        pitcher_scores["num_pairs"] / 10,
        1
    )
)


pitcher_scores["trusted_tunneling_score"] = (
    pitcher_scores["raw_tunneling_score"]
    *
    pitcher_scores["confidence"]
)



# scale 0-100 (percentile rank)
#
# Percentile rank instead of min-max scaling: min-max makes every
# pitcher's score entirely dependent on wherever the single most
# extreme pitcher happens to sit, which means "50" doesn't reliably
# mean "average" and scores aren't comparable across components.
# Percentile rank fixes both: 50 = exact median, 90 = better than
# 90% of qualifying pitchers, consistently across every PAOM
# component (this one is reference-only / dashboard display, not
# part of paom_score, but kept on the same scale convention).

pitcher_scores["tunneling_score"] = (
    pitcher_scores["trusted_tunneling_score"].rank(pct=True) * 100
)



print("\nTop tunneling pitchers")

print(
    pitcher_scores
    .sort_values(
        "tunneling_score",
        ascending=False
    )
    .head(20)
)



pitcher_scores.to_csv(
    "PAOM_tunneling_component.csv",
    index=False
)



print("\nSaved:")
print("- PAOM_pair_deception_scores.csv")
print("- PAOM_tunneling_component.csv")

# ============================================================
# Correlation Check: Tunneling vs Effectiveness
# ============================================================

print("\n==============================")
print("Effectiveness vs Tunneling Correlation")
print("==============================")


corr_df = (
    pitcher_scores
    .merge(
        pitch_data[
            [
                "player_name",
                "trusted_effectiveness"
            ]
        ],
        on="player_name",
        how="inner"
    )
)


print("\nRows matched:")
print(len(corr_df))


print("\nCorrelation Matrix")
print("------------------")

print(
    corr_df[
        [
            "trusted_tunneling_score",
            "trusted_effectiveness"
        ]
    ]
    .corr()
)


print("\nCorrelation:")
print(
    corr_df[
        "trusted_tunneling_score"
    ]
    .corr(
        corr_df[
            "trusted_effectiveness"
        ]
    )
)


corr_df.to_csv(
    "PAOM_tunneling_effectiveness_correlation.csv",
    index=False
)

# ============================================================
# VALIDATION CHECKS
# ============================================================

print("\n==============================")
print("TUNNELING VALIDATION")
print("==============================")

# 1. Number of pitch pairs
print("\nAverage pairs per pitcher:")
print(pitcher_scores["num_pairs"].describe())

# 2. Relationship with effectiveness
print("\nCorrelation with effectiveness:")
print(
    corr_df["trusted_tunneling_score"].corr(
        corr_df["trusted_effectiveness"]
    )
)

# 3. Relationship with repertoire size
pitch_counts = (
    pitch_data
    .groupby("player_name")
    .size()
    .reset_index(name="num_pitch_types")
)

validation = (
    pitcher_scores
    .merge(pitch_counts, on="player_name")
)

print("\nCorrelation with repertoire size:")
print(
    validation["trusted_tunneling_score"].corr(
        validation["num_pitch_types"]
    )
)

# 4. Relationship with average pitch usage
usage_summary = (
    pitch_data
    .groupby("player_name")["usage"]
    .mean()
    .reset_index(name="avg_pitch_usage")
)

validation = validation.merge(
    usage_summary,
    on="player_name"
)

print("\nCorrelation with average pitch usage:")
print(
    validation["trusted_tunneling_score"].corr(
        validation["avg_pitch_usage"]
    )
)

validation.to_csv(
    "PAOM_tunneling_validation.csv",
    index=False
)