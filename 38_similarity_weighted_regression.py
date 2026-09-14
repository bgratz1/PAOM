"""
38_similarity_weighted_regression.py

Purpose:
--------
The actual predictive tool this whole thread has been building toward:
for a target pitcher considering a specific arsenal change (e.g. "add
a sweeper"), predict the effect on a chosen outcome metric (WAR, FIP,
xwOBA-against), weighted by BOTH how similar historical pitchers were
to the target's own pre-change profile AND how reliable each
historical outcome was -- while also surfacing the real, named
historical comps driving that number, so the prediction isn't a black
box.

DESIGN, tying together the whole thread's discussion:
- Target: LAGGED outcome model -- Outcome(t+1) regressed on
  Outcome(t) + the change itself, not a naive before/after
  difference (differencing two noisy variables roughly doubles
  noise variance; including the lagged outcome as a predictor
  handles this within the model instead).
- Similarity: Euclidean distance in the standardized feature space
  from 37_similarity_feature_space.py (Movement-style geometry +
  Velocity + release point), computed against each historical
  pitcher's PRE-CHANGE (season_from) profile -- comparing the
  target's CURRENT arsenal shape to what similar historical
  pitchers looked like right before they made a comparable change.
- Reliability: the SAME hyperbolic n/(n+k) confidence form already
  validated for Effectiveness (26_effectiveness_confidence_sweep.py),
  applied to each historical outcome's own BIP -- a noisy, thin-
  sample historical result should influence the prediction less,
  the same principle used everywhere else in this project.
- FILTERED TO THE SAME PITCH TYPE AND CHANGE TYPE being simulated
  (e.g. predicting "add a slider" only draws on historical ADD-SL
  events) -- pooling across different pitch types would let
  "similar pitchers who added a curveball" influence a slider
  prediction, which doesn't make baseball sense regardless of how
  similar the pitchers' broader profiles are.

SIMILARITY KERNEL BANDWIDTH is self-calibrating (median pairwise
distance within the filtered comparison group), not a fixed guessed
number -- adapts to how spread out each specific (pitch_type,
change_type) group actually is, rather than one bandwidth assumed
to fit every group equally.

FALLBACK FOR SMALL GROUPS: if a (pitch_type, change_type) has too
few historical events for a stable weighted regression, falls back
to a simple similarity/reliability-weighted average of the outcome
change instead of a full regression -- a less sophisticated but
still real, honest estimate rather than an unstable fitted
coefficient from too little data.

Output:
-------
Two functions -- predict_arsenal_change_effect() and
show_similar_comps() -- meant to be called per target pitcher/
candidate change, demonstrated below with example predictions using
this project's real data.
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

CHANGE_EVENTS_FILE = "arsenal_change_events_pitch_level.csv"
SIMILARITY_FEATURES_FILE = "pitcher_similarity_features_2020_2025.csv"
OUTCOMES_FILE = "pitcher_season_metrics_2020_2025.csv"
NEW_PITCH_DISTANCE_FILE = "arsenal_change_new_pitch_distance_features.csv"
SYNERGY_FILE = "arsenal_synergy_features.csv"
ARSENAL_FILE = "pitcher_arsenal_evolution_2020_2025.csv"

SIMILARITY_FEATURE_COLS = [
    "hull_area_z", "avg_nn_distance_z", "fastball_relative_break_z",
    "velocity_range_z", "max_velo_z", "release_x_z", "release_z_z"
]

# TIERED SIMILARITY WEIGHTS -- explicit priority, not implicit equal-
# weighting. Previously every standardized feature contributed
# equally to the distance calculation just by virtue of being
# standardized -- standardizing puts features on the same SCALE, it
# doesn't establish PRIORITY between them. These weights are a
# judgment call, not empirically validated -- flagged the same way
# every other unvalidated choice in this project has been.
#
# Arsenal shape (hull area, nearest-neighbor spacing, fastball-
# relative break, velocity range/max) is weighted highest -- the
# most direct, established read on "what does this pitcher's stuff
# look like." Release point/arm slot is weighted lower -- a real,
# requested factor, but a secondary one. The two NEW pitch-
# composition features below get the HIGHEST weight of all -- they
# are the most literal, direct interpretation of "similar arsenal":
# do these two pitchers throw the same KINDS of pitches, and are
# those SPECIFIC shared pitches shaped similarly.
FEATURE_WEIGHTS = {
    "hull_area_z": 1.0,
    "avg_nn_distance_z": 1.0,
    "fastball_relative_break_z": 1.0,
    "velocity_range_z": 1.0,
    "max_velo_z": 1.0,
    "release_x_z": 0.5,
    "release_z_z": 0.5,
}
WEIGHT_PITCH_COMPOSITION = 1.5  # Jaccard overlap of qualifying pitch
                                  # types -- do they throw the same
                                  # KINDS of pitches at all
WEIGHT_SHARED_SHAPE = 1.5        # for pitch types BOTH throw, how
                                  # close are those SPECIFIC pitches'
                                  # movement profiles to each other
WEIGHT_USAGE_LEVEL = 2.0         # for DROP/USAGE_INCREASE/USAGE_
                                  # DECREASE only: how close was the
                                  # historical comp's STARTING usage
                                  # of this pitch type to the
                                  # target's own current usage -- a
                                  # pitcher going 20%->29% is a
                                  # different decision than one going
                                  # 37%->48%, even with an identical
                                  # delta (real gap found while
                                  # reviewing 45's output on José
                                  # Cuas). Not applied to ADD, where
                                  # usage_pct_before is always ~0 by
                                  # definition for both target and
                                  # comps -- near-zero-variance and
                                  # uninformative there.
                                  #
                                  # TESTED against the real Cuas case
                                  # (SI usage 61%): even a 4.0 weight
                                  # barely moved the top comps, while
                                  # real comps DO exist near 61% in
                                  # the data (9-14 of 207 SI-increase
                                  # events start above 55-50%). This
                                  # suggests arsenal SHAPE and
                                  # starting USAGE LEVEL are fairly
                                  # independent dimensions in the real
                                  # data -- a structural limit, not
                                  # purely a weight-tuning problem.
                                  # Settled at 2.0 as a reasonable
                                  # middle ground given diminishing
                                  # returns above 1.0; the underlying
                                  # tension (shape-similar vs. usage-
                                  # similar comps sometimes being
                                  # different pitchers) may need a
                                  # different approach than reweighting
                                  # alone to fully resolve.

MEANINGFUL_USAGE_FLOOR = 5.0  # SAME threshold used throughout this
                                # project (09/34/37/40) for "is this
                                # pitch really part of the arsenal"

PITCH_TYPES = ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS"]

RELIABILITY_K = 100  # SAME starting value as 32_pull_pitcher_season_
                       # metrics.py's original shrinkage attempt --
                       # not independently re-validated for this
                       # specific use, worth sweeping later the same
                       # way Effectiveness's k was

MIN_GROUP_SIZE_FOR_REGRESSION = 40  # RAISED from 15, based on real
                                      # evidence from 39_validate_
                                      # pitch_type_sensitivity.py: KC
                                      # (16 events, barely clearing
                                      # the old floor of 15) produced
                                      # wildly unstable predictions
                                      # every single time it was
                                      # tested (e.g. -3.32 vs. the
                                      # next-most-extreme value of
                                      # -1.01 for the same target) --
                                      # every other pitch type tested
                                      # had 61-222 events and behaved
                                      # far more stably. 40 is a
                                      # reasonable, still NOT fully
                                      # swept threshold (same honest
                                      # caveat as every other
                                      # unvalidated cutoff in this
                                      # project) -- worth a proper
                                      # sweep later, the same way
                                      # Effectiveness's cutoffs were.

TOP_K_COMPS = 5


# ============================================================
# LOAD AND BUILD TRAINING DATA
# ============================================================

print("Loading and merging training data...")

change_events = pd.read_csv(CHANGE_EVENTS_FILE)
similarity_features = pd.read_csv(SIMILARITY_FEATURES_FILE)
outcomes = pd.read_csv(OUTCOMES_FILE)
arsenal_raw = pd.read_csv(ARSENAL_FILE)

print(f"Change events: {len(change_events):,}")
print(f"Similarity features: {len(similarity_features):,}")
print(f"Outcomes: {len(outcomes):,}")
print(f"Raw arsenal data: {len(arsenal_raw):,}")

# indexed for fast pairwise lookup by (player_id, season) -- the new
# pitch-composition similarity needs each comparison's RAW per-pitch-
# type data, not the aggregate features 37_similarity_feature_space
# .py already computed
arsenal_raw_indexed = arsenal_raw.set_index(["player_id", "season"])

missing_sim_cols = [c for c in SIMILARITY_FEATURE_COLS if c not in similarity_features.columns]
if missing_sim_cols:
    raise ValueError(
        f"Missing similarity feature columns: {missing_sim_cols}. "
        f"Check 37_similarity_feature_space.py's output."
    )

# training data = change events + the pitcher's PRE-CHANGE (season_from)
# similarity profile + the outcome BEFORE (season_from) and AFTER
# (season_to) the change
training = change_events.merge(
    similarity_features[["player_id", "season"] + SIMILARITY_FEATURE_COLS],
    left_on=["player_id", "season_from"], right_on=["player_id", "season"],
    how="inner", suffixes=("", "_simfeat")
)
# drop the redundant "season" column this merge introduces (it's
# just a duplicate of season_from at this point) -- otherwise it
# collides with season_before/season_after from the outcome merges
# below, since it isn't a real join key on the LEFT side going
# forward, just leftover baggage from the right side of this join
training = training.drop(columns=["season"])

training = training.merge(
    outcomes[["player_id", "season", "bip"] + [c for c in ["war", "fip", "xwoba_against"] if c in outcomes.columns]],
    left_on=["player_id", "season_from"], right_on=["player_id", "season"],
    how="inner", suffixes=("", "_before")
)
training = training.rename(columns={"season": "season_before_matched"})

training = training.merge(
    outcomes[["player_id", "season", "bip"] + [c for c in ["war", "fip", "xwoba_against"] if c in outcomes.columns]],
    left_on=["player_id", "season_to"], right_on=["player_id", "season"],
    how="inner", suffixes=("_before", "_after")
)
training = training.rename(columns={"season": "season_after_matched"})

# merge in new-pitch-distance features (40_new_pitch_distance_
# features.py) -- ADD events built specifically to test the mean-
# reversion-vs-change-specific hypothesis from 39's Check 4. Only
# ADD events will get real values here; DROP/USAGE_INCREASE/
# USAGE_DECREASE rows correctly get NaN, since this feature isn't
# defined for those change types.
try:
    new_pitch_distance = pd.read_csv(NEW_PITCH_DISTANCE_FILE)
    training = training.merge(
        new_pitch_distance[
            ["player_id", "season_from", "season_to", "pitch_type",
             "new_pitch_nn_distance_from_existing"]
        ],
        on=["player_id", "season_from", "season_to", "pitch_type"],
        how="left"
    )
    n_with_distance_feature = training["new_pitch_nn_distance_from_existing"].notna().sum()
    print(
        f"{n_with_distance_feature:,} of {len(training):,} training "
        f"rows have the new-pitch-distance feature (ADD events only, "
        f"by design)"
    )
except FileNotFoundError:
    print(
        f"\nNOTE: {NEW_PITCH_DISTANCE_FILE} not found -- proceeding "
        f"without the new-pitch-distance feature. Run "
        f"40_new_pitch_distance_features.py first to enable it."
    )
    training["new_pitch_nn_distance_from_existing"] = np.nan

# merge in arsenal-synergy features (42_arsenal_synergy_features.py)
# -- measures whether the pitcher's OTHER pitches improved or
# declined after the new pitch was added. Originally built to test a
# tunneling/deception hypothesis (a new pitch making OTHER pitches
# play up by mechanical association). RESOLVED via 66/67_synergy_
# persistence_test.py: an improvement-persists-a-season-later version
# of this feature was placebo-tested (67) and found INDISTINGUISHABLE
# from its own shuffled placebo (magnitude 0.0035 real vs. 0.0022
# placebo, consistency 61.1% real vs. 63.9% placebo -- placebo
# actually scored HIGHER on one measure). This favors the SHARED-
# CAUSE explanation over the causal one: the strong same-season
# correlation this feature captures most likely reflects a pitcher
# having a broadly good year for unrelated reasons (health,
# mechanics, luck), showing up simultaneously as better whiff rates
# elsewhere AND a better overall outcome -- not a genuine, lasting
# effect from the new pitch itself. The feature remains a REAL,
# validated PREDICTOR of same-season outcomes (its own placebo test,
# 43, proved that decisively, and it contributed to real backtest
# accuracy in 58/64) -- what changed is the INTERPRETATION: a
# recommendation with a strong synergy score should be read as "this
# pitcher's situation correlates with good things happening," not as
# "this specific pitch addition will make your other pitches better
# too." Same ADD-events-only scope as the distance feature above.
try:
    synergy = pd.read_csv(SYNERGY_FILE)
    training = training.merge(
        synergy[
            ["player_id", "season_from", "season_to", "pitch_type",
             "arsenal_synergy_whiff_delta"]
        ],
        on=["player_id", "season_from", "season_to", "pitch_type"],
        how="left"
    )
    n_with_synergy_feature = training["arsenal_synergy_whiff_delta"].notna().sum()
    print(
        f"{n_with_synergy_feature:,} of {len(training):,} training "
        f"rows have the arsenal-synergy feature (ADD events only, "
        f"by design)"
    )
except FileNotFoundError:
    print(
        f"\nNOTE: {SYNERGY_FILE} not found -- proceeding without the "
        f"arsenal-synergy feature. Run 42_arsenal_synergy_features.py "
        f"first to enable it."
    )
    training["arsenal_synergy_whiff_delta"] = np.nan

print(f"\nFinal training dataset: {len(training):,} historical change events with full profile + outcome data")


# ============================================================
# RELIABILITY WEIGHTING (hyperbolic, same form as Effectiveness's)
# ============================================================

def reliability_weight(bip, k=RELIABILITY_K):
    bip = np.asarray(bip, dtype=float)
    return bip / (bip + k)


# ============================================================
# PITCH-COMPOSITION SIMILARITY (NEW -- addresses a real blind spot:
# two pitchers can land close together on AGGREGATE shape summary
# numbers by coincidence even if they throw completely different
# pitch types. This measures composition directly.)
# ============================================================

def get_qualifying_pitch_types(arsenal_row):
    """Pitch types clearing MEANINGFUL_USAGE_FLOOR for one pitcher-season."""
    qualifying = []
    for pt in PITCH_TYPES:
        usage = arsenal_row.get(f"{pt}_usage_pct", np.nan)
        if pd.notna(usage) and usage >= MEANINGFUL_USAGE_FLOOR:
            qualifying.append(pt)
    return qualifying


def compute_pitch_composition_similarity(row_a, row_b):
    """
    row_a, row_b: raw arsenal rows (from arsenal_raw_indexed) for two
    specific pitcher-seasons.

    Returns (jaccard_similarity, avg_shared_shape_distance):
    - jaccard_similarity: |shared pitch types| / |union of pitch
      types| -- 1.0 if identical composition, 0.0 if no overlap
      at all
    - avg_shared_shape_distance: for pitch types BOTH throw, average
      Euclidean distance between their (pfx_x, pfx_z) shapes --
      np.nan if there's no overlap to measure (handled by the caller,
      not treated as zero distance -- "no shared pitches" is a
      DIFFERENT thing than "shared pitches with identical shape")
    """
    types_a = set(get_qualifying_pitch_types(row_a))
    types_b = set(get_qualifying_pitch_types(row_b))

    union = types_a | types_b
    shared = types_a & types_b

    jaccard_similarity = len(shared) / len(union) if len(union) > 0 else 0.0

    if len(shared) == 0:
        return jaccard_similarity, np.nan

    shape_distances = []
    for pt in shared:
        ax, az = row_a.get(f"{pt}_avg_pfx_x", np.nan), row_a.get(f"{pt}_avg_pfx_z", np.nan)
        bx, bz = row_b.get(f"{pt}_avg_pfx_x", np.nan), row_b.get(f"{pt}_avg_pfx_z", np.nan)
        if pd.notna(ax) and pd.notna(az) and pd.notna(bx) and pd.notna(bz):
            shape_distances.append(np.sqrt((ax - bx) ** 2 + (az - bz) ** 2))

    avg_shared_shape_distance = np.mean(shape_distances) if shape_distances else np.nan

    return jaccard_similarity, avg_shared_shape_distance


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_arsenal_change_effect(
    target_features, pitch_type, change_type, target_outcome_before,
    outcome_metric, target_player_id, target_season, training_df=training
):
    """
    target_features: dict of {feature_col: value} for the target
        pitcher's CURRENT standardized similarity profile
    pitch_type: e.g. "SL"
    change_type: "ADD", "DROP", "USAGE_INCREASE", "USAGE_DECREASE"
    target_outcome_before: target pitcher's own current value of
        outcome_metric (the lagged-outcome predictor)
    outcome_metric: "war", "fip", or "xwoba_against"
    target_player_id, target_season: identifies the target's raw
        arsenal composition (needed for the pitch-composition
        similarity -- which SPECIFIC pitches they throw, not just
        aggregate shape summary numbers)

    Returns: dict with predicted_outcome, predicted_change, method
        used, and the weighted comparison group (for show_similar_
        comps to use)
    """
    outcome_before_col = f"{outcome_metric}_before"
    outcome_after_col = f"{outcome_metric}_after"

    group = training_df[
        (training_df["pitch_type"] == pitch_type)
        & (training_df["change_type"] == change_type)
        & training_df[outcome_before_col].notna()
        & training_df[outcome_after_col].notna()
    ].copy()

    # drop rows with incomplete similarity features (e.g. the ~2%
    # of pitcher-seasons never matched to any release-point source)
    # -- a single NaN anywhere in a row's feature vector makes its
    # distance to the target NaN, which corrupts the ENTIRE weight
    # array for the whole group, not just that one row
    n_before_nan_drop = len(group)
    group = group.dropna(subset=SIMILARITY_FEATURE_COLS)
    n_dropped_for_nan = n_before_nan_drop - len(group)
    if n_dropped_for_nan > 0:
        print(
            f"  (excluded {n_dropped_for_nan} historical event(s) with "
            f"incomplete similarity features)"
        )

    if len(group) == 0:
        return {
            "predicted_outcome": None,
            "method": "no_historical_precedent",
            "n_historical_events": 0,
            "group": group,
            "mean_reversion_coef": None,
            "change_specific_coef": None,
            "new_pitch_distance_coef": None,
            "synergy_coef": None,
            "used_distance_feature": False,
            "used_synergy_feature": False
        }

    # similarity distance -- Euclidean in the standardized feature
    # space, against each historical event's PRE-CHANGE profile
    target_vec = np.array([target_features[c] for c in SIMILARITY_FEATURE_COLS])
    if np.isnan(target_vec).any():
        missing = [c for c, v in zip(SIMILARITY_FEATURE_COLS, target_vec) if np.isnan(v)]
        return {
            "predicted_outcome": None,
            "method": "incomplete_target_profile",
            "n_historical_events": len(group),
            "group": group,
            "error": f"Target profile missing: {missing}",
            "mean_reversion_coef": None,
            "change_specific_coef": None,
            "new_pitch_distance_coef": None,
            "synergy_coef": None,
            "used_distance_feature": False,
            "used_synergy_feature": False
        }

    hist_vecs = group[SIMILARITY_FEATURE_COLS].values

    # look up target's raw arsenal composition -- needed for the
    # pitch-composition similarity (which SPECIFIC pitches, not just
    # aggregate shape numbers)
    target_arsenal_key = (target_player_id, target_season)
    if target_arsenal_key not in arsenal_raw_indexed.index:
        return {
            "predicted_outcome": None,
            "method": "no_target_arsenal_data",
            "n_historical_events": len(group),
            "group": group,
            "mean_reversion_coef": None,
            "change_specific_coef": None,
            "new_pitch_distance_coef": None,
            "synergy_coef": None,
            "used_distance_feature": False,
            "used_synergy_feature": False
        }
    target_arsenal_row = arsenal_raw_indexed.loc[target_arsenal_key]
    if isinstance(target_arsenal_row, pd.DataFrame):
        target_arsenal_row = target_arsenal_row.iloc[0]

    # pairwise pitch-composition similarity against EACH historical
    # row's PRE-CHANGE (season_from) arsenal -- same "compare against
    # the pre-change profile" convention as the aggregate similarity
    # features
    jaccard_sims = []
    shared_shape_dists = []
    for _, hist_row in group.iterrows():
        hist_key = (hist_row["player_id"], hist_row["season_from"])
        if hist_key in arsenal_raw_indexed.index:
            hist_arsenal_row = arsenal_raw_indexed.loc[hist_key]
            if isinstance(hist_arsenal_row, pd.DataFrame):
                hist_arsenal_row = hist_arsenal_row.iloc[0]
            j, d = compute_pitch_composition_similarity(target_arsenal_row, hist_arsenal_row)
        else:
            j, d = 0.0, np.nan
        jaccard_sims.append(j)
        shared_shape_dists.append(d)

    jaccard_sims = np.array(jaccard_sims)
    shared_shape_dists = np.array(shared_shape_dists)
    jaccard_distances = 1 - jaccard_sims  # convert similarity (1=identical) to distance (0=identical)

    # standardize shared_shape_dists WITHIN this group -- self-
    # calibrating, same philosophy as the Gaussian kernel bandwidth
    # below. NaN (no shared pitch types at all between this
    # historical pitcher and the target) is filled with the group's
    # WORST observed value, not the mean -- "no shared pitches" is
    # informative (genuinely dissimilar), not an "average" case that
    # mean-imputation would incorrectly treat as neutral
    valid_shape_dists = shared_shape_dists[~np.isnan(shared_shape_dists)]
    if len(valid_shape_dists) > 0:
        fill_value = valid_shape_dists.max()
        shared_shape_dists_filled = np.where(np.isnan(shared_shape_dists), fill_value, shared_shape_dists)
        shape_mean = shared_shape_dists_filled.mean()
        shape_std = shared_shape_dists_filled.std()
        shape_std = shape_std if shape_std > 0 else 1.0
        shared_shape_z = (shared_shape_dists_filled - shape_mean) / shape_std
    else:
        # no valid pitch-shape comparisons anywhere in this group --
        # this component can't contribute anything, treat as neutral
        # rather than crashing
        shared_shape_z = np.zeros(len(group))

    group["jaccard_similarity"] = jaccard_sims
    group["shared_shape_distance"] = shared_shape_dists

    # STARTING-USAGE-LEVEL similarity -- only meaningful for DROP/
    # USAGE_INCREASE/USAGE_DECREASE, where the target's own current
    # usage of this pitch type is a real, variable quantity. Not
    # applied to ADD, where usage_pct_before is always ~0 by
    # definition for both target and every historical comp.
    use_usage_level = change_type != "ADD"
    if use_usage_level:
        target_usage_for_pt = target_arsenal_row.get(f"{pitch_type}_usage_pct", np.nan)
        if pd.isna(target_usage_for_pt):
            target_usage_for_pt = 0.0

        usage_level_dists = (group["usage_pct_before"] - target_usage_for_pt).abs().values

        usage_mean = usage_level_dists.mean()
        usage_std = usage_level_dists.std()
        usage_std = usage_std if usage_std > 0 else 1.0
        usage_level_z = (usage_level_dists - usage_mean) / usage_std

        group["usage_level_distance"] = usage_level_dists
    else:
        usage_level_z = np.zeros(len(group))

    # WEIGHTED combined distance -- standard weighted-Euclidean form
    # (each dimension's squared difference scaled by its own weight
    # before summing), covering the existing 7 aggregate features
    # (their tiered FEATURE_WEIGHTS), the two pitch-composition
    # components (WEIGHT_PITCH_COMPOSITION, WEIGHT_SHARED_SHAPE),
    # PLUS starting-usage-level for DROP/INCREASE/DECREASE
    # (WEIGHT_USAGE_LEVEL) -- replaces the old unweighted Euclidean
    # distance, which treated every dimension as equally important
    # just by virtue of being standardized
    weighted_sq_sum = np.zeros(len(group))
    for i, col in enumerate(SIMILARITY_FEATURE_COLS):
        w = FEATURE_WEIGHTS.get(col, 1.0)
        weighted_sq_sum += w * (hist_vecs[:, i] - target_vec[i]) ** 2

    weighted_sq_sum += WEIGHT_PITCH_COMPOSITION * (jaccard_distances ** 2)
    weighted_sq_sum += WEIGHT_SHARED_SHAPE * (shared_shape_z ** 2)
    weighted_sq_sum += WEIGHT_USAGE_LEVEL * (usage_level_z ** 2)

    distances = np.sqrt(weighted_sq_sum)

    # self-calibrating Gaussian kernel bandwidth -- median pairwise
    # distance WITHIN this specific comparison group, not one fixed
    # number assumed to fit every (pitch_type, change_type) group
    bandwidth = np.median(distances) if len(distances) > 1 else 1.0
    bandwidth = max(bandwidth, 1e-6)  # avoid divide-by-zero for a
                                        # single-event group
    similarity_wt = np.exp(-(distances ** 2) / (2 * bandwidth ** 2))

    reliability_wt = reliability_weight(group["bip_after"].values)

    combined_wt = similarity_wt * reliability_wt
    group["similarity_weight"] = similarity_wt
    group["reliability_weight"] = reliability_wt
    group["combined_weight"] = combined_wt

    if combined_wt.sum() == 0:
        return {
            "predicted_outcome": None,
            "method": "zero_total_weight",
            "n_historical_events": len(group),
            "group": group,
            "mean_reversion_coef": None,
            "change_specific_coef": None,
            "new_pitch_distance_coef": None,
            "synergy_coef": None,
            "used_distance_feature": False,
            "used_synergy_feature": False
        }

    mean_reversion_coef = None
    change_specific_coef = None
    new_pitch_distance_coef = None
    synergy_coef = None

    # each extra feature is checked INDEPENDENTLY against the
    # original group's own coverage -- distance and synergy features
    # happened to have identical (100%) coverage in this project's
    # real data, but the code doesn't assume that; a future data
    # update where they diverge should still work correctly
    use_distance_feature = (
        change_type == "ADD"
        and "new_pitch_nn_distance_from_existing" in group.columns
        and group["new_pitch_nn_distance_from_existing"].notna().sum() >= MIN_GROUP_SIZE_FOR_REGRESSION
    )
    use_synergy_feature = (
        change_type == "ADD"
        and "arsenal_synergy_whiff_delta" in group.columns
        and group["arsenal_synergy_whiff_delta"].notna().sum() >= MIN_GROUP_SIZE_FOR_REGRESSION
    )

    dropna_cols = []
    if use_distance_feature:
        dropna_cols.append("new_pitch_nn_distance_from_existing")
    if use_synergy_feature:
        dropna_cols.append("arsenal_synergy_whiff_delta")
    if dropna_cols:
        group = group.dropna(subset=dropna_cols)
        combined_wt = group["combined_weight"].values

    if len(group) >= MIN_GROUP_SIZE_FOR_REGRESSION:
        # weighted regression: Outcome(t+1) ~ Outcome(t) + usage_delta
        # (+ new_pitch_nn_distance_from_existing, + arsenal_synergy_
        # whiff_delta, for ADD events with enough coverage of each --
        # see 39_validate_pitch_type_sensitivity.py's Check 4, which
        # found usage_pct_delta ALONE wasn't enough to give the
        # change-specific term a stable, meaningful signal)
        #
        # Predictors are STANDARDIZED before fitting -- all of these
        # are on very different raw scales, so comparing the RAW
        # coefficients directly would be comparing different units,
        # not genuine relative importance. Standardizing first makes
        # all coefficients comparable in magnitude -- this is what
        # 39's coefficient comparison actually needs to be meaningful.
        predictor_cols = [outcome_before_col, "usage_pct_delta"]
        if use_distance_feature:
            predictor_cols.append("new_pitch_nn_distance_from_existing")
        if use_synergy_feature:
            predictor_cols.append("arsenal_synergy_whiff_delta")

        X_raw = group[predictor_cols].values
        y = group[outcome_after_col].values

        X_mean = X_raw.mean(axis=0)
        X_std = X_raw.std(axis=0)
        X_std[X_std == 0] = 1.0  # avoid divide-by-zero for a
                                   # degenerate no-variance column
        X_scaled = (X_raw - X_mean) / X_std

        model = LinearRegression()
        model.fit(X_scaled, y, sample_weight=combined_wt)

        mean_reversion_coef = model.coef_[0]
        change_specific_coef = model.coef_[1]
        next_coef_idx = 2
        if use_distance_feature:
            new_pitch_distance_coef = model.coef_[next_coef_idx]
            next_coef_idx += 1
        if use_synergy_feature:
            synergy_coef = model.coef_[next_coef_idx]
            next_coef_idx += 1

        target_usage_delta = target_features.get("usage_pct_delta", group["usage_pct_delta"].median())
        target_predictors = [target_outcome_before, target_usage_delta]
        if use_distance_feature:
            target_distance = target_features.get(
                "new_pitch_nn_distance_from_existing",
                group["new_pitch_nn_distance_from_existing"].median()
            )
            target_predictors.append(target_distance)
        if use_synergy_feature:
            target_synergy = target_features.get(
                "arsenal_synergy_whiff_delta",
                group["arsenal_synergy_whiff_delta"].median()
            )
            target_predictors.append(target_synergy)

        target_X_raw = np.array([target_predictors])
        target_X_scaled = (target_X_raw - X_mean) / X_std
        predicted = model.predict(target_X_scaled)[0]

        extras = []
        if use_distance_feature:
            extras.append("distance")
        if use_synergy_feature:
            extras.append("synergy")
        method = "weighted_regression_with_" + "_and_".join(extras) if extras else "weighted_regression"
    else:
        # too few events for a stable fit -- weighted average of the
        # outcome CHANGE, applied to the target's own baseline
        outcome_deltas = group[outcome_after_col].values - group[outcome_before_col].values
        weighted_avg_delta = np.average(outcome_deltas, weights=combined_wt)
        predicted = target_outcome_before + weighted_avg_delta

        method = "weighted_average_fallback"

    return {
        "predicted_outcome": predicted,
        "predicted_change": predicted - target_outcome_before,
        "method": method,
        "n_historical_events": len(group),
        "group": group,
        "mean_reversion_coef": mean_reversion_coef,
        "change_specific_coef": change_specific_coef,
        "new_pitch_distance_coef": new_pitch_distance_coef,
        "synergy_coef": synergy_coef,
        "used_distance_feature": use_distance_feature,
        "used_synergy_feature": use_synergy_feature
    }


def show_similar_comps(result, top_k=TOP_K_COMPS):
    """
    Given a predict_arsenal_change_effect() result, print the top-K
    most similar (by combined weight) real historical comps.
    """
    group = result["group"]
    if len(group) == 0 or "combined_weight" not in group.columns:
        print("No comparison group available.")
        return

    top = group.nlargest(top_k, "combined_weight")
    display_cols = [
        "player_name", "season_from", "season_to", "usage_pct_before",
        "usage_pct_after", "similarity_weight", "reliability_weight"
    ]
    if "jaccard_similarity" in top.columns:
        display_cols += ["jaccard_similarity", "shared_shape_distance"]
    print(top[display_cols].to_string(index=False))


# ============================================================
# DEMONSTRATION (only runs when this file is executed directly,
# not when its functions are imported elsewhere -- see
# 39_validate_pitch_type_sensitivity.py, which imports training,
# predict_arsenal_change_effect, and show_similar_comps from here)
# ============================================================

if __name__ == "__main__":
    print("\n\n==============================")
    print("DEMONSTRATION")
    print("==============================")

    if len(training) > 0:
        # use a real pitcher's real profile as the target, to demonstrate
        # against real data rather than a fabricated example
        example_target_row = similarity_features.iloc[0]
        target_features = {c: example_target_row[c] for c in SIMILARITY_FEATURE_COLS}
        target_features["usage_pct_delta"] = 10.0  # simulate adding ~10% usage

        print(f"\nTarget (demonstration): {example_target_row['player_name']}, {example_target_row['season']} profile")

        # DEFAULT METRIC PREFERENCE, based on real evidence from
        # 41_outcome_metric_comparison.py: xwoba_against showed the
        # highest sign consistency (75.0% vs. WAR's 56.3%) and a
        # strong mean-reversion-to-change ratio -- WAR carries a
        # whole season's worth of context (defense, luck, innings
        # distribution) unrelated to any single arsenal change,
        # diluting the signal relative to metrics closer to what a
        # pitch itself does. fip is the fallback (best ratio, though
        # not the best consistency); war is last since it was worst
        # on both measures.
        available_metrics = [c for c in ["xwoba_against", "fip", "war"] if f"{c}_before" in training.columns]

        if available_metrics:
            metric = available_metrics[0]
            outcome_before_col = f"{metric}_before"
            target_baseline = outcomes[outcomes["player_id"] == example_target_row["player_id"]]
            target_outcome_before = (
                target_baseline[metric].iloc[0] if len(target_baseline) > 0 and metric in target_baseline.columns
                else training[outcome_before_col].median()
            )

            for pitch_type, change_type in [("SL", "ADD"), ("ST", "ADD")]:
                print(f"\n--- Simulating: {change_type} {pitch_type}, predicting {metric} ---")
                result = predict_arsenal_change_effect(
                    target_features, pitch_type, change_type,
                    target_outcome_before, metric,
                    example_target_row["player_id"], example_target_row["season"]
                )
                print(
                    f"Method: {result['method']}, "
                    f"historical events available: {result['n_historical_events']}"
                )
                if result["predicted_outcome"] is not None:
                    print(
                        f"Predicted {metric}: {result['predicted_outcome']:.3f} "
                        f"(change of {result['predicted_change']:+.3f} from "
                        f"baseline {target_outcome_before:.3f})"
                    )
                    print(f"\nTop {TOP_K_COMPS} most similar historical comps:")
                    show_similar_comps(result)
        else:
            print("No outcome metrics available in the outcomes file to demonstrate with.")
    else:
        print("Training dataset is empty -- cannot demonstrate.")
