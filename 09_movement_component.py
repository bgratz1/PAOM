"""
09_movement_component.py

Purpose:
--------
Create the PAOM Movement Component.

Goal:
-----
Measure how much of the movement landscape a pitcher's arsenal
covers, independent of outcomes. This is NOT outcome-regressed —
unlike Effectiveness (07d) and Tunneling (08), Movement is a
descriptive/latent construct: it describes the SHAPE of an
arsenal, not its results.

Method:
-------
1. Aggregate to pitcher x pitch type from master_pitch_table_2025.csv
   (needed because PAOM_effectiveness_component.csv does not carry
   HB / IVB / velo through its own aggregation).
2. Filter out pitch types below a usage floor so rarely-thrown
   show-me pitches don't distort the geometry.
3. For each pitcher, compute five raw geometric features:
       - horizontal_coverage   (max HB - min HB)
       - vertical_coverage     (max IVB - min IVB)
       - convex_hull_area      (area in HB-IVB space; bounding-box
                                 fallback for <3 pitch types)
       - fastball_relative_break (unweighted avg distance from the
                                 pitcher's own FF/SI anchor pitch to
                                 their other pitches -- measures
                                 spread from the pitch a hitter is
                                 calibrated to expect, not from an
                                 abstract arsenal-centered point)
       - avg_nn_distance       (avg distance to each pitch's
                                 nearest neighbor in movement space)
   (velocity_range and quadrant_coverage_count are also computed,
   kept as reference columns only -- velocity lives in its own
   dedicated 21_velocity_component.py; quadrant_coverage_count was
   the weakest performer of an earlier six-feature version.)
4. Dampen EACH of the five features individually against
   n_pitch_types (partial removal, DAMPENING_FACTOR=0.6 -- "more
   pitch types is partly real value," not pure noise to strip out).
   NO orthogonalization between features this version -- see the
   design note further down for why three different orthogonalization
   attempts (dampen-then-orthogonalize, orthogonalize-then-dampen, a
   joint regression combining both) each traded one problem for
   another, and kept discarding features that likely still carried
   real information Ridge just couldn't cleanly attribute credit to.
5. Combine ALL FIVE dampened features into ONE single PCA (PC1) --
   not split into groups, not orthogonalized against each other.
   Two prior designs tried splitting by concept (Coverage vs.
   Distinctness) and full orthogonalization; both kept hitting the
   same pattern -- whichever features were most correlated with
   each other lost their individual credit to whichever one Ridge
   favored, no matter how the split was drawn. That consistent
   pattern across multiple structural attempts suggests the
   features are correlated because they reflect a genuinely shared
   underlying fact (a deep, well-built arsenal), not because of a
   fixable measurement artifact -- so this version stops trying to
   force separation and lets one PCA find the shared axis directly.
6. Percentile rank + confidence shrink PC1 -- trusted_movement_score
   is the single resulting Movement score, merged into
   10_paom_score.py as one direct input (same pattern as
   Effectiveness/Command/Velocity), not several candidates to
   combine or exclude from a display composite.

Output:
-------
PAOM_movement_component.csv
"""

import pandas as pd
import numpy as np

from scipy.spatial import ConvexHull, QhullError

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "master_pitch_table_2025.csv"

OUTPUT_FILE = "PAOM_movement_component.csv"

MIN_PITCHES = 50          # per pitch type, looser than effectiveness's
                           # 75/50 since confidence shrink handles the rest
MIN_PITCH_TYPE_USAGE = 0.05   # exclude sub-5% usage pitches from geometry

CONFIDENCE_PITCH_TYPES = 4    # num_pitch_types at which confidence saturates


# ============================================================
# LOAD DATA
# ============================================================

print("Loading master pitch table...")

df = pd.read_csv(INPUT_FILE)

print(f"Initial rows: {len(df):,}")


required_cols = [
    "player_name",
    "pitcher",
    "pitch_type",
    "p_throws",
    "pitches",
    "velo",
    "HB",
    "IVB",
    "usage"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# FILTER SAMPLE SIZE + USAGE FLOOR
# ============================================================

df = df[df["pitches"] >= MIN_PITCHES].copy()

print(f"After {MIN_PITCHES} pitch filter: {len(df):,}")

geo_df = df[df["usage"] >= MIN_PITCH_TYPE_USAGE].copy()

print(
    f"After {MIN_PITCH_TYPE_USAGE:.0%} usage floor "
    f"(used for geometry): {len(geo_df):,}"
)


# ============================================================
# CONVEX HULL AREA (with fallback for <3 points)
# ============================================================

def hull_area(points):
    """
    points: array of shape (n, 2) -> (HB, IVB)

    Returns (area, method) where method is
    'hull' (>=3 non-collinear points) or
    'bbox' (fallback for 2-pitch arsenals or
    degenerate/collinear cases).
    """

    n = len(points)

    if n < 3:
        hb_range = points[:, 0].max() - points[:, 0].min()
        ivb_range = points[:, 1].max() - points[:, 1].min()
        return hb_range * ivb_range, "bbox"

    try:
        hull = ConvexHull(points)
        return hull.volume, "hull"   # .volume = area in 2D
    except QhullError:
        # collinear points -> degenerate hull
        hb_range = points[:, 0].max() - points[:, 0].min()
        ivb_range = points[:, 1].max() - points[:, 1].min()
        return hb_range * ivb_range, "bbox"


# ============================================================
# PER-PITCHER GEOMETRY
# ============================================================

print("\nComputing arsenal geometry...")

records = []

ANCHOR_PRIORITY = ["FF", "SI"]  # four-seam preferred, sinker as fallback

for pitcher, group in geo_df.groupby("player_name"):

    hb = group["HB"].values
    ivb = group["IVB"].values
    velo = group["velo"].values
    pitch_types = group["pitch_type"].values

    n_pitch_types = len(group)

    horizontal_coverage = hb.max() - hb.min()
    vertical_coverage = ivb.max() - ivb.min()
    velocity_range = velo.max() - velo.min()

    points = np.column_stack([hb, ivb])
    area, hull_method = hull_area(points)

    # nearest-neighbor gap distance
    #
    # horizontal/vertical coverage and hull_area are all defined
    # by whichever pitches sit on the OUTER BOUNDARY of the
    # movement cloud -- an interior pitch (e.g. a changeup or
    # cutter sitting between a fastball and slider) can add real,
    # distinct separation without ever being the horizontal or
    # vertical extreme, and those three features are blind to it.
    #
    # For each pitch, find the distance to its closest neighbor
    # in HB-IVB space, then average across the arsenal (unweighted,
    # consistent with the rest of the structural-capability
    # features). This rewards arsenals with no big empty gaps
    # between pitches, independent of hull boundary membership.
    if n_pitch_types >= 2:
        nn_distances = []
        for i in range(n_pitch_types):
            other_idx = [j for j in range(n_pitch_types) if j != i]
            d = np.sqrt(
                (hb[i] - hb[other_idx]) ** 2
                + (ivb[i] - ivb[other_idx]) ** 2
            )
            nn_distances.append(d.min())
        avg_nn_distance = np.mean(nn_distances)
    else:
        # single-pitch arsenal -- no neighbor to measure against
        avg_nn_distance = 0.0

    # fastball-relative break
    #
    # REPLACES centroid_dispersion (removed after a diagnostic --
    # 22_movement_diagnostics.py -- found centroid_dispersion was
    # the single most redundant feature in the set, correlating
    # 0.5-0.8 with three of the other four). Instead of measuring
    # spread from the arsenal's own center of mass, this measures
    # spread from the pitcher's OWN fastball specifically -- the
    # pitch a hitter is calibrated to expect by default, so what
    # matters for a secondary pitch is how much it deviates from
    # THAT anchor, not from an abstract arsenal-centered point.
    #
    # Anchor is restricted to FF/SI (not "whichever pitch is thrown
    # most" for everyone) -- both are legitimately "fastball family"
    # from a hitter's timing perspective; a breaking ball becoming
    # the anchor for a breaking-ball-heavy pitcher would just
    # reintroduce an arbitrary reference point, the same redundancy
    # problem that got centroid_dispersion removed. When a pitcher
    # throws BOTH FF and SI, anchor on whichever they actually throw
    # MORE (not a fixed FF-always-wins rule) -- better reflects what
    # hitters are actually seeing most from that specific pitcher.
    available_anchors = [pt for pt in ANCHOR_PRIORITY if pt in pitch_types]

    if len(available_anchors) == 0:
        anchor_type = None
    elif len(available_anchors) == 1:
        anchor_type = available_anchors[0]
    else:
        pitches_count = group["pitches"].values
        ff_pitches = pitches_count[pitch_types == "FF"][0]
        si_pitches = pitches_count[pitch_types == "SI"][0]
        anchor_type = "FF" if ff_pitches >= si_pitches else "SI"

    no_anchor_available = anchor_type is None

    if not no_anchor_available:
        anchor_idx = np.where(pitch_types == anchor_type)[0][0]
        anchor_hb, anchor_ivb = hb[anchor_idx], ivb[anchor_idx]

        other_idx = [i for i in range(n_pitch_types) if i != anchor_idx]

        if len(other_idx) > 0:
            fb_distances = np.sqrt(
                (hb[other_idx] - anchor_hb) ** 2
                + (ivb[other_idx] - anchor_ivb) ** 2
            )
            fastball_relative_break = fb_distances.mean()
        else:
            # anchor was this pitcher's ONLY qualifying pitch -- a
            # genuine zero (no secondary pitches exist to measure
            # separation for), not missing data. Same convention as
            # avg_nn_distance's single-pitch-arsenal case above.
            fastball_relative_break = 0.0
    else:
        # no FF or SI available at all -- this IS missing data, not
        # a genuine zero. Filled in with the league average after
        # the loop (see below) instead of a fake 0.0, which would
        # otherwise shove these pitchers to the bottom of this
        # feature's distribution for a measurement gap, not a real
        # finding.
        fastball_relative_break = np.nan

    # quadrant coverage
    #
    # A different KIND of signal than the other five features:
    # discrete/binary-per-region rather than continuous area or
    # distance. Counts how many of the four movement-space
    # quadrants (sign of HB, sign of IVB -- the same zero-crosshair
    # reference already shown on the dashboard's Movement Map) this
    # pitcher's arsenal actually touches (1-4). Tested standalone
    # first (25_quadrant_coverage_component.py): correlates 0.569
    # with n_pitch_types (above the redundancy threshold, gets the
    # same dampening treatment as the other confounded features
    # below) but under 0.5 with every existing Movement feature
    # individually -- a real, structurally distinct signal, even
    # though its coarseness (85% of real pitchers land in just 2 of
    # the 4 possible values) limits how much fine differentiation
    # it can add.
    hb_sign = hb >= 0
    ivb_sign = ivb >= 0
    quadrant_coverage_count = len(set(zip(hb_sign, ivb_sign)))

    records.append({
        "player_name": pitcher,
        "n_pitch_types": n_pitch_types,
        "horizontal_coverage": horizontal_coverage,
        "vertical_coverage": vertical_coverage,
        "velocity_range": velocity_range,
        "hull_area": area,
        "hull_method": hull_method,
        "fastball_relative_break": fastball_relative_break,
        "anchor_type": anchor_type,
        "no_anchor_available": no_anchor_available,
        "avg_nn_distance": avg_nn_distance,
        "quadrant_coverage_count": quadrant_coverage_count
    })

movement_df = pd.DataFrame(records)

n_no_anchor = movement_df["no_anchor_available"].sum()

if n_no_anchor > 0:
    league_avg_fb_break = movement_df.loc[
        ~movement_df["no_anchor_available"], "fastball_relative_break"
    ].mean()

    movement_df.loc[
        movement_df["no_anchor_available"], "fastball_relative_break"
    ] = league_avg_fb_break

    print(
        f"\n{n_no_anchor:,} pitcher(s) had no qualifying FF or SI -- "
        f"filled fastball_relative_break with the league average "
        f"({league_avg_fb_break:.2f}) rather than 0.0, since a "
        f"missing anchor is a measurement gap, not evidence of zero "
        f"separation."
    )

print(f"Pitchers with movement geometry: {len(movement_df):,}")

print(
    "\nHull method breakdown:\n"
    f"{movement_df['hull_method'].value_counts()}"
)


# ============================================================
# DAMPENED PITCH-COUNT ADJUSTMENT
# ============================================================

"""
A diagnostic (22_movement_diagnostics.py) found four of Movement's
five raw features correlate 0.55-0.69 with n_pitch_types -- they
mechanically grow with more pitch types to draw a boundary/spread
around, largely independent of whether the shape itself is any
good. Only avg_nn_distance tested independent (-0.20) and is left
untouched.

Fully removing that relationship (group-mean subtraction) would
treat "throws more pitch types" as pure noise to strip out -- but
covering more distinct shapes is partly REAL value (a hitter
genuinely has more to prepare for), not just a measurement
artifact. So this dampens the group-level trend rather than fully
removing it:

    adjusted = feature - DAMPENING_FACTOR * (predicted_trend - feature_mean)

DAMPENING_FACTOR=0 leaves the feature untouched (old behavior).
DAMPENING_FACTOR=1 fully removes the linear n_pitch_types
relationship (equivalent to full group-mean demeaning). At any
value in between, a pitcher's OWN deviation from what their own
pitch count would predict (the real "is this shape well-built"
signal) is NEVER touched -- only the group-level baseline itself
is proportionally dampened. Default 0.6 removes most of the
mechanical inflation while keeping some legitimate credit for
genuinely covering more shapes.
"""

DAMPENING_FACTOR = 0.6
APPLY_DAMPENING = True  # hull_area dropped entirely (see
                          # CONFOUNDED_FEATURES note below); the
                          # dampen-after-orthogonalize instability
                          # that broke hull_area specifically may
                          # not affect this new feature set the same
                          # way. horizontal_coverage/vertical_coverage
                          # in particular had strong original
                          # pitch-count confounds (0.69/0.61 raw) and
                          # need dampening to be viable candidates at
                          # all -- set False to re-run the
                          # no-dampening comparison if needed.

APPLY_CONFIDENCE_SHRINKAGE = True  # REVERTED after a real-data test:
                          # with shrinkage disabled, Movement's ENTIRE
                          # top 20 became 2-pitch-type pitchers (no
                          # multi-pitch starters at all), and pitch-
                          # count correlation got WORSE (0.04 -> 0.14)
                          # -- unlike Velocity (see 21_velocity_
                          # component.py), Movement's features
                          # (hull_area, coverage, avg_nn_distance) are
                          # shape/spread statistics that are genuinely
                          # degenerate with very few points (a "hull
                          # area" from 2 points isn't a real area),
                          # so they need this protection. Velocity's
                          # features (a single extremal value, and a
                          # max-minus-min) don't have the same
                          # structural dependency on point count, and
                          # its shrinkage stays disabled as a result.

CONFOUNDED_FEATURES = [
    "hull_area",
    "horizontal_coverage",
    "vertical_coverage",
    "fastball_relative_break",
    "avg_nn_distance"
]

# Every raw feature gets dampened individually against n_pitch_types
# (simple single-predictor formula, no orthogonalization at all this
# time -- see the design note below for why).


# ============================================================
# INDIVIDUAL DAMPENING (NO ORTHOGONALIZATION THIS VERSION)
# ============================================================

"""
FOURTH design, after three different orthogonalization-based
attempts each traded one problem for another (dampen-then-
orthogonalize let orthogonalization undo dampening's pitch-count
fix; orthogonalize-then-dampen reintroduced severe redundancy
between shared-pitch-count-dependent features; the joint regression
fixed both but diluted pitch-count control and destabilized
fastball_relative_break). Each fix, and each attempt to drop
"unhelpful" features afterward, kept losing real information --
horizontal_coverage/vertical_coverage/hull_area getting a weak or
unstable Ridge coefficient doesn't necessarily mean they carry no
real signal; it may just mean linear Ridge on top of full
orthogonalization is the wrong tool for extracting value from
CORRELATED features that are still conceptually meaningful.

FIX: stop trying to force every feature into one mutually-
orthogonal set. Group by CONCEPT instead, and let PCA -- built
specifically for correlated inputs -- handle redundancy within each
group, the way it was originally designed to be used (this project
only abandoned PCA earlier because it was being fed already-fully-
orthogonalized features with nothing left to compress; that's not
true here, since no orthogonalization happens in this version).

Two concept groups:
  - COVERAGE: hull_area, horizontal_coverage, vertical_coverage --
    all fundamentally answer "how much space does the arsenal
    occupy," just measured three different ways.
  - DISTINCTNESS: fastball_relative_break, avg_nn_distance -- both
    answer "how differentiated are the pitches from each other,"
    a genuinely different question from Coverage.

Each of the five raw features is dampened against n_pitch_types
INDIVIDUALLY (same partial-credit philosophy as before -- 0.6
factor, "more pitch types is partly real value") -- but NOT
orthogonalized against each other. Redundancy WITHIN each concept
group is handled by PCA below, not by forcing zero correlation
first.
"""

print(f"\nDampening each feature individually against n_pitch_types...")

n_pitch_types_vals = movement_df["n_pitch_types"].values.astype(float)

for feature in CONFOUNDED_FEATURES:
    y = movement_df[feature].values.astype(float)

    slope, intercept = np.polyfit(n_pitch_types_vals, y, 1)
    pre_corr = np.corrcoef(y, n_pitch_types_vals)[0, 1]

    if APPLY_DAMPENING:
        predicted_trend = intercept + slope * n_pitch_types_vals
        feature_mean = y.mean()
        adjusted = y - DAMPENING_FACTOR * (predicted_trend - feature_mean)
    else:
        adjusted = y.copy()

    FLOOR = 0.01
    n_clipped = (adjusted < FLOOR).sum()
    adjusted = np.maximum(adjusted, FLOOR)

    movement_df[f"{feature}_pre_dampening"] = movement_df[feature]
    movement_df[feature] = adjusted

    post_corr = np.corrcoef(adjusted, n_pitch_types_vals)[0, 1]
    clip_note = f", {n_clipped} pitcher(s) floored at {FLOOR}" if n_clipped > 0 else ""
    damp_note = "" if APPLY_DAMPENING else " (dampening disabled)"

    print(
        f"  {feature}: correlation with n_pitch_types "
        f"{pre_corr:.3f} -> {post_corr:.3f}{damp_note}{clip_note}"
    )


# ============================================================
# SINGLE UNIFIED PCA (SIXTH DESIGN)
# ============================================================

"""
velocity_range REMOVED from this feature set entirely -- pulled out
into its own dedicated component (21_velocity_component.py). See
that script's docstring for why (a fundamentally different kind of
signal -- speed spread, not break shape -- with independent evidence
from the Tunneling investigation that it deserves to stand alone).

SIXTH design. The prior COVERAGE/DISTINCTNESS grouped-PCA version
fixed pitch-count control (correlation with n_pitch_types improved
to the best result of this whole investigation) but the same
redundancy problem just moved up a level: the two group scores
still correlated at 0.57 with each other, and Coverage still lost
its own credit to Distinctness in the actual Ridge fit -- the
identical "one correlated thing eats the other's contribution"
pattern seen with individual features, individual pairs, and now
whole groups, across five different structural attempts.

That consistent pattern across every attempted structure is itself
informative: Coverage and Distinctness likely aren't measurement
artifacts to be engineered apart -- they're probably genuinely
correlated FACTS about real MLB arsenals (a deep, well-built
repertoire tends to score well on both "covers a lot of movement
space" and "pitches are well-differentiated from the fastball,"
because both trace back to the same underlying "this is a good
arsenal"). No amount of restructuring fully separates two things
that are correlated in reality, not just in how they're measured.

FIX: stop trying to force separation. One single PCA across all
five dampened (not orthogonalized, not group-split) features
together -- close to Movement's very original design, but now with
the validated individual pitch-count dampening step it never had.
Let PCA find whatever single dominant axis captures the shared
"good arsenal" signal directly, rather than forcing artificial
groups that still end up correlated with each other anyway.

This also simplifies 10_paom_score.py considerably: Movement goes
back to being ONE column (trusted_movement_score) merged in
directly, the same way Effectiveness/Command/Velocity already are
-- no more multi-candidate merging or composite-display logic
needed, since there's only one Movement number now, not several to
combine or exclude from a display composite.
"""

MOVEMENT_FEATURES_FOR_PCA = [
    "hull_area",
    "horizontal_coverage",
    "vertical_coverage",
    "fastball_relative_break",
    "avg_nn_distance"
]

# shared confidence -- based on n_pitch_types
movement_df["confidence_score"] = (
    np.minimum(
        movement_df["n_pitch_types"] / CONFIDENCE_PITCH_TYPES,
        1
    )
    * 100
)

print(f"\nRunning single unified PCA across: {MOVEMENT_FEATURES_FOR_PCA}")

X = movement_df[MOVEMENT_FEATURES_FOR_PCA].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=len(MOVEMENT_FEATURES_FOR_PCA))
components = pca.fit_transform(X_scaled)

pc1_explained = pca.explained_variance_ratio_[0]

print(f"PC1 explains {pc1_explained:.1%} of variance")

if pc1_explained < 0.40:
    print(
        "WARNING: PC1 explains less than 40% of variance -- these "
        "features may not share a single clean axis."
    )

loadings = pd.DataFrame({
    "feature": MOVEMENT_FEATURES_FOR_PCA,
    "PC1_loading": pca.components_[0]
})
print("\nPCA loadings")
print(loadings)

raw_index = components[:, 0]

# sign-correct so higher = better, using hull_area as the reference
# direction (matches historical convention -- bigger hull area
# should always mean a higher score)
if np.corrcoef(raw_index, movement_df["hull_area"])[0, 1] < 0:
    raw_index = -raw_index
    loadings["PC1_loading"] *= -1

movement_df["raw_movement_score"] = raw_index

def percentile_rank(series):
    return series.rank(pct=True) * 100

movement_df["movement_score"] = percentile_rank(movement_df["raw_movement_score"])

league_average = movement_df["movement_score"].mean()

if APPLY_CONFIDENCE_SHRINKAGE:
    movement_df["trusted_movement_score"] = (
        movement_df["movement_score"] * (movement_df["confidence_score"] / 100)
        + league_average * (1 - movement_df["confidence_score"] / 100)
    )
else:
    movement_df["trusted_movement_score"] = movement_df["movement_score"]

print(
    f"\nConfidence shrinkage {'ENABLED' if APPLY_CONFIDENCE_SHRINKAGE else 'DISABLED (experiment)'} "
    f"for trusted_movement_score."
)


# ============================================================
# FINAL OUTPUT
# ============================================================

final_columns = [
    "player_name",
    "n_pitch_types",
    "horizontal_coverage",
    "horizontal_coverage_pre_dampening",
    "vertical_coverage",
    "vertical_coverage_pre_dampening",
    "velocity_range",
    "hull_area",
    "hull_area_pre_dampening",
    "hull_method",
    "fastball_relative_break",
    "fastball_relative_break_pre_dampening",
    "anchor_type",
    "no_anchor_available",
    "avg_nn_distance",
    "avg_nn_distance_pre_dampening",
    "quadrant_coverage_count",
    "confidence_score",
    "raw_movement_score",
    "movement_score",
    "trusted_movement_score"
]

movement_output = (
    movement_df[final_columns]
    .sort_values("trusted_movement_score", ascending=False)
)

movement_output.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("Movement Component Saved")
print("==============================")

print(
    f"""
Pitchers scored:
{len(movement_output):,}

Average movement score:
{movement_output['trusted_movement_score'].mean():.2f}

Average confidence:
{movement_output['confidence_score'].mean():.2f}
"""
)

print("\nTop 20 movement scores")
print("-----------------------")

print(
    movement_output[
        [
            "player_name",
            "n_pitch_types",
            "hull_area",
            "trusted_movement_score"
        ]
    ]
    .head(20)
)

print("\nSaved:")
print(f"- {OUTPUT_FILE}")


# ============================================================
# VALIDATION CHECKS
# ============================================================

print("\n==============================")
print("MOVEMENT VALIDATION")
print("==============================")

print("\nCorrelation with number of pitch types:")
print(
    movement_output["trusted_movement_score"].corr(
        movement_output["n_pitch_types"]
    )
)

# Merge effectiveness to confirm movement is NOT just re-deriving it
try:
    effectiveness = pd.read_csv("PAOM_effectiveness_component.csv")

    eff_pitcher = (
        effectiveness
        .groupby("player_name")["trusted_effectiveness"]
        .mean()
        .reset_index()
    )

    check = movement_output.merge(
        eff_pitcher,
        on="player_name",
        how="inner"
    )

    print("\nCorrelation with effectiveness (should be low):")
    print(
        check["trusted_movement_score"].corr(
            check["trusted_effectiveness"]
        )
    )

    check.to_csv("PAOM_movement_effectiveness_correlation.csv", index=False)

except FileNotFoundError:
    print("\nPAOM_effectiveness_component.csv not found -- skipping check.")
