"""
18_release_consistency_component.py

Purpose:
--------
Measure release point consistency: how tightly a pitcher's release
position clusters, both WITHIN each pitch type (mechanical
repeatability) and ACROSS pitch types (whether every pitch comes
from the same slot, or the release itself tips off which pitch is
coming).

This is distinct from Command (11_command_component.py), which
measures where a pitch LANDS (plate_x/plate_z) -- release
consistency measures where the ball LEAVES THE HAND (release_pos_x/
release_pos_z), a physically earlier and conceptually different
signal. A pitcher can have excellent command (consistently hits his
locations) while still tipping pitches with an inconsistent release,
or vice versa.

Two deliberate differences from Command's design:

1. NO handedness split. Command splits by batter side because
   pitchers often INTENTIONALLY target different locations against
   same- vs opposite-handed batters -- a real skill. Release point
   has no equivalent legitimate reason to differ by batter side; a
   pitcher's release mechanics for a given pitch type should be the
   same regardless of who's in the box. Splitting here would just
   dilute sample size for no real reason.

2. Both dispersion measures are UNWEIGHTED by usage at the pitcher
   level (mirroring Movement's "structural capability" philosophy,
   not Command's "usage-experienced" philosophy). A rarely-thrown
   pitch with a release point that stands out from the rest of the
   arsenal is arguably a BIGGER tell, not a smaller one, since
   hitters have less practice recognizing it -- diluting it by
   usage would understate exactly the pattern this component exists
   to catch.

Method:
-------
1. WITHIN-TYPE dispersion: for each pitcher x pitch type, cluster
   (release_x, release_z) points with the same best-of-1-or-2-
   component GMM approach used in Command (BIC-selected), then
   measure average distance to each point's assigned cluster
   center. Unlike Command, a detected second cluster here is more
   likely a genuine mechanical inconsistency (or two subtly
   different deliveries of the "same" pitch) than a deliberate,
   skillful pattern -- intentionally releasing one pitch type from
   two different slots isn't a recognized skill the way
   intentionally locating to two different spots is.

2. ACROSS-TYPE dispersion: treat each pitch type's own MEAN release
   point as a single point, then measure each pitch type's distance
   from the pitcher's overall (unweighted) release centroid. This
   captures whether the whole arsenal releases from one consistent
   slot -- the more deception-relevant of the two measures, and the
   one existing components don't capture (Tunneling's pairwise
   release_distance feeds into pair-level deception scoring, but
   isn't exposed as a standalone arsenal-wide consistency measure).

3. Combine both (standardized) via PCA into one score, sign-correct
   so lower dispersion = higher score, scale 0-100, apply a
   confidence adjustment based on total qualifying pitch volume.

Output:
-------
PAOM_release_consistency_component.csv
"""

import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "clean_statcast_2025.csv"
OUTPUT_FILE = "PAOM_release_consistency_component.csv"

MIN_PITCHES = 50               # per pitch type, same floor as Command
MIN_PITCHES_FOR_2_CLUSTER = 40  # only try a 2nd cluster above this size

CONFIDENCE_SATURATION_PITCHES = 300


# ============================================================
# LOAD DATA
# ============================================================

print("Loading cleaned Statcast...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Loaded {len(df):,} pitches")

required_cols = [
    "player_name", "pitch_type", "release_pos_x", "release_pos_z"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required_cols).copy()

print(f"Rows with complete release data: {len(df):,}")


# ============================================================
# FILTER TO PITCHER x PITCH TYPE WITH ENOUGH VOLUME
# ============================================================

pitch_counts = (
    df.groupby(["player_name", "pitch_type"])
    .size()
    .reset_index(name="n")
)

qualifying = pitch_counts[pitch_counts["n"] >= MIN_PITCHES]

df = df.merge(
    qualifying[["player_name", "pitch_type"]],
    on=["player_name", "pitch_type"],
    how="inner"
)

print(
    f"Pitcher x pitch type groups meeting {MIN_PITCHES}-pitch "
    f"minimum: {len(qualifying):,}"
)


# ============================================================
# BEST-OF-1-OR-2-COMPONENT GMM DISPERSION (same method as Command)
# ============================================================

def cluster_dispersion(points):
    """
    Fits a 1-component GMM and, if there's enough data, a
    2-component GMM, picks whichever has the lower BIC, and returns
    the average distance from each point to its own assigned
    cluster's center.
    """

    n = len(points)

    gmm1 = GaussianMixture(n_components=1, random_state=42)
    gmm1.fit(points)
    best_model = gmm1
    best_bic = gmm1.bic(points)

    if n >= MIN_PITCHES_FOR_2_CLUSTER:
        try:
            gmm2 = GaussianMixture(n_components=2, random_state=42)
            gmm2.fit(points)
            bic2 = gmm2.bic(points)

            if bic2 < best_bic:
                best_model = gmm2
                best_bic = bic2
        except Exception:
            pass

    labels = best_model.predict(points)
    centers = best_model.means_[labels]

    distances = np.sqrt(((points - centers) ** 2).sum(axis=1))

    return distances.mean(), best_model.n_components


# ============================================================
# PER-PITCHER RELEASE CONSISTENCY
# ============================================================

print("\nComputing release consistency...")

records = []

for pitcher, group in df.groupby("player_name"):

    within_dispersions = []
    within_weights = []
    within_n_clusters = []

    pitch_type_means = []

    for pitch_type, sub in group.groupby("pitch_type"):
        points = sub[["release_pos_x", "release_pos_z"]].values

        disp, n_clusters = cluster_dispersion(points)

        within_dispersions.append(disp)
        within_weights.append(len(sub))
        within_n_clusters.append(n_clusters)

        pitch_type_means.append({
            "pitch_type": pitch_type,
            "mean_x": points[:, 0].mean(),
            "mean_z": points[:, 1].mean()
        })

    # within-type dispersion: pitch-COUNT-weighted average across
    # this pitcher's pitch types -- this weighting is about sample
    # reliability per pitch type (a 400-pitch fastball's dispersion
    # estimate is more trustworthy than a 55-pitch curveball's),
    # NOT a usage-importance judgment; the arsenal-wide combination
    # below stays unweighted per the docstring's design rationale
    within_type_dispersion = np.average(
        within_dispersions, weights=within_weights
    )

    # across-type dispersion: unweighted mean of pitch-type mean
    # release points -- see docstring
    means_df = pd.DataFrame(pitch_type_means)

    centroid_x = means_df["mean_x"].mean()
    centroid_z = means_df["mean_z"].mean()

    across_distances = np.sqrt(
        (means_df["mean_x"] - centroid_x) ** 2
        + (means_df["mean_z"] - centroid_z) ** 2
    )

    across_type_dispersion = across_distances.mean()

    records.append({
        "player_name": pitcher,
        "n_pitch_types": len(group["pitch_type"].unique()),
        "total_pitches": len(group),
        "within_type_dispersion": within_type_dispersion,
        "across_type_dispersion": across_type_dispersion,
        "avg_clusters_per_pitch_type": np.mean(within_n_clusters)
    })

release_df = pd.DataFrame(records)

print(f"Pitchers scored: {len(release_df):,}")

single_type = (release_df["n_pitch_types"] == 1).sum()
print(
    f"Single-pitch-type pitchers (across_type_dispersion trivially "
    f"0 for these): {single_type:,}"
)

print(
    "\nAverage clusters per pitch type (>1 suggests real "
    "within-type inconsistency detected somewhere):"
)
print(release_df["avg_clusters_per_pitch_type"].describe())


# ============================================================
# STANDARDIZE + PCA
# ============================================================

print("\nCombining both dispersion measures via PCA...")

pca_features = ["within_type_dispersion", "across_type_dispersion"]

X = release_df[pca_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

print("\nExplained variance ratio")
print(pca.explained_variance_ratio_)

pc1_explained = pca.explained_variance_ratio_[0]

print(f"PC1 explains {pc1_explained:.1%} of variance")

if pc1_explained < 0.40:
    print(
        "\nWARNING: PC1 explains less than 40% of variance -- "
        "within-type and across-type dispersion may be measuring "
        "genuinely independent things rather than a single "
        "'consistency' axis. Inspect before trusting PC1 alone "
        "(same threshold convention as 09_movement_component.py)."
    )

release_df["raw_release_consistency_index"] = components[:, 0]

# sign-correct: PCA doesn't guarantee direction -- higher index
# should mean MORE consistent (lower dispersion on BOTH measures).
# Check alignment against both features, not just one -- if they
# disagree after the flip, that's a real signal PC1 doesn't cleanly
# represent "overall consistency" and is worth flagging rather than
# silently trusting the sign of just one feature.
within_corr = release_df["raw_release_consistency_index"].corr(
    release_df["within_type_dispersion"]
)

if within_corr > 0:
    release_df["raw_release_consistency_index"] *= -1

across_corr_after_flip = release_df["raw_release_consistency_index"].corr(
    release_df["across_type_dispersion"]
)

print(
    f"\nPost-flip correlation with within_type_dispersion: "
    f"{-abs(within_corr):.3f} (should be negative)"
)
print(
    f"Post-flip correlation with across_type_dispersion: "
    f"{across_corr_after_flip:.3f} (should also be negative)"
)

if across_corr_after_flip > 0:
    print(
        "\nWARNING: after sign-correcting for within_type_dispersion, "
        "the index correlates POSITIVELY with across_type_dispersion "
        "-- PC1 does not cleanly represent 'overall consistency' on "
        "both measures at once. Consider a simple standardized "
        "average of the two features instead of PCA if this persists "
        "on the real population."
    )


# ============================================================
# SCALE 0-100 (PERCENTILE RANK)
# ============================================================

"""
Percentile rank instead of min-max scaling: min-max makes every
pitcher's score entirely dependent on wherever the single most
extreme pitcher happens to sit, which means "50" doesn't reliably
mean "average" and scores aren't comparable across components.
Percentile rank fixes both: 50 = exact median, 90 = better than
90% of qualifying pitchers, consistently across every PAOM
component.
"""

def percentile_rank(series):
    return series.rank(pct=True) * 100

release_df["release_consistency_score"] = percentile_rank(
    release_df["raw_release_consistency_index"]
)


# ============================================================
# CONFIDENCE ADJUSTMENT
# ============================================================

release_df["confidence_score"] = (
    np.minimum(
        release_df["total_pitches"] / CONFIDENCE_SATURATION_PITCHES,
        1
    )
    * 100
)

league_average = release_df["release_consistency_score"].mean()

release_df["trusted_release_consistency_score"] = (
    release_df["release_consistency_score"]
    * (release_df["confidence_score"] / 100)
    + league_average * (1 - release_df["confidence_score"] / 100)
)


# ============================================================
# SAVE OUTPUT
# ============================================================

final_columns = [
    "player_name",
    "n_pitch_types",
    "total_pitches",
    "within_type_dispersion",
    "across_type_dispersion",
    "avg_clusters_per_pitch_type",
    "raw_release_consistency_index",
    "release_consistency_score",
    "confidence_score",
    "trusted_release_consistency_score"
]

output = (
    release_df[final_columns]
    .sort_values("trusted_release_consistency_score", ascending=False)
)

output.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("Release Consistency Component Saved")
print("==============================")

print(
    f"""
Pitchers scored:
{len(output):,}

Average release consistency score:
{output['trusted_release_consistency_score'].mean():.2f}

Average confidence:
{output['confidence_score'].mean():.2f}
"""
)

print("\nTop 20 release consistency scores")
print("------------------------------------")
print(
    output[
        [
            "player_name", "n_pitch_types", "total_pitches",
            "within_type_dispersion", "across_type_dispersion",
            "trusted_release_consistency_score"
        ]
    ]
    .head(20)
)

print("\nSaved:")
print(f"- {OUTPUT_FILE}")
