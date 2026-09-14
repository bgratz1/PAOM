"""
11_command_component.py

Purpose:
--------
Measure command: how consistently a pitcher locates a given pitch
type, as a skill distinct from Effectiveness (does the pitch work)
and Movement (does the arsenal cover enough shape).

The problem with raw location variance:
-----------------------------------------
A pitcher who deliberately works a pitch to two different spots
(e.g. backdoor to same-side batters, backfoot to opposite-side
batters -- or simply up in the zone for swing-and-miss vs. down
for ground balls) is showing GOOD command of two targets, not bad
command of one. Naive variance in plate_x/plate_z across all
throws of a pitch type would score that identically to a pitcher
who just can't repeat his release. Two fixes, applied together:

Fix 1 -- handedness-conditioned dispersion:
    The most common driver of intentional multi-locating is batter
    handedness. Measure dispersion SEPARATELY within same-side and
    opposite-side batters, not pooled -- so two tight, different
    targets (one per side) score as precise, not as scatter.

Fix 2 -- cluster-based dispersion within each split:
    Handedness doesn't explain every intentional pattern (e.g.
    up-in-zone vs down-in-zone against the same batter side). A
    small Gaussian Mixture Model (1 vs 2 components, selected by
    BIC) is fit to each handedness split's (plate_x, plate_z)
    points. Dispersion is measured as each pitch's distance to ITS
    OWN cluster's center, not to one overall center -- so a
    genuinely bimodal but each-mode-tight pattern still scores as
    precise. BIC selection means a 2nd cluster is only used when it
    meaningfully improves the fit -- a pitcher who's just scattered
    doesn't get an artificial second cluster invented to excuse it.

Method:
-------
1. For each pitcher x pitch type x batter-side split (with enough
   pitches), fit the best-of-1-or-2-component GMM.
2. Compute each pitch's distance to its assigned cluster center.
3. Average within the split, then combine splits (pitch-count
   weighted) to one dispersion number per pitcher x pitch type.
4. Convert to a 0-100 score (lower dispersion = higher score),
   with a confidence adjustment for pitch volume.

Output:
-------
PAOM_command_component.csv
"""

import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "clean_statcast_2025.csv"
OUTPUT_FILE = "PAOM_command_component.csv"

MIN_PITCHES = 50          # per pitcher x pitch type, overall
MIN_SPLIT_PITCHES = 20    # per handedness split, to attempt its own GMM
MIN_PITCHES_FOR_2_CLUSTER = 40  # only try a 2nd cluster above this size

CONFIDENCE_SATURATION_PITCHES = 300


# ============================================================
# LOAD DATA
# ============================================================

print("Loading cleaned Statcast...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Loaded {len(df):,} pitches")

required_cols = [
    "player_name", "pitch_type", "stand", "plate_x", "plate_z"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required_cols).copy()

print(f"Rows with complete location data: {len(df):,}")


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
# BEST-OF-1-OR-2-COMPONENT GMM DISPERSION
# ============================================================

def cluster_dispersion(points):
    """
    points: array of shape (n, 2) -> (plate_x, plate_z)

    Fits a 1-component GMM (equivalent to distance from the mean)
    and, if there's enough data, a 2-component GMM. Picks whichever
    has the lower BIC, then returns the average distance from each
    point to its own assigned cluster's center.
    """

    n = len(points)

    gmm1 = GaussianMixture(n_components=1, random_state=42)
    gmm1.fit(points)
    bic1 = gmm1.bic(points)
    best_model = gmm1
    best_bic = bic1

    if n >= MIN_PITCHES_FOR_2_CLUSTER:
        try:
            gmm2 = GaussianMixture(n_components=2, random_state=42)
            gmm2.fit(points)
            bic2 = gmm2.bic(points)

            if bic2 < best_bic:
                best_model = gmm2
                best_bic = bic2
        except Exception:
            # degenerate fit (e.g. near-duplicate points) -- fall
            # back to the 1-component result
            pass

    labels = best_model.predict(points)
    centers = best_model.means_[labels]

    distances = np.sqrt(
        ((points - centers) ** 2).sum(axis=1)
    )

    return distances.mean(), best_model.n_components


# ============================================================
# PER PITCHER x PITCH TYPE x HANDEDNESS-SPLIT DISPERSION
# ============================================================

print("\nComputing command dispersion...")

records = []

for (pitcher, pitch_type), group in df.groupby(
    ["player_name", "pitch_type"]
):

    split_dispersions = []
    split_weights = []
    split_n_clusters = []

    for stand_val, sub in group.groupby("stand"):

        if len(sub) < MIN_SPLIT_PITCHES:
            continue

        points = sub[["plate_x", "plate_z"]].values

        disp, n_clusters = cluster_dispersion(points)

        split_dispersions.append(disp)
        split_weights.append(len(sub))
        split_n_clusters.append(n_clusters)

    if split_dispersions:
        # pitch-count-weighted combination across handedness splits
        command_dispersion = np.average(
            split_dispersions, weights=split_weights
        )
        used_handedness_split = True
    else:
        # neither handedness split had enough pitches on its own --
        # fall back to clustering the whole pooled group so we're
        # not throwing away pitchers with just moderate volume
        points = group[["plate_x", "plate_z"]].values
        command_dispersion, _ = cluster_dispersion(points)
        used_handedness_split = False

    records.append({
        "player_name": pitcher,
        "pitch_type": pitch_type,
        "pitches": len(group),
        "command_dispersion": command_dispersion,
        "used_handedness_split": used_handedness_split,
        "avg_clusters_per_split": (
            np.mean(split_n_clusters) if split_n_clusters else np.nan
        )
    })

command_df = pd.DataFrame(records)

print(f"Pitcher x pitch type rows scored: {len(command_df):,}")

print(
    f"\nUsed handedness split: "
    f"{command_df['used_handedness_split'].sum():,} / {len(command_df):,}"
)

print(
    "\nAverage clusters per split (>1 means real multi-locating "
    "detected somewhere):"
)
print(command_df["avg_clusters_per_split"].describe())


# ============================================================
# CONVERT TO 0-100 SCORE (LOWER DISPERSION = HIGHER SCORE)
# ============================================================

"""
Dispersion is normalized RELATIVE TO EACH PITCH TYPE'S OWN
population, not pooled across all pitch types globally. A sweeper
is inherently harder to command precisely than a changeup -- pooling
them into one global z-score meant a pitcher's score partly
reflected WHICH pitch types they throw, not how good their command
actually is relative to peers throwing the same pitch. Comparing a
sweeper's dispersion only to other sweepers' isolates "how well do
you command this pitch, given how hard this pitch type generally
is to command" -- a more honest, difficulty-adjusted signal.

Falls back to the old global z-score for any pitch type with too
few pitchers (MIN_PITCHERS_PER_TYPE) to establish a reliable
pitch-type-specific baseline -- a rare pitch type (e.g. eephus,
screwball) with only 2-3 pitchers can't support its own stable mean/
std, so those fall back to comparison against the whole population
rather than producing an unstable or undefined z-score.
"""

MIN_PITCHERS_PER_TYPE = 5

global_mean = command_df["command_dispersion"].mean()
global_std = command_df["command_dispersion"].std()

command_df["dispersion_z_global"] = (
    command_df["command_dispersion"] - global_mean
) / global_std

def relative_z(group):
    if len(group) < MIN_PITCHERS_PER_TYPE or group.std() == 0:
        return pd.Series(np.nan, index=group.index)
    return (group - group.mean()) / group.std()

command_df["dispersion_z_relative"] = (
    command_df.groupby("pitch_type")["command_dispersion"]
    .transform(relative_z)
)

n_using_relative = command_df["dispersion_z_relative"].notna().sum()
print(
    f"\nPitch-type-relative normalization used for "
    f"{n_using_relative:,} / {len(command_df):,} rows "
    f"(rest fall back to global, due to sub-{MIN_PITCHERS_PER_TYPE}-pitcher "
    f"pitch types)"
)

command_df["dispersion_z"] = (
    command_df["dispersion_z_relative"]
    .fillna(command_df["dispersion_z_global"])
)

# flip sign: lower dispersion should mean a HIGHER command score
command_df["raw_command_score"] = -command_df["dispersion_z"]


def percentile_rank(series):
    """
    Percentile rank instead of min-max scaling: min-max makes every
    pitcher's score entirely dependent on wherever the single most
    extreme pitcher happens to sit, which means "50" doesn't
    reliably mean "average" and scores aren't comparable across
    components. Percentile rank fixes both: 50 = exact median,
    90 = better than 90% of qualifying pitchers, consistently
    across every PAOM component.
    """
    return series.rank(pct=True) * 100

command_df["command_score"] = percentile_rank(command_df["raw_command_score"])


# ============================================================
# CONFIDENCE ADJUSTMENT
# ============================================================

"""
(Reverted after a shrinkage-removal experiment showed clear
evidence this protection was doing real work -- see paom.md /
conversation history for the comparison.)
"""

command_df["confidence_score"] = (
    np.minimum(
        command_df["pitches"] / CONFIDENCE_SATURATION_PITCHES,
        1
    )
    * 100
)

league_average = command_df["command_score"].mean()

command_df["trusted_command_score"] = (
    command_df["command_score"] * (command_df["confidence_score"] / 100)
    + league_average * (1 - command_df["confidence_score"] / 100)
)


# ============================================================
# SAVE OUTPUT
# ============================================================

final_columns = [
    "player_name",
    "pitch_type",
    "pitches",
    "command_dispersion",
    "dispersion_z",
    "dispersion_z_relative",
    "used_handedness_split",
    "avg_clusters_per_split",
    "raw_command_score",
    "command_score",
    "confidence_score",
    "trusted_command_score"
]

output = (
    command_df[final_columns]
    .sort_values("trusted_command_score", ascending=False)
)

output.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("Command Component Saved")
print("==============================")

print(
    f"""
Pitcher x pitch type rows:
{len(output):,}

Average command score:
{output['trusted_command_score'].mean():.2f}

Average confidence:
{output['confidence_score'].mean():.2f}
"""
)

print("\nTop 20 command scores")
print("-----------------------")
print(
    output[
        [
            "player_name", "pitch_type", "pitches",
            "command_dispersion", "avg_clusters_per_split",
            "trusted_command_score"
        ]
    ]
    .head(20)
)

print("\nSaved:")
print(f"- {OUTPUT_FILE}")
