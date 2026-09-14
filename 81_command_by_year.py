"""
81_command_by_year.py

Purpose:
--------
The REAL 11_command_component.py, parameterized by YEAR -- all GMM/
handedness-split/pitch-type-relative-normalization logic UNCHANGED
from the original, only file names now vary by year.

Requires clean_statcast_{YEAR}.csv (from 75) -- stand/plate_x/
plate_z are standard raw Statcast columns that pass through 75's
cleaning step untouched, so no changes to 75 were needed for this.
"""

import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

import sys
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

INPUT_FILE = f"clean_statcast_{YEAR}.csv"
OUTPUT_FILE = f"PAOM_command_component_{YEAR}.csv"

MIN_PITCHES = 50
MIN_SPLIT_PITCHES = 20
MIN_PITCHES_FOR_2_CLUSTER = 40

CONFIDENCE_SATURATION_PITCHES = 300


# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading cleaned Statcast for {YEAR}...")

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
        command_dispersion = np.average(
            split_dispersions, weights=split_weights
        )
        used_handedness_split = True
    else:
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


# ============================================================
# CONVERT TO 0-100 SCORE (LOWER DISPERSION = HIGHER SCORE)
# ============================================================

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
    f"{n_using_relative:,} / {len(command_df):,} rows"
)

command_df["dispersion_z"] = (
    command_df["dispersion_z_relative"]
    .fillna(command_df["dispersion_z_global"])
)

command_df["raw_command_score"] = -command_df["dispersion_z"]


def percentile_rank(series):
    return series.rank(pct=True) * 100

command_df["command_score"] = percentile_rank(command_df["raw_command_score"])


# ============================================================
# CONFIDENCE ADJUSTMENT
# ============================================================

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

command_df["season"] = YEAR

final_columns = [
    "player_name",
    "season",
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
print(f"Command Component Saved ({YEAR})")
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

print("\nSaved:")
print(f"- {OUTPUT_FILE}")
