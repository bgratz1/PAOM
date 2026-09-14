"""
20_league_benchmarks_export.py

Purpose:
--------
Export same-handed league-average benchmarks for the dashboard's
Pitcher Detail page -- lets a pitcher's own Movement Map and
Release Point Map show how their shapes compare to the realistic
peer group (pitchers who throw with the same hand), rather than
viewing their arsenal in isolation.

Deliberately NOT mirroring lefties/righties onto a shared sign
convention -- the data stays exactly as Statcast reports it. The
benchmark itself is already scoped to the same hand, so there's no
mismatch to correct: a lefty's benchmark is built only from other
lefties, in the same raw coordinate convention that lefty's own
data uses.

Two benchmarks, two different granularities (deliberately):
- Movement: one average per (throwing hand, pitch type) -- shape
  is pitch-type-specific, so the benchmark should be too.
- Release point: one average per throwing hand only, NOT split by
  pitch type -- matches 18_release_consistency_component.py's own
  established principle that release mechanics shouldn't legitimately
  vary by pitch type in a way that needs separate benchmarking; a
  single overall release-point benchmark per hand is the right
  granularity here.

"League average" is computed as the average of each qualifying
PITCHER's own per-pitch-type average -- not pooled across every
individual pitch -- so a high-volume pitcher doesn't drown out
others in the benchmark. Same MIN_PITCHES floor used elsewhere in
the pipeline, so the benchmark population matches who's already
being scored throughout.

Output:
-------
PAOM_league_movement_benchmark.csv   (p_throws x pitch_type -> avg HB/IVB)
PAOM_league_release_benchmark.csv    (p_throws -> avg release_x/z)
PAOM_pitcher_hands.csv               (player_name -> p_throws lookup,
                                       unfiltered -- every pitcher who
                                       threw a tracked pitch, since this
                                       is just an identity lookup, not
                                       a scored/thresholded value)
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

MASTER_FILE = "master_pitch_table_2025.csv"
RELEASE_MEANS_FILE = "PAOM_release_map_means.csv"

MOVEMENT_BENCHMARK_OUTPUT = "PAOM_league_movement_benchmark.csv"
RELEASE_BENCHMARK_OUTPUT = "PAOM_league_release_benchmark.csv"
PITCHER_HANDS_OUTPUT = "PAOM_pitcher_hands.csv"

MIN_PITCHES = 50  # same floor as Movement/Release exports


# ============================================================
# PITCHER HAND LOOKUP (unfiltered -- pure identity lookup)
# ============================================================

print("Loading master pitch table...")

master = pd.read_csv(MASTER_FILE)

required_cols = ["player_name", "pitch_type", "p_throws", "HB", "IVB", "pitches"]

missing = [c for c in required_cols if c not in master.columns]

if missing:
    raise ValueError(f"Missing required columns in {MASTER_FILE}: {missing}")

hands = (
    master
    .groupby("player_name")["p_throws"]
    .first()
    .reset_index()
)

hands.to_csv(PITCHER_HANDS_OUTPUT, index=False)

print(f"Saved {len(hands):,} pitcher hand lookups")


# ============================================================
# MOVEMENT BENCHMARK: (p_throws x pitch_type) -> avg HB/IVB
# ============================================================

print("\nBuilding movement benchmark...")

qualifying = master[master["pitches"] >= MIN_PITCHES].copy()

# one row per pitcher x pitch_type (already effectively this grain,
# but groupby+mean is a harmless no-op if there are duplicates)
pitcher_pitch_avg = (
    qualifying
    .groupby(["player_name", "p_throws", "pitch_type"])
    .agg(HB=("HB", "mean"), IVB=("IVB", "mean"))
    .reset_index()
)

movement_benchmark = (
    pitcher_pitch_avg
    .groupby(["p_throws", "pitch_type"])
    .agg(
        avg_HB=("HB", "mean"),
        avg_IVB=("IVB", "mean"),
        n_pitchers=("player_name", "nunique")
    )
    .reset_index()
)

movement_benchmark.to_csv(MOVEMENT_BENCHMARK_OUTPUT, index=False)

print(f"Saved {len(movement_benchmark):,} (hand x pitch type) movement benchmarks")
print(movement_benchmark.sort_values("n_pitchers", ascending=False).head(10))


# ============================================================
# RELEASE BENCHMARK: p_throws -> avg release_x/z (all pitch types combined)
# ============================================================

print("\nBuilding release point benchmark...")

release_means = pd.read_csv(RELEASE_MEANS_FILE)

release_means = release_means.merge(hands, on="player_name", how="inner")

# per-pitcher overall release point: unweighted average across that
# pitcher's own pitch types, matching 18_release_consistency_component.py's
# convention (a rarely-thrown pitch's release point counts fully,
# not diluted by usage)
pitcher_overall_release = (
    release_means
    .groupby(["player_name", "p_throws"])
    .agg(
        release_x=("mean_release_x", "mean"),
        release_z=("mean_release_z", "mean")
    )
    .reset_index()
)

release_benchmark = (
    pitcher_overall_release
    .groupby("p_throws")
    .agg(
        avg_release_x=("release_x", "mean"),
        avg_release_z=("release_z", "mean"),
        n_pitchers=("player_name", "nunique")
    )
    .reset_index()
)

release_benchmark.to_csv(RELEASE_BENCHMARK_OUTPUT, index=False)

print(f"Saved {len(release_benchmark):,} (hand) release point benchmarks")
print(release_benchmark)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("League Benchmarks Saved")
print("==============================")

print("\nSaved:")
print(f"- {MOVEMENT_BENCHMARK_OUTPUT}")
print(f"- {RELEASE_BENCHMARK_OUTPUT}")
print(f"- {PITCHER_HANDS_OUTPUT}")
