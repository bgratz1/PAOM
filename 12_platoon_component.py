"""
12_platoon_component.py

Purpose:
--------
Score each pitcher x pitch type's effectiveness SEPARATELY against
same-handed and opposite-handed batters, on the same 0-100 scale
as the overall Effectiveness component -- so a specific platoon
weakness (e.g. "this changeup is average overall but well below
average specifically vs left-handed batters") can be identified
directly.

This is reference data for the recommendation engine, not a
paom_score component -- a platoon gap is a targeting detail (WHICH
batters an arsenal struggles against), not an overall quality
rating the way Effectiveness/Movement/Command are.

Method:
-------
1. Refit the SAME regression as 07d_final_effectiveness.py
   (csw_rate, chase_rate, hard_hit_rate, gb_rate -> xwoba) on the
   full pitcher x pitch type population, so there's one consistent
   model.

2. Compute the same four features separately for same-handed and
   opposite-handed batters (stand vs p_throws), then score each
   split through THAT SAME fitted model.

3. Scale using the OVERALL effectiveness score's min/max (from
   PAOM_effectiveness_component.csv) rather than re-deriving new
   min/max from the split data -- so a platoon score of 50 means
   the same thing as an overall effectiveness score of 50, and
   splits are directly comparable to the overall number.

4. Compute platoon_gap = opposite-side score minus same-side score,
   plus which side is the weaker one -- the actual recommendation-
   relevant signal.

Output:
-------
PAOM_platoon_component.csv
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "clean_statcast_2025.csv"
EFFECTIVENESS_FILE = "PAOM_effectiveness_component.csv"

OUTPUT_FILE = "PAOM_platoon_component.csv"

MIN_PITCHES = 75          # same overall pitch-type filter as 07d
MIN_SWINGS = 50
MIN_SPLIT_PITCHES = 30    # per platoon split, looser since it's a subset

FEATURES = ["csw_rate", "chase_rate", "hard_hit_rate", "gb_rate"]
TARGET = "xwoba"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading cleaned Statcast...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Loaded {len(df):,} pitches")

required_cols = [
    "player_name", "pitcher", "pitch_type", "p_throws", "stand",
    "is_swing", "is_whiff", "is_called_strike", "is_hard_hit",
    "is_chase", "is_outside_zone", "is_in_play", "launch_angle",
    "estimated_woba_using_speedangle"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[df["pitch_type"].notna()]
df = df[df["estimated_woba_using_speedangle"].notna()]

df["matchup_type"] = np.where(
    df["stand"] == df["p_throws"], "same", "opposite"
)


# ============================================================
# HELPER: BUILD RATE FEATURES FROM A GROUP OF PITCHES
# ============================================================

def build_rates(group_cols, min_pitches, min_swings=None):
    """
    Aggregates raw pitch-level flags into the same rate features
    used by 07d, grouped by whatever columns are passed in.
    """

    agg = (
        df
        .groupby(group_cols)
        .agg(
            pitches=("pitch_type", "count"),
            xwoba=("estimated_woba_using_speedangle", "mean"),
            swings=("is_swing", "sum"),
            whiffs=("is_whiff", "sum"),
            called_strikes=("is_called_strike", "sum"),
            hard_hits=("is_hard_hit", "sum"),
            chase_swings=("is_chase", "sum"),
            outside_zone=("is_outside_zone", "sum"),
            balls_in_play=("is_in_play", "sum"),
            ground_balls=("launch_angle", lambda x: (x < 10).sum())
        )
        .reset_index()
    )

    agg = agg[agg["pitches"] >= min_pitches]

    if min_swings is not None:
        agg = agg[agg["swings"] >= min_swings]

    agg["csw_rate"] = (
        (agg["whiffs"] + agg["called_strikes"]) / agg["pitches"]
    )
    agg["chase_rate"] = agg["chase_swings"] / agg["outside_zone"]
    agg["hard_hit_rate"] = agg["hard_hits"] / agg["pitches"]
    agg["gb_rate"] = agg["ground_balls"] / agg["balls_in_play"]

    agg = agg.replace([np.inf, -np.inf], np.nan)
    agg = agg.dropna(subset=FEATURES + [TARGET])

    return agg


# ============================================================
# STEP 1: REFIT THE SAME MODEL AS 07d, ON THE OVERALL POPULATION
# ============================================================

print("\nRefitting the effectiveness model (same method as 07d)...")

overall = build_rates(
    ["player_name", "pitcher", "pitch_type"],
    MIN_PITCHES,
    MIN_SWINGS
)

print(f"Pitcher x pitch type rows used to fit model: {len(overall):,}")

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 50)))
    ]
)

model.fit(overall[FEATURES], overall[TARGET])

print("Model refit complete.")


# ============================================================
# STEP 2: SCALING REFERENCE FROM THE OVERALL EFFECTIVENESS FILE
# ============================================================

"""
Anchor the 0-100 scale to the ALREADY-SAVED overall effectiveness
population's raw_effectiveness_index distribution, so a platoon-
split score of 50 means the same thing as an overall
trusted_effectiveness of 50 -- not a freshly re-derived scale from
just the split data.

Uses percentile-against-reference (not min-max): each platoon
split's raw_effectiveness_index gets the percentile it WOULD fall
at within the overall effectiveness population's distribution --
the same "50 = median, 90 = better than 90%" interpretation used
everywhere else in PAOM, just computed against an external
reference population instead of the split data's own population
(since the split population is a different, smaller set of rows
than the population the reference scale should represent).
"""

effectiveness_ref = pd.read_csv(EFFECTIVENESS_FILE)

ref_values = np.sort(effectiveness_ref["raw_effectiveness_index"].values)

print(
    f"\nScaling reference (from overall effectiveness): "
    f"{len(ref_values):,} pitch types, "
    f"min={ref_values.min():.4f}, max={ref_values.max():.4f}"
)


def scale_to_overall_range(raw_index):
    ranks = np.searchsorted(ref_values, raw_index, side="right")
    percentiles = ranks / len(ref_values) * 100
    return pd.Series(percentiles, index=raw_index.index).clip(0, 100)


# ============================================================
# STEP 3: SCORE EACH PLATOON SPLIT THROUGH THE SAME MODEL
# ============================================================

print("\nComputing platoon splits...")

splits = build_rates(
    ["player_name", "pitcher", "pitch_type", "matchup_type"],
    MIN_SPLIT_PITCHES
)

print(f"Pitcher x pitch type x matchup rows: {len(splits):,}")

splits["predicted_xwoba"] = model.predict(splits[FEATURES])
splits["raw_effectiveness_index"] = -splits["predicted_xwoba"]
splits["platoon_effectiveness_score"] = scale_to_overall_range(
    splits["raw_effectiveness_index"]
)


# ============================================================
# STEP 4: PIVOT TO ONE ROW PER PITCHER x PITCH TYPE
# ============================================================

pivot = splits.pivot_table(
    index=["player_name", "pitcher", "pitch_type"],
    columns="matchup_type",
    values=["platoon_effectiveness_score", "pitches"]
)

pivot.columns = ["_".join(col) for col in pivot.columns]
pivot = pivot.reset_index()

rename_map = {
    "platoon_effectiveness_score_same": "score_vs_same",
    "platoon_effectiveness_score_opposite": "score_vs_opposite",
    "pitches_same": "pitches_vs_same",
    "pitches_opposite": "pitches_vs_opposite"
}

pivot = pivot.rename(columns=rename_map)

for col in [
    "score_vs_same", "score_vs_opposite",
    "pitches_vs_same", "pitches_vs_opposite"
]:
    if col not in pivot.columns:
        pivot[col] = np.nan

# only keep rows where BOTH sides cleared the sample filter --
# a gap is only meaningful if both sides are separately trustworthy
pivot = pivot.dropna(subset=["score_vs_same", "score_vs_opposite"])


# ============================================================
# CONFIDENCE-BASED SHRINKAGE (matching Effectiveness/Command)
# ============================================================

"""
Platoon was the one component in the pipeline with NO protection
against small-sample noise -- MIN_SPLIT_PITCHES=30 was the only
gate, and a score could sit at an extreme value purely because it
just cleared that floor. A shrinkage-removal experiment on
Effectiveness/Command showed clearly that this kind of protection
does real, visible work (compressed/inflated leaderboard placements
for thin samples) -- Platoon needed the same fix.

CONFIDENCE_SATURATION_SPLIT_PITCHES=150 is a deliberate choice,
not derived from anything: roughly the same "floor-to-saturation"
ratio (5x) used elsewhere (Effectiveness: 75->500 pitches/swings,
~6.7x; Command: 50->300, 6x), scaled down to match a platoon
split's inherently smaller sample sizes. Each side (same/opposite)
gets its OWN confidence, based on its OWN pitch count -- a pitch
can be well-established against same-handed batters while still
being a thin, unreliable sample against opposite-handed ones.

Shrinkage target is a SINGLE pooled league_average across both
same- and opposite-handed scores together (not two separate means)
-- both sides are already on the same effectiveness-anchored 0-100
scale, so one consistent shrinkage target is more defensible than
inventing two.

platoon_gap is computed from the SHRUNK (trusted) scores, not the
raw ones -- the gap itself is exactly the number that was blowing
out to 80-90 points on thin samples before this fix, so it needs
the same protection as the scores it's derived from.
"""

CONFIDENCE_SATURATION_SPLIT_PITCHES = 150

pivot["confidence_vs_same"] = (
    np.minimum(pivot["pitches_vs_same"] / CONFIDENCE_SATURATION_SPLIT_PITCHES, 1) * 100
)
pivot["confidence_vs_opposite"] = (
    np.minimum(pivot["pitches_vs_opposite"] / CONFIDENCE_SATURATION_SPLIT_PITCHES, 1) * 100
)

league_average = pd.concat(
    [pivot["score_vs_same"], pivot["score_vs_opposite"]]
).mean()

pivot["trusted_score_vs_same"] = (
    pivot["score_vs_same"] * (pivot["confidence_vs_same"] / 100)
    + league_average * (1 - pivot["confidence_vs_same"] / 100)
)
pivot["trusted_score_vs_opposite"] = (
    pivot["score_vs_opposite"] * (pivot["confidence_vs_opposite"] / 100)
    + league_average * (1 - pivot["confidence_vs_opposite"] / 100)
)

pivot["platoon_gap"] = (
    pivot["trusted_score_vs_opposite"] - pivot["trusted_score_vs_same"]
)

pivot["weaker_side"] = np.where(
    pivot["platoon_gap"] > 0, "same", "opposite"
)

pivot["platoon_gap_abs"] = pivot["platoon_gap"].abs()


# ============================================================
# SAVE OUTPUT
# ============================================================

output = pivot[
    [
        "player_name", "pitcher", "pitch_type",
        "pitches_vs_same", "pitches_vs_opposite",
        "score_vs_same", "score_vs_opposite",
        "confidence_vs_same", "confidence_vs_opposite",
        "trusted_score_vs_same", "trusted_score_vs_opposite",
        "platoon_gap", "platoon_gap_abs", "weaker_side"
    ]
].sort_values("platoon_gap_abs", ascending=False)

output.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print("Platoon Component Saved")
print("==============================")

print(
    f"""
Pitcher x pitch type rows with both sides scored:
{len(output):,}

Average |platoon gap|:
{output['platoon_gap_abs'].mean():.2f}
"""
)

print("\nTop 20 largest platoon gaps")
print("-----------------------------")
print(output.head(20))

print("\nSaved:")
print(f"- {OUTPUT_FILE}")
