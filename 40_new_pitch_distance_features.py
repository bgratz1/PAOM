"""
40_new_pitch_distance_features.py

Purpose:
--------
39_validate_pitch_type_sensitivity.py's Check 4 directly confirmed
that the regression in 38_similarity_weighted_regression.py is
mostly picking up mean-reversion (coefficient ~6x larger, far more
stable across pitch types) rather than a genuine pitch-specific
effect (small, sign-inconsistent 25% of the time). Likely cause:
usage_pct_delta only tells the model HOW MUCH of a pitch was added,
not WHAT KIND, relative to what the pitcher already throws. Adding
a slider nearly identical to an existing cutter is a very different
move than adding a genuinely new weapon -- the model currently can't
tell those apart.

This computes two new features for each ADD event, reusing the
SAME validated geometric concepts already established in
09_movement_component.py / 37_similarity_feature_space.py:

1. new_pitch_nn_distance_from_existing: distance from the newly
   added pitch to its NEAREST existing pitch in the arsenal (at
   season_to, when the new pitch actually shows up with real data).
   Small = redundant with something already thrown. Large = fills a
   genuine gap. Same nearest-neighbor concept as avg_nn_distance,
   just directed specifically at "distance from the new pitch to
   what was already there," not an arsenal-wide average.

2. new_pitch_fastball_relative_break: distance from the newly added
   pitch to the pitcher's own FF/SI anchor (same usage-based anchor
   selection as 09_movement_component.py / 37_similarity_feature_
   space.py). Answers a DIFFERENT question than #1 -- not "is this
   redundant with the rest of the arsenal" but "is this well-
   differentiated from what a hitter's timing is calibrated to by
   default."

SCOPE: ADD events only. DROP/USAGE_INCREASE/USAGE_DECREASE events
don't have a clean "new pitch" to measure -- a parallel "how
redundant was the DROPPED pitch with what remains" feature is a
reasonable future extension, not built here.

Output:
-------
arsenal_change_new_pitch_distance_features.csv -- one row per
(player_id, season_from, season_to, pitch_type) for ADD events only,
with the two new distance features. A SEPARATE supplementary file,
not a modification of arsenal_change_events_pitch_level.csv itself
-- 38_similarity_weighted_regression.py merges this in as an
additional predictor.
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

CHANGE_EVENTS_FILE = "arsenal_change_events_pitch_level.csv"
ARSENAL_FILE = "pitcher_arsenal_evolution_2020_2025.csv"

OUTPUT_FILE = "arsenal_change_new_pitch_distance_features.csv"

MEANINGFUL_USAGE_FLOOR = 5.0  # SAME threshold used throughout this
                                # project (09_movement_component.py's
                                # MIN_PITCH_TYPE_USAGE,
                                # 34_arsenal_change_events.py,
                                # 37_similarity_feature_space.py)

PITCH_TYPES = ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS"]

ANCHOR_PRIORITY = ["FF", "SI"]  # SAME as 09/37


# ============================================================
# LOAD
# ============================================================

print("Loading change events and arsenal data...")

change_events = pd.read_csv(CHANGE_EVENTS_FILE)
arsenal = pd.read_csv(ARSENAL_FILE)

add_events = change_events[change_events["change_type"] == "ADD"].copy()

print(f"Loaded {len(add_events):,} ADD events")


# ============================================================
# COMPUTE DISTANCE FEATURES PER ADD EVENT
# ============================================================

print("\nComputing new-pitch distance features...")

records = []

for _, event in add_events.iterrows():
    player_id = event["player_id"]
    season_to = event["season_to"]
    new_pitch_type = event["pitch_type"]

    # look up this pitcher's FULL arsenal at season_to (when the
    # new pitch actually shows up with real data)
    arsenal_row = arsenal[
        (arsenal["player_id"] == player_id) & (arsenal["season"] == season_to)
    ]

    if len(arsenal_row) == 0:
        continue  # shouldn't happen given how add_events was built,
                    # but skip defensively rather than crash

    arsenal_row = arsenal_row.iloc[0]

    new_pitch_x = arsenal_row.get(f"{new_pitch_type}_avg_pfx_x", np.nan)
    new_pitch_z = arsenal_row.get(f"{new_pitch_type}_avg_pfx_z", np.nan)

    if pd.isna(new_pitch_x) or pd.isna(new_pitch_z):
        continue  # new pitch's own movement data is missing -- skip

    # the REST of the arsenal at season_to, EXCLUDING the newly
    # added pitch type itself
    existing_types = []
    for pt in PITCH_TYPES:
        if pt == new_pitch_type:
            continue
        usage = arsenal_row.get(f"{pt}_usage_pct", np.nan)
        if pd.notna(usage) and usage >= MEANINGFUL_USAGE_FLOOR:
            existing_types.append(pt)

    # --- FEATURE 1: nearest-neighbor distance to the EXISTING arsenal ---
    if len(existing_types) > 0:
        existing_x = np.array([arsenal_row[f"{pt}_avg_pfx_x"] for pt in existing_types])
        existing_z = np.array([arsenal_row[f"{pt}_avg_pfx_z"] for pt in existing_types])

        valid = ~(np.isnan(existing_x) | np.isnan(existing_z))
        existing_x, existing_z = existing_x[valid], existing_z[valid]

        if len(existing_x) > 0:
            distances_to_existing = np.sqrt(
                (existing_x - new_pitch_x) ** 2 + (existing_z - new_pitch_z) ** 2
            )
            nn_distance_from_existing = distances_to_existing.min()
        else:
            nn_distance_from_existing = np.nan
    else:
        # the added pitch is the pitcher's ONLY qualifying pitch --
        # no existing arsenal to compare against
        nn_distance_from_existing = np.nan

    # --- FEATURE 2: distance from the FF/SI anchor ---
    # same usage-based anchor selection as 09/37, but the anchor
    # must come from the EXISTING arsenal (excluding the new pitch
    # itself, in case someone is literally adding a new fastball
    # variant that would otherwise BE the anchor)
    available_anchors = [pt for pt in ANCHOR_PRIORITY if pt in existing_types]

    if len(available_anchors) == 0:
        anchor_type = None
    elif len(available_anchors) == 1:
        anchor_type = available_anchors[0]
    else:
        ff_usage = arsenal_row.get("FF_usage_pct", 0)
        si_usage = arsenal_row.get("SI_usage_pct", 0)
        anchor_type = "FF" if ff_usage >= si_usage else "SI"

    if anchor_type is not None:
        anchor_x = arsenal_row.get(f"{anchor_type}_avg_pfx_x", np.nan)
        anchor_z = arsenal_row.get(f"{anchor_type}_avg_pfx_z", np.nan)
        if pd.notna(anchor_x) and pd.notna(anchor_z):
            new_pitch_fastball_relative_break = np.sqrt(
                (new_pitch_x - anchor_x) ** 2 + (new_pitch_z - anchor_z) ** 2
            )
        else:
            new_pitch_fastball_relative_break = np.nan
    else:
        # no FF/SI in the existing arsenal at all (e.g. the pitcher
        # is adding a sinker that becomes their first fastball-type
        # pitch) -- this feature doesn't apply cleanly here
        new_pitch_fastball_relative_break = np.nan

    records.append({
        "player_id": player_id,
        "season_from": event["season_from"],
        "season_to": season_to,
        "pitch_type": new_pitch_type,
        "n_existing_pitch_types": len(existing_types),
        "new_pitch_nn_distance_from_existing": nn_distance_from_existing,
        "anchor_type_used": anchor_type,
        "new_pitch_fastball_relative_break": new_pitch_fastball_relative_break,
    })

result_df = pd.DataFrame(records)

print(f"Computed distance features for {len(result_df):,} ADD events")


# ============================================================
# SAVE
# ============================================================

result_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n==============================")
print(f"Saved: {OUTPUT_FILE}")
print(f"==============================")

n_no_existing_comparison = result_df["new_pitch_nn_distance_from_existing"].isna().sum()
n_no_fastball_comparison = result_df["new_pitch_fastball_relative_break"].isna().sum()
print(
    f"\n{n_no_existing_comparison:,} events have no nearest-neighbor "
    f"comparison (added pitch was the pitcher's only qualifying pitch)"
)
print(
    f"{n_no_fastball_comparison:,} events have no fastball-relative "
    f"comparison (no FF/SI in the existing arsenal)"
)

print(f"\nDistribution of new_pitch_nn_distance_from_existing:")
print(result_df["new_pitch_nn_distance_from_existing"].describe())

print(f"\nSample rows -- most REDUNDANT additions (smallest nn_distance):")
print(
    result_df.nsmallest(10, "new_pitch_nn_distance_from_existing")
    [["player_id", "season_to", "pitch_type", "n_existing_pitch_types",
      "new_pitch_nn_distance_from_existing", "new_pitch_fastball_relative_break"]]
    .to_string(index=False)
)

print(f"\nSample rows -- most DIFFERENTIATED additions (largest nn_distance):")
print(
    result_df.nlargest(10, "new_pitch_nn_distance_from_existing")
    [["player_id", "season_to", "pitch_type", "n_existing_pitch_types",
      "new_pitch_nn_distance_from_existing", "new_pitch_fastball_relative_break"]]
    .to_string(index=False)
)
