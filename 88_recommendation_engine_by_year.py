"""
88_recommendation_engine_by_year.py

Purpose:
--------
The REAL 16_recommendation_engine.py, parameterized by YEAR -- all
Usage/Drop/Add logic UNCHANGED from the original.

Requires PAOM_effectiveness_component_{YEAR}.csv (76), pitch_
relationships_{YEAR}.csv (86), master_pitch_table_{YEAR}.csv (77),
PAOM_pitcher_similarity_{YEAR}.csv (87). Command, platoon, and
tunnel-pair files are all optional in the original (try/except) --
no year-parameterized platoon or tunnel-pair pipeline exists yet,
so those stay gracefully skipped, exactly matching the original's
own handling of a missing optional file.

Output:
-------
PAOM_usage_recommendations_{YEAR}.csv
PAOM_drop_recommendations_{YEAR}.csv
PAOM_add_recommendations_{YEAR}.csv
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

import sys
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

EFFECTIVENESS_FILE = f"PAOM_effectiveness_component_{YEAR}.csv"
RELATIONSHIPS_FILE = f"pitch_relationships_{YEAR}.csv"
TUNNEL_PAIRS_FILE = f"PAOM_pair_deception_scores_{YEAR}.csv"  # optional, likely absent
MOVEMENT_FILE = f"PAOM_movement_component_{YEAR}.csv"  # defined but unused in the
                                                          # original too -- kept for
                                                          # parity, harmless
SIMILARITY_FILE = f"PAOM_pitcher_similarity_{YEAR}.csv"
MASTER_FILE = f"master_pitch_table_{YEAR}.csv"
COMMAND_FILE = f"PAOM_command_component_{YEAR}.csv"
PLATOON_FILE = f"PAOM_platoon_component_{YEAR}.csv"  # optional, likely absent

USAGE_OUTPUT = f"PAOM_usage_recommendations_{YEAR}.csv"
DROP_OUTPUT = f"PAOM_drop_recommendations_{YEAR}.csv"
ADD_OUTPUT = f"PAOM_add_recommendations_{YEAR}.csv"

USAGE_GAP_THRESHOLD = 0.30
DROP_EFFECTIVENESS_PCTL = 0.35
DROP_REDUNDANCY_PCTL = 0.35

N_COMPS_FOR_ADD = 10
MIN_ARSENAL_PITCHES = 50
MIN_ARSENAL_USAGE = 0.05
TOP_ADD_CANDIDATES = 3


# ============================================================
# LOAD SHARED DATA
# ============================================================

print(f"Loading component files for {YEAR}...")

effectiveness = pd.read_csv(EFFECTIVENESS_FILE)
relationships = pd.read_csv(RELATIONSHIPS_FILE)
master = pd.read_csv(MASTER_FILE)

print(f"Effectiveness rows: {len(effectiveness):,}")
print(f"Pairwise relationships: {len(relationships):,}")
print(f"Master pitch table rows: {len(master):,}")

try:
    command = pd.read_csv(COMMAND_FILE)
except FileNotFoundError:
    command = None

try:
    platoon = pd.read_csv(PLATOON_FILE)
except FileNotFoundError:
    platoon = None


# ============================================================
# WITHIN-PITCHER PERCENTILES (shared by Usage and Drop)
# ============================================================

eff = effectiveness.copy()

eff["n_pitch_types"] = eff.groupby("player_name")["pitch_type"].transform("count")

eff["effectiveness_pctl"] = (
    eff.groupby("player_name")["trusted_effectiveness"]
    .rank(pct=True)
)

eff["usage_pctl"] = (
    eff.groupby("player_name")["usage"]
    .rank(pct=True)
)


# ============================================================
# 1. USAGE RECOMMENDATIONS
# ============================================================

print("\n==============================")
print("Building Usage Recommendations")
print("==============================")

usage_df = eff[eff["n_pitch_types"] >= 2].copy()

usage_df["pctl_gap"] = usage_df["effectiveness_pctl"] - usage_df["usage_pctl"]

def usage_flag(gap):
    if gap >= USAGE_GAP_THRESHOLD:
        return "increase usage"
    elif gap <= -USAGE_GAP_THRESHOLD:
        return "decrease usage"
    else:
        return "maintain"

usage_df["recommendation"] = usage_df["pctl_gap"].apply(usage_flag)

if command is not None:
    usage_df = usage_df.merge(
        command[["player_name", "pitch_type", "trusted_command_score"]],
        on=["player_name", "pitch_type"],
        how="left"
    )

if platoon is not None:
    usage_df = usage_df.merge(
        platoon[["player_name", "pitch_type", "platoon_gap_abs", "weaker_side"]],
        on=["player_name", "pitch_type"],
        how="left"
    )

usage_output = usage_df[
    [
        "player_name", "pitch_type", "usage",
        "trusted_effectiveness", "effectiveness_pctl", "usage_pctl",
        "pctl_gap", "recommendation"
    ]
    + (["trusted_command_score"] if command is not None else [])
    + (["platoon_gap_abs", "weaker_side"] if platoon is not None else [])
].sort_values("pctl_gap", ascending=False)

usage_output["season"] = YEAR
usage_output.to_csv(USAGE_OUTPUT, index=False)

print(f"Saved {len(usage_output):,} usage recommendation rows")
print(usage_output["recommendation"].value_counts())


# ============================================================
# 2. DROP RECOMMENDATIONS
# ============================================================

print("\n==============================")
print("Building Drop Recommendations")
print("==============================")

drop_pool = eff[eff["n_pitch_types"] >= 3].copy()

nearest_records = []

for (pitcher, pitch), _ in drop_pool.groupby(["player_name", "pitch_type"]):
    own_pairs = relationships[
        (relationships["player_name"] == pitcher)
        & (
            (relationships["pitch_1"] == pitch)
            | (relationships["pitch_2"] == pitch)
        )
    ]
    if len(own_pairs) == 0:
        continue
    nearest_records.append({
        "player_name": pitcher,
        "pitch_type": pitch,
        "nearest_teammate_distance": own_pairs["movement_distance"].min()
    })

nearest_df = pd.DataFrame(nearest_records)

drop_pool = drop_pool.merge(
    nearest_df, on=["player_name", "pitch_type"], how="left"
)

drop_pool = drop_pool.dropna(subset=["nearest_teammate_distance"])

drop_pool["redundancy_pctl"] = (
    drop_pool.groupby("player_name")["nearest_teammate_distance"]
    .rank(pct=True)
)

try:
    tunnel_pairs = pd.read_csv(TUNNEL_PAIRS_FILE)

    tunnel_records = []
    for (pitcher, pitch), _ in drop_pool.groupby(["player_name", "pitch_type"]):
        own_pairs = tunnel_pairs[
            (tunnel_pairs["player_name"] == pitcher)
            & (
                (tunnel_pairs["pitch_1"] == pitch)
                | (tunnel_pairs["pitch_2"] == pitch)
            )
        ]
        if len(own_pairs) == 0 or "tunnel_differential" not in own_pairs.columns:
            continue
        tunnel_records.append({
            "player_name": pitcher,
            "pitch_type": pitch,
            "avg_tunnel_contribution": own_pairs["tunnel_differential"].mean()
        })

    tunnel_df = pd.DataFrame(tunnel_records)
    drop_pool = drop_pool.merge(
        tunnel_df, on=["player_name", "pitch_type"], how="left"
    )
except (FileNotFoundError, KeyError):
    drop_pool["avg_tunnel_contribution"] = np.nan


drop_pool["drop_candidate"] = (
    (drop_pool["effectiveness_pctl"] <= DROP_EFFECTIVENESS_PCTL)
    & (drop_pool["redundancy_pctl"] <= DROP_REDUNDANCY_PCTL)
)

drop_output = drop_pool[
    [
        "player_name", "pitch_type", "usage",
        "trusted_effectiveness", "effectiveness_pctl",
        "nearest_teammate_distance", "redundancy_pctl",
        "avg_tunnel_contribution", "drop_candidate"
    ]
].sort_values(
    ["drop_candidate", "effectiveness_pctl"], ascending=[False, True]
)

drop_output["season"] = YEAR
drop_output.to_csv(DROP_OUTPUT, index=False)

print(f"Saved {len(drop_output):,} drop-evaluated rows")
print(f"Flagged as drop candidates: {drop_output['drop_candidate'].sum():,}")


# ============================================================
# 3. ADD RECOMMENDATIONS
# ============================================================

print("\n==============================")
print("Building Add Recommendations")
print("==============================")

similarity = pd.read_csv(SIMILARITY_FILE)

arsenal = master[
    (master["pitches"] >= MIN_ARSENAL_PITCHES)
    & (master["usage"] >= MIN_ARSENAL_USAGE)
][["player_name", "pitch_type", "HB", "IVB", "velo"]].copy()


def bbox_area(points):
    hb_range = points[:, 0].max() - points[:, 0].min()
    ivb_range = points[:, 1].max() - points[:, 1].min()
    return hb_range * ivb_range


add_records = []

pitchers_with_comps = similarity["player_name"].unique()

for pitcher in pitchers_with_comps:

    own_pitches = arsenal[arsenal["player_name"] == pitcher]

    if len(own_pitches) == 0:
        continue

    own_pitch_types = set(own_pitches["pitch_type"])
    own_points = own_pitches[["HB", "IVB"]].values
    own_coverage_area = bbox_area(own_points)

    comps = (
        similarity[similarity["player_name"] == pitcher]
        .sort_values("rank")
        .head(N_COMPS_FOR_ADD)
    )

    comp_names = comps["comp_player_name"].tolist()

    comp_pitches = arsenal[
        arsenal["player_name"].isin(comp_names)
        & ~arsenal["pitch_type"].isin(own_pitch_types)
    ]

    if len(comp_pitches) == 0:
        continue

    for candidate_type, group in comp_pitches.groupby("pitch_type"):

        candidate_hb = group["HB"].mean()
        candidate_ivb = group["IVB"].mean()
        candidate_velo = group["velo"].mean()
        n_comps_throwing = group["player_name"].nunique()

        distances_to_own = np.sqrt(
            (own_points[:, 0] - candidate_hb) ** 2
            + (own_points[:, 1] - candidate_ivb) ** 2
        )
        nearest_own_distance = distances_to_own.min()

        new_points = np.vstack([
            own_points, [[candidate_hb, candidate_ivb]]
        ])
        new_coverage_area = bbox_area(new_points)
        coverage_gain = new_coverage_area - own_coverage_area

        candidate_eff = effectiveness[
            (effectiveness["player_name"].isin(group["player_name"]))
            & (effectiveness["pitch_type"] == candidate_type)
        ]["trusted_effectiveness"].mean()

        add_records.append({
            "player_name": pitcher,
            "candidate_pitch_type": candidate_type,
            "n_comps_throwing": n_comps_throwing,
            "candidate_avg_velo": candidate_velo,
            "candidate_avg_effectiveness": candidate_eff,
            "nearest_own_pitch_distance": nearest_own_distance,
            "coverage_gain": coverage_gain
        })

add_df = pd.DataFrame(add_records)

print(f"Candidate pitch evaluations: {len(add_df):,}")

if len(add_df) == 0:
    print(
        "\nNo ADD candidates found for any pitcher this year -- saving "
        "an empty (but correctly-columned) output file rather than "
        "crashing on a fully-empty dataframe."
    )
    add_output = pd.DataFrame(columns=[
        "player_name", "candidate_pitch_type", "n_comps_throwing",
        "candidate_avg_velo", "candidate_avg_effectiveness",
        "nearest_own_pitch_distance", "coverage_gain",
        "effectiveness_scaled", "coverage_gain_scaled", "add_score",
        "rank_within_pitcher", "season"
    ])
    add_output.to_csv(ADD_OUTPUT, index=False)
    print(f"Saved empty ADD output for {YEAR}")
else:
    add_df = add_df.dropna(subset=["candidate_avg_effectiveness"])


    def minmax_within_group(s):
        if s.max() == s.min():
            return pd.Series(50.0, index=s.index)
        return 100 * (s - s.min()) / (s.max() - s.min())

    add_df["effectiveness_scaled"] = add_df.groupby("player_name")[
        "candidate_avg_effectiveness"
    ].transform(minmax_within_group)

    add_df["coverage_gain_scaled"] = add_df.groupby("player_name")[
        "coverage_gain"
    ].transform(minmax_within_group)

    add_df["add_score"] = (
        0.6 * add_df["effectiveness_scaled"]
        + 0.4 * add_df["coverage_gain_scaled"]
    )

    add_df["rank_within_pitcher"] = (
        add_df.groupby("player_name")["add_score"]
        .rank(ascending=False, method="first")
    )

    add_output = (
        add_df[add_df["rank_within_pitcher"] <= TOP_ADD_CANDIDATES]
        .sort_values(["player_name", "rank_within_pitcher"])
    )

    add_output["season"] = YEAR
    add_output.to_csv(ADD_OUTPUT, index=False)

    print(f"Saved top-{TOP_ADD_CANDIDATES} add recommendations for "
          f"{add_output['player_name'].nunique():,} pitchers")


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print(f"Recommendation Engine Complete ({YEAR})")
print("==============================")

print("\nSaved:")
print(f"- {USAGE_OUTPUT}")
print(f"- {DROP_OUTPUT}")
print(f"- {ADD_OUTPUT}")
