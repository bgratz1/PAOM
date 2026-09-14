"""
82_paom_leaderboard.py

Purpose:
--------
Year-filterable PAOM leaderboard, "every season shown separately"
(per direct decision) -- the same pitcher can appear multiple times
if they had strong seasons in different years. Built and tested now
against synthetic multi-year data so the query logic is ready the
moment real historical paom_score_{YEAR}.csv files exist (Phase 2
of the roadmap -- requires 10_paom_score.py extended per-year first,
which hasn't been shared/rebuilt yet).

DESIGN NOTE for the eventual switch-to-best-season-only comparison
mentioned as a next step: get_leaderboard() below has a
DEDUPLICATE_TO_BEST_SEASON toggle already wired in, off by default
-- flipping it on collapses to one row per pitcher (their single
best season) without changing any other logic, so comparing both
views side by side later is a one-line change, not a rebuild.

Expects, once real data exists, a combined multi-year paom_score
file (one row per player_name + season) with at minimum:
player_name, season, paom_score, paom_confidence. REAL data
(83_paom_score_by_year.py) is keyed by player_name only -- no
player_id column exists in the real files, confirmed directly
rather than assumed.

Output:
-------
get_leaderboard() -- a reusable, callable function. Demonstrated
below on synthetic data.
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

COMBINED_PAOM_FILE = "paom_score_all_years.csv"  # expected once
                                                    # Phase 2 exists


# ============================================================
# LEADERBOARD QUERY
# ============================================================

def get_leaderboard(
    df,
    year=None,
    top_n=25,
    deduplicate_to_best_season=False,
    min_confidence=None,
):
    """
    year: None = all years combined (every season shown separately,
          the current decision); an int = filter to just that season.
    deduplicate_to_best_season: if True, collapses to one row per
          player_name (their single highest paom_score across all
          years) -- the alternate view mentioned as worth comparing
          once real data exists. Off by default, matching the
          current decision.
    min_confidence: optional floor on paom_confidence -- worth
          having available given paom_confidence = min() of all four
          components' own confidence scores (a thin-data pitcher in
          any ONE component drags the whole row down), so a real,
          low-confidence season could otherwise crowd out a
          genuinely well-supported one on a small-sample leaderboard.
    """
    result = df.copy()

    if year is not None:
        result = result[result["season"] == year]

    if min_confidence is not None:
        result = result[result["paom_confidence"] >= min_confidence]

    if deduplicate_to_best_season:
        result = (
            result
            .sort_values("paom_score", ascending=False)
            .drop_duplicates(subset=["player_name"], keep="first")
        )

    result = result.sort_values("paom_score", ascending=False).head(top_n)

    return result.reset_index(drop=True)


# ============================================================
# DEMONSTRATION -- synthetic multi-year data, confirms the
# mechanics work correctly ahead of real data being available
# ============================================================

if __name__ == "__main__":
    print("Building synthetic multi-year paom_score data for testing...")

    np.random.seed(11)
    rows = []
    for season in range(2020, 2026):
        for pid in range(600000, 600030):
            rows.append({
                "player_id": pid,
                "player_name": f"pitcher_{pid}",
                "season": season,
                "paom_score": np.random.normal(50, 15),
                "paom_confidence": np.random.uniform(40, 100),
            })
    synthetic_df = pd.DataFrame(rows)

    print(f"Built {len(synthetic_df):,} synthetic pitcher-seasons across "
          f"{synthetic_df['season'].nunique()} years\n")

    print("=" * 60)
    print("All years combined, every season shown separately, top 10")
    print("=" * 60)
    print(get_leaderboard(synthetic_df, year=None, top_n=10).to_string(index=False))

    print("\n" + "=" * 60)
    print("Filtered to a single year (2023), top 10")
    print("=" * 60)
    print(get_leaderboard(synthetic_df, year=2023, top_n=10).to_string(index=False))

    print("\n" + "=" * 60)
    print("Deduplicated to best season per pitcher (the alternate view), top 10")
    print("=" * 60)
    print(get_leaderboard(synthetic_df, year=None, top_n=10, deduplicate_to_best_season=True).to_string(index=False))

    print("\n" + "=" * 60)
    print("With a confidence floor applied (min_confidence=70), top 10")
    print("=" * 60)
    print(get_leaderboard(synthetic_df, year=None, top_n=10, min_confidence=70).to_string(index=False))
