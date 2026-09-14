"""
data_loader.py

Shared data-loading utilities for the PAOM dashboard. All pages
import from here rather than reading CSVs directly, so caching and
missing-file handling stay consistent across pages.

All files are expected to sit in the same folder as app.py (i.e.
your PAOM project folder where the pipeline scripts wrote their
outputs).
"""

import pandas as pd
import streamlit as st
import os


DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _path(filename):
    return os.path.join(DATA_DIR, filename)


def _safe_load(filename, required=True):
    """
    Loads a CSV if present. If missing and required=True, shows a
    Streamlit error and stops that page (rather than crashing with
    a raw traceback). If required=False, returns None silently so
    a page can render a reduced view instead of failing entirely.
    """
    path = _path(filename)
    if not os.path.exists(path):
        if required:
            st.error(
                f"Missing file: **{filename}**. Make sure you've "
                f"run the full PAOM pipeline and that this dashboard "
                f"is in the same folder as your pipeline outputs."
            )
            st.stop()
        return None
    return pd.read_csv(path)


@st.cache_data
def load_final_scores():
    return _safe_load("PAOM_final_scores.csv")


@st.cache_data
def load_final_scores_all_years():
    """
    Combines the real, per-year PAOM_final_scores_{YEAR}.csv files
    (2020-2025, from 83_paom_score_by_year.py's historical rebuild)
    into one dataframe with a 'season' column -- the real, validated
    historical scoring, as opposed to load_final_scores()'s single,
    undated file (which may or may not be the same underlying
    methodology; this loader is explicit about which year's real
    data backs every row, so there's no ambiguity about what's being
    shown). Missing years are skipped gracefully, not required.
    """
    frames = []
    for year in range(2020, 2026):
        df = _safe_load(f"PAOM_final_scores_{year}.csv", required=False)
        if df is not None:
            df = df.copy()
            df["season"] = year
            frames.append(df)

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


@st.cache_data
def load_effectiveness():
    return _safe_load("PAOM_effectiveness_component.csv")


@st.cache_data
def load_command():
    return _safe_load("PAOM_command_component.csv", required=False)


@st.cache_data
def load_platoon():
    return _safe_load("PAOM_platoon_component.csv", required=False)


@st.cache_data
def load_tunneling():
    return _safe_load("PAOM_tunneling_component.csv", required=False)


@st.cache_data
def load_movement_points():
    return _safe_load("PAOM_movement_map_points.csv", required=False)


@st.cache_data
def load_movement_scores():
    return _safe_load("PAOM_movement_component.csv", required=False)


@st.cache_data
def load_movement_hulls():
    return _safe_load("PAOM_movement_map_hulls.csv", required=False)


@st.cache_data
def load_location_points():
    return _safe_load("PAOM_location_map_points.csv", required=False)


@st.cache_data
def load_location_means():
    return _safe_load("PAOM_location_map_means.csv", required=False)


@st.cache_data
def load_release_points():
    return _safe_load("PAOM_release_map_points.csv", required=False)


@st.cache_data
def load_release_means():
    return _safe_load("PAOM_release_map_means.csv", required=False)


@st.cache_data
def load_league_movement_benchmark():
    return _safe_load("PAOM_league_movement_benchmark.csv", required=False)


@st.cache_data
def load_league_release_benchmark():
    return _safe_load("PAOM_league_release_benchmark.csv", required=False)


@st.cache_data
def load_pitcher_hands():
    return _safe_load("PAOM_pitcher_hands.csv", required=False)


@st.cache_data
def load_usage_recs():
    return _safe_load("PAOM_usage_recommendations.csv", required=False)


@st.cache_data
def load_drop_recs():
    return _safe_load("PAOM_drop_recommendations.csv", required=False)


@st.cache_data
def load_add_recs():
    return _safe_load("PAOM_add_recommendations.csv", required=False)


@st.cache_data
def load_historical_recommendations():
    """
    The historical-precedent engine's pre-computed, ranked
    recommendations (94_batch_precompute_historical_recommendations
    .py) -- replaces PAOM's own Usage/Drop/Add recommendation logic
    on the dashboard, per the validation work that found PAOM's
    recommendations didn't hold up against real historical outcomes.
    One row per (pitcher, rank), covering all four change types
    (ADD, DROP, USAGE_INCREASE, USAGE_DECREASE, plus a NO_CHANGE
    baseline row) in a single ranked list per pitcher, rather than
    three separate recommendation types.
    """
    return _safe_load("historical_recommendations.csv", required=False)


@st.cache_data
def load_historical_recommendation_comps():
    """
    Real historical comps behind each historical-engine
    recommendation (from the same batch script) -- the pitchers a
    given recommendation's prediction is actually based on, shown
    by default rather than hidden behind a click.
    """
    return _safe_load("historical_recommendation_comps.csv", required=False)


@st.cache_data
def load_similarity_features():
    """
    The historical-precedent engine's own raw similarity features
    (38_similarity_weighted_regression.py's real 7-feature space),
    loaded directly and lightweight -- NOT via the full 45/60 import
    chain, which also loads the entire training dataset and would be
    much heavier than a dashboard page needs just to compute a
    general similarity ranking. Real column names, matching 38's own
    SIMILARITY_FEATURE_COLS exactly: hull_area_z, avg_nn_distance_z,
    fastball_relative_break_z, velocity_range_z, max_velo_z,
    release_x_z, release_z_z.
    """
    return _safe_load("pitcher_similarity_features_2020_2025.csv", required=False)


@st.cache_data
def load_arsenal_usage():
    """
    Real per-pitch-type usage percentages (pitcher_arsenal_evolution
    _2020_2025.csv), loaded directly and lightweight -- used to
    determine each pitcher-season's real qualifying pitch types
    (>=5% usage, matching this project's MEANINGFUL_USAGE_FLOOR
    convention throughout), for the pitch-composition overlap
    (Jaccard) term in the general similarity ranking.
    """
    return _safe_load("pitcher_arsenal_evolution_2020_2025.csv", required=False)


@st.cache_data
def load_mechanical_similarity():
    return _safe_load("PAOM_pitcher_similarity.csv", required=False)


@st.cache_data
def load_mechanical_map():
    return _safe_load("PAOM_pitcher_map_coordinates.csv", required=False)


@st.cache_data
def load_mechanical_similarity_by_year(year):
    """
    Real, year-specific mechanical similarity (87_pitcher_similarity
    _by_year.py's PAOM_pitcher_similarity_{YEAR}.csv). Unlike the
    arsenal-shape map below, this pipeline WAS extended to all six
    years (2020-2025) and tested -- a real, year-aware alternative
    to the single, undated load_mechanical_similarity() above.
    """
    return _safe_load(f"PAOM_pitcher_similarity_{year}.csv", required=False)


@st.cache_data
def load_mechanical_map_by_year(year):
    """
    Real, year-specific mechanical map coordinates (87's PAOM_
    pitcher_map_coordinates_{YEAR}.csv). See load_mechanical_
    similarity_by_year's docstring.
    """
    return _safe_load(f"PAOM_pitcher_map_coordinates_{year}.csv", required=False)


@st.cache_data
def load_arsenal_similarity():
    return _safe_load("PAOM_arsenal_similarity.csv", required=False)


@st.cache_data
def load_arsenal_map():
    return _safe_load("PAOM_arsenal_map_coordinates.csv", required=False)


def all_pitcher_names():
    """
    Master list of pitchers for search/select widgets -- the UNION
    of names across every major component file, not just
    PAOM_final_scores.csv.

    Using only final_scores silently excludes any pitcher who
    doesn't clear ALL THREE of Effectiveness/Movement/Command's
    combined thresholds (paom_score's population, ~394) even though
    they may have real, viewable data in a broader population --
    Movement alone covers ~722 pitchers, for instance. That made
    such pitchers completely unselectable anywhere in the dashboard,
    despite pages like Pitcher Detail already having graceful
    handling for "no PAOM score available" that could never
    actually trigger, since the pitcher could never be chosen in
    the first place.

    Also includes load_final_scores_all_years() and
    load_historical_recommendations(), since the original five
    sources below are all single, undated (current-season-only)
    files -- a pitcher who appears in a real historical
    recommendation example, or who scored in an earlier real season
    but not the current one, was previously unselectable anywhere on
    the dashboard even though their name is cited directly in real
    recommendation output.
    """
    name_sets = []

    for loader in [
        load_final_scores,
        load_final_scores_all_years,
        load_movement_points,
        load_command,
        load_release_means,
        load_location_means,
        load_historical_recommendations,
    ]:
        df = loader()
        if df is not None and "player_name" in df.columns:
            name_sets.append(set(df["player_name"].unique()))

    if not name_sets:
        return []

    all_names = set().union(*name_sets)
    return sorted(all_names)
