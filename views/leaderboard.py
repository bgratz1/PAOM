"""
views/leaderboard.py

PAOM Dashboard -- league-wide leaderboard (the default page,
registered in app.py).

UPDATED: now sources from the real, historically-rebuilt, per-year
PAOM_final_scores_{YEAR}.csv files (2020-2025, from 83_paom_score_
by_year.py) via load_final_scores_all_years(), rather than a single,
undated PAOM_final_scores.csv whose exact provenance/methodology
version could be ambiguous. Every row on this page is now explicitly
labeled with its real season, so there's no doubt about which
version of paom_score is being shown.

Also fixes a real, confirmed bug: Velocity was never included in
this page's own display column list at all (not a backend/data
issue -- just missing from display_cols), which is why it appeared
to be "missing" from the dashboard. Added here directly.

Year filtering: "every season shown separately" by default (the
explicit design decision made earlier in this project), with a
toggle to collapse to each pitcher's own single best season instead
-- both options directly reuse 82_paom_leaderboard.py's already-
built, already-tested get_leaderboard() logic (embedded here rather
than imported, matching this dashboard's convention of self-
contained page logic).
"""

import streamlit as st
import pandas as pd

from data_loader import load_final_scores_all_years, all_pitcher_names
from styling import render_welcome_dialog, render_pitcher_indicator


col_title, col_about = st.columns([5, 1])
with col_title:
    st.title("PAOM -- Pitch Arsenal Optimization Model")
with col_about:
    st.write("")  # small vertical nudge to align with the title
    if st.button("ℹ️ About This Dashboard"):
        render_welcome_dialog()

st.caption(
    "League-wide pitcher ratings combining pitch effectiveness, "
    "arsenal movement coverage, command, and velocity."
)
render_pitcher_indicator()


# ============================================================
# LEADERBOARD QUERY LOGIC (from 82_paom_leaderboard.py)
# ============================================================

def get_leaderboard(df, year=None, top_n=None, deduplicate_to_best_season=False, min_confidence=None):
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

    result = result.sort_values("paom_score", ascending=False)

    if top_n is not None:
        result = result.head(top_n)

    return result.reset_index(drop=True)


# ============================================================
# LOAD DATA
# ============================================================

scores = load_final_scores_all_years()

if scores is None:
    st.error(
        "No real, per-year PAOM_final_scores_{YEAR}.csv files found "
        "for any year (2020-2025). Run 83_paom_score_by_year.py / "
        "84_run_paom_score_all_years.py first."
    )
    st.stop()

available_years = sorted(scores["season"].unique())

# try to attach total pitch volume for workload context -- paom_score
# is a rate-based quality measure, not a season-value measure
try:
    master = pd.read_csv("master_pitch_table_2025.csv")
    volume = (
        master.groupby("player_name")["pitches"].sum()
        .reset_index(name="total_pitches")
    )
    scores = scores.merge(volume, on="player_name", how="left")
except FileNotFoundError:
    scores["total_pitches"] = None


# ============================================================
# YEAR / VIEW CONTROLS
# ============================================================

st.subheader("View options")

col_year, col_mode, col_conf = st.columns([1, 1.3, 1])

with col_year:
    year_choice = st.selectbox(
        "Season",
        options=["All years (every season shown separately)"] + [str(y) for y in available_years],
        index=0
    )

with col_mode:
    best_season_only = st.checkbox(
        "Show each pitcher's best season only",
        value=False,
        help=(
            "Off (default): the same pitcher can appear multiple "
            "times if they had strong seasons in different years. "
            "On: collapses to one row per pitcher, their single "
            "highest paom_score across whichever years are in view."
        )
    )

with col_conf:
    min_confidence = st.slider(
        "Minimum PAOM confidence",
        min_value=0, max_value=100, value=0, step=5,
        help=(
            "paom_confidence is the MINIMUM of all four components' "
            "own confidence scores -- a thin-data season in even one "
            "component pulls this down. Filters out low-confidence "
            "rows entirely."
        )
    )

selected_year = None if year_choice.startswith("All years") else int(year_choice)

leaderboard_data = get_leaderboard(
    scores,
    year=selected_year,
    top_n=None,
    deduplicate_to_best_season=best_season_only,
    min_confidence=min_confidence if min_confidence > 0 else None
)

st.caption(
    f"Showing real data from {'all years (' + ', '.join(str(y) for y in available_years) + ')' if selected_year is None else str(selected_year)} "
    f"-- {len(leaderboard_data):,} row(s)."
)

st.divider()


# ============================================================
# SUMMARY METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

col1.metric("Pitcher-Seasons Shown", f"{len(leaderboard_data):,}")
col2.metric(
    "Average PAOM Score",
    f"{leaderboard_data['paom_score'].mean():.1f}" if len(leaderboard_data) > 0 else "-"
)
col3.metric(
    "Score Range",
    f"{leaderboard_data['paom_score'].min():.0f} - {leaderboard_data['paom_score'].max():.0f}"
    if len(leaderboard_data) > 0 else "-"
)

st.divider()


# ============================================================
# JUMP TO A PITCHER
# ============================================================

st.subheader("Look up a pitcher")

selected = st.selectbox(
    "Search by name",
    options=all_pitcher_names(),
    index=None,
    placeholder="Start typing a name..."
)

if selected:
    if st.button(f"View {selected}", type="primary"):
        st.session_state["selected_pitcher"] = selected
        st.switch_page("views/pitcher_detail.py")

st.divider()


# ============================================================
# LEADERBOARD
# ============================================================

st.subheader("Leaderboard")

display_cols = [
    "player_name", "season", "paom_score", "pitcher_effectiveness",
    "trusted_movement_score", "pitcher_command", "trusted_velocity_score",
]

display_cols = [c for c in display_cols if c in leaderboard_data.columns]

rename_map = {
    "player_name": "Pitcher",
    "season": "Season",
    "paom_score": "PAOM Score",
    "pitcher_effectiveness": "Effectiveness",
    "trusted_movement_score": "Movement",
    "pitcher_command": "Command",
    "trusted_velocity_score": "Velocity",
}

leaderboard = (
    leaderboard_data[display_cols]
    .rename(columns=rename_map)
    .reset_index(drop=True)
)

leaderboard.index = leaderboard.index + 1

st.caption(
    "Click a column header to sort. PAOM Score is a rate-based "
    "arsenal-quality rating, not a season-value metric. When "
    "\"every season shown separately\" is on, the same pitcher may "
    "appear more than once. Hover a column header for what it "
    "measures. Confidence and workload (Total Pitches) for this "
    "score are shown on the Pitcher Detail page."
)

st.dataframe(
    leaderboard.style.format({
        "PAOM Score": "{:.1f}",
        "Effectiveness": "{:.1f}",
        "Movement": "{:.1f}",
        "Command": "{:.1f}",
        "Velocity": "{:.1f}",
    }),
    width='stretch',
    height=600,
    column_config={
        "PAOM Score": st.column_config.NumberColumn(
            "PAOM Score",
            help="Overall arsenal-quality rating, 0-100, higher is better."
        ),
        "Effectiveness": st.column_config.NumberColumn(
            "Effectiveness",
            help="Pitch-quality component, 0-100, higher is better."
        ),
        "Movement": st.column_config.NumberColumn(
            "Movement",
            help="Arsenal movement-coverage component, 0-100, higher is better."
        ),
        "Command": st.column_config.NumberColumn(
            "Command",
            help="Location-consistency component, 0-100, higher is better."
        ),
        "Velocity": st.column_config.NumberColumn(
            "Velocity",
            help="Velocity component, 0-100, higher is better."
        ),
    }
)
