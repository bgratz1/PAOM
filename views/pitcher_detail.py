"""
pages/1_Pitcher_Detail.py

PAOM Dashboard -- single-pitcher deep dive: component scores,
movement map, recommendations (usage/drop/add), platoon splits,
and similar pitchers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_loader import (
    load_final_scores, load_movement_points, load_movement_hulls,
    load_movement_scores, load_effectiveness,
    load_location_points, load_location_means,
    load_release_points, load_release_means,
    load_league_movement_benchmark, load_league_release_benchmark,
    load_pitcher_hands,
    load_historical_recommendations, load_historical_recommendation_comps,
    load_platoon, load_command,
    load_mechanical_similarity, load_arsenal_similarity,
    load_arsenal_usage, load_final_scores_all_years,
    all_pitcher_names
)
from styling import get_pitch_type_color, format_pitch_type, render_pitch_type_key, format_player_name, apply_chart_theme, render_handedness_badge


# ============================================================
# PITCHER SELECTION
# ============================================================

names = all_pitcher_names()

default_pitcher = st.session_state.get("selected_pitcher", None)
default_index = (
    names.index(default_pitcher) if default_pitcher in names else 0
)

pitcher = st.selectbox(
    "Pitcher", options=names, index=default_index
)

st.session_state["selected_pitcher"] = pitcher

st.title(pitcher)

_pitcher_hands = load_pitcher_hands()
if _pitcher_hands is not None:
    _hand_row = _pitcher_hands[_pitcher_hands["player_name"] == pitcher]
    if len(_hand_row) > 0:
        render_handedness_badge(_hand_row.iloc[0]["p_throws"])


# ============================================================
# COMPONENT SCORES
# ============================================================

"""
Each component score is looked up on its own, independently of
whether this pitcher qualifies for a full PAOM Score, which
requires clearing every component's threshold at once. A pitcher
who qualifies for some components but not all still shows real,
viewable scores for the ones they clear -- with a dash for
whichever is missing, rather than hiding every score just because
one wasn't available.

Effectiveness and Command are originally computed per pitch type,
so they're rolled up here into a single, usage-weighted score per
pitcher, using the same approach applied throughout the dashboard
-- so these numbers match what would go into a full PAOM Score if
this pitcher qualified for one.
"""

def rollup_effectiveness(pitcher_name):
    eff = load_effectiveness()
    if eff is None:
        return None
    p_eff = eff[eff["player_name"] == pitcher_name]
    if len(p_eff) == 0:
        return None
    return np.average(p_eff["trusted_effectiveness"], weights=p_eff["usage"])


def rollup_command(pitcher_name):
    cmd = load_command()
    if cmd is None:
        return None
    p_cmd = cmd[cmd["player_name"] == pitcher_name]
    if len(p_cmd) == 0:
        return None
    weights = p_cmd["pitches"] / p_cmd["pitches"].sum()
    return np.average(p_cmd["trusted_command_score"], weights=weights)


def lookup_movement(pitcher_name):
    mv = load_movement_scores()
    if mv is None:
        return None
    p_mv = mv[mv["player_name"] == pitcher_name]
    if len(p_mv) == 0:
        return None
    return p_mv.iloc[0]["trusted_movement_score"]


scores = load_final_scores()
final_row = scores[scores["player_name"] == pitcher]

paom_value = (
    final_row.iloc[0]["paom_score"] if len(final_row) > 0 else None
)
effectiveness_value = rollup_effectiveness(pitcher)
movement_value = lookup_movement(pitcher)
command_value = rollup_command(pitcher)


def format_metric(value):
    return f"{value:.1f}" if value is not None else "-"


col1, col2, col3, col4 = st.columns(4)
col1.metric("PAOM Score", format_metric(paom_value))
col2.metric("Effectiveness", format_metric(effectiveness_value))
col3.metric("Movement", format_metric(movement_value))
col4.metric("Command", format_metric(command_value))

# Confidence and Total Pitches -- real context for the score above,
# but neither is a direct input to the paom_score formula itself.
# Moved here from the Leaderboard, which was showing too many
# columns without room to properly explain either one.
total_pitches_value = None
try:
    master = pd.read_csv("master_pitch_table_2025.csv")
    p_master = master[master["player_name"] == pitcher]
    if len(p_master) > 0:
        total_pitches_value = p_master["pitches"].sum()
except FileNotFoundError:
    pass

confidence_value = None
weakest_component = None
if paom_value is not None and "paom_confidence" in final_row.columns:
    r = final_row.iloc[0]
    confidence_value = r["paom_confidence"]
    component_confidences = {
        "Effectiveness": r.get("effectiveness_confidence"),
        "Movement": r.get("movement_confidence"),
        "Command": r.get("command_confidence")
    }
    weakest_component = min(component_confidences, key=lambda k: component_confidences[k])

col_conf, col_pitches = st.columns(2)
col_conf.metric("Confidence", format_metric(confidence_value))
col_pitches.metric(
    "Total Pitches (2025)",
    f"{total_pitches_value:,.0f}" if total_pitches_value is not None else "-"
)

with st.expander("What are Confidence and Total Pitches?"):
    st.markdown(
        f"""
**Neither of these is a direct input to the PAOM Score formula above** --
PAOM Score is a weighted combination of only the four real components
(Effectiveness, Movement, Command, Velocity). These two are shown
here as context for interpreting that score, not as ingredients in it.

**Confidence** -- how much real data backs this pitcher's PAOM Score,
on a 0-100 scale. It's calculated as the **minimum** of the four
components' own individual confidence values (each based on real
sample size) -- the overall score is only as trustworthy as its
single weakest input, not an average across all four.
{f"For this pitcher, that weakest link is **{weakest_component}**." if weakest_component else ""}
A low confidence number means this score is leaning more on
league-average shrinkage than on this pitcher's own real data --
it's a *reliability* indicator, not a quality measure, and a low
score here doesn't mean the pitcher is bad, just that there's less
real data behind the number.

**Total Pitches (2025)** -- the pitcher's real, total pitch count
for the season, summed directly from real pitch-level data. PAOM
Score is a **rate-based** quality rating (how good is this arsenal,
per pitch), not a season-value metric -- it says nothing about
workload. A reliever who threw 200 total pitches can have a higher
PAOM Score than a 3,000-pitch starter; Total Pitches is shown here
so that distinction doesn't get lost.
        """
    )

if paom_value is None:
    missing = [
        name for name, val in [
            ("Effectiveness", effectiveness_value),
            ("Movement", movement_value),
            ("Command", command_value)
        ] if val is None
    ]
    if missing:
        st.caption(
            f"No overall PAOM score -- missing: {', '.join(missing)} "
            f"(didn't clear that component's minimum sample size)."
        )
    else:
        st.caption(
            "No overall PAOM score, even though all three components "
            "are present -- this pitcher may not have cleared the "
            "combined population's sample requirements. Re-run "
            "10_paom_score.py if this looks wrong."
        )

st.divider()


# ============================================================
# MOVEMENT MAP + LOCATION MAP
# ============================================================

col_movement, col_location = st.columns(2)

with col_movement:
    st.subheader("Movement Map")

    points = load_movement_points()
    hulls = load_movement_hulls()

    if points is not None:
        p_points = points[points["player_name"] == pitcher]

        if len(p_points) == 0:
            st.info("No movement map data available for this pitcher.")
        else:
            fig = go.Figure()

            # hull/bbox boundary, drawn first so points sit on top
            if hulls is not None:
                p_hull = hulls[hulls["player_name"] == pitcher].sort_values(
                    "vertex_order"
                )
                if len(p_hull) > 0:
                    hb_vals = p_hull["HB"].tolist()
                    ivb_vals = p_hull["IVB"].tolist()
                    # close the polygon
                    hb_vals.append(hb_vals[0])
                    ivb_vals.append(ivb_vals[0])
                    fig.add_trace(go.Scatter(
                        x=hb_vals, y=ivb_vals,
                        mode="lines",
                        fill="toself",
                        fillcolor="rgba(99,110,250,0.15)",
                        line=dict(color="rgba(99,110,250,0.6)"),
                        name="Coverage boundary",
                        hoverinfo="skip"
                    ))

            fig.add_trace(go.Scatter(
                x=p_points["HB"], y=p_points["IVB"],
                mode="markers+text",
                text=p_points["pitch_type"],
                textposition="top center",
                marker=dict(
                    size=p_points["usage"] * 80 + 10,
                    color=p_points["velo"],
                    colorscale="Bluered",
                    showscale=True,
                    colorbar=dict(title="Velo")
                ),
                name="Pitches",
                customdata=p_points["pitch_type"].apply(format_pitch_type),
                hovertemplate=(
                    "%{customdata}<br>HB: %{x:.1f}<br>IVB: %{y:.1f}<extra></extra>"
                )
            ))

            # same-handed league-average benchmark, one marker per
            # pitch type this pitcher actually throws. Marker SHAPE
            # (not color) distinguishes it from the pitcher's own
            # points, since color here already encodes velocity --
            # an open diamond keeps the benchmark visually distinct
            # without competing with that colorscale.
            movement_benchmark = load_league_movement_benchmark()
            pitcher_hands = load_pitcher_hands()

            if movement_benchmark is not None and pitcher_hands is not None:
                hand_row = pitcher_hands[pitcher_hands["player_name"] == pitcher]
                if len(hand_row) > 0:
                    p_hand = hand_row.iloc[0]["p_throws"]
                    own_pitch_types = p_points["pitch_type"].unique()
                    bench = movement_benchmark[
                        (movement_benchmark["p_throws"] == p_hand)
                        & (movement_benchmark["pitch_type"].isin(own_pitch_types))
                    ]
                    if len(bench) > 0:
                        fig.add_trace(go.Scatter(
                            x=bench["avg_HB"], y=bench["avg_IVB"],
                            mode="markers",
                            marker=dict(
                                size=12, symbol="diamond-open",
                                color="#E8EAED", line=dict(width=2)
                            ),
                            name=f"League avg ({p_hand}HP)",
                            text=bench["pitch_type"],
                            customdata=bench["pitch_type"].apply(format_pitch_type),
                            hovertemplate=(
                                f"League avg %{{customdata}} ({p_hand}HP)<br>"
                                "HB: %{x:.1f}<br>IVB: %{y:.1f}"
                                "<extra></extra>"
                            )
                        ))

            fig.update_layout(
                xaxis_title="Horizontal Break (in)",
                yaxis_title="Induced Vertical Break (in)",
                height=550,
                legend=dict(
                    orientation="h",
                    yanchor="top", y=-0.15,
                    xanchor="left", x=0
                )
            )

            apply_chart_theme(fig, zeroline=True)
            # extra bottom margin for the legend, now positioned
            # below the chart -- apply_chart_theme's own default
            # margin isn't tall enough to fit it without overlap
            fig.update_layout(margin=dict(b=90))
            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Bubble size = usage share. Color = average velocity. "
                "Shaded region = movement-space coverage (bounding box "
                "for 2-pitch arsenals, true hull otherwise). Open black "
                "diamonds = league-average shape for that pitch type, "
                "among pitchers who throw with the same hand."
            )
    else:
        st.info("Movement map data not found.")


with col_location:
    st.subheader("Location Map")

    loc_means = load_location_means()
    loc_points = load_location_points()

    if loc_means is not None:
        p_means = loc_means[loc_means["player_name"] == pitcher]

        if len(p_means) == 0:
            st.info("No location map data available for this pitcher.")
        else:
            matchup_options = ["All"] + sorted(
                p_means["matchup_type"].unique().tolist()
            )
            matchup_filter = st.radio(
                "vs. batters",
                options=matchup_options,
                horizontal=True,
                key="location_matchup_filter",
                help=(
                    "'same' = batters that share this pitcher's "
                    "throwing hand. 'opposite' = the reverse."
                )
            )

            show_raw = st.checkbox(
                "Show individual pitches (detail view)",
                value=False,
                key="location_show_raw"
            )

            if matchup_filter != "All":
                p_means_filtered = p_means[
                    p_means["matchup_type"] == matchup_filter
                ].copy()
            else:
                # "All" collapses same/opposite back into one
                # pitch-count-weighted average per pitch type, so
                # the default view is exactly one point per pitch
                # type -- not two overlapping points
                p_means_filtered = (
                    p_means
                    .groupby("pitch_type")
                    .apply(lambda x: pd.Series({
                        "pitches": x["pitches"].sum(),
                        "mean_plate_x": (
                            (x["mean_plate_x"] * x["pitches"]).sum()
                            / x["pitches"].sum()
                        ),
                        "mean_plate_z": (
                            (x["mean_plate_z"] * x["pitches"]).sum()
                            / x["pitches"].sum()
                        ),
                    }))
                    .reset_index()
                )

            fig2 = go.Figure()

            # strike zone -- approximate, using standard plate width
            # (+ ball radius, the common public-visualization
            # convention) and average vertical bounds. Actual zone
            # varies by batter height; this is a reference outline,
            # not this pitcher's batter-specific zone.
            fig2.add_shape(
                type="rect",
                x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
                line=dict(color="#E8EAED", width=2),
                fillcolor="rgba(0,0,0,0)"
            )

            color_map = {
                pt: get_pitch_type_color(pt)
                for pt in p_means_filtered["pitch_type"].unique()
            }

            # optional raw-pitch detail, drawn first so mean markers
            # sit on top
            if show_raw and loc_points is not None:
                p_points = loc_points[loc_points["player_name"] == pitcher]
                if matchup_filter != "All":
                    p_points = p_points[p_points["matchup_type"] == matchup_filter]
                for pitch_type, group in p_points.groupby("pitch_type"):
                    fig2.add_trace(go.Scatter(
                        x=group["plate_x"], y=group["plate_z"],
                        mode="markers",
                        marker=dict(
                            size=4, color=color_map.get(pitch_type, "gray")
                        ),
                        opacity=0.25,
                        showlegend=False,
                        hoverinfo="skip"
                    ))

            # mean marker, one per pitch type -- no spread ellipse
            # (dropped: added visual clutter without much payoff,
            # given hover detail and the raw-points toggle above
            # already cover that need)
            for _, row in p_means_filtered.iterrows():
                pitch_type = row["pitch_type"]
                color = color_map.get(pitch_type, "gray")

                fig2.add_trace(go.Scatter(
                    x=[row["mean_plate_x"]], y=[row["mean_plate_z"]],
                    mode="markers+text",
                    marker=dict(size=14, color=color, line=dict(color="#E8EAED", width=1)),
                    text=[pitch_type],
                    textposition="top center",
                    name=pitch_type,
                    hovertemplate=(
                        f"{format_pitch_type(pitch_type)}<br>"
                        f"{int(row['pitches']):,} real pitches<br>"
                        "x: %{x:.2f}<br>z: %{y:.2f}<extra></extra>"
                    )
                ))

            fig2.update_layout(
                xaxis_title="Horizontal Location (ft, catcher's view)",
                yaxis_title="Height (ft)",
                xaxis=dict(range=[-2.5, 2.5]),
                yaxis=dict(range=[0, 5]),
                height=550,
                showlegend=False
            )
            fig2.update_yaxes(scaleanchor="x", scaleratio=1)

            apply_chart_theme(fig2, zeroline=False)
            st.plotly_chart(fig2, width='stretch')
            st.caption(
                "Each labeled dot is a pitch type's average location. "
                "Rectangle is an approximate strike zone (standard "
                "width, average height bounds) -- not this pitcher's "
                "batter-specific zone."
            )
    else:
        st.info("Location map data not found.")

st.divider()


# ============================================================
# RELEASE POINT MAP
# ============================================================

"""
Same base design as the Location Map above: mean point per pitch
type, with raw detail available as an opt-in. This is the
visualization home for release-point consistency, which isn't
part of the aggregate PAOM Score itself -- it's a real, stable
signal, but not one that earns enough additional predictive value
on its own to justify inclusion in the combined score. No
handedness split here, unlike Location -- release mechanics have
no legitimate reason to vary by batter side.
"""

st.subheader("Release Point Map")

rel_means = load_release_means()
rel_points = load_release_points()

if rel_means is not None:
    p_rel_means = rel_means[rel_means["player_name"] == pitcher]

    if len(p_rel_means) == 0:
        st.info("No release map data available for this pitcher.")
    else:
        show_raw_release = st.checkbox(
            "Show individual pitches (detail view)",
            value=False,
            key="release_show_raw"
        )

        fig3 = go.Figure()

        rel_color_map = {
            pt: get_pitch_type_color(pt)
            for pt in p_rel_means["pitch_type"].unique()
        }

        if show_raw_release and rel_points is not None:
            p_rel_points = rel_points[rel_points["player_name"] == pitcher]
            for pitch_type, group in p_rel_points.groupby("pitch_type"):
                fig3.add_trace(go.Scatter(
                    x=group["release_pos_x"], y=group["release_pos_z"],
                    mode="markers",
                    marker=dict(
                        size=4, color=rel_color_map.get(pitch_type, "gray")
                    ),
                    opacity=0.25,
                    showlegend=False,
                    hoverinfo="skip"
                ))

        for _, row in p_rel_means.iterrows():
            pitch_type = row["pitch_type"]
            color = rel_color_map.get(pitch_type, "gray")

            fig3.add_trace(go.Scatter(
                x=[row["mean_release_x"]], y=[row["mean_release_z"]],
                mode="markers+text",
                marker=dict(size=14, color=color, line=dict(color="#E8EAED", width=1)),
                text=[pitch_type],
                textposition="top center",
                name=pitch_type,
                hovertemplate=(
                    f"{format_pitch_type(pitch_type)}<br>"
                    f"{int(row['pitches']):,} real pitches<br>"
                    "x: %{x:.2f}<br>z: %{y:.2f}<extra></extra>"
                )
            ))

        # same-handed league-average release point -- ONE marker for
        # the whole arsenal (not per pitch type), matching Release
        # Consistency's own established principle that release
        # mechanics shouldn't be benchmarked per pitch type
        release_benchmark = load_league_release_benchmark()
        pitcher_hands = load_pitcher_hands()

        if release_benchmark is not None and pitcher_hands is not None:
            hand_row = pitcher_hands[pitcher_hands["player_name"] == pitcher]
            if len(hand_row) > 0:
                p_hand = hand_row.iloc[0]["p_throws"]
                bench = release_benchmark[release_benchmark["p_throws"] == p_hand]
                if len(bench) > 0:
                    b = bench.iloc[0]
                    fig3.add_trace(go.Scatter(
                        x=[b["avg_release_x"]], y=[b["avg_release_z"]],
                        mode="markers+text",
                        marker=dict(
                            size=16, symbol="diamond-open",
                            color="#E8EAED", line=dict(width=2)
                        ),
                        text=["Lg Avg"],
                        textposition="bottom center",
                        name=f"League avg ({p_hand}HP)",
                        hovertemplate=(
                            f"League avg release ({p_hand}HP)<br>"
                            f"{int(b['n_pitchers']):,} real pitchers<br>"
                            "x: %{x:.2f}<br>z: %{y:.2f}<extra></extra>"
                        )
                    ))

        fig3.update_layout(
            xaxis_title="Release Position, Horizontal (ft)",
            yaxis_title="Release Position, Height (ft)",
            height=550,
            showlegend=False
        )
        fig3.update_yaxes(scaleanchor="x", scaleratio=1)

        apply_chart_theme(fig3, zeroline=False)
        st.plotly_chart(fig3, width='stretch')
        st.caption(
            "Each labeled dot is a pitch type's average release "
            "position. Pitch types releasing from noticeably "
            "different points may be easier for hitters to "
            "distinguish before the ball even leaves the hand. "
            "Open black diamond = league-average release point "
            "(all pitch types combined) among pitchers who throw "
            "with the same hand."
        )
else:
    st.info("Release map data not found.")

st.divider()


# ============================================================
# ARSENAL EVOLUTION -- real, multi-year (2020-2025) trends
#
# Uses pitcher_arsenal_evolution_2020_2025.csv (real per-pitch-type
# usage and movement, every real season) and the real, per-year PAOM
# component scores -- both already power other parts of the
# dashboard (Similar Pitchers Map's pitch-composition factor, the
# Leaderboard's year filter), reused here directly rather than any
# new data pipeline.
#
# CROSSWALK NOTE: this pitcher's name here is PAOM-style ("Last,
# First"), but the arsenal evolution data uses its own, different
# lowercase "first last" naming (same Kaggle source as elsewhere on
# this page) -- resolved via historical_recommendations.csv's real
# player_id <-> player_name mapping, the same approach already used
# for the Recommendations section below.
# ============================================================

st.subheader("Arsenal Evolution")
st.caption(
    "How this pitcher's real pitch mix, movement, and PAOM scores "
    "have changed across every season with real data (2020-2025)."
)

arsenal_usage = load_arsenal_usage()
hist_recs_for_crosswalk = load_historical_recommendations()

evo_target_id = None
if hist_recs_for_crosswalk is not None:
    evo_crosswalk_match = hist_recs_for_crosswalk[
        hist_recs_for_crosswalk["player_name"] == pitcher
    ]
    if len(evo_crosswalk_match) > 0:
        evo_target_id = evo_crosswalk_match.iloc[0]["player_id"]

if arsenal_usage is None:
    st.info("Arsenal evolution data not found.")
elif evo_target_id is None:
    st.info(f"Couldn't resolve {pitcher} to a real player ID for this data source.")
else:
    p_arsenal = arsenal_usage[arsenal_usage["player_id"] == evo_target_id].sort_values("season")

    if len(p_arsenal) == 0:
        st.info(f"No arsenal evolution data found for {pitcher}.")
    else:
        tab_usage, tab_movement, tab_scores = st.tabs(
            ["Usage Over Time", "Movement Over Time", "PAOM Score Over Time"]
        )

        PITCH_TYPE_COLS_PREFIX = ["CH", "CS", "CU", "EP", "FA", "FC", "FF", "FO", "FS",
                                    "KC", "KN", "PO", "SC", "SI", "SL", "ST", "SV", "UN"]
        MEANINGFUL_USAGE_FLOOR = 5.0

        # real pitch types this pitcher has thrown meaningfully
        # (>=5% usage, matching this project's convention) in ANY
        # real season -- so a pitch dropped years ago still shows
        # its full real history, not just years it cleared the floor
        real_pitch_types = set()
        for pt in PITCH_TYPE_COLS_PREFIX:
            col = f"{pt}_usage_pct"
            if col in p_arsenal.columns and (p_arsenal[col] >= MEANINGFUL_USAGE_FLOOR).any():
                real_pitch_types.add(pt)

        # ------------------------------------------------------
        # USAGE OVER TIME
        # ------------------------------------------------------

        with tab_usage:
            if not real_pitch_types:
                st.info("No pitch type cleared the usage floor in any real season.")
            else:
                fig_usage = go.Figure()
                for pt in sorted(real_pitch_types):
                    col = f"{pt}_usage_pct"
                    if col not in p_arsenal.columns:
                        continue
                    fig_usage.add_trace(go.Scatter(
                        x=p_arsenal["season"], y=p_arsenal[col],
                        mode="lines+markers",
                        name=format_pitch_type(pt),
                        line=dict(color=get_pitch_type_color(pt)),
                        connectgaps=False,  # a real gap (pitch not
                                              # thrown that season)
                                              # should show as a real
                                              # gap, not a misleading
                                              # straight line across it
                        hovertemplate=(
                            f"{format_pitch_type(pt)}<br>Season: "
                            "%{x}<br>Usage: %{y:.1f}%<extra></extra>"
                        )
                    ))

                fig_usage.update_layout(
                    xaxis_title="Season",
                    yaxis_title="Usage (%)",
                    xaxis=dict(tickmode="linear", dtick=1),
                    height=500
                )
                apply_chart_theme(fig_usage, zeroline=False)
                st.plotly_chart(fig_usage, width='stretch')
                st.caption(
                    "A gap in a line means that pitch wasn't thrown "
                    "meaningfully (below a 5% usage floor) that "
                    "season -- not necessarily zero, just below the "
                    "threshold this project uses throughout for a "
                    "'real' pitch in the arsenal."
                )

        # ------------------------------------------------------
        # MOVEMENT OVER TIME
        # ------------------------------------------------------

        with tab_movement:
            if not real_pitch_types:
                st.info("No pitch type cleared the usage floor in any real season.")
            else:
                fig_movement = go.Figure()
                for pt in sorted(real_pitch_types):
                    x_col, y_col = f"{pt}_avg_pfx_x", f"{pt}_avg_pfx_z"
                    if x_col not in p_arsenal.columns or y_col not in p_arsenal.columns:
                        continue
                    pt_data = p_arsenal.dropna(subset=[x_col, y_col])
                    if len(pt_data) == 0:
                        continue

                    color = get_pitch_type_color(pt)
                    fig_movement.add_trace(go.Scatter(
                        x=pt_data[x_col], y=pt_data[y_col],
                        mode="lines+markers+text",
                        text=pt_data["season"].astype(int).astype(str),
                        textposition="top center",
                        name=format_pitch_type(pt),
                        line=dict(color=color, dash="dot"),
                        marker=dict(size=9, color=color),
                        hovertemplate=(
                            f"{format_pitch_type(pt)}<br>Season: "
                            "%{text}<br>x: %{x:.2f}<br>z: %{y:.2f}<extra></extra>"
                        )
                    ))

                fig_movement.update_layout(
                    xaxis_title="Horizontal Break (in)",
                    yaxis_title="Induced Vertical Break (in)",
                    height=550
                )
                apply_chart_theme(fig_movement, zeroline=True)
                st.plotly_chart(fig_movement, width='stretch')
                st.caption(
                    "Each dotted path traces one pitch type's real "
                    "average shape from season to season -- a season "
                    "missing from a path means that pitch didn't "
                    "clear the usage floor that year, not that its "
                    "shape is unknown."
                )

        # ------------------------------------------------------
        # PAOM SCORE OVER TIME
        # ------------------------------------------------------

        with tab_scores:
            all_year_scores = load_final_scores_all_years()
            if all_year_scores is None:
                st.info("Multi-year PAOM score data not found.")
            else:
                p_scores = all_year_scores[
                    all_year_scores["player_name"] == pitcher
                ].sort_values("season")

                if len(p_scores) == 0:
                    st.info(f"No real PAOM score history found for {pitcher}.")
                else:
                    SCORE_COLS = {
                        "paom_score": "PAOM Score",
                        "pitcher_effectiveness": "Effectiveness",
                        "trusted_movement_score": "Movement",
                        "pitcher_command": "Command",
                        "trusted_velocity_score": "Velocity",
                    }
                    fig_scores = go.Figure()
                    for col, label in SCORE_COLS.items():
                        if col not in p_scores.columns:
                            continue
                        fig_scores.add_trace(go.Scatter(
                            x=p_scores["season"], y=p_scores[col],
                            mode="lines+markers",
                            name=label,
                            line=dict(width=3 if col == "paom_score" else 1.5),
                            hovertemplate=f"{label}<br>Season: %{{x}}<br>Score: %{{y:.1f}}<extra></extra>"
                        ))

                    fig_scores.update_layout(
                        xaxis_title="Season",
                        yaxis_title="Score (0-100, higher = better)",
                        xaxis=dict(tickmode="linear", dtick=1),
                        height=500
                    )
                    apply_chart_theme(fig_scores, zeroline=False)
                    st.plotly_chart(fig_scores, width='stretch')
                    st.caption(
                        "PAOM Score (bold line) and its four real "
                        "components, each on a 0-100 scale where "
                        "higher is better."
                    )

render_pitch_type_key()

st.divider()


# ============================================================
# RECOMMENDATIONS -- historical-precedent engine
# ============================================================

"""
Replaces the previous PAOM Usage/Drop/Add recommendation tabs.
Validation work (see project notes) compared PAOM's own
recommendation logic against real historical outcomes across all
four change types, using two independent methods -- rank-agreement
against the historical engine, and direct validation of PAOM's
followed recommendations against real subsequent outcomes. Both
came back consistently weak-to-negative across large real samples,
so PAOM's recommendation logic no longer appears on the dashboard,
in any form.

Single ranked list instead of three separate tabs -- the historical
engine evaluates ADD/DROP/USAGE_INCREASE/USAGE_DECREASE candidates
on the same predicted-outcome scale, so they're directly comparable
and rankable together, unlike PAOM's three structurally different
scoring systems. NO_CHANGE is always included as a real row in this
list (from the source data, not added here) -- the honest baseline
every other candidate should be judged against, never hidden.
"""

st.subheader("Recommendations")

hist_recs = load_historical_recommendations()
hist_comps = load_historical_recommendation_comps()

if hist_recs is not None:
    p_recs = hist_recs[hist_recs["player_name"] == pitcher].sort_values("rank")

    if len(p_recs) == 0:
        st.info("No historical-precedent recommendations available for this pitcher.")
    else:
        CONFIDENCE_COLORS = {
            "Strong evidence": "🟢",
            "Moderate evidence": "🟡",
            "Limited evidence": "⚪",
            "n/a": "⚪",
        }

        top = p_recs.iloc[0]

        # ------------------------------------------------------
        # TOP RECOMMENDATION -- highlighted, with comps shown
        # by default, not behind a click
        # ------------------------------------------------------

        if top["change_type"] == "NO_CHANGE":
            st.info(
                "**Top recommendation: no change.** No candidate arsenal "
                "change beat this pitcher's own real-data baseline."
            )
        else:
            tier_icon = CONFIDENCE_COLORS.get(top["confidence_tier"], "⚪")
            usage_note = (
                f", usage change {top['estimated_usage_delta']:+.1f} percentage points"
                if pd.notna(top.get("estimated_usage_delta")) else ""
            )
            st.success(
                f"**Top recommendation: {top['change_type']} {format_pitch_type(top['pitch_type'])}**"
                f"{usage_note}  \n"
                f"Predicted change in xwOBA-against: {top['predicted_change']:+.3f} "
                f"*(lower is better)* "
                f"— based on {top['n_historical_events']:.0f} real historical comparison events "
                f"&nbsp;&nbsp;{tier_icon} {top['confidence_tier']}"
            )

            if top.get("exceeds_historical_precedent"):
                st.warning(
                    "This recommendation's magnitude exceeds anything seen "
                    "in real historical data -- treat with extra caution."
                )
            if top.get("is_high_usage_drop"):
                st.warning(
                    "This recommends dropping a pitch this pitcher currently "
                    "throws at an unusually high rate -- rare but not "
                    "unprecedented in real historical data (real pitchers "
                    "have dropped pitches thrown at this usage level before, "
                    "just not often)."
                )
            if pd.notna(top.get("fallback_values_used")) and top["fallback_values_used"] != "none":
                st.caption(
                    f"Some inputs used fallback/estimated values, not this "
                    f"pitcher's own data: {top['fallback_values_used']}"
                )

            if hist_comps is not None:
                top_comps = hist_comps[
                    (hist_comps["player_name"] == pitcher)
                    & (hist_comps["change_type"] == top["change_type"])
                    & (hist_comps["pitch_type"] == top["pitch_type"])
                ].sort_values("comp_rank")

                if len(top_comps) > 0:
                    st.caption("Real historical pitchers this recommendation is based on:")
                    top_comps_display = top_comps.copy()
                    top_comps_display["comp_player_name"] = top_comps_display["comp_player_name"].apply(format_player_name)
                    st.dataframe(
                        top_comps_display[[
                            "comp_player_name", "comp_season_from", "comp_season_to",
                            "comp_usage_before", "comp_usage_after", "similarity_weight"
                        ]].rename(columns={
                            "comp_player_name": "Pitcher",
                            "comp_season_from": "From",
                            "comp_season_to": "To",
                            "comp_usage_before": "Usage Before (%)",
                            "comp_usage_after": "Usage After (%)",
                            "similarity_weight": "Similarity"
                        }).style.format({
                            "Usage Before (%)": "{:.1f}%",
                            "Usage After (%)": "{:.1f}%",
                            "Similarity": "{:.2f}"
                        }),
                        width='stretch', hide_index=True
                    )
                    st.caption(
                        "Similarity is a relative score (higher = more similar "
                        "to this pitcher), not a physical unit -- it's only "
                        "meaningful for comparing rows within this same table."
                    )

        st.divider()

        # ------------------------------------------------------
        # FULL RANKED LIST
        # ------------------------------------------------------

        st.markdown("**All candidates, ranked**")

        def build_flags(row):
            flags = []
            if row.get("exceeds_historical_precedent") is True:
                flags.append("⚠️ unprecedented magnitude")
            if row.get("is_high_usage_drop") is True:
                flags.append("⚠️ high-usage drop")
            return ", ".join(flags) if flags else ""

        p_recs_with_flags = p_recs.copy()
        p_recs_with_flags["Flags"] = p_recs_with_flags.apply(build_flags, axis=1)
        p_recs_with_flags["pitch_type"] = p_recs_with_flags["pitch_type"].apply(format_pitch_type)

        display_recs = p_recs_with_flags[[
            "rank", "change_type", "pitch_type", "estimated_usage_delta",
            "predicted_change", "confidence_tier", "n_historical_events", "Flags"
        ]].rename(columns={
            "rank": "Rank",
            "change_type": "Change",
            "pitch_type": "Pitch",
            "estimated_usage_delta": "Usage Change (pp)",
            "predicted_change": "Predicted Change (xwOBA, lower=better)",
            "confidence_tier": "Evidence",
            "n_historical_events": "Real Comparison Events"
        })

        st.dataframe(
            display_recs.style.format({
                "Usage Change (pp)": lambda v: f"{v:+.1f}" if pd.notna(v) else "-",
                "Predicted Change (xwOBA, lower=better)": "{:+.3f}",
                "Real Comparison Events": lambda v: f"{v:.0f}" if pd.notna(v) else "-"
            }),
            width='stretch', hide_index=True
        )
        st.caption(
            "\"Usage Change (pp)\" is in percentage points (e.g. +10.0 means "
            "throwing that pitch 10 percentage points more often). \"Evidence\" "
            "reflects how many real historical comparison events back a "
            "prediction -- more events generally means a more trustworthy "
            "estimate."
        )
        if p_recs_with_flags["Flags"].str.len().gt(0).any():
            st.caption(
                "⚠️ high-usage drop: recommends dropping a pitch thrown at "
                "an unusually high rate -- rare but real precedent exists. "
                "⚠️ unprecedented magnitude: this candidate's predicted "
                "outcome exceeds anything seen in real historical data."
            )

        render_pitch_type_key()

        # ------------------------------------------------------
        # COMPS FOR ANY OTHER CANDIDATE
        # ------------------------------------------------------

        if hist_comps is not None:
            other_candidates = p_recs[p_recs["change_type"] != "NO_CHANGE"]
            if len(other_candidates) > 0:
                with st.expander("See real comps for a specific candidate"):
                    candidate_labels = [
                        f"#{r['rank']:.0f} -- {r['change_type']} {format_pitch_type(r['pitch_type'])}"
                        for _, r in other_candidates.iterrows()
                    ]
                    chosen = st.selectbox(
                        "Candidate", options=candidate_labels, key="rec_comp_selector"
                    )
                    chosen_row = other_candidates.iloc[candidate_labels.index(chosen)]

                    chosen_comps = hist_comps[
                        (hist_comps["player_name"] == pitcher)
                        & (hist_comps["change_type"] == chosen_row["change_type"])
                        & (hist_comps["pitch_type"] == chosen_row["pitch_type"])
                    ].sort_values("comp_rank")

                    if len(chosen_comps) > 0:
                        chosen_comps_display = chosen_comps.copy()
                        chosen_comps_display["comp_player_name"] = chosen_comps_display["comp_player_name"].apply(format_player_name)
                        st.dataframe(
                            chosen_comps_display[[
                                "comp_player_name", "comp_season_from", "comp_season_to",
                                "comp_usage_before", "comp_usage_after", "similarity_weight"
                            ]].rename(columns={
                                "comp_player_name": "Pitcher",
                                "comp_season_from": "From",
                                "comp_season_to": "To",
                                "comp_usage_before": "Usage Before (%)",
                                "comp_usage_after": "Usage After (%)",
                                "similarity_weight": "Similarity"
                            }).style.format({
                                "Usage Before (%)": "{:.1f}%",
                                "Usage After (%)": "{:.1f}%",
                                "Similarity": "{:.2f}"
                            }),
                            width='stretch', hide_index=True
                        )
                    else:
                        st.info("No comp data available for this candidate.")
else:
    st.info(
        "Historical-precedent recommendation data not found -- run "
        "94_batch_precompute_historical_recommendations.py first."
    )

st.divider()


# ============================================================
# PLATOON SPLITS
# ============================================================

st.subheader("Platoon Splits")

platoon = load_platoon()
if platoon is not None:
    p_platoon = platoon[platoon["player_name"] == pitcher].drop(
        columns=["player_name", "pitcher"], errors="ignore"
    )
    if len(p_platoon) > 0:
        p_platoon = p_platoon.sort_values("platoon_gap_abs", ascending=False).copy()
        p_platoon["pitch_type"] = p_platoon["pitch_type"].apply(format_pitch_type)
        p_platoon["weaker_side"] = p_platoon["weaker_side"].map(
            {"same": "Same-handed", "opposite": "Opposite-handed"}
        )

        display_platoon = p_platoon[[
            "pitch_type", "trusted_score_vs_same", "trusted_score_vs_opposite",
            "confidence_vs_same", "confidence_vs_opposite",
            "platoon_gap", "weaker_side"
        ]].rename(columns={
            "pitch_type": "Pitch",
            "trusted_score_vs_same": "Effectiveness vs Same-Handed",
            "trusted_score_vs_opposite": "Effectiveness vs Opposite-Handed",
            "confidence_vs_same": "Confidence (Same)",
            "confidence_vs_opposite": "Confidence (Opposite)",
            "platoon_gap": "Platoon Gap",
            "weaker_side": "Weaker Against"
        })

        st.dataframe(
            display_platoon.style.format({
                "Effectiveness vs Same-Handed": "{:.1f}",
                "Effectiveness vs Opposite-Handed": "{:.1f}",
                "Confidence (Same)": "{:.0f}",
                "Confidence (Opposite)": "{:.0f}",
                "Platoon Gap": "{:+.1f}",
            }),
            width='stretch', hide_index=True
        )

        with st.expander("What do these columns mean?"):
            st.markdown(
                """
**Effectiveness vs Same/Opposite-Handed** -- this pitch's real
effectiveness (0-100, same scale as the overall Effectiveness
component above), computed **separately** for batters standing on
the same side as this pitcher throws vs. the opposite side. These
are the real, trustworthy numbers to read directly -- already
shrunk toward league average based on how much real data backs each
side (see Confidence below), not raw values.

**Confidence (Same / Opposite)** -- how much real data backs *each
side's* score specifically, 0-100. These are tracked **separately**
on purpose: a pitch can be well-established against same-handed
batters while still being a thin, unreliable sample against
opposite-handed ones (or vice versa) -- one combined number would
hide that.

**Platoon Gap** -- the real difference between the two trusted
scores (opposite-handed minus same-handed). A large positive number
means this pitch performs meaningfully better against opposite-
handed batters; a large negative number means the reverse. Computed
from the trusted (shrunk) scores specifically, so a big gap
reflects a real pattern, not just a thin-sample fluke.

**Weaker Against** -- which side this pitch is actually the weaker
pitch against, based on the sign of the Platoon Gap above.
                """
            )
    else:
        st.info("No platoon split data available for this pitcher.")
else:
    st.info("Platoon split data not found.")

st.divider()


# ============================================================
# SIMILAR PITCHERS
# ============================================================

st.subheader("Similar Pitchers")

col_mech, col_arsenal = st.columns(2)

with col_mech:
    st.markdown("**Mechanical comps** (same throwing hand)")
    mech = load_mechanical_similarity()
    if mech is not None:
        p_mech = mech[mech["player_name"] == pitcher].sort_values("rank").copy()
        if len(p_mech) > 0:
            p_mech["comp_player_name"] = p_mech["comp_player_name"].apply(format_player_name)
            st.dataframe(
                p_mech[["comp_player_name", "rank", "physical_distance"]],
                width='stretch', hide_index=True
            )
        else:
            st.info("No mechanical comps found.")
    else:
        st.info("Mechanical similarity data not found.")

with col_arsenal:
    st.markdown("**Arsenal-shape comps** (both hands)")
    arsenal = load_arsenal_similarity()
    if arsenal is not None:
        p_arsenal = arsenal[arsenal["player_name"] == pitcher].sort_values(
            "rank"
        ).copy()
        if len(p_arsenal) > 0:
            p_arsenal["comp_player_name"] = p_arsenal["comp_player_name"].apply(format_player_name)
            st.dataframe(
                p_arsenal[
                    ["comp_player_name", "same_hand", "rank", "arsenal_distance"]
                ],
                width='stretch', hide_index=True
            )
        else:
            st.info("No arsenal comps found.")
    else:
        st.info("Arsenal similarity data not found.")
