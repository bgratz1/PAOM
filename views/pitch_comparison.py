"""
views/pitch_comparison.py

PAOM Dashboard -- select two pitcher x pitch pairs (can be the same
pitcher's two pitches, or two different pitchers) and compare them:
a full stat table plus Movement, Location, and Release Point
context charts, reusing the same-handed league benchmarks built for
the Pitcher Detail page.

Design:
-------
Unlike the arsenal-level Pitcher Comparison tab (which overlays
whole arsenals -- multiple pitch types per pitcher, enough points
to make a scatter worth reading), this tab is scoped to exactly ONE
pitch per side. Two lonely points on a chart don't carry much
information on their own, so each context chart also plots the
relevant league benchmark (same-handed average for that pitch type/
overall release) as a reference -- "how does this specific slider
compare to a typical same-handed slider" is the actual useful
question here, not just "where do these two dots sit relative to
each other."

Marker convention: SHAPE distinguishes which side (circle = side A,
square = side B) -- avoids clashing with the "open diamond = league
benchmark" convention already established on the Pitcher Detail
page. COLOR still encodes pitch type via the shared palette, same
as everywhere else in the dashboard.

Nearly all of this is built from data that already exists at
exactly pitcher x pitch-type grain (Effectiveness, Command, Platoon,
Movement, Release, Location) -- this page is mostly new UI/layout,
not new data pipeline work.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_loader import (
    load_effectiveness, load_command, load_platoon,
    load_movement_points, load_release_means, load_location_means,
    load_league_movement_benchmark, load_league_release_benchmark,
    load_pitcher_hands, all_pitcher_names
)
from styling import get_pitch_type_color, apply_chart_theme, render_pitcher_indicator, render_handedness_badge


st.title("Pitch-Level Comparison")
render_pitcher_indicator()
st.caption(
    "Compare two specific pitches -- from two different pitchers, "
    "or the same pitcher's own two pitches -- against each other "
    "and against the same-handed league average."
)


# ============================================================
# HELPERS
# ============================================================

def pitcher_pitch_types(pitcher_name):
    """
    Union of pitch types available for this pitcher across the main
    per-pitch-type sources, so the dropdown reflects everything
    viewable here, not just whichever single file happens to be
    checked first.
    """
    types = set()
    for loader in [load_effectiveness, load_command, load_movement_points]:
        df = loader()
        if df is not None:
            p = df[df["player_name"] == pitcher_name]
            if len(p) > 0 and "pitch_type" in p.columns:
                types.update(p["pitch_type"].unique())
    return sorted(types)


def get_pitcher_hand(pitcher_name):
    hands = load_pitcher_hands()
    if hands is None:
        return None
    row = hands[hands["player_name"] == pitcher_name]
    return row.iloc[0]["p_throws"] if len(row) > 0 else None


def collapse_location(pitcher_name, pitch_type):
    """
    Pitch-count-weighted average across same/opposite matchup
    splits, matching the "All" default used elsewhere in the
    dashboard.
    """
    loc = load_location_means()
    if loc is None:
        return None
    rows = loc[
        (loc["player_name"] == pitcher_name)
        & (loc["pitch_type"] == pitch_type)
    ]
    if len(rows) == 0:
        return None
    total_pitches = rows["pitches"].sum()
    return {
        "mean_plate_x": (rows["mean_plate_x"] * rows["pitches"]).sum() / total_pitches,
        "mean_plate_z": (rows["mean_plate_z"] * rows["pitches"]).sum() / total_pitches,
        "pitches": total_pitches
    }


def gather_pitch_data(pitcher_name, pitch_type):
    """
    Pulls every available metric for one (pitcher, pitch_type) pair
    from its own source, independently -- missing pieces stay None
    rather than the whole lookup failing.
    """
    data = {"player_name": pitcher_name, "pitch_type": pitch_type}

    eff = load_effectiveness()
    if eff is not None:
        row = eff[(eff["player_name"] == pitcher_name) & (eff["pitch_type"] == pitch_type)]
        if len(row) > 0:
            data["effectiveness"] = row.iloc[0]["trusted_effectiveness"]
            data["usage"] = row.iloc[0].get("usage")

    cmd = load_command()
    if cmd is not None:
        row = cmd[(cmd["player_name"] == pitcher_name) & (cmd["pitch_type"] == pitch_type)]
        if len(row) > 0:
            data["command"] = row.iloc[0]["trusted_command_score"]

    mv = load_movement_points()
    if mv is not None:
        row = mv[(mv["player_name"] == pitcher_name) & (mv["pitch_type"] == pitch_type)]
        if len(row) > 0:
            data["HB"] = row.iloc[0]["HB"]
            data["IVB"] = row.iloc[0]["IVB"]
            data["velo"] = row.iloc[0]["velo"]

    rel = load_release_means()
    if rel is not None:
        row = rel[(rel["player_name"] == pitcher_name) & (rel["pitch_type"] == pitch_type)]
        if len(row) > 0:
            data["release_x"] = row.iloc[0]["mean_release_x"]
            data["release_z"] = row.iloc[0]["mean_release_z"]

    loc = collapse_location(pitcher_name, pitch_type)
    if loc is not None:
        data["plate_x"] = loc["mean_plate_x"]
        data["plate_z"] = loc["mean_plate_z"]

    platoon = load_platoon()
    if platoon is not None:
        row = platoon[
            (platoon["player_name"] == pitcher_name)
            & (platoon["pitch_type"] == pitch_type)
        ]
        if len(row) > 0:
            data["score_vs_same"] = row.iloc[0].get("trusted_score_vs_same")
            data["score_vs_opposite"] = row.iloc[0].get("trusted_score_vs_opposite")
            data["platoon_gap_abs"] = row.iloc[0].get("platoon_gap_abs")
            data["weaker_side"] = row.iloc[0].get("weaker_side")

    data["p_throws"] = get_pitcher_hand(pitcher_name)

    return data


def fmt(value, decimals=1):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{value:.{decimals}f}"


# ============================================================
# SELECTION
# ============================================================

names = all_pitcher_names()

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Side A**")
    pitcher_a = st.selectbox("Pitcher", options=names, index=0, key="pc_pitcher_a")
    pitches_a = pitcher_pitch_types(pitcher_a)
    if not pitches_a:
        st.warning(f"No pitch-level data found for {pitcher_a}.")
        st.stop()
    pitch_a = st.selectbox("Pitch", options=pitches_a, key="pc_pitch_a")

with col_b:
    st.markdown("**Side B**")
    default_b_index = 1 if len(names) > 1 else 0
    pitcher_b = st.selectbox(
        "Pitcher", options=names, index=default_b_index, key="pc_pitcher_b"
    )
    pitches_b = pitcher_pitch_types(pitcher_b)
    if not pitches_b:
        st.warning(f"No pitch-level data found for {pitcher_b}.")
        st.stop()
    default_pitch_b_index = 1 if len(pitches_b) > 1 else 0
    pitch_b = st.selectbox(
        "Pitch", options=pitches_b, index=default_pitch_b_index, key="pc_pitch_b"
    )

data_a = gather_pitch_data(pitcher_a, pitch_a)
data_b = gather_pitch_data(pitcher_b, pitch_b)

label_a = f"{pitcher_a} ({pitch_a})"
label_b = f"{pitcher_b} ({pitch_b})"

badge_col_a, badge_col_b = st.columns(2)
with badge_col_a:
    render_handedness_badge(data_a.get("p_throws"))
with badge_col_b:
    render_handedness_badge(data_b.get("p_throws"))

SHAPE_A = "circle"
SHAPE_B = "square"

st.divider()


# ============================================================
# STAT TABLE
# ============================================================

st.subheader("Stat Comparison")

metric_rows = [
    ("Usage", "usage", lambda v: f"{v:.0%}" if v is not None else "-"),
    ("Effectiveness", "effectiveness", lambda v: fmt(v)),
    ("Command", "command", lambda v: fmt(v)),
    ("Velocity (mph)", "velo", lambda v: fmt(v)),
    ("Horizontal Break (in)", "HB", lambda v: fmt(v)),
    ("Induced Vert. Break (in)", "IVB", lambda v: fmt(v)),
    ("Effectiveness vs Same-Handed", "score_vs_same", lambda v: fmt(v)),
    ("Effectiveness vs Opposite-Handed", "score_vs_opposite", lambda v: fmt(v)),
    ("Platoon Gap (abs)", "platoon_gap_abs", lambda v: fmt(v)),
    ("Weaker Side", "weaker_side", lambda v: {"same": "Same-Handed", "opposite": "Opposite-Handed"}.get(v, "-")),
]

table_rows = []
for label, key, formatter in metric_rows:
    table_rows.append({
        "Metric": label,
        label_a: formatter(data_a.get(key)),
        label_b: formatter(data_b.get(key))
    })

stat_table = pd.DataFrame(table_rows).set_index("Metric")

st.dataframe(stat_table, width='stretch')
st.caption(
    "Effectiveness vs Same/Opposite-Handed is this pitch's real "
    "effectiveness score computed separately against same-handed "
    "and opposite-handed batters. Platoon Gap (abs) is the size of "
    "the real difference between those two; Weaker Side names which "
    "one this pitch actually performs worse against."
)

st.divider()


# ============================================================
# CONTEXT CHARTS
# ============================================================

st.subheader("Context Charts")

tab_movement, tab_location, tab_release = st.tabs(
    ["Movement", "Location", "Release"]
)


def add_side_marker(fig, data, label, shape, x_key, y_key):
    if data.get(x_key) is None or data.get(y_key) is None:
        return False
    color = get_pitch_type_color(data["pitch_type"])
    fig.add_trace(go.Scatter(
        x=[data[x_key]], y=[data[y_key]],
        mode="markers+text",
        marker=dict(size=16, color=color, symbol=shape, line=dict(color="#E8EAED", width=1)),
        text=[label],
        textposition="top center",
        showlegend=False,
        hovertemplate=f"{label}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>"
    ))
    return True


# ------------------------------------------------------------
# MOVEMENT TAB
# ------------------------------------------------------------

with tab_movement:
    fig = go.Figure()

    plotted_a = add_side_marker(fig, data_a, f"A: {pitch_a}", SHAPE_A, "HB", "IVB")
    plotted_b = add_side_marker(fig, data_b, f"B: {pitch_b}", SHAPE_B, "HB", "IVB")

    movement_benchmark = load_league_movement_benchmark()
    if movement_benchmark is not None:
        for data, side_label in [(data_a, "A"), (data_b, "B")]:
            if data.get("p_throws") is None:
                continue
            bench = movement_benchmark[
                (movement_benchmark["p_throws"] == data["p_throws"])
                & (movement_benchmark["pitch_type"] == data["pitch_type"])
            ]
            if len(bench) > 0:
                b = bench.iloc[0]
                fig.add_trace(go.Scatter(
                    x=[b["avg_HB"]], y=[b["avg_IVB"]],
                    mode="markers",
                    marker=dict(size=12, symbol="diamond-open", color="#E8EAED", line=dict(width=2)),
                    showlegend=False,
                    hovertemplate=(
                        f"League avg {data['pitch_type']} ({data['p_throws']}HP)<br>"
                        "x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>"
                    )
                ))

    if not plotted_a and not plotted_b:
        st.info("No movement data available for either selection.")
    else:
        fig.update_layout(
            xaxis_title="Horizontal Break (in)",
            yaxis_title="Induced Vertical Break (in)",
            height=500
        )
        apply_chart_theme(fig, zeroline=True)
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Circle = Side A, square = Side B. Open black diamonds = "
            "same-handed league average for that pitch type."
        )


# ------------------------------------------------------------
# LOCATION TAB
# ------------------------------------------------------------

with tab_location:
    fig2 = go.Figure()

    fig2.add_shape(
        type="rect",
        x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
        line=dict(color="#E8EAED", width=2),
        fillcolor="rgba(0,0,0,0)"
    )

    plotted_a = add_side_marker(fig2, data_a, f"A: {pitch_a}", SHAPE_A, "plate_x", "plate_z")
    plotted_b = add_side_marker(fig2, data_b, f"B: {pitch_b}", SHAPE_B, "plate_x", "plate_z")

    if not plotted_a and not plotted_b:
        st.info("No location data available for either selection.")
    else:
        fig2.update_layout(
            xaxis_title="Horizontal Location (ft, catcher's view)",
            yaxis_title="Height (ft)",
            xaxis=dict(range=[-2.5, 2.5]),
            yaxis=dict(range=[0, 5]),
            height=500
        )
        fig2.update_yaxes(scaleanchor="x", scaleratio=1)
        apply_chart_theme(fig2, zeroline=False)
        st.plotly_chart(fig2, width='stretch')
        st.caption(
            "Circle = Side A, square = Side B. Average location "
            "combined across batter handedness. Rectangle is an "
            "approximate strike zone, not batter-specific."
        )


# ------------------------------------------------------------
# RELEASE TAB
# ------------------------------------------------------------

with tab_release:
    fig3 = go.Figure()

    plotted_a = add_side_marker(fig3, data_a, f"A: {pitch_a}", SHAPE_A, "release_x", "release_z")
    plotted_b = add_side_marker(fig3, data_b, f"B: {pitch_b}", SHAPE_B, "release_x", "release_z")

    release_benchmark = load_league_release_benchmark()
    if release_benchmark is not None:
        seen_hands = set()
        for data in [data_a, data_b]:
            if data.get("p_throws") is None or data["p_throws"] in seen_hands:
                continue
            seen_hands.add(data["p_throws"])
            bench = release_benchmark[release_benchmark["p_throws"] == data["p_throws"]]
            if len(bench) > 0:
                b = bench.iloc[0]
                fig3.add_trace(go.Scatter(
                    x=[b["avg_release_x"]], y=[b["avg_release_z"]],
                    mode="markers+text",
                    marker=dict(size=14, symbol="diamond-open", color="#E8EAED", line=dict(width=2)),
                    text=[f"Lg Avg ({data['p_throws']}HP)"],
                    textposition="bottom center",
                    showlegend=False,
                    hovertemplate=(
                        f"League avg release ({data['p_throws']}HP)<br>"
                        "x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>"
                    )
                ))

    if not plotted_a and not plotted_b:
        st.info("No release point data available for either selection.")
    else:
        fig3.update_layout(
            xaxis_title="Release Position, Horizontal (ft)",
            yaxis_title="Release Position, Height (ft)",
            height=500
        )
        fig3.update_yaxes(scaleanchor="x", scaleratio=1)
        apply_chart_theme(fig3, zeroline=False)
        st.plotly_chart(fig3, width='stretch')
        st.caption(
            "Circle = Side A, square = Side B. Open black diamonds = "
            "same-handed league-average release point (all pitch "
            "types combined)."
        )
