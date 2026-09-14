"""
views/pitcher_comparison.py

PAOM Dashboard -- select two pitchers and compare their PAOM
scores plus their Movement, Location, and Release Point maps
overlaid on the same chart.

Design:
-------
Overlay rather than side-by-side separate charts -- since each map
already shows one point per pitch type (not raw scatter clutter),
overlaying two pitchers' points stays readable and makes direct
comparison much easier than jumping between two separate charts
with potentially different axis scales.

Two-dimensional categorical encoding: COLOR = pitch type (fixed,
shared across the whole dashboard via styling.py, so e.g. "SL" is
always the same color for both pitchers), SHAPE = which pitcher
(circle vs diamond). This lets you see both "how do their arsenals
compare shape-for-shape" and "whose point is whose" at a glance,
without needing per-pitcher-per-pitch-type legend entries.

No spread indicators (ellipses, std dev, etc.) on any of the three
maps here -- with two pitchers' points already overlaid on one
chart, adding spread on top would make it noticeably harder to
read, not easier.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_loader import (
    load_final_scores, load_movement_points,
    load_location_means, load_release_means,
    load_pitcher_hands,
    all_pitcher_names
)
from styling import get_pitch_type_color, apply_chart_theme, render_pitcher_indicator, render_handedness_badge


st.title("Pitcher Comparison")
render_pitcher_indicator()
st.caption(
    "Compare two pitchers' PAOM scores and arsenals side by side. "
    "Color = pitch type (consistent across both pitchers). "
    "Shape = which pitcher."
)


# ============================================================
# PITCHER SELECTION
# ============================================================

names = all_pitcher_names()

col1, col2 = st.columns(2)

with col1:
    pitcher_a = st.selectbox(
        "Pitcher A", options=names, index=0, key="compare_pitcher_a"
    )

with col2:
    default_b_index = 1 if len(names) > 1 else 0
    pitcher_b = st.selectbox(
        "Pitcher B", options=names, index=default_b_index, key="compare_pitcher_b"
    )

if pitcher_a == pitcher_b:
    st.warning("Pick two different pitchers to compare.")
    st.stop()

SHAPE_A = "circle"
SHAPE_B = "diamond"

st.divider()


# ============================================================
# PAOM SCORE COMPARISON
# ============================================================

st.subheader("PAOM Scores")

scores = load_final_scores()

row_a = scores[scores["player_name"] == pitcher_a]
row_b = scores[scores["player_name"] == pitcher_b]

score_cols = st.columns(2)

pitcher_hands = load_pitcher_hands()

for col, name, row in zip(score_cols, [pitcher_a, pitcher_b], [row_a, row_b]):
    with col:
        st.markdown(f"**{name}**")
        if pitcher_hands is not None:
            hand_row = pitcher_hands[pitcher_hands["player_name"] == name]
            if len(hand_row) > 0:
                render_handedness_badge(hand_row.iloc[0]["p_throws"])
        if len(row) == 0:
            st.info("No PAOM score available (didn't clear minimum sample size).")
        else:
            r = row.iloc[0]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("PAOM", f"{r['paom_score']:.1f}")
            m2.metric("Eff.", f"{r['pitcher_effectiveness']:.1f}")
            m3.metric("Mvmt.", f"{r['trusted_movement_score']:.1f}")
            m4.metric("Cmd.", f"{r['pitcher_command']:.1f}")
            m5.metric("Vel.", f"{r['trusted_velocity_score']:.1f}")

st.divider()


# ============================================================
# SHARED LEGEND HELPER
# ============================================================

def add_pitcher_shape_legend(fig, name_a, name_b):
    """
    Two invisible-data, legend-only traces explaining the shape
    convention (gray marker, since color is reserved for pitch
    type elsewhere on the chart) -- avoids a cluttered legend with
    one entry per pitcher x pitch type combination.
    """
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol=SHAPE_A, color="#9CA3AF", size=12),
        name=name_a, showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol=SHAPE_B, color="#9CA3AF", size=12),
        name=name_b, showlegend=True
    ))


def add_pitcher_points(fig, df, x_col, y_col, name, shape, label_col="pitch_type",
                        hover_extra=None):
    """
    Adds one marker per row (pitch type) for a single pitcher,
    color-coded by pitch type, using the given marker shape to
    distinguish this pitcher from the other one on the same chart.
    """
    for _, row in df.iterrows():
        pitch_type = row[label_col]
        color = get_pitch_type_color(pitch_type)

        hover = f"{name} - {pitch_type}"
        if hover_extra:
            hover += f"<br>{hover_extra(row)}"

        fig.add_trace(go.Scatter(
            x=[row[x_col]], y=[row[y_col]],
            mode="markers+text",
            marker=dict(
                size=14, color=color, symbol=shape,
                line=dict(color="#E8EAED", width=1)
            ),
            text=[pitch_type],
            textposition="top center",
            showlegend=False,
            hovertemplate=hover + "<extra></extra>"
        ))


# ============================================================
# TABS: MOVEMENT / LOCATION / RELEASE
# ============================================================

tab_movement, tab_location, tab_release = st.tabs(
    ["Movement", "Location", "Release"]
)


# ------------------------------------------------------------
# MOVEMENT TAB
# ------------------------------------------------------------

with tab_movement:
    movement_points = load_movement_points()

    if movement_points is None:
        st.info("Movement map data not found.")
    else:
        p_a = movement_points[movement_points["player_name"] == pitcher_a]
        p_b = movement_points[movement_points["player_name"] == pitcher_b]

        if len(p_a) == 0 and len(p_b) == 0:
            st.info("No movement data available for either pitcher.")
        else:
            fig = go.Figure()

            add_pitcher_points(
                fig, p_a, "HB", "IVB", pitcher_a, SHAPE_A,
                hover_extra=lambda r: f"usage: {r['usage']:.0%}, velo: {r['velo']:.1f}"
            )
            add_pitcher_points(
                fig, p_b, "HB", "IVB", pitcher_b, SHAPE_B,
                hover_extra=lambda r: f"usage: {r['usage']:.0%}, velo: {r['velo']:.1f}"
            )
            add_pitcher_shape_legend(fig, pitcher_a, pitcher_b)

            fig.update_layout(
                xaxis_title="Horizontal Break (in)",
                yaxis_title="Induced Vertical Break (in)",
                height=550
            )
            apply_chart_theme(fig, zeroline=True)
            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Each point is one pitch type's average movement. "
                "Color = pitch type, shape = pitcher."
            )


# ------------------------------------------------------------
# LOCATION TAB
# ------------------------------------------------------------

with tab_location:
    location_means = load_location_means()

    if location_means is None:
        st.info("Location map data not found.")
    else:
        def collapse_to_all(df):
            """
            Collapses same/opposite matchup splits into one
            pitch-count-weighted point per pitch type, matching the
            "All" default on the single-pitcher detail page.
            """
            if len(df) == 0:
                return df
            return (
                df.groupby("pitch_type")
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

        p_a = collapse_to_all(
            location_means[location_means["player_name"] == pitcher_a]
        )
        p_b = collapse_to_all(
            location_means[location_means["player_name"] == pitcher_b]
        )

        if len(p_a) == 0 and len(p_b) == 0:
            st.info("No location data available for either pitcher.")
        else:
            fig = go.Figure()

            fig.add_shape(
                type="rect",
                x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
                line=dict(color="#E8EAED", width=2),
                fillcolor="rgba(0,0,0,0)"
            )

            add_pitcher_points(
                fig, p_a, "mean_plate_x", "mean_plate_z", pitcher_a, SHAPE_A,
                hover_extra=lambda r: f"{int(r['pitches']):,} real pitches"
            )
            add_pitcher_points(
                fig, p_b, "mean_plate_x", "mean_plate_z", pitcher_b, SHAPE_B,
                hover_extra=lambda r: f"{int(r['pitches']):,} real pitches"
            )
            add_pitcher_shape_legend(fig, pitcher_a, pitcher_b)

            fig.update_layout(
                xaxis_title="Horizontal Location (ft, catcher's view)",
                yaxis_title="Height (ft)",
                xaxis=dict(range=[-2.5, 2.5]),
                yaxis=dict(range=[0, 5]),
                height=550
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

            apply_chart_theme(fig, zeroline=False)
            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Each point is one pitch type's average location "
                "(combined across batter handedness). Rectangle is "
                "an approximate strike zone, not batter-specific."
            )


# ------------------------------------------------------------
# RELEASE TAB
# ------------------------------------------------------------

with tab_release:
    release_means = load_release_means()

    if release_means is None:
        st.info("Release map data not found.")
    else:
        p_a = release_means[release_means["player_name"] == pitcher_a]
        p_b = release_means[release_means["player_name"] == pitcher_b]

        if len(p_a) == 0 and len(p_b) == 0:
            st.info("No release data available for either pitcher.")
        else:
            fig = go.Figure()

            add_pitcher_points(
                fig, p_a, "mean_release_x", "mean_release_z", pitcher_a, SHAPE_A,
                hover_extra=lambda r: f"{int(r['pitches']):,} real pitches"
            )
            add_pitcher_points(
                fig, p_b, "mean_release_x", "mean_release_z", pitcher_b, SHAPE_B,
                hover_extra=lambda r: f"{int(r['pitches']):,} real pitches"
            )
            add_pitcher_shape_legend(fig, pitcher_a, pitcher_b)

            fig.update_layout(
                xaxis_title="Release Position, Horizontal (ft)",
                yaxis_title="Release Position, Height (ft)",
                height=550
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

            apply_chart_theme(fig, zeroline=False)
            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Each point is one pitch type's average release "
                "position. Color = pitch type, shape = pitcher."
            )
