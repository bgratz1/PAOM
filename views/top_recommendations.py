"""
views/top_recommendations.py

PAOM Dashboard -- league-wide "Top Recommendations" tab. Aggregates
the historical-precedent engine's real, pre-computed recommendations
(94_batch_precompute_historical_recommendations.py's output) across
every pitcher, ranked by a CONFIDENCE-WEIGHTED score rather than raw
predicted_change alone.

WHY CONFIDENCE-WEIGHTED, NOT RAW MAGNITUDE:
A recommendation backed by 150 real historical comps and one backed
by 41 comps shouldn't compete purely on predicted magnitude -- 65's
real calibration work (this project's own validation) confirmed
error genuinely, if modestly, differs by evidence volume (MAE 0.0266
at the lowest tier vs. 0.0240 at the highest, walk-forward validated,
~10% relative difference). This surfaces that finding directly in
the ranking: each recommendation's magnitude is shrunk toward zero
(no predicted benefit) by a modest, tier-based factor before
sorting, so a large-but-thin-evidence prediction doesn't
automatically dominate over a smaller, better-supported one.

The shrinkage factors below are DELIBERATELY MODEST (0.70/0.85/1.0),
matching how modest the real, underlying calibration effect actually
was -- this mirrors the same "low-key, not high/low confidence"
labeling discipline already used for confidence_tier itself, applied
now to how tiers affect ranking, not just how they're displayed.

Raw predicted_change is ALWAYS shown alongside the weighted score --
nothing is hidden, only re-ordered.
"""

import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_historical_recommendations, load_historical_recommendation_comps
from styling import format_pitch_type, render_pitch_type_key, render_pitcher_indicator


st.title("Top Recommendations")
render_pitcher_indicator()
st.caption(
    "League-wide, ranked by a confidence-weighted score -- a "
    "recommendation backed by more real historical comps is "
    "weighted more heavily than one of similar predicted magnitude "
    "backed by less real evidence, not sorted by raw magnitude "
    "alone."
)


# ============================================================
# CONFIDENCE-WEIGHTING
# ============================================================

# Modest, real-evidence-grounded shrinkage -- NOT a hard filter.
# Mirrors 65_confidence_calibration.py's own real finding (a clean,
# monotonic, but modest ~10% relative MAE difference across tiers,
# walk-forward validated) rather than an arbitrary or dramatic
# discount.
CONFIDENCE_WEIGHTS = {
    "Strong evidence": 1.00,
    "Moderate evidence": 0.85,
    "Limited evidence": 0.70,
    "n/a": 0.0,  # NO_CHANGE rows -- excluded from this ranked view
                  # entirely below, this weight is never actually used
}


def compute_weighted_score(row, lower_is_better=True):
    """
    Shrinks predicted_change toward zero (no predicted benefit) by
    the tier's real weight -- a genuinely large predicted
    improvement from a thin-evidence recommendation ends up
    comparable to a smaller improvement from a well-supported one,
    rather than automatically outranking it.
    """
    weight = CONFIDENCE_WEIGHTS.get(row["confidence_tier"], 0.70)
    return row["predicted_change"] * weight


# ============================================================
# LOAD AND FILTER
# ============================================================

hist_recs = load_historical_recommendations()

if hist_recs is None:
    st.info(
        "Historical-precedent recommendation data not found -- run "
        "94_batch_precompute_historical_recommendations.py first."
    )
    st.stop()

# Only real candidates (exclude NO_CHANGE rows, and only each
# pitcher's OWN top-ranked real candidate -- this view is "best
# recommendation per pitcher, ranked league-wide", not every
# candidate for every pitcher)
real_candidates = hist_recs[hist_recs["change_type"] != "NO_CHANGE"].copy()
top_per_pitcher = (
    real_candidates.sort_values("rank")
    .groupby("player_name", as_index=False)
    .first()
)

top_per_pitcher["weighted_score"] = top_per_pitcher.apply(compute_weighted_score, axis=1)

# lower predicted_change = better (xwoba_against convention, matching
# 45's own LOWER_IS_BETTER_METRICS) -- so a lower (more negative)
# weighted_score ranks higher
top_per_pitcher = top_per_pitcher.sort_values("weighted_score", ascending=True).reset_index(drop=True)


# ============================================================
# FILTERS
# ============================================================

col1, col2 = st.columns([1, 2])

with col1:
    min_tier = st.selectbox(
        "Minimum evidence tier",
        options=["Limited evidence", "Moderate evidence", "Strong evidence"],
        index=0,
        help="Filters out recommendations below this evidence level entirely."
    )

tier_order = {"Limited evidence": 0, "Moderate evidence": 1, "Strong evidence": 2}
min_tier_rank = tier_order[min_tier]
top_per_pitcher = top_per_pitcher[
    top_per_pitcher["confidence_tier"].map(tier_order).fillna(-1) >= min_tier_rank
]

with col2:
    change_type_filter = st.multiselect(
        "Change type",
        options=["ADD", "DROP", "USAGE_INCREASE", "USAGE_DECREASE", "SWAP"],
        default=["ADD", "DROP", "USAGE_INCREASE", "USAGE_DECREASE", "SWAP"]
    )

top_per_pitcher = top_per_pitcher[top_per_pitcher["change_type"].isin(change_type_filter)]

N_SHOW = st.slider("Number to show", min_value=10, max_value=100, value=25, step=5)


# ============================================================
# DISPLAY
# ============================================================

st.divider()
st.subheader(f"Top {min(N_SHOW, len(top_per_pitcher))} recommendations")

CONFIDENCE_ICONS = {
    "Strong evidence": "🟢",
    "Moderate evidence": "🟡",
    "Limited evidence": "⚪",
}

display_df = top_per_pitcher.head(N_SHOW).copy()
display_df["Evidence"] = display_df["confidence_tier"].map(
    lambda t: f"{CONFIDENCE_ICONS.get(t, '⚪')} {t}"
)
display_df["pitch_type"] = display_df["pitch_type"].apply(format_pitch_type)

display_cols = display_df[[
    "player_name", "change_type", "pitch_type", "estimated_usage_delta",
    "predicted_change", "weighted_score", "Evidence", "n_historical_events"
]].rename(columns={
    "player_name": "Pitcher",
    "change_type": "Change",
    "pitch_type": "Pitch",
    "estimated_usage_delta": "Usage Change (pp)",
    "predicted_change": "Predicted Change (xwOBA, lower=better)",
    "weighted_score": "Weighted Score (xwOBA, lower=better)",
    "n_historical_events": "Real Comparison Events"
})

st.dataframe(
    display_cols.style.format({
        "Usage Change (pp)": lambda v: f"{v:+.1f}" if pd.notna(v) else "-",
        "Predicted Change (xwOBA, lower=better)": "{:+.3f}",
        "Weighted Score (xwOBA, lower=better)": "{:+.3f}",
        "Real Comparison Events": lambda v: f"{v:.0f}" if pd.notna(v) else "-"
    }),
    width='stretch', hide_index=True, height=600
)

st.caption(
    "\"Usage Change (pp)\" is in percentage points. Predicted and "
    "weighted change are both xwOBA-against, where lower is better."
)

render_pitch_type_key()

st.caption(
    "\"Predicted Change\" is the historical engine's direct "
    "output -- unmodified. \"Weighted Score\" is what this ranking "
    "actually sorts by, after the modest, tier-based shrinkage "
    "described above. A pitcher only appears once here, showing "
    "their own single best real candidate -- see the Pitcher Detail "
    "page for that pitcher's full ranked list."
)


# ============================================================
# JUMP TO A PITCHER
# ============================================================

st.divider()
selected = st.selectbox(
    "View a pitcher's full detail page",
    options=display_df["player_name"].tolist(),
    index=None,
    placeholder="Pick a pitcher from the table above..."
)

if selected:
    if st.button(f"View {selected}", type="primary"):
        st.session_state["selected_pitcher"] = selected
        st.switch_page("views/pitcher_detail.py")
