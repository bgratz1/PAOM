"""
app.py

PAOM Dashboard -- entrypoint.

Run with:
    streamlit run app.py

Expects this file (and the views/ folder) to sit in the same
directory as your PAOM pipeline's output CSVs.

Uses st.Page + st.navigation (the modern, recommended multipage
API) rather than the older automatic pages/ folder convention --
the older convention paired with st.switch_page has documented,
version- and OS-dependent path-resolution bugs (see Streamlit
GitHub issues #8070, #8607 among others), which is what this
avoids.
"""

import streamlit as st

from styling import render_welcome_dialog

st.set_page_config(
    page_title="PAOM Dashboard",
    page_icon="⚾",
    layout="wide"
)

# Hide Streamlit's default chrome (the dev-oriented hamburger menu
# and "Made with Streamlit" footer) -- standard practice for a
# polished, portfolio-facing dashboard rather than an obviously
# framework-default page. Settings/theme access isn't needed by an
# end user here, since the app already ships its own real theme.
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# First-open orientation popup -- shown once per app session (resets
# on a fresh browser reload/restart, not remembered permanently
# across sessions). Can always be reopened later via the "About
# This Dashboard" button on the Leaderboard page.
if "has_seen_welcome" not in st.session_state:
    st.session_state["has_seen_welcome"] = True
    render_welcome_dialog()

leaderboard_page = st.Page(
    "views/leaderboard.py",
    title="Leaderboard",
    icon="🏆",
    default=True
)

top_recommendations_page = st.Page(
    "views/top_recommendations.py",
    title="Top Recommendations",
    icon="🎯"
)

pitcher_detail_page = st.Page(
    "views/pitcher_detail.py",
    title="Pitcher Detail",
    icon="🔍"
)

similar_map_page = st.Page(
    "views/similar_pitchers_map.py",
    title="Similar Pitchers Map",
    icon="🗺️"
)

comparison_page = st.Page(
    "views/pitcher_comparison.py",
    title="Pitcher Comparison",
    icon="⚖️"
)

pitch_comparison_page = st.Page(
    "views/pitch_comparison.py",
    title="Pitch Comparison",
    icon="🆚"
)

methodology_page = st.Page(
    "views/methodology.py",
    title="Methodology",
    icon="📖"
)

pg = st.navigation([
    leaderboard_page,
    top_recommendations_page,
    pitcher_detail_page,
    similar_map_page,
    comparison_page,
    pitch_comparison_page,
    methodology_page
])

pg.run()
