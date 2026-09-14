"""
styling.py

Shared visual constants used across dashboard pages -- primarily a
FIXED pitch-type-to-color mapping, so e.g. "SL" is always the same
color everywhere in the dashboard, not just consistent within a
single pitcher's own chart. This matters most for the Pitcher
Comparison page, where two different pitchers' arsenals are
overlaid on one chart and need genuinely comparable colors -- but
it also fixes a latent inconsistency on the single-pitcher detail
page, where colors were previously assigned per-pitcher based on
alphabetical order within that pitcher's own arsenal (meaning "SL"
could be a different color on two different pitchers' charts).
"""

_PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
    "#FF97FF", "#FECB52", "#8C564B", "#17BECF"
]

# canonical, fixed order -- common MLB pitch type codes first, so
# they get stable, well-separated colors; anything else falls back
# to a deterministic hash-based assignment from the same palette
_KNOWN_PITCH_TYPES = [
    "FF", "SI", "SL", "CH", "CU", "FC", "ST", "FS",
    "KC", "SV", "CS", "EP", "KN", "SC", "FO"
]

_KNOWN_COLOR_MAP = {
    pt: _PALETTE[i % len(_PALETTE)]
    for i, pt in enumerate(_KNOWN_PITCH_TYPES)
}


def get_pitch_type_color(pitch_type):
    """
    Returns a fixed color for a given pitch type code. Known MLB
    codes get a stable, well-separated color from _KNOWN_COLOR_MAP.
    Anything not in that list still gets a consistent color (same
    input always produces the same output) via a deterministic
    hash into the same palette, rather than crashing or defaulting
    to a single fallback color for every unknown code.
    """
    if pitch_type in _KNOWN_COLOR_MAP:
        return _KNOWN_COLOR_MAP[pitch_type]
    idx = sum(ord(c) for c in str(pitch_type)) % len(_PALETTE)
    return _PALETTE[idx]


def hex_to_rgba(hex_color, alpha=0.15):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ============================================================
# PITCH TYPE FULL NAMES
#
# Standard MLB/Statcast pitch type codes, mapped to their real,
# properly capitalized full names -- used to make abbreviations
# readable across the dashboard (e.g. "SL" -> "Slider"), and to
# power the shared Pitch Type Key component below.
# ============================================================

PITCH_NAME_MAP = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "SL": "Slider",
    "CH": "Changeup",
    "CU": "Curveball",
    "FC": "Cutter",
    "ST": "Sweeper",
    "FS": "Splitter",
    "KC": "Knuckle Curve",
    "SV": "Slurve",
    "CS": "Slow Curve",
    "EP": "Eephus",
    "KN": "Knuckleball",
    "SC": "Screwball",
    "FO": "Forkball",
    "FA": "Fastball",
    "UN": "Unknown",
    "PO": "Pitch Out",
}


def get_pitch_type_full_name(pitch_type):
    """
    Returns the real, full pitch name for a given code (e.g. "SL"
    -> "Slider"). Falls back to just the raw code, unchanged, for
    anything not in PITCH_NAME_MAP -- never crashes or hides an
    unrecognized code.
    """
    return PITCH_NAME_MAP.get(pitch_type, str(pitch_type))


def format_pitch_type(pitch_type):
    """
    "SL" -> "SL (Slider)" -- the standard, dashboard-wide format for
    showing a pitch type with its full name attached, used anywhere
    a raw abbreviation would otherwise appear alone. Falls back to
    just the code if it's not a recognized type.
    """
    full_name = get_pitch_type_full_name(pitch_type)
    if full_name == str(pitch_type):
        return str(pitch_type)
    return f"{pitch_type} ({full_name})"


def render_pitch_type_key():
    """
    A small, reusable "Pitch Type Key" expander -- every abbreviation
    used anywhere on the dashboard, mapped to its real full name.
    Meant to be dropped into any page that displays pitch type codes,
    repeated per-page rather than centralized, so it's always
    available without navigating away. Imports streamlit locally
    (not at module level) since styling.py is otherwise pure
    constants/helpers with no UI dependency of its own.
    """
    import streamlit as st

    with st.expander("Pitch Type Key"):
        key_rows = [
            {"Code": code, "Full Name": name}
            for code, name in sorted(PITCH_NAME_MAP.items())
        ]
        st.dataframe(key_rows, width='stretch', hide_index=True)


def format_player_name(name):
    """
    Capitalizes a player name for display -- some real data sources
    on this dashboard use their own lowercase "first last" naming
    (the historical-precedent engine's own Kaggle-sourced data),
    while others use PAOM's own, already-capitalized "Last, First"
    style. Python's str.title() handles both correctly and safely:
    confirmed idempotent on already-correct names (e.g. "Verlander,
    Justin" stays exactly "Verlander, Justin"), and confirmed correct
    on real, tricky lowercase examples from this project's actual
    data -- accented characters ("josé cisnero" -> "José Cisnero"),
    apostrophes ("o'neil cruz" -> "O'Neil Cruz"), and multi-initial
    names ("j. p. france" -> "J. P. France"). Safe to apply
    everywhere a player name is displayed, regardless of which
    source it came from. Returns the input unchanged if it isn't a
    string (e.g. NaN).
    """
    if not isinstance(name, str):
        return name
    return name.title()


def render_welcome_dialog():
    """
    The dashboard's first-open orientation popup -- a real Streamlit
    modal (st.dialog), not just styled inline text. Organized as
    tabs internally since there's genuinely a lot of ground to cover
    (purpose, PAOM explained, every dashboard tab, how scores are
    built, reliability/confidence) -- one continuous scroll would be
    overwhelming. Deliberately brief on methodology depth: each
    metric gets a plain-language summary here, with a pointer to
    where the FULL real explanation already lives on that metric's
    own page, rather than duplicating it.

    Called from two places: app.py (auto-shown once per session, via
    a session_state flag), and a manual "About This Dashboard"
    button on the Leaderboard page, so it can always be reopened
    after being closed.
    """
    import streamlit as st

    @st.dialog("Welcome to the PAOM Dashboard", width="large")
    def _dialog():
        tab_overview, tab_tabs, tab_scores, tab_reliability = st.tabs(
            ["Overview", "The Tabs", "How Scores Are Built", "Reliability & Confidence"]
        )

        with tab_overview:
            st.markdown(
                """
This dashboard has two real, connected purposes:

1. **Rate how good a pitcher's current arsenal is** -- PAOM Score,
   a single 0-100 number combining pitch effectiveness, arsenal
   movement coverage, command, and velocity into one overall
   quality rating.

2. **Recommend real, evidence-based changes to that arsenal** --
   this is the dashboard's main feature. Rather than a hand-written
   rule ("low effectiveness means drop it"), recommendations come
   from a real, validated engine that finds actual historical
   pitchers who made a similar change and predicts from what
   genuinely happened to them. Every recommendation is grounded in
   real precedent, not a guess.

Everything else on this dashboard exists to support one of those
two goals -- rating the current arsenal, or recommending what to
do about it.
                """
            )

        with tab_tabs:
            st.markdown(
                """
**Leaderboard** -- every pitcher's PAOM Score, filterable by season,
sorted and compared league-wide.

**Top Recommendations** -- the best real arsenal-change
recommendations across the whole league at once, weighted by how
much real evidence backs each one.

**Pitcher Detail** -- everything about one pitcher: their PAOM
Score and components, how their arsenal has evolved across real
seasons, and their full, ranked list of recommended changes with
the real historical comps behind each one. **This is where the
recommendation engine's full output lives.**

**Similar Pitchers Map** -- who a pitcher is mechanically and
stylistically similar to, using several different real similarity
measures (mechanics, arsenal shape, release point, and the
recommendation engine's own similarity space).

**Pitcher Comparison / Pitch Comparison** -- put two pitchers, or
two specific pitches, directly side by side across every available
real metric.
                """
            )

        with tab_scores:
            st.markdown(
                """
Brief summaries -- each page with a score has its own expander
with the full, real methodology; this is just the plain-language
version.

**Effectiveness** -- how good a pitch actually performs (whiffs,
called strikes, hard contact allowed, ground balls), modeled
against real outcomes.

**Movement** -- how much distinct movement-shape territory a
pitcher's whole arsenal covers.

**Command** -- how consistently a pitcher locates a pitch where
they're aiming.

**Velocity** -- a pitcher's velocity profile, including how much
range they have across their arsenal.

**PAOM Score** -- a weighted combination of all four, each first
converted to a 0-100 percentile scale so they're directly
comparable.

**Recommendation predictions** -- built from real historical
pitchers who made a similar change, weighted by how similar they
are to the pitcher in question, not a fixed formula applied to
everyone alike.
                """
            )

        with tab_reliability:
            st.markdown(
                """
**Confidence** -- how much real data backs a score, 0-100. For
PAOM Score, this is the *minimum* of all four components' own
confidence -- a score is only as trustworthy as its weakest input.
A low number means the score leans more on league-average
shrinkage than this pitcher's own real data -- it's a reliability
indicator, not a quality judgment.

**Evidence tiers on recommendations** (Limited / Moderate / Strong)
-- how many real historical comparison events back a specific
recommendation. More real events generally means a more trustworthy
prediction.

**Disclosed flags** -- some recommendations carry an explicit
warning (like an unusually high-usage pitch being dropped) rather
than being silently excluded or silently allowed. These are shown
so you can weigh them yourself, not hidden either way.

Full detail on any of these lives on the page where that number
actually appears -- this is just the map to get you there.
                """
            )

        if st.button("Got it", type="primary"):
            st.session_state["has_seen_welcome"] = True
            st.rerun()

    _dialog()


# ============================================================
# UNIFIED CHART THEME
# ============================================================

CHART_FONT_FAMILY = "Arial, sans-serif"
CHART_GRIDLINE_COLOR = "rgba(255,255,255,0.12)"
CHART_ZEROLINE_COLOR = "#6B7280"
CHART_TEXT_COLOR = "#E8EAED"


def apply_chart_theme(fig, zeroline=False):
    """
    Applies a single, consistent visual theme across every Plotly
    chart on the dashboard -- font, background, and gridline
    treatment -- so charts look like they belong to one product
    rather than having been built independently, page by page,
    across a long development session. Call this on a figure right
    before st.plotly_chart(), after any chart-specific layout
    settings (titles, axis ranges, etc.) are already set, so this
    doesn't need to duplicate or guess at those.

    Backgrounds are transparent rather than a hardcoded color, so
    every chart automatically follows the dashboard's real theme
    (.streamlit/config.toml) rather than needing to be kept in sync
    with it manually.

    zeroline: pass True for charts where a real zero line is
    meaningful (e.g. movement charts, where 0 horizontal/vertical
    break is a real, interpretable reference point) -- matches the
    zeroline convention already used on several of these charts.
    """
    fig.update_layout(
        font=dict(family=CHART_FONT_FAMILY, size=12, color=CHART_TEXT_COLOR),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=40, l=50, r=30),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=CHART_GRIDLINE_COLOR,
        zeroline=zeroline, zerolinewidth=2, zerolinecolor=CHART_ZEROLINE_COLOR,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=CHART_GRIDLINE_COLOR,
        zeroline=zeroline, zerolinewidth=2, zerolinecolor=CHART_ZEROLINE_COLOR,
    )
    return fig


# ============================================================
# PERSISTENT "CURRENTLY VIEWING" INDICATOR
# ============================================================

def render_pitcher_indicator():
    """
    A small, persistent indicator showing which pitcher is currently
    selected (st.session_state["selected_pitcher"]), with a quick
    link back to their full Pitcher Detail page -- meant for pages
    OTHER than Pitcher Detail itself (that page already shows the
    pitcher's name as its own title, so this would be redundant
    there). Shows nothing if no pitcher has been selected yet, since
    an empty or placeholder indicator would just be visual clutter
    on a first visit.
    """
    import streamlit as st

    selected = st.session_state.get("selected_pitcher")
    if not selected:
        return

    col_indicator, col_link = st.columns([4, 1])
    with col_indicator:
        st.caption(f"👤 Currently viewing: **{selected}**")
    with col_link:
        if st.button("Full profile →", key="pitcher_indicator_link"):
            st.switch_page("views/pitcher_detail.py")


# ============================================================
# HANDEDNESS BADGE
# ============================================================

def render_handedness_badge(p_throws):
    """
    Renders a real pitcher's throwing hand as a small, visually
    distinct badge (st.badge) rather than plain text -- "L"/"R" is
    genuinely useful at a glance, especially on comparison pages
    where two pitchers' handedness can differ and directly affects
    how to read a movement or release chart. Handles missing/
    unknown handedness gracefully rather than crashing or showing
    a blank badge.
    """
    import streamlit as st

    if p_throws == "L":
        st.badge("Left-Handed", icon="🫱", color="blue")
    elif p_throws == "R":
        st.badge("Right-Handed", icon="🫲", color="orange")
    else:
        st.badge("Handedness unknown", color="gray")
