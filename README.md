# PAOM Dashboard

A Streamlit dashboard for the Pitch Arsenal Optimization Model.

## Setup

1. Place this whole folder (`app.py`, `data_loader.py`, `views/`,
   `requirements.txt`) directly inside your PAOM project folder --
   the same folder where your pipeline scripts (`01_...py` through
   `16_...py`) write their output CSVs. The dashboard reads those
   CSVs directly from that folder.

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run:
   ```
   streamlit run app.py
   ```

## Structure

This uses Streamlit's modern `st.Page` + `st.navigation` API
(rather than the older automatic `pages/` directory convention).
That older convention has documented, version- and OS-dependent
bugs with `st.switch_page` -- if you ever hit
`StreamlitAPIException: Could not find page: ...`, it's almost
always that older mechanism; this structure avoids it.

- `app.py` -- entrypoint. Defines the three pages and runs
  whichever one is currently active.
- `views/leaderboard.py` -- league-wide `paom_score` ranking,
  with a search box to jump to any pitcher's detail page. This is
  the default page.
- `views/pitcher_detail.py` -- component score breakdown,
  movement map, Usage/Drop/Add recommendations, platoon splits,
  and similar-pitcher comps for one pitcher.
- `views/similar_pitchers_map.py` -- 2D scatter of either
  mechanical (release) or arsenal-shape similarity across all
  pitchers.
- `data_loader.py` -- shared, cached data loading used by all
  three views.

## Required files

The dashboard needs `PAOM_final_scores.csv` and
`master_pitch_table_2025.csv` to run at all. Every other file is
optional -- if a given output (e.g. `PAOM_platoon_component.csv`)
isn't present, that section of the Pitcher Detail page will show
an info message instead of failing.

## Notes

- Data is cached with `st.cache_data`. If you re-run pipeline
  scripts and want the dashboard to pick up fresh output, restart
  the Streamlit app (or use the "Clear cache" option in the app's
  menu).
- Tested with Streamlit 1.60 using the `AppTest` framework --
  including the full select-a-pitcher -> click -> switch_page
  navigation flow -- against representative synthetic data
  covering every input file. Zero exceptions.
