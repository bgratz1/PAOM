# PAOM: Pitch Arsenal Optimization Model

Rating MLB pitch arsenals and recommending evidence-based changes, built from Baseball Savant Statcast data and a Kaggle-sourced arsenal-evolution dataset.

**Live dashboard:** https://paomdashboard.streamlit.app/

---

## What This Is

Pitch-quality metrics like Stuff+ and pitch-level xwOBA answer a narrow question: how good is this pitch, right now, in isolation? PAOM answers a different question: what should a pitcher actually do about their arsenal?

PAOM has two connected parts:

1. **PAOM Score**, a single 0-100 rating of how good a pitcher's current arsenal is, built from four components: Effectiveness, Movement, Command, and Velocity.
2. **A historical-precedent recommendation engine**, which finds similar pitchers who made a given kind of arsenal change in the past and predicts the outcome from what happened to them, instead of applying a fixed rule to every pitcher alike.

Both are built into a full interactive dashboard.

## Key Results

- **4,381** historical arsenal-change events across **6 MLB seasons** (2020-2025) back every recommendation the engine produces.
- **78.9% ranking concordance** on genuinely held-out future seasons for swap predictions, 64-71% for single-change predictions, against a 50% baseline for random chance.
- The choice of which outcome metric to predict was settled by direct evidence, not assumption: xwOBA-against showed 83.3% sign consistency in a direct test against WAR (56.3%) and FIP (58.3%).
- A prior, purely structural recommendation system was tested directly against real outcomes and found to underperform a simple baseline across every change type checked. It has been retired from the dashboard as a result.
- Every one of PAOM Score's four components is refit fresh each season; the weights, coefficients, and leaderboard all shift year to year rather than staying fixed.

See the full writeup for complete methodology, validation, and results.

## Repository Structure

```
PAOM/
├── app.py                     Dashboard entry point (Streamlit)
├── data_loader.py             Shared data loading utilities
├── styling.py                 Shared dashboard theme and UI helpers
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
├── views/                     Dashboard pages
│   ├── leaderboard.py
│   ├── top_recommendations.py
│   ├── pitcher_detail.py
│   ├── similar_pitchers_map.py
│   ├── pitcher_comparison.py
│   ├── pitch_comparison.py
│   └── methodology.py
├── archive/                    Earlier design iterations, kept for documented comparison
│   ├── 16_recommendation_engine.py   The original rule-based engine PAOM replaced
│   └── ...                          (other earlier, superseded scripts)
│
└── (repo root)                Data pipeline, recommendation engine, and validation
                                scripts, numbered roughly in build order. Not yet split
                                into subfolders. The pipeline runs 74-84 (Statcast pull
                                through PAOM Score, year-parameterized 2020-2025); the
                                recommendation engine is built from 34, 37, 38, 40, 42,
                                45, 50, 54, 60, 63, 70, 94, 96; everything else in that
                                range is a real validation, placebo test, or diagnostic
                                script referenced directly in the writeup.
```

Numbered scripts follow the rough chronological order they were built in. The full writeup explains the reasoning behind each major design decision, including the ones that were tried and rejected.

## Setup

```bash
pip install -r requirements.txt
```

Raw Statcast data is not stored in this repository. The pipeline scripts regenerate it directly from Baseball Savant.

```bash
python 74_raw_statcast_pull_by_year.py 2025
python 75_clean_statcast_by_year.py 2025
python 76_effectiveness_by_year.py 2025
python 77_master_pitch_table_by_year.py 2025
python 78_velocity_by_year.py 2025
python 80_movement_by_year.py 2025
python 81_command_by_year.py 2025
python 83_paom_score_by_year.py 2025
```

or run the full pipeline for a single season at once:

```bash
python 79_run_all_years.py
```

To launch the dashboard once component data exists:

```bash
streamlit run app.py
```

## Data Sources

- **Baseball Savant Statcast**, pulled via [pybaseball](https://github.com/jldbc/pybaseball), 2020-2025 MLB seasons.
- **MLB Pitcher Arsenal Evolution (2020-2025)**, a Kaggle dataset used for the recommendation engine's historical arsenal-change events. https://www.kaggle.com/datasets/yasunorim/mlb-pitcher-arsenal-2020-2025

## Methodology

The full writeup covers:

- How each of the four PAOM Score components is built and validated, including six years of real coefficient and weight trends
- How the recommendation engine finds comps, predicts outcomes, and ranks candidates across change types
- Placebo tests, walk-forward validation, and a direct comparison against the prior recommendation system
- Known limitations and disclosed gaps


## Author

Ben Gratz
