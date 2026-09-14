"""
63_recommendation_engine_fully_integrated.py

Purpose:
--------
The complete, fully-integrated recommendation engine: 45's search-
and-rank logic, with EVERY candidate type now using a validated
Stage 1 signal -- ADD via 54's regression-based prediction (since
the new pitch doesn't exist yet), DROP/USAGE_INCREASE/USAGE_DECREASE
via 60's direct-lookup extension (since those pitches already exist
and their real quality is directly observable). Both cleared their
own real backtests (58/61 for ADD, 62 for the other three) before
being wired in here as the new default.

predict_arsenal_change_with_stage1_all_types (60) correctly handles
ALL FOUR change types internally, so it's passed as BOTH the
add_prediction_fn and other_change_prediction_fn arguments -- one
function, used consistently everywhere a candidate is evaluated.

Output:
-------
Printed ranked recommendations for a demonstration target pitcher,
using the fully-integrated Stage 1 signal for every candidate type.
"""

import importlib.util
import sys

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT 45 (fresh) AND 60 (fresh) INDEPENDENTLY
# ============================================================

print("Loading recommendation engine (45) and all-types Stage 1 integration (60)...")

rec_spec = importlib.util.spec_from_file_location(
    "similarity_recommendation_engine", "45_similarity_recommendation_engine.py"
)
rec_engine = importlib.util.module_from_spec(rec_spec)
sys.modules["similarity_recommendation_engine"] = rec_engine
rec_spec.loader.exec_module(rec_engine)

recommend_arsenal_changes = rec_engine.recommend_arsenal_changes
show_similar_comps = rec_engine.show_similar_comps
similarity_features = rec_engine.similarity_features
SIMILARITY_FEATURE_COLS = rec_engine.SIMILARITY_FEATURE_COLS

stage60_spec = importlib.util.spec_from_file_location(
    "stage1_all_change_types", "60_stage1_all_change_types.py"
)
stage60 = importlib.util.module_from_spec(stage60_spec)
sys.modules["stage1_all_change_types"] = stage60
stage60_spec.loader.exec_module(stage60)

predict_arsenal_change_with_stage1_all_types = stage60.predict_arsenal_change_with_stage1_all_types


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("\n\n==============================")
    print("DEMONSTRATION: fully-integrated recommendation engine")
    print("==============================")

    demo_row = similarity_features.dropna(subset=SIMILARITY_FEATURE_COLS).iloc[
        len(similarity_features) // 3
    ]
    demo_player_id = demo_row["player_id"]
    demo_season = demo_row["season"]

    recommendations, result_objects = recommend_arsenal_changes(
        demo_player_id, demo_season, outcome_metric="xwoba_against",
        add_prediction_fn=predict_arsenal_change_with_stage1_all_types,
        other_change_prediction_fn=predict_arsenal_change_with_stage1_all_types
    )

    if len(recommendations) > 0:
        print("\n\n==============================")
        print("Ranked recommendations (every candidate type uses Stage 1)")
        print("==============================")
        print(recommendations.to_string(index=False))

        top_row = recommendations.iloc[0]
        top_key = (top_row["change_type"], top_row["pitch_type"])
        print(
            f"\n\nTop recommendation: {top_row['change_type']} {top_row['pitch_type']} "
            f"(estimated usage delta: {top_row['estimated_usage_delta']:+.1f}pp)"
        )
        if top_row["change_type"] != "NO_CHANGE":
            print(f"Most similar historical comps who made this change:")
            show_similar_comps(result_objects[top_key])
