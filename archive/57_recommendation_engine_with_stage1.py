"""
57_recommendation_engine_with_stage1.py

Purpose:
--------
The actual payoff of the whole two-stage effort: 45's recommendation
engine, now using Stage 1's validated pitch-quality prediction for
ADD candidates instead of the base regression. Confirmed on TWO
independent real-data samples (55, 56) that stage1_quality_coef
carries real signal exceeding mean_reversion_coef itself -- the
first change-specific feature in this whole project to do so.

Built via dependency injection (45's add_prediction_fn parameter),
NOT by modifying 45 or 38 directly -- avoids the circular import
that would result from 45 trying to import from 54 (54 already
imports FROM 45, via 50).

Output:
-------
Printed ranked recommendations for a demonstration target pitcher,
using the Stage-1-integrated prediction for every ADD candidate.
"""

import importlib.util
import sys

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT 45 (fresh) AND 54 (fresh) INDEPENDENTLY
# ============================================================

print("Loading recommendation engine (45) and Stage 1 integration (54)...")

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

stage2_spec = importlib.util.spec_from_file_location(
    "stage2_integration", "54_stage2_integration.py"
)
stage2 = importlib.util.module_from_spec(stage2_spec)
sys.modules["stage2_integration"] = stage2
stage2_spec.loader.exec_module(stage2)

predict_arsenal_change_with_stage1 = stage2.predict_arsenal_change_with_stage1


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("\n\n==============================")
    print("DEMONSTRATION: recommendation engine WITH Stage 1 integration")
    print("==============================")

    demo_row = similarity_features.dropna(subset=SIMILARITY_FEATURE_COLS).iloc[
        len(similarity_features) // 3
    ]
    demo_player_id = demo_row["player_id"]
    demo_season = demo_row["season"]

    recommendations, result_objects = recommend_arsenal_changes(
        demo_player_id, demo_season, outcome_metric="xwoba_against",
        add_prediction_fn=predict_arsenal_change_with_stage1
    )

    if len(recommendations) > 0:
        print("\n\n==============================")
        print("Ranked recommendations (ADD candidates use Stage 1 integration)")
        print("==============================")
        print(recommendations.to_string(index=False))

        top_row = recommendations.iloc[0]
        top_key = (top_row["change_type"], top_row["pitch_type"])
        print(
            f"\n\nTop recommendation: {top_row['change_type']} {top_row['pitch_type']} "
            f"(estimated usage delta: {top_row['estimated_usage_delta']:+.1f}pp)"
        )
        if top_row["change_type"] == "ADD":
            top_result = result_objects[top_key]
            print(
                f"Stage 1 predicted quality for this new pitch: "
                f"{top_result.get('stage1_predicted_quality_used', 'n/a')}"
            )
        print(f"Most similar historical comps who made this change:")
        show_similar_comps(result_objects[top_key])
