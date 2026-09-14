import pandas as pd

arsenal = pd.read_csv("pitcher_arsenal_evolution_2020_2025.csv")
combined = pd.read_csv("pitcher_release_features_combined_2020_2025.csv")

broader = set(zip(arsenal["player_id"], arsenal["season"]))
covered = set(zip(combined["player_id"], combined["season"]))

true_coverage = len(broader & covered)
print(f"True coverage: {true_coverage:,} of {len(broader):,} ({true_coverage/len(broader):.1%})")

n_uncovered = len(broader - covered)
print(f"Still uncovered: {n_uncovered:,}")