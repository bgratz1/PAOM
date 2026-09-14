"""
89_run_paom_recommendations_all_years.py

Purpose:
--------
Loops 86_pairwise_metrics_by_year.py -> 87_pitcher_similarity_by_year
.py -> 88_recommendation_engine_by_year.py across all six years
(2020-2025), so this doesn't require running nine commands by hand.

Same error-handling pattern as 79/79b/84: a failed year is logged
and skipped, not allowed to crash the rest of the run. Order matters
within each year -- 87 needs 86's output, 88 needs both.
"""

import subprocess
import sys
import time

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

STAGES = [
    "86_pairwise_metrics_by_year.py",
    "87_pitcher_similarity_by_year.py",
    "88_recommendation_engine_by_year.py",
]

results = {}

print(f"Running {STAGES} for {len(YEARS)} years: {YEARS}\n")

overall_start = time.time()

for year in YEARS:
    print(f"\n{'='*60}")
    print(f"YEAR {year}")
    print(f"{'='*60}")

    year_start = time.time()
    year_failed_at = None

    for stage in STAGES:
        print(f"\n--- Running {stage} for {year} ---")
        result = subprocess.run(
            [sys.executable, stage, str(year)],
            capture_output=False
        )
        if result.returncode != 0:
            print(f"\nFAILED: {stage} for {year} (exit code {result.returncode})")
            year_failed_at = stage
            break

    year_elapsed = time.time() - year_start

    if year_failed_at is None:
        results[year] = {"status": "success", "elapsed_seconds": year_elapsed}
        print(f"\n{year}: SUCCESS ({year_elapsed:.0f}s)")
    else:
        results[year] = {"status": "failed", "failed_at": year_failed_at, "elapsed_seconds": year_elapsed}
        print(f"\n{year}: FAILED at {year_failed_at} ({year_elapsed:.0f}s) -- moving to next year")

overall_elapsed = time.time() - overall_start


# ============================================================
# SUMMARY
# ============================================================

print(f"\n\n{'='*60}")
print("FULL RUN SUMMARY")
print(f"{'='*60}")
print(f"\nTotal time: {overall_elapsed/60:.1f} minutes")

n_success = sum(1 for r in results.values() if r["status"] == "success")
n_failed = sum(1 for r in results.values() if r["status"] == "failed")

print(f"\nSucceeded: {n_success} of {len(YEARS)} years")
print(f"Failed: {n_failed} of {len(YEARS)} years")

for year, r in results.items():
    if r["status"] == "success":
        print(f"  {year}: OK ({r['elapsed_seconds']:.0f}s)")
    else:
        print(f"  {year}: FAILED at {r['failed_at']} ({r['elapsed_seconds']:.0f}s)")

if n_failed > 0:
    failed_years = [y for y, r in results.items() if r["status"] == "failed"]
    print(f"\nTo retry just the failed years, run each stage manually for those years, e.g.:")
    for y in failed_years:
        print(f"  python 86_pairwise_metrics_by_year.py {y}")
