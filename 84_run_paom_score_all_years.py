"""
84_run_paom_score_all_years.py

Purpose:
--------
Loops 83_paom_score_by_year.py across all six years, now that every
input it needs (Effectiveness, Movement, Command, Velocity, master
pitch table, cleaned Statcast) genuinely exists for 2020-2025.
Produces real, historical paom_score values -- the actual unlock
for Phase 3 of the roadmap (the blend test against the old
recommendation system).

Same error-handling pattern as 79/79b: a failed year is logged and
skipped, not allowed to crash the rest of the run.
"""

import subprocess
import sys
import time

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

results = {}

print(f"Running 83_paom_score_by_year.py for {len(YEARS)} years: {YEARS}\n")

overall_start = time.time()

for year in YEARS:
    print(f"\n{'='*60}")
    print(f"YEAR {year}")
    print(f"{'='*60}")

    year_start = time.time()

    result = subprocess.run(
        [sys.executable, "83_paom_score_by_year.py", str(year)],
        capture_output=False
    )

    year_elapsed = time.time() - year_start

    if result.returncode == 0:
        results[year] = {"status": "success", "elapsed_seconds": year_elapsed}
        print(f"\n{year}: SUCCESS ({year_elapsed:.0f}s)")
    else:
        results[year] = {"status": "failed", "elapsed_seconds": year_elapsed}
        print(f"\n{year}: FAILED (exit code {result.returncode}, {year_elapsed:.0f}s) -- moving to next year")

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
    status_label = "OK" if r["status"] == "success" else "FAILED"
    print(f"  {year}: {status_label} ({r['elapsed_seconds']:.0f}s)")

if n_failed > 0:
    failed_years = [y for y, r in results.items() if r["status"] == "failed"]
    print(f"\nTo retry just the failed years:")
    for y in failed_years:
        print(f"  python 83_paom_score_by_year.py {y}")
