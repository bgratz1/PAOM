"""
79_run_all_years.py

Purpose:
--------
COMPONENTS-ONLY VERSION: skips 74 (raw pull) and 75 (cleaning) --
both already exist for all six years from the earlier full run.
Re-running them here would waste real time re-pulling 4+ million
pitches already on disk. Runs only the four component-scoring
stages (77 -> 78 -> 76 -> 80 -> 81) against the EXISTING statcast_
{YEAR}_raw.csv / clean_statcast_{YEAR}.csv files. If those files
are missing for a given year, this will fail at 77 -- run
74_raw_statcast_pull_by_year.py and 75_clean_statcast_by_year.py
for that year first.

HONEST SCALE WARNING: each year's raw pull (74) alone can take real
time (a full season is 700,000+ pitches). Running all six years back
to back is a genuinely long operation -- expect this to run for a
significant while, not minutes.

ERROR HANDLING, deliberately: if any stage fails for a given year
(e.g. a network hiccup during the raw pull), that year is logged as
failed and the loop moves on to the NEXT year rather than crashing
the whole run -- a bad year shouldn't cost you the years that
already succeeded or the years still to come. A full summary prints
at the end showing exactly which years fully succeeded and which
didn't, so failed years can be re-run individually afterward
(each script still works standalone with a single year argument,
e.g. `python 74_raw_statcast_pull_by_year.py 2021`).

Order matters: 77 (master pitch table) depends on 75's output, and
78 (velocity) and 80 (movement) both depend on 77's output. 76
(effectiveness) and 81 (command) only need 75's output directly, so
they're placed wherever convenient in the sequence, not by a real
dependency requirement.
"""

import subprocess
import sys
import time

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

STAGES = [
    "77_master_pitch_table_by_year.py",
    "78_velocity_by_year.py",
    "76_effectiveness_by_year.py",
    "80_movement_by_year.py",
    "81_command_by_year.py",
]

results = {}

print(f"Running full pipeline for {len(YEARS)} years: {YEARS}")
print(f"Stages per year: {STAGES}")
print("\nHONEST WARNING: this is a genuinely long operation -- each year's")
print("raw pull alone can take real time. Expect this to run for a while.\n")

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
            capture_output=False  # let output stream live, so
                                     # progress is visible during
                                     # long-running stages like 74
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
    print(
        f"\nTo retry just the failed years, run each stage manually "
        f"for those years, e.g.:"
    )
    for y in failed_years:
        print(f"  python 74_raw_statcast_pull_by_year.py {y}")
