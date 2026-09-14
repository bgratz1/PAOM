"""
74_raw_statcast_pull_by_year.py

Purpose:
--------
Direct extension of the ORIGINAL raw pull script (confirmed by the
user, previously used for 2025's statcast_2025_raw.csv) to any other
season -- same monthly-chunk methodology, unchanged, just
parameterized by year instead of hardcoded to 2025.

This REPLACES 73_effectiveness_raw_pull.py's approach entirely --
73 used a single bulk date-range call and its OWN reconstructed
rate definitions, which turned out to not match the real original
(hard_hit_rate denominator, ground-ball definition). This script
matches the proven original exactly.

Output:
-------
statcast_{YEAR}_raw.csv -- raw, pitch-by-pitch Statcast data for the
given season, monthly chunks concatenated, matching the original
2025 file's exact structure.
"""

from pybaseball import statcast, cache
import pandas as pd
import time

cache.enable()


# ============================================================
# SETTINGS -- change YEAR and re-run for each season needed
# ============================================================

import sys
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

# same month boundaries as the original 2025 pull, shifted to YEAR --
# regular season typically late March/early April through September,
# matching the original's late-March start and September end (the
# original's 2025 pull also included October, covering postseason;
# kept here for consistency)
MONTHS = [
    (f"{YEAR}-03-27", f"{YEAR}-04-30"),
    (f"{YEAR}-05-01", f"{YEAR}-05-31"),
    (f"{YEAR}-06-01", f"{YEAR}-06-30"),
    (f"{YEAR}-07-01", f"{YEAR}-07-31"),
    (f"{YEAR}-08-01", f"{YEAR}-08-31"),
    (f"{YEAR}-09-01", f"{YEAR}-09-28"),
    (f"{YEAR}-10-01", f"{YEAR}-10-31"),
]

OUTPUT_FILE = f"statcast_{YEAR}_raw.csv"


# ============================================================
# PULL, MONTH BY MONTH -- same pattern as the original
# ============================================================

frames = []

for start, end in MONTHS:
    print(f"Downloading {start} - {end}")
    df = statcast(start_dt=start, end_dt=end)
    frames.append(df)
    time.sleep(5)

raw = pd.concat(frames, ignore_index=True)

raw.to_csv(OUTPUT_FILE, index=False)

print(raw.shape)
print(f"Saved: {OUTPUT_FILE}")
