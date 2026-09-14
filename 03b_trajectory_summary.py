"""
03b_trajectory_summary.py

Purpose:
--------
Compute each pitch's position at the hitter's DECISION POINT --
not just at release or at the plate -- using the pitch's actual
flight kinematics. This is the foundation for a rigorous tunnel
differential: how similar do two pitches look at the point a
hitter has to commit to swinging, versus how different are they
by the time they cross the plate.

Statcast models pitch flight as constant acceleration from release
to plate. Given a pitch's initial position (release_pos_x/y/z),
initial velocity (vx0, vy0, vz0), and acceleration (ax, ay, az),
standard kinematics lets us solve for the ball's (x, z) position
at any distance from the plate -- including the decision point,
which Statcast does not report directly.

Coordinate convention:
-----------------------
Statcast's y-axis runs from the pitcher (y ~ 50-55 ft, release
point) toward home plate. plate_x / plate_z (already in the data)
are measured at y = 1.417 ft (front edge of the plate). We solve
for each pitch's position at y = 1.417 + DECISION_DISTANCE_FT --
i.e. DECISION_DISTANCE_FT in front of the plate, back toward the
pitcher, which approximates where a hitter must commit to a swing
decision.

DECISION_DISTANCE_FT = 23.8 is a commonly-cited approximation in
public tunneling research (roughly the point ~175ms before the
pitch would reach the plate for a typical fastball). It's a
deliberate simplification -- true swing-commit timing varies by
pitch speed and hitter -- but it's a fixed, consistent reference
point that lets every pitch be compared on equal footing.

Method:
-------
For each pitch, solve the kinematic quadratic
    0.5*ay*t^2 + vy0*t + (release_pos_y - y_decision) = 0
for t (time to reach the decision point), then compute
    x(t) = release_pos_x + vx0*t + 0.5*ax*t^2
    z(t) = release_pos_z + vz0*t + 0.5*az*t^2

Aggregate decision-point (x, z) and plate (x, z) to pitcher x
pitch type level (mean), matching the existing pipeline's
aggregation grain.

Output:
-------
pitch_trajectory_summary.csv
"""

import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "clean_statcast_2025.csv"
OUTPUT_FILE = "pitch_trajectory_summary.csv"

PLATE_Y = 1.417            # front edge of home plate, Statcast convention
DECISION_DISTANCE_FT = 23.8  # distance in front of plate, back toward pitcher

Y_DECISION = PLATE_Y + DECISION_DISTANCE_FT

MIN_PITCHES = 50


# ============================================================
# LOAD DATA
# ============================================================

print("Loading cleaned Statcast...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Loaded {len(df):,} pitches")


required_cols = [
    "player_name",
    "pitch_type",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "plate_x",
    "plate_z"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}. "
        f"Make sure 02_clean_statcast.py has been updated to keep "
        f"vx0/vy0/vz0/ax/ay/az and re-run it before this script."
    )


# ============================================================
# DROP ROWS WITH MISSING TRAJECTORY DATA
# ============================================================

df = df.dropna(subset=required_cols).copy()

print(f"Rows with complete trajectory data: {len(df):,}")


# ============================================================
# SOLVE FOR TIME TO DECISION POINT
# ============================================================

"""
Quadratic: 0.5*ay*t^2 + vy0*t + (y0 - y_decision) = 0

Solve with the standard quadratic formula, vectorized across
the whole dataframe. Physically, we want the smaller positive
root -- the ball passes through the decision-point plane once
on its way to home plate (a second mathematical root, if it
exists, corresponds to a time outside the physical flight).
"""

y0 = df["release_pos_y"].values
vy0 = df["vy0"].values
ay = df["ay"].values

a_coef = 0.5 * ay
b_coef = vy0
c_coef = y0 - Y_DECISION

discriminant = b_coef ** 2 - 4 * a_coef * c_coef

# guard against tiny negative discriminants from floating point noise
discriminant = np.clip(discriminant, 0, None)

sqrt_disc = np.sqrt(discriminant)

t1 = (-b_coef + sqrt_disc) / (2 * a_coef)
t2 = (-b_coef - sqrt_disc) / (2 * a_coef)

# take the smaller positive root; where ay is ~0, fall back to
# the linear approximation t = (y0 - y_decision) / -vy0
t_candidates = np.column_stack([t1, t2])
t_candidates = np.where(t_candidates > 0, t_candidates, np.inf)
t_decision = np.min(t_candidates, axis=1)

linear_fallback = (y0 - Y_DECISION) / -vy0

near_zero_accel = np.abs(ay) < 1e-6

t_decision = np.where(
    near_zero_accel | ~np.isfinite(t_decision),
    linear_fallback,
    t_decision
)

print(
    f"\nTime-to-decision-point summary (seconds):\n"
    f"{pd.Series(t_decision).describe()}"
)


# ============================================================
# POSITION AT DECISION POINT
# ============================================================

x0 = df["release_pos_x"].values
z0 = df["release_pos_z"].values
vx0 = df["vx0"].values
vz0 = df["vz0"].values
ax = df["ax"].values
az = df["az"].values

df["decision_x"] = x0 + vx0 * t_decision + 0.5 * ax * t_decision ** 2
df["decision_z"] = z0 + vz0 * t_decision + 0.5 * az * t_decision ** 2

# sanity filter -- decision point should occur before the pitch
# reaches the plate; drop physically invalid solves
df = df[
    np.isfinite(df["decision_x"])
    & np.isfinite(df["decision_z"])
].copy()

print(f"\nRows with valid decision-point solve: {len(df):,}")


# ============================================================
# AGGREGATE TO PITCHER x PITCH TYPE
# ============================================================

print("\nAggregating to pitcher x pitch type...")

summary = (
    df
    .groupby(["player_name", "pitch_type"])
    .agg(
        pitches=("pitch_type", "count"),
        decision_x=("decision_x", "mean"),
        decision_z=("decision_z", "mean"),
        plate_x=("plate_x", "mean"),
        plate_z=("plate_z", "mean")
    )
    .reset_index()
)

summary = summary[summary["pitches"] >= MIN_PITCHES].copy()

print(f"Pitcher x pitch type rows: {len(summary):,}")


# ============================================================
# SAVE
# ============================================================

summary = summary.round(4)

summary.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved: {OUTPUT_FILE}")
print("\nSample:")
print(summary.head())
