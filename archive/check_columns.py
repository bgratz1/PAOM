import pandas as pd

needed = ["vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z", "release_pos_x", "release_pos_y", "release_pos_z"]

# just read the header row -- no need to load the whole file
cols = pd.read_csv("statcast_2025_raw.csv", nrows=0).columns.tolist()

print("Total columns in raw file:", len(cols))
print()
for c in needed:
    print(f"{c:20s} {'FOUND' if c in cols else 'MISSING'}")
