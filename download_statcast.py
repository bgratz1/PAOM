from pybaseball import statcast, cache
import pandas as pd
import time

cache.enable()

months = [
    ('2025-03-27', '2025-04-30'),
    ('2025-05-01', '2025-05-31'),
    ('2025-06-01', '2025-06-30'),
    ('2025-07-01', '2025-07-31'),
    ('2025-08-01', '2025-08-31'),
    ('2025-09-01', '2025-09-28'),
    ('2025-10-01', '2025-10-31'),
]

frames = []

for start, end in months:

    print(f"Downloading {start} - {end}")

    df = statcast(start_dt=start, end_dt=end)

    frames.append(df)

    time.sleep(5)

raw = pd.concat(frames, ignore_index=True)

raw.to_csv("statcast_2025_raw.csv", index=False)

print(raw.shape)