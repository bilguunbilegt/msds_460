import pandas as pd
import numpy as np

INPUT  = "taxi_data.csv"
OUTPUT = "hourly_zone_demand.csv"

# 1) Load
df = pd.read_csv(INPUT)

# 2) Normalize column names
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(r"\s+", "_", regex=True)
)

# 2.1) Ensure required columns exist (try common aliases if they don't)
required = {"trip_start_timestamp", "pickup_community_area"}
missing = required - set(df.columns)

if missing:
    aliases = {
        # trip_start_timestamp
        "trip_start": "trip_start_timestamp",
        "trip_start_time": "trip_start_timestamp",
        "start_timestamp": "trip_start_timestamp",
        "tpep_pickup_datetime": "trip_start_timestamp",  # NYC-style
        # pickup_community_area
        "pickup_community": "pickup_community_area",
        "pickup_area": "pickup_community_area",
        "pickup_community_area_id": "pickup_community_area",
    }
    for src, dest in aliases.items():
        if dest in missing and src in df.columns:
            df[dest] = df[src]

    still_missing = required - set(df.columns)
    if still_missing:
        raise ValueError(
            f"Missing required column(s): {sorted(still_missing)}. "
            f"Available columns: {list(df.columns)}"
        )

# Optional sanity: fleet size
if "taxi_id" in df.columns:
    print(f"Unique taxi IDs in the file: {df['taxi_id'].nunique()}")

# 3) Keep only what we need
df = df[["trip_start_timestamp", "pickup_community_area"]].copy()

# 4) Parse timestamp (be flexible) and clean pickup area
df["trip_start_timestamp"] = pd.to_datetime(
    df["trip_start_timestamp"], errors="coerce"
)
df["pickup_community_area"] = pd.to_numeric(
    df["pickup_community_area"], errors="coerce"
)

df = df.dropna(subset=["trip_start_timestamp", "pickup_community_area"])
df["pickup_community_area"] = df["pickup_community_area"].astype(int)

# 4.1) Derive date & hour
df["date"] = df["trip_start_timestamp"].dt.date
df["hour"] = df["trip_start_timestamp"].dt.hour

# 5) Count trips per (date, area, hour)
daily = (
    df.groupby(["date", "pickup_community_area", "hour"], as_index=False)
      .size()
      .rename(columns={"size": "trips"})
)

# 6) Typical-day demand = mean over dates for each (area, hour)
demand = (
    daily.groupby(["pickup_community_area", "hour"], as_index=False)["trips"]
         .mean()
         .rename(columns={"trips": "demand"})
)

# 6.1) Ensure all (area, hour) combos appear
areas = np.sort(df["pickup_community_area"].unique())
hours = np.arange(24, dtype=int)
full = (
    pd.MultiIndex.from_product([areas, hours],
                               names=["pickup_community_area", "hour"])
      .to_frame(index=False)
)

demand = full.merge(demand, on=["pickup_community_area", "hour"], how="left")
demand["demand"] = demand["demand"].fillna(0).round(3)

# 7) Save
demand.to_csv(OUTPUT, index=False)
print(f"Saved: {OUTPUT} (typical-day mean trips/hour)")
