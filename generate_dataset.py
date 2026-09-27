"""
Smart Grid AI Project - Synthetic Dataset Generator
-----------------------------------------------------
Ye script 1 saal ka hourly data generate karta hai:
1. Electricity demand (residential + industrial)
2. Solar power production
3. Wind power production
4. Battery storage level
5. Equipment failure events (anomalies)

Usage: python generate_dataset.py
Output: data/ folder mein CSV files ban jayengi
"""

import numpy as np
import pandas as pd
from datetime import datetime

np.random.seed(42)

# ----------------------------
# 1. Time range banao (1 saal, hourly)
# ----------------------------
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 12, 31, 23, 0, 0)
timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
n = len(timestamps)

print(f"Generating {n} hourly records for 1 year...")

df = pd.DataFrame({"timestamp": timestamps})
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# ----------------------------
# 2. Electricity Demand
# ----------------------------
hourly_pattern = 1 + 0.5 * np.sin((df["hour"] - 6) * np.pi / 12) + \
                  0.3 * np.sin((df["hour"] - 18) * np.pi / 6)
hourly_pattern = np.clip(hourly_pattern, 0.3, None)
seasonal_pattern = 1 + 0.25 * np.cos((df["month"] - 7) * np.pi / 6)
weekend_factor = np.where(df["is_weekend"] == 1, 0.85, 1.0)

base_residential = 5000
df["residential_demand_mw"] = (
    base_residential * hourly_pattern * seasonal_pattern * weekend_factor
    + np.random.normal(0, 150, n)
).clip(lower=500)

base_industrial = 3500
industrial_pattern = 1 + 0.2 * np.sin((df["hour"] - 10) * np.pi / 12)
df["industrial_demand_mw"] = (
    base_industrial * industrial_pattern * np.where(df["is_weekend"] == 1, 0.6, 1.0)
    + np.random.normal(0, 100, n)
).clip(lower=300)

df["total_demand_mw"] = df["residential_demand_mw"] + df["industrial_demand_mw"]

# ----------------------------
# 3. Weather Data
# ----------------------------
df["temperature_c"] = (
    20 + 10 * np.cos((df["month"] - 7) * np.pi / 6)
    + 5 * np.sin((df["hour"] - 14) * np.pi / 12)
    + np.random.normal(0, 2, n)
)
df["cloud_cover_pct"] = np.clip(
    30 + 40 * np.abs(np.sin(np.arange(n) / 50)) + np.random.normal(0, 15, n), 0, 100
)
df["wind_speed_mps"] = np.clip(
    8 + 4 * np.sin(np.arange(n) / 80) + np.random.normal(0, 2, n), 0, None
)

# ----------------------------
# 4. Solar Power Production
# ----------------------------
solar_daylight = np.clip(np.sin((df["hour"] - 6) * np.pi / 12), 0, None)
solar_capacity_mw = 1200
cloud_reduction = 1 - (df["cloud_cover_pct"] / 100) * 0.7
df["solar_production_mw"] = (
    solar_capacity_mw * solar_daylight * cloud_reduction + np.random.normal(0, 20, n)
).clip(lower=0)

# ----------------------------
# 5. Wind Power Production
# ----------------------------
wind_capacity_mw = 600
wind_power_raw = (df["wind_speed_mps"] ** 2) * 8
df["wind_production_mw"] = np.clip(wind_power_raw, 0, wind_capacity_mw) + np.random.normal(0, 10, n)
df["wind_production_mw"] = df["wind_production_mw"].clip(lower=0)

# ----------------------------
# 6. Battery Storage Level
# ----------------------------
battery_capacity_mwh = 800
battery_level = [battery_capacity_mwh * 0.5]
renewable_total = df["solar_production_mw"] + df["wind_production_mw"]
net_supply = renewable_total - df["total_demand_mw"] * 0.15

for i in range(1, n):
    change = net_supply.iloc[i] * 0.05
    new_level = np.clip(battery_level[-1] + change, 0, battery_capacity_mwh)
    battery_level.append(new_level)

df["battery_level_mwh"] = battery_level
df["battery_level_pct"] = (df["battery_level_mwh"] / battery_capacity_mwh) * 100

# ----------------------------
# 7. Equipment / Failure Data
# ----------------------------
df["equipment_id"] = np.random.choice([f"TRANSFORMER_{i:03d}" for i in range(1, 51)], size=n)
df["voltage_kv"] = 220 + np.random.normal(0, 3, n)
df["frequency_hz"] = 50 + np.random.normal(0, 0.1, n)
df["load_pct"] = np.clip(
    (df["total_demand_mw"] / df["total_demand_mw"].max()) * 100 + np.random.normal(0, 5, n), 0, 120
)

n_failures = int(n * 0.01)
failure_indices = np.random.choice(n, size=n_failures, replace=False)
df["is_failure"] = 0
df.loc[failure_indices, "is_failure"] = 1
df.loc[failure_indices, "voltage_kv"] += np.random.normal(15, 5, n_failures)
df.loc[failure_indices, "frequency_hz"] += np.random.normal(0.5, 0.2, n_failures)
df.loc[failure_indices, "load_pct"] = np.clip(
    df.loc[failure_indices, "load_pct"] + np.random.normal(25, 10, n_failures), 0, 150
)

# Failure type (for Failure Prediction module requirements: Transformer/Line/Voltage/Frequency/Overload)
failure_types = []
for idx in df.index:
    if df.loc[idx, "is_failure"] == 0:
        failure_types.append("none")
    else:
        v, f, l = df.loc[idx, "voltage_kv"], df.loc[idx, "frequency_hz"], df.loc[idx, "load_pct"]
        if l > 100:
            failure_types.append("equipment_overload")
        elif abs(v - 220) > 20:
            failure_types.append("voltage_instability")
        elif abs(f - 50) > 0.5:
            failure_types.append("frequency_deviation")
        else:
            failure_types.append("transformer_failure")
df["failure_type"] = failure_types

# ----------------------------
# 8. Save Files
# ----------------------------
main_cols = [
    "timestamp", "hour", "day_of_week", "month", "is_weekend",
    "residential_demand_mw", "industrial_demand_mw", "total_demand_mw",
    "temperature_c", "cloud_cover_pct", "wind_speed_mps",
    "solar_production_mw", "wind_production_mw",
    "battery_level_mwh", "battery_level_pct"
]
df[main_cols].to_csv("data/grid_demand_renewable.csv", index=False)

failure_cols = [
    "timestamp", "equipment_id", "voltage_kv", "frequency_hz",
    "load_pct", "is_failure", "failure_type"
]
df[failure_cols].to_csv("data/equipment_failures.csv", index=False)

print("\nDone! Files created:")
print("  - data/grid_demand_renewable.csv")
print("  - data/equipment_failures.csv")
print(f"\nTotal records: {n} | Failure events: {n_failures}")
