"""
Smart Grid AI Project - Module 1: Smart Meter Data Processing
------------------------------------------------------------------
Covers:
  - Smart Meter Data Collection (loading raw data)
  - Data Cleaning (missing values, duplicates, outliers)
  - Consumption Analysis (usage patterns)
  - Load Aggregation (hourly -> daily/weekly)
  - Streaming Data Processing (simulated real-time feed)

Usage: python data_processing.py
Output: data/cleaned_grid_data.csv + printed analysis
"""

import pandas as pd
import numpy as np
import time

# ============================================================
# 1. SMART METER DATA COLLECTION
# ============================================================
print("=" * 55)
print("1. SMART METER DATA COLLECTION")
print("=" * 55)
raw_df = pd.read_csv("data/grid_demand_renewable.csv", parse_dates=["timestamp"])
print(f"Collected {len(raw_df)} smart meter readings.")
print(f"Date range: {raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}")

# ============================================================
# 2. DATA CLEANING
# ============================================================
print("\n" + "=" * 55)
print("2. DATA CLEANING")
print("=" * 55)

df = raw_df.copy()

# a) Missing values check & fill
missing_before = df.isnull().sum().sum()
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
print(f"Missing values found: {missing_before} -> filled with column mean")

# b) Duplicate rows remove
dupes_before = df.duplicated(subset=["timestamp"]).sum()
df = df.drop_duplicates(subset=["timestamp"])
print(f"Duplicate rows removed: {dupes_before}")

# c) Outlier clipping (values 3 std dev se zyada door -> clip)
for col in ["total_demand_mw", "solar_production_mw", "wind_production_mw"]:
    mean, std = df[col].mean(), df[col].std()
    lower, upper = mean - 3 * std, mean + 3 * std
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"Outliers clipped in '{col}': {outliers} values")

df.to_csv("data/cleaned_grid_data.csv", index=False)
print("Cleaned data saved to: data/cleaned_grid_data.csv")

# ============================================================
# 3. CONSUMPTION ANALYSIS
# ============================================================
print("\n" + "=" * 55)
print("3. CONSUMPTION ANALYSIS")
print("=" * 55)

avg_by_hour = df.groupby("hour")["total_demand_mw"].mean().sort_values(ascending=False)
print("Peak consumption hours (top 3):")
print(avg_by_hour.head(3).round(2))

avg_by_day = df.groupby("day_of_week")["total_demand_mw"].mean()
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
avg_by_day.index = days
print("\nAverage demand by day of week:")
print(avg_by_day.round(2))

weekday_avg = df[df["is_weekend"] == 0]["total_demand_mw"].mean()
weekend_avg = df[df["is_weekend"] == 1]["total_demand_mw"].mean()
print(f"\nWeekday average demand: {weekday_avg:.2f} MW")
print(f"Weekend average demand: {weekend_avg:.2f} MW")
print(f"Residential vs Industrial split: "
      f"{df['residential_demand_mw'].mean():.0f} MW (Residential) / "
      f"{df['industrial_demand_mw'].mean():.0f} MW (Industrial)")

# ============================================================
# 4. LOAD AGGREGATION
# ============================================================
print("\n" + "=" * 55)
print("4. LOAD AGGREGATION")
print("=" * 55)

agg_df = df.set_index("timestamp")

daily_load = agg_df["total_demand_mw"].resample("D").sum()
weekly_load = agg_df["total_demand_mw"].resample("W").sum()
monthly_load = agg_df["total_demand_mw"].resample("ME").sum()

daily_load.to_csv("data/daily_load_aggregated.csv")
weekly_load.to_csv("data/weekly_load_aggregated.csv")
monthly_load.to_csv("data/monthly_load_aggregated.csv")

print(f"Daily aggregation: {len(daily_load)} days -> data/daily_load_aggregated.csv")
print(f"Weekly aggregation: {len(weekly_load)} weeks -> data/weekly_load_aggregated.csv")
print(f"Monthly aggregation: {len(monthly_load)} months -> data/monthly_load_aggregated.csv")

# ============================================================
# 5. STREAMING DATA PROCESSING (Simulated Real-Time Feed)
# ============================================================
print("\n" + "=" * 55)
print("5. STREAMING DATA PROCESSING (Simulation)")
print("=" * 55)
print("Simulating real-time smart meter data stream (10 readings)...\n")


def stream_meter_readings(dataframe, n_readings=10, delay=0.2):
    """
    Generator function jo real-time streaming simulate karta hai.
    Real deployment mein ye Kafka/MQTT se aata, yahan hum simulate kar rahe hain.
    """
    for _, row in dataframe.head(n_readings).iterrows():
        yield {
            "timestamp": str(row["timestamp"]),
            "demand_mw": round(row["total_demand_mw"], 2),
            "solar_mw": round(row["solar_production_mw"], 2),
            "wind_mw": round(row["wind_production_mw"], 2),
        }
        time.sleep(delay)


for reading in stream_meter_readings(df, n_readings=5, delay=0.1):
    print(f"[STREAM] {reading}")

print("\nModule 1 (Smart Meter Data Processing) complete.")
