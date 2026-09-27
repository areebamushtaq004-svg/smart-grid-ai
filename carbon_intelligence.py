"""
Smart Grid AI Project - Module 7: Carbon Intelligence
------------------------------------------------------------
Covers:
  - Carbon Emissions
  - Renewable Energy Usage
  - Energy Efficiency Score
  - Sustainability Report

Usage: python carbon_intelligence.py
Output: docs/sustainability_report.txt
"""

import pandas as pd
import numpy as np

print("Loading data...")
df = pd.read_csv("data/cleaned_grid_data.csv", parse_dates=["timestamp"])

# Emission factor: kitni CO2 (tons) per MWh non-renewable generation se paida hoti hai
# (Ye industry-standard average hai fossil-fuel grid mix ke liye)
EMISSION_FACTOR_TON_PER_MWH = 0.45

# ============================================================
# 1. CARBON EMISSIONS
# ============================================================
print("\n" + "=" * 55)
print("1. CARBON EMISSIONS")
print("=" * 55)

df["renewable_mw"] = df["solar_production_mw"] + df["wind_production_mw"]
df["non_renewable_mw"] = (df["total_demand_mw"] - df["renewable_mw"]).clip(lower=0)
df["carbon_emissions_tons"] = df["non_renewable_mw"] * EMISSION_FACTOR_TON_PER_MWH

total_emissions = df["carbon_emissions_tons"].sum()
avg_daily_emissions = df.groupby(df["timestamp"].str.slice(0, 10) if df["timestamp"].dtype == object
                                  else pd.to_datetime(df["timestamp"]).dt.date)["carbon_emissions_tons"].sum().mean()

print(f"Total carbon emissions (1 year): {total_emissions:,.0f} tons CO2")
print(f"Average daily emissions: {avg_daily_emissions:,.0f} tons CO2/day")

# ============================================================
# 2. RENEWABLE ENERGY USAGE
# ============================================================
print("\n" + "=" * 55)
print("2. RENEWABLE ENERGY USAGE")
print("=" * 55)

total_demand = df["total_demand_mw"].sum()
total_renewable = df["renewable_mw"].sum()
renewable_pct = (total_renewable / total_demand) * 100

print(f"Total demand served: {total_demand:,.0f} MWh")
print(f"Total renewable generated: {total_renewable:,.0f} MWh")
print(f"Renewable Energy Usage: {renewable_pct:.2f}% of total demand")

solar_pct = (df["solar_production_mw"].sum() / total_demand) * 100
wind_pct = (df["wind_production_mw"].sum() / total_demand) * 100
print(f"  - Solar contribution: {solar_pct:.2f}%")
print(f"  - Wind contribution: {wind_pct:.2f}%")

# ============================================================
# 3. ENERGY EFFICIENCY SCORE
# ============================================================
print("\n" + "=" * 55)
print("3. ENERGY EFFICIENCY SCORE")
print("=" * 55)

# Simplified efficiency score: renewable usage + low variance in battery usage + low failure rate
failures = pd.read_csv("data/equipment_failures.csv")
failure_rate = failures["is_failure"].mean()

# Score out of 100: renewable_pct (40%) + (1-failure_rate) (30%) + battery utilization stability (30%)
battery_stability = 1 - (df["battery_level_pct"].std() / 100)
battery_stability = np.clip(battery_stability, 0, 1)

efficiency_score = (
    (renewable_pct / 100) * 40 +
    (1 - failure_rate) * 30 +
    battery_stability * 30
)
print(f"Renewable usage component: {(renewable_pct/100)*40:.1f} / 40")
print(f"Reliability component (low failure rate): {(1-failure_rate)*30:.1f} / 30")
print(f"Battery stability component: {battery_stability*30:.1f} / 30")
print(f"\nOverall Energy Efficiency Score: {efficiency_score:.1f} / 100")

# ============================================================
# 4. SUSTAINABILITY REPORT
# ============================================================
print("\n" + "=" * 55)
print("4. SUSTAINABILITY REPORT")
print("=" * 55)

report = f"""
SMART GRID SUSTAINABILITY REPORT
==================================
Reporting Period: {df['timestamp'].min()} to {df['timestamp'].max()}

CARBON EMISSIONS
  Total CO2 Emissions: {total_emissions:,.0f} tons
  Average Daily Emissions: {avg_daily_emissions:,.0f} tons/day

RENEWABLE ENERGY
  Renewable Usage: {renewable_pct:.2f}% of total demand
  Solar Contribution: {solar_pct:.2f}%
  Wind Contribution: {wind_pct:.2f}%

GRID RELIABILITY
  Equipment Failure Rate: {failure_rate*100:.2f}%
  Battery Utilization Stability: {battery_stability*100:.1f}%

OVERALL EFFICIENCY SCORE: {efficiency_score:.1f} / 100

RECOMMENDATIONS
  - {"Increase solar/wind capacity to further reduce carbon footprint" if renewable_pct < 20 else "Renewable adoption is progressing well"}
  - {"Investigate high failure rate equipment for preventive maintenance" if failure_rate > 0.02 else "Grid reliability is within acceptable range"}
  - Continue monitoring battery charge cycles for optimal storage efficiency
"""

print(report)

with open("docs/sustainability_report.txt", "w") as f:
    f.write(report)
print("Saved: docs/sustainability_report.txt")

print("\n" + "=" * 55)
print("Module 7 (Carbon Intelligence) complete.")
print("=" * 55)
