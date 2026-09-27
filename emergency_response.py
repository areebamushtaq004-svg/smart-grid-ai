"""
Smart Grid AI Project - Module 6: Emergency Response Engine
------------------------------------------------------------------
Covers (rule-based recommendations using outputs from other modules):
  - Load Redistribution
  - Backup Power Activation
  - Controlled Load Shedding
  - Emergency Dispatch
  - Maintenance Scheduling

Usage: python emergency_response.py
"""

import pandas as pd
import numpy as np

print("Loading data...")
df = pd.read_csv("data/cleaned_grid_data.csv", parse_dates=["timestamp"])
failures = pd.read_csv("data/equipment_failures.csv", parse_dates=["timestamp"])

GRID_CAPACITY_MW = 16000  # assumed max grid capacity (based on max observed demand + margin)
LOW_RENEWABLE_THRESHOLD = 300  # MW
HIGH_LOAD_THRESHOLD = 95  # % load


def emergency_response_engine(demand, renewable_supply, battery_pct, load_pct, failure_risk):
    """
    Rule-based Emergency Response Engine.
    Sab modules (Demand Forecast, Renewable Forecast, Failure Prediction) ke
    outputs ko dekh kar automatic recommendations deta hai.
    """
    recommendations = []

    total_supply_estimate = renewable_supply + (GRID_CAPACITY_MW - demand) * 0  # grid tops up rest
    grid_needed = max(0, demand - renewable_supply)

    # 1. Controlled Load Shedding
    if demand > GRID_CAPACITY_MW * 0.95:
        recommendations.append({
            "action": "Controlled Load Shedding",
            "reason": f"Demand ({demand:.0f} MW) grid capacity ke 95% se zyada hai",
            "priority": "CRITICAL"
        })

    # 2. Backup Power Activation
    if renewable_supply < LOW_RENEWABLE_THRESHOLD and battery_pct < 20:
        recommendations.append({
            "action": "Backup Power Activation",
            "reason": f"Renewable supply kam hai ({renewable_supply:.0f} MW) aur battery bhi low hai ({battery_pct:.0f}%)",
            "priority": "HIGH"
        })

    # 3. Load Redistribution
    if load_pct > HIGH_LOAD_THRESHOLD:
        recommendations.append({
            "action": "Load Redistribution",
            "reason": f"Ek area mein load {load_pct:.0f}% hai - doosray kam-load regions mein shift karo",
            "priority": "HIGH"
        })

    # 4. Emergency Dispatch
    if failure_risk == 1:
        recommendations.append({
            "action": "Emergency Dispatch",
            "reason": "Equipment failure predict hui hai - maintenance team turant bhejo",
            "priority": "CRITICAL"
        })

    # 5. Maintenance Scheduling (preventive, lower priority)
    if 0.5 < load_pct / 100 <= 0.95 and failure_risk == 0:
        recommendations.append({
            "action": "Maintenance Scheduling",
            "reason": "Load moderate hai - is waqt preventive maintenance schedule karna behtar hai",
            "priority": "LOW"
        })

    if not recommendations:
        recommendations.append({
            "action": "No Action Needed",
            "reason": "Sab parameters normal range mein hain",
            "priority": "NONE"
        })

    return recommendations


# ============================================================
# Test the engine on sample scenarios from the data
# ============================================================
print("\n" + "=" * 60)
print("EMERGENCY RESPONSE ENGINE - Sample Scenarios")
print("=" * 60)

# Merge failure info into main df on nearest timestamp (simplified: sample rows)
sample_rows = df.sample(5, random_state=42).reset_index(drop=True)
failure_sample = failures.sample(5, random_state=1).reset_index(drop=True)

for i in range(5):
    row = sample_rows.iloc[i]
    frow = failure_sample.iloc[i]

    demand = row["total_demand_mw"]
    renewable = row["solar_production_mw"] + row["wind_production_mw"]
    battery_pct = row["battery_level_pct"]
    load_pct = frow["load_pct"]
    failure_risk = frow["is_failure"]

    print(f"\n--- Scenario {i+1} ({row['timestamp']}) ---")
    print(f"Demand: {demand:.0f} MW | Renewable: {renewable:.0f} MW | "
          f"Battery: {battery_pct:.0f}% | Load: {load_pct:.0f}% | Failure risk: {bool(failure_risk)}")

    recs = emergency_response_engine(demand, renewable, battery_pct, load_pct, failure_risk)
    for r in recs:
        print(f"  [{r['priority']}] {r['action']} -> {r['reason']}")

# Force a critical scenario to demonstrate all recommendation types
print("\n--- Simulated CRITICAL Scenario (forced high load + failure) ---")
recs = emergency_response_engine(demand=15800, renewable_supply=150, battery_pct=10, load_pct=98, failure_risk=1)
for r in recs:
    print(f"  [{r['priority']}] {r['action']} -> {r['reason']}")

print("\n" + "=" * 60)
print("Module 6 (Emergency Response Engine) complete.")
print("=" * 60)
