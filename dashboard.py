"""
Smart Grid AI Project - Module 8: Executive Dashboard
------------------------------------------------------------
Covers:
  - National Grid Health
  - Live Energy Flow
  - Renewable Production
  - Demand Forecast
  - High Risk Zones
  - Power Loss Analytics
  - Carbon Dashboard
  - Incident Map

Usage: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="National Smart Energy Grid Dashboard", layout="wide")

st.title("National Smart Energy Grid Intelligence Platform")
st.caption("Executive Dashboard - Real-time Grid Monitoring & AI Insights")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_grid_data.csv", parse_dates=["timestamp"])
    failures = pd.read_csv("data/equipment_failures.csv", parse_dates=["timestamp"])
    return df, failures

df, failures = load_data()
df["renewable_mw"] = df["solar_production_mw"] + df["wind_production_mw"]

# ============================================================
# 1. NATIONAL GRID HEALTH (Top KPI row)
# ============================================================
st.header("National Grid Health")

latest = df.iloc[-1]
failure_rate = failures["is_failure"].mean() * 100
grid_health_score = max(0, 100 - failure_rate * 10)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Grid Health Score", f"{grid_health_score:.1f}/100",
            "Healthy" if grid_health_score > 80 else "Needs Attention")
col2.metric("Current Demand", f"{latest['total_demand_mw']:.0f} MW")
col3.metric("Renewable Supply", f"{latest['renewable_mw']:.0f} MW")
col4.metric("Battery Level", f"{latest['battery_level_pct']:.0f}%")

st.divider()

# ============================================================
# 2. LIVE ENERGY FLOW + DEMAND FORECAST
# ============================================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Live Energy Flow (Last 7 Days)")
    recent = df.tail(24 * 7)
    st.line_chart(recent.set_index("timestamp")[["total_demand_mw", "renewable_mw"]])

with col_right:
    st.subheader("Demand Forecast Pattern (Full Year)")
    monthly = df.set_index("timestamp")["total_demand_mw"].resample("D").mean()
    st.line_chart(monthly)

st.divider()

# ============================================================
# 3. RENEWABLE PRODUCTION
# ============================================================
st.header("Renewable Production")
col1, col2, col3 = st.columns(3)
col1.metric("Total Solar (Year)", f"{df['solar_production_mw'].sum()/1000:,.0f} GWh")
col2.metric("Total Wind (Year)", f"{df['wind_production_mw'].sum()/1000:,.0f} GWh")
renewable_pct = (df["renewable_mw"].sum() / df["total_demand_mw"].sum()) * 100
col3.metric("Renewable Share", f"{renewable_pct:.1f}%")

recent_renewable = df.tail(24 * 3).set_index("timestamp")[["solar_production_mw", "wind_production_mw"]]
st.bar_chart(recent_renewable)

st.divider()

# ============================================================
# 4. HIGH RISK ZONES + INCIDENT MAP (simplified table-based)
# ============================================================
st.header("High Risk Zones / Incident Overview")

risk_by_equipment = failures.groupby("equipment_id").agg(
    total_readings=("is_failure", "count"),
    failure_count=("is_failure", "sum")
).reset_index()
risk_by_equipment["failure_rate_pct"] = (risk_by_equipment["failure_count"] / risk_by_equipment["total_readings"]) * 100
high_risk = risk_by_equipment[risk_by_equipment["failure_count"] > 0].sort_values("failure_count", ascending=False)

col_map, col_list = st.columns([2, 1])
with col_map:
    st.subheader("Incident Map (Equipment Failure Locations - Simulated Grid Positions)")
    np.random.seed(1)
    map_data = pd.DataFrame({
        "lat": 30.0 + np.random.randn(len(high_risk)) * 2,
        "lon": 70.0 + np.random.randn(len(high_risk)) * 2,
    })
    st.map(map_data)

with col_list:
    st.subheader("Top High-Risk Equipment")
    st.dataframe(high_risk.head(10)[["equipment_id", "failure_count", "failure_rate_pct"]], hide_index=True)

st.divider()

# ============================================================
# 5. POWER LOSS ANALYTICS
# ============================================================
st.header("Power Loss Analytics")
# Simplified: transmission loss assumed ~3-5% of total demand
df["estimated_loss_mw"] = df["total_demand_mw"] * 0.04
total_loss = df["estimated_loss_mw"].sum()
col1, col2 = st.columns(2)
col1.metric("Estimated Annual Transmission Loss", f"{total_loss/1000:,.0f} GWh")
col2.metric("Loss as % of Total Demand", "4.0%")

st.divider()

# ============================================================
# 6. CARBON DASHBOARD
# ============================================================
st.header("Carbon Dashboard")
EMISSION_FACTOR = 0.45
df["non_renewable_mw"] = (df["total_demand_mw"] - df["renewable_mw"]).clip(lower=0)
df["carbon_tons"] = df["non_renewable_mw"] * EMISSION_FACTOR
total_carbon = df["carbon_tons"].sum()

col1, col2 = st.columns(2)
col1.metric("Total Carbon Emissions (Year)", f"{total_carbon:,.0f} tons CO2")
col2.metric("Renewable Offset", f"{renewable_pct:.1f}% demand covered by clean energy")

monthly_carbon = df.set_index("timestamp")["carbon_tons"].resample("ME").sum()
st.bar_chart(monthly_carbon)

st.caption("Smart Grid AI Platform | National Smart Energy Grid Intelligence & Demand Forecasting")

# ============================================================
# 7. DIGITAL TWIN (Simulation Mode)
# ============================================================
st.divider()
st.header("Digital Twin - Grid Simulation Mode")
st.caption("Adjust the sliders below to simulate different grid conditions and see live system response.")

import pickle

sim_col1, sim_col2, sim_col3 = st.columns(3)
with sim_col1:
    sim_hour = st.slider("Hour of Day", 0, 23, 12)
    sim_temp = st.slider("Temperature (°C)", -5, 45, 25)
with sim_col2:
    sim_cloud = st.slider("Cloud Cover (%)", 0, 100, 30)
    sim_wind = st.slider("Wind Speed (m/s)", 0.0, 20.0, 8.0)
with sim_col3:
    sim_weekend = st.selectbox("Day Type", ["Weekday", "Weekend"])
    sim_month = st.slider("Month", 1, 12, 6)

is_weekend_val = 1 if sim_weekend == "Weekend" else 0
day_of_week_val = 5 if is_weekend_val else 1

try:
    with open("models/demand_hourly.pkl", "rb") as f:
        demand_model = pickle.load(f)
    with open("models/renewable_solar.pkl", "rb") as f:
        solar_model = pickle.load(f)
    with open("models/renewable_wind.pkl", "rb") as f:
        wind_model = pickle.load(f)

    demand_input = pd.DataFrame([{
        "hour": sim_hour, "day_of_week": day_of_week_val, "month": sim_month,
        "is_weekend": is_weekend_val, "temperature_c": sim_temp,
        "cloud_cover_pct": sim_cloud, "wind_speed_mps": sim_wind
    }])
    renewable_input = pd.DataFrame([{
        "hour": sim_hour, "month": sim_month, "temperature_c": sim_temp,
        "cloud_cover_pct": sim_cloud, "wind_speed_mps": sim_wind
    }])

    predicted_demand = demand_model.predict(demand_input)[0]
    predicted_solar = solar_model.predict(renewable_input)[0]
    predicted_wind = wind_model.predict(renewable_input)[0]
    predicted_renewable = predicted_solar + predicted_wind
    grid_gap = max(0, predicted_demand - predicted_renewable)

    st.subheader("Simulated Outcome")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Predicted Demand", f"{predicted_demand:.0f} MW")
    r2.metric("Predicted Renewable Supply", f"{predicted_renewable:.0f} MW")
    r3.metric("Grid Power Needed", f"{grid_gap:.0f} MW")
    coverage = min(100, (predicted_renewable / predicted_demand) * 100) if predicted_demand > 0 else 0
    r4.metric("Renewable Coverage", f"{coverage:.1f}%")

    if grid_gap > predicted_demand * 0.9:
        st.error("⚠️ High Risk: Grid under significant stress in this scenario - emergency protocols would activate.")
    elif coverage > 30:
        st.success("✅ Good renewable coverage in this scenario.")
    else:
        st.info("ℹ️ Normal grid operation expected in this scenario.")

except FileNotFoundError:
    st.warning("Run demand_forecasting.py and renewable_forecasting.py first to enable the Digital Twin simulator.")

