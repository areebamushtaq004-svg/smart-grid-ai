"""
Smart Grid AI Project - Module 3: Renewable Energy Forecasting
--------------------------------------------------------------------
Covers:
  - Solar Production
  - Wind Generation
  - Battery Storage Levels
  - Weather Impact (correlation analysis)
  - Renewable Availability (combined solar + wind forecast)

Usage: python renewable_forecasting.py
Output: models/solar.pkl, wind.pkl, battery.pkl + weather impact report
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import os

os.makedirs("models", exist_ok=True)

print("Loading cleaned data...")
df = pd.read_csv("data/cleaned_grid_data.csv", parse_dates=["timestamp"])

feature_cols = ["hour", "month", "temperature_c", "cloud_cover_pct", "wind_speed_mps"]


def train_model(target_col, model_name, save_path):
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{model_name:25s} | MAE: {mae:8.2f} | R²: {r2:.4f}")
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    return model, preds, y_test


print(f"\n{'Model':25s} | {'MAE':>12s} | {'R² Score'}")
print("-" * 55)

# ============================================================
# 1. SOLAR PRODUCTION FORECAST
# ============================================================
solar_model, solar_preds, solar_actual = train_model(
    "solar_production_mw", "1. Solar Production", "models/renewable_solar.pkl"
)

# ============================================================
# 2. WIND GENERATION FORECAST
# ============================================================
wind_model, wind_preds, wind_actual = train_model(
    "wind_production_mw", "2. Wind Generation", "models/renewable_wind.pkl"
)

# ============================================================
# 3. BATTERY STORAGE LEVEL FORECAST
# ============================================================
battery_feature_cols = ["hour", "month", "solar_production_mw", "wind_production_mw", "total_demand_mw"]
X_batt = df[battery_feature_cols]
y_batt = df["battery_level_pct"]
X_train, X_test, y_train, y_test = train_test_split(X_batt, y_batt, test_size=0.2, random_state=42, shuffle=False)
battery_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
battery_model.fit(X_train, y_train)
batt_preds = battery_model.predict(X_test)
batt_mae = mean_absolute_error(y_test, batt_preds)
batt_r2 = r2_score(y_test, batt_preds)
print(f"{'3. Battery Storage Level':25s} | MAE: {batt_mae:8.2f} | R²: {batt_r2:.4f}")
with open("models/renewable_battery.pkl", "wb") as f:
    pickle.dump(battery_model, f)

# ============================================================
# 4. WEATHER IMPACT ANALYSIS
# ============================================================
print("\n" + "=" * 55)
print("4. WEATHER IMPACT ANALYSIS")
print("=" * 55)

weather_impact = pd.DataFrame({
    "Temperature vs Demand": [df["temperature_c"].corr(df["total_demand_mw"])],
    "Cloud Cover vs Solar": [df["cloud_cover_pct"].corr(df["solar_production_mw"])],
    "Wind Speed vs Wind Production": [df["wind_speed_mps"].corr(df["wind_production_mw"])],
}).T
weather_impact.columns = ["Correlation"]
print(weather_impact.round(3))
weather_impact.to_csv("data/weather_impact_analysis.csv")
print("\nSaved: data/weather_impact_analysis.csv")

# ============================================================
# 5. RENEWABLE AVAILABILITY (Combined Solar + Wind Forecast)
# ============================================================
print("\n" + "=" * 55)
print("5. RENEWABLE AVAILABILITY FORECAST")
print("=" * 55)

total_renewable_predicted = solar_preds + wind_preds
total_renewable_actual = solar_actual.values + wind_actual.values

avail_mae = mean_absolute_error(total_renewable_actual, total_renewable_predicted)
avail_r2 = r2_score(total_renewable_actual, total_renewable_predicted)
print(f"Combined Renewable Availability | MAE: {avail_mae:.2f} MW | R²: {avail_r2:.4f}")

avg_renewable_pct = (df["solar_production_mw"] + df["wind_production_mw"]) / df["total_demand_mw"] * 100
print(f"Average renewable coverage of total demand: {avg_renewable_pct.mean():.2f}%")

print("\n" + "=" * 55)
print("All Renewable Energy Forecasting models complete.")
print("=" * 55)
