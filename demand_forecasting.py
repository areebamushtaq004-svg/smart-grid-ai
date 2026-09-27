"""
Smart Grid AI Project - Module 2: Demand Forecasting
------------------------------------------------------------
Covers:
  - Hourly Demand
  - Daily Demand
  - Weekly Demand
  - Seasonal Demand
  - Industrial Demand
  - Residential Demand

Model: XGBoost (Time-Series Forecasting - AI Component requirement)

Usage: python demand_forecasting.py
Output: models/demand_*.pkl (6 trained models)
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
df = df.set_index("timestamp")


def train_model(X, y, model_name, save_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{model_name:30s} | MAE: {mae:10.2f} | R²: {r2:.4f}")
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    return r2


print(f"\n{'Model':30s} | {'MAE':>13s} | {'R² Score'}")
print("-" * 65)

results = {}

# 1. HOURLY DEMAND
feat_hourly = ["hour", "day_of_week", "month", "is_weekend", "temperature_c", "cloud_cover_pct", "wind_speed_mps"]
results["Hourly Demand"] = train_model(
    df[feat_hourly], df["total_demand_mw"], "1. Hourly Demand", "models/demand_hourly.pkl"
)

# 2. RESIDENTIAL DEMAND
results["Residential Demand"] = train_model(
    df[feat_hourly], df["residential_demand_mw"], "2. Residential Demand", "models/demand_residential.pkl"
)

# 3. INDUSTRIAL DEMAND
results["Industrial Demand"] = train_model(
    df[feat_hourly], df["industrial_demand_mw"], "3. Industrial Demand", "models/demand_industrial.pkl"
)

# 4. DAILY DEMAND (resampled)
daily_df = df.resample("D").agg({
    "total_demand_mw": "sum", "temperature_c": "mean", "cloud_cover_pct": "mean",
    "wind_speed_mps": "mean", "is_weekend": "first", "month": "first", "day_of_week": "first"
}).dropna()
feat_daily = ["day_of_week", "month", "is_weekend", "temperature_c", "cloud_cover_pct", "wind_speed_mps"]
results["Daily Demand"] = train_model(
    daily_df[feat_daily], daily_df["total_demand_mw"], "4. Daily Demand", "models/demand_daily.pkl"
)

# 5. WEEKLY DEMAND (resampled)
weekly_df = df.resample("W").agg({
    "total_demand_mw": "sum", "temperature_c": "mean", "cloud_cover_pct": "mean",
    "wind_speed_mps": "mean", "month": "first"
}).dropna()
feat_weekly = ["month", "temperature_c", "cloud_cover_pct", "wind_speed_mps"]
results["Weekly Demand"] = train_model(
    weekly_df[feat_weekly], weekly_df["total_demand_mw"], "5. Weekly Demand", "models/demand_weekly.pkl"
)

# 6. SEASONAL DEMAND
def get_season(month):
    if month in [12, 1, 2]:
        return 0
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    return 3

df["season"] = df["month"].apply(get_season)
feat_seasonal = ["season", "month", "hour", "temperature_c", "is_weekend"]
results["Seasonal Demand"] = train_model(
    df[feat_seasonal], df["total_demand_mw"], "6. Seasonal Demand", "models/demand_seasonal.pkl"
)

seasonal_summary = df.groupby("season")["total_demand_mw"].agg(["mean", "sum", "max"])
seasonal_summary.index = ["Winter", "Spring", "Summer", "Fall"]
print("\nSeasonal Demand Summary:")
print(seasonal_summary.round(2))

print("\n" + "=" * 65)
print("All 6 demand forecasting models trained and saved to models/")
print("=" * 65)
