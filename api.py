"""
Smart Grid AI Project - REST API Backend (FastAPI)
------------------------------------------------------------
Exposes all trained AI models as REST API endpoints.

Usage: uvicorn api:app --reload
Then open: http://localhost:8000/docs  (auto-generated interactive API docs)
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI(
    title="National Smart Energy Grid Intelligence API",
    description="REST API for demand forecasting, renewable forecasting, failure prediction, "
                "and grid optimization models.",
    version="1.0.0"
)

# ----------------------------
# Load all trained models once at startup
# ----------------------------
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

models = {
    "demand_hourly": load_model("models/demand_hourly.pkl"),
    "demand_residential": load_model("models/demand_residential.pkl"),
    "demand_industrial": load_model("models/demand_industrial.pkl"),
    "solar": load_model("models/renewable_solar.pkl"),
    "wind": load_model("models/renewable_wind.pkl"),
    "battery": load_model("models/renewable_battery.pkl"),
    "failure_classifier": load_model("models/failure_classifier.pkl"),
}


# ----------------------------
# Request schemas (define what input data looks like)
# ----------------------------
class DemandRequest(BaseModel):
    hour: int
    day_of_week: int
    month: int
    is_weekend: int
    temperature_c: float
    cloud_cover_pct: float
    wind_speed_mps: float


class RenewableRequest(BaseModel):
    hour: int
    month: int
    temperature_c: float
    cloud_cover_pct: float
    wind_speed_mps: float


class FailureRequest(BaseModel):
    voltage_kv: float
    frequency_hz: float
    load_pct: float


# ============================================================
# ROOT
# ============================================================
@app.get("/")
def root():
    return {
        "message": "National Smart Energy Grid Intelligence API",
        "docs": "/docs",
        "endpoints": [
            "/forecast/demand", "/forecast/renewable", "/predict/failure",
            "/emergency/recommend", "/health"
        ]
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}


# ============================================================
# 1. DEMAND FORECASTING
# ============================================================
@app.post("/forecast/demand")
def forecast_demand(request: DemandRequest):
    input_df = pd.DataFrame([request.dict()])
    hourly_pred = models["demand_hourly"].predict(input_df)[0]
    residential_pred = models["demand_residential"].predict(input_df)[0]
    industrial_pred = models["demand_industrial"].predict(input_df)[0]

    return {
        "hourly_demand_mw": round(float(hourly_pred), 2),
        "residential_demand_mw": round(float(residential_pred), 2),
        "industrial_demand_mw": round(float(industrial_pred), 2)
    }


# ============================================================
# 2. RENEWABLE FORECASTING
# ============================================================
@app.post("/forecast/renewable")
def forecast_renewable(request: RenewableRequest):
    input_df = pd.DataFrame([request.dict()])
    solar_pred = models["solar"].predict(input_df)[0]
    wind_pred = models["wind"].predict(input_df)[0]

    return {
        "solar_production_mw": round(float(solar_pred), 2),
        "wind_production_mw": round(float(wind_pred), 2),
        "total_renewable_mw": round(float(solar_pred + wind_pred), 2)
    }


# ============================================================
# 3. FAILURE PREDICTION
# ============================================================
@app.post("/predict/failure")
def predict_failure(request: FailureRequest):
    input_df = pd.DataFrame([request.dict()])
    prediction = models["failure_classifier"].predict(input_df)[0]
    probability = models["failure_classifier"].predict_proba(input_df)[0]

    return {
        "failure_predicted": bool(prediction),
        "failure_probability": round(float(probability[1]), 4),
        "risk_level": "HIGH" if probability[1] > 0.5 else ("MEDIUM" if probability[1] > 0.2 else "LOW")
    }


# ============================================================
# 4. EMERGENCY RESPONSE RECOMMENDATION
# ============================================================
class EmergencyRequest(BaseModel):
    demand_mw: float
    renewable_supply_mw: float
    battery_pct: float
    load_pct: float
    failure_risk: int


@app.post("/emergency/recommend")
def emergency_recommend(request: EmergencyRequest):
    GRID_CAPACITY_MW = 16000
    recommendations = []

    if request.demand_mw > GRID_CAPACITY_MW * 0.95:
        recommendations.append({"action": "Controlled Load Shedding", "priority": "CRITICAL"})

    if request.renewable_supply_mw < 300 and request.battery_pct < 20:
        recommendations.append({"action": "Backup Power Activation", "priority": "HIGH"})

    if request.load_pct > 95:
        recommendations.append({"action": "Load Redistribution", "priority": "HIGH"})

    if request.failure_risk == 1:
        recommendations.append({"action": "Emergency Dispatch", "priority": "CRITICAL"})

    if not recommendations:
        recommendations.append({"action": "No Action Needed", "priority": "NONE"})

    return {"recommendations": recommendations}


# ============================================================
# 5. CARBON INTELLIGENCE
# ============================================================
class CarbonRequest(BaseModel):
    demand_mw: float
    renewable_mw: float


@app.post("/carbon/calculate")
def carbon_calculate(request: CarbonRequest):
    EMISSION_FACTOR = 0.45
    non_renewable = max(0, request.demand_mw - request.renewable_mw)
    carbon_tons_per_hour = non_renewable * EMISSION_FACTOR
    renewable_pct = min(100, (request.renewable_mw / request.demand_mw) * 100) if request.demand_mw > 0 else 0

    return {
        "carbon_emissions_tons_per_hour": round(carbon_tons_per_hour, 2),
        "renewable_usage_pct": round(renewable_pct, 2)
    }
