# National Smart Energy Grid Intelligence & Demand Forecasting Platform

AI-powered platform for demand forecasting, renewable energy forecasting, grid optimization,
failure prediction, emergency response, and carbon intelligence — built for a simulated national
power grid (250 power stations, 40 solar farms, 18 wind farms, 12 battery facilities).

## Note on Data

Real-world data for a national grid of this scale (15 million smart meters) is not publicly
available and no real utility publishes this data due to privacy/security. This project uses
**synthetically generated data** (`generate_dataset.py`) that models realistic demand, weather,
solar/wind production, and equipment failure patterns for one full year (hourly resolution).

## Project Structure

```
smart-grid-ai/
├── generate_dataset.py        # Generates 1 year of synthetic hourly grid data
├── data_processing.py          # Module 1: Smart Meter Data Processing
├── demand_forecasting.py       # Module 2: Demand Forecasting (6 types)
├── renewable_forecasting.py    # Module 3: Renewable Energy Forecasting
├── grid_optimization.py        # Module 4: AI Grid Optimization (LP + RL)
├── failure_prediction.py       # Module 5: Failure Prediction (Anomaly Detection + GNN + XAI)
├── emergency_response.py       # Module 6: Emergency Response Engine
├── carbon_intelligence.py      # Module 7: Carbon Intelligence
├── dashboard.py                 # Module 8: Executive Dashboard (Streamlit)
├── requirements.txt
├── data/                        # Generated datasets (CSV)
├── models/                      # Trained AI models (pickle/pt files)
└── docs/                        # Reports (sustainability report, etc.)
```

## How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Generate the dataset (run this first)
```
python generate_dataset.py
```

### 3. Run each module in order
```
python data_processing.py
python demand_forecasting.py
python renewable_forecasting.py
python failure_prediction.py
python grid_optimization.py
python emergency_response.py
python carbon_intelligence.py
```

### 4. Launch the dashboard
```
streamlit run dashboard.py
```

## AI Components Used

| Component | Where Used |
|---|---|
| Time-Series Forecasting | Demand, Solar, Wind, Battery forecasting (XGBoost) |
| Graph Neural Networks | Grid topology-based failure risk prediction (`failure_prediction.py`) |
| Reinforcement Learning | Battery charging strategy (PPO agent, `grid_optimization.py`) |
| Anomaly Detection | Equipment failure detection (Isolation Forest) |
| Explainable AI | SHAP values for failure prediction interpretability |
| Optimization Algorithms | Linear Programming for grid load distribution (PuLP) |

## Core Modules Coverage

- **Smart Meter Data Processing**: Collection, Cleaning, Consumption Analysis, Load Aggregation, Streaming Simulation
- **Demand Forecasting**: Hourly, Daily, Weekly, Seasonal, Industrial, Residential
- **Renewable Energy Forecasting**: Solar, Wind, Battery Storage, Weather Impact, Renewable Availability
- **AI Grid Optimization**: Energy Distribution, Load Balancing, Power Routing, Battery Strategy, Renewable Allocation
- **Failure Prediction**: Transformer, Transmission Line, Voltage, Frequency, Equipment Overload
- **Emergency Response Engine**: Load Redistribution, Backup Activation, Load Shedding, Emergency Dispatch, Maintenance Scheduling
- **Carbon Intelligence**: Emissions, Renewable Usage, Efficiency Score, Sustainability Report
- **Executive Dashboard**: Grid Health, Live Flow, Renewable Production, Forecasts, Risk Zones, Loss Analytics, Carbon Dashboard, Incident Map

## Deployment Notes

This project can be containerized using Docker. A `Dockerfile` and `docker-compose.yml` can be
added to run the FastAPI backend, PostgreSQL database, and Streamlit dashboard as separate
services for a production-style deployment.
