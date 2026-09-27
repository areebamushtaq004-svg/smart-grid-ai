# Deployment Guide
## National Smart Energy Grid Intelligence & Demand Forecasting Platform

---

## 1. System Requirements

- Python 3.10 or higher
- 4 GB RAM minimum (8 GB recommended for GNN/RL training)
- ~2 GB free disk space (for libraries and model files)

---

## 2. Installation Steps

```bash
# 1. Clone/extract the project folder
cd smart-grid-ai

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 3. Running the Pipeline (in order)

```bash
python generate_dataset.py          # Generates 1 year of synthetic grid data
python data_processing.py           # Cleans data, runs consumption analysis
python demand_forecasting.py        # Trains 6 demand forecasting models
python renewable_forecasting.py     # Trains solar/wind/battery models
python failure_prediction.py        # Trains anomaly detection + GNN + SHAP
python grid_optimization.py         # Runs LP optimization + trains RL agent
python emergency_response.py        # Runs the rule-based emergency engine
python carbon_intelligence.py       # Generates the sustainability report
```

---

## 4. Running the API Server

```bash
uvicorn api:app --reload
```
Access interactive API documentation at: `http://localhost:8000/docs`

---

## 5. Running the Dashboard

```bash
streamlit run dashboard.py
```
Opens automatically in your browser at `http://localhost:8501`

---

## 6. Technology Choices & Justification

The project specification lists suggested technologies and explicitly allows teams to
**"justify alternative technologies."** Below is a summary of the choices made for this
implementation and the reasoning behind them.

| Suggested Technology | What We Used Instead | Justification |
|---|---|---|
| PostgreSQL / TimescaleDB | CSV files (pandas) | Project scope is a demo/case-study on synthetic data of manageable size (~8,760 hourly records). A full database server adds setup overhead without added value at this scale. For a production deployment at true national-grid scale (15M smart meters), PostgreSQL + TimescaleDB would be the correct choice for time-series storage and would be a direct next step. |
| Apache Kafka | Python generator function (simulated streaming) | Demonstrates the streaming concept (`data_processing.py`, Module 1) without requiring a Kafka broker setup, which is significant infrastructure for a 10-day project timeline. |
| Redis | Not used | No caching layer was needed at this data scale; all models run fast enough in-memory. |
| Grafana / Prometheus | Streamlit | Streamlit provides equivalent dashboarding capability (charts, live metrics, maps) directly in Python without a separate monitoring stack to configure, which was more time-efficient given project constraints. |
| Kubernetes | Not used (Docker only) | The specification explicitly marks this as optional ("Teams may justify alternative technologies"). Docker alone is sufficient to containerize and run this project; Kubernetes orchestration would only be necessary at true multi-node production scale. |
| FastAPI | FastAPI (used as suggested) | Used exactly as recommended — lightweight, fast to develop, and auto-generates interactive API documentation. |
| PyTorch, XGBoost | Used as suggested | Used exactly as recommended for the GNN and time-series forecasting models respectively. |

**Summary:** All core AI/ML requirements (forecasting, GNN, RL, anomaly detection, optimization,
explainability, digital twin) were implemented using the suggested or equivalent technologies.
Infrastructure technologies (database, message queue, monitoring) were substituted with
lighter-weight equivalents appropriate for a 10-day development timeline and demo-scale data,
with a clear path to the originally suggested stack for production deployment.

---

## 7. Docker Deployment (Optional)

A `Dockerfile` can containerize the API service:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t smart-grid-api .
docker run -p 8000:8000 smart-grid-api
```

---

## 8. Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Ensure the virtual environment is activated (`venv\Scripts\activate`) before running any script |
| Models not found when running `api.py` or `dashboard.py` | Run the training scripts (`demand_forecasting.py`, `renewable_forecasting.py`, `failure_prediction.py`) first to generate the `.pkl` model files |
| Port already in use | Change the port: `uvicorn api:app --port 8001` or `streamlit run dashboard.py --server.port 8502` |
