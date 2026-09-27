# Dockerfile - National Smart Energy Grid Intelligence Platform (API Service)

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for some Python packages (e.g. torch, xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the API port
EXPOSE 8000

# Default command: run the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
