from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import mlflow.sklearn
import pandas as pd
import numpy as np
import time
import os

# ─────────────────────────────────────────
# App setup
# ─────────────────────────────────────────
app = FastAPI(title="NYC Taxi Drift Detection API")

mlflow.set_tracking_uri("mlflow")

FEATURES = ['trip_distance', 'fare_amount', 'PULocationID',
            'DOLocationID', 'passenger_count']

# ─────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────
PREDICTION_COUNT   = Counter(
    'predictions_total',
    'Total number of predictions made'
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction'
)
DRIFT_ALERT_COUNT  = Counter(
    'drift_alerts_total',
    'Total number of drift alerts triggered'
)
PREDICTION_VALUE   = Histogram(
    'prediction_value_minutes',
    'Distribution of predicted trip durations',
    buckets=[1, 5, 10, 15, 20, 30, 45, 60, 90, 120]
)

# ─────────────────────────────────────────
# Load model
# ─────────────────────────────────────────
MODEL = None

def load_model():
    global MODEL
    try:
        runs = mlflow.search_runs(
            experiment_names=["nyc-taxi-drift-detection"],
            order_by=["start_time DESC"]
        )
        if runs.empty:
            raise Exception("No runs found")
        run_id = runs.iloc[0]['run_id']
        MODEL  = mlflow.sklearn.load_model(f"mlflow/0/{run_id}/artifacts/model")
        print(f"Model loaded from run: {run_id}")
    except Exception as e:
        print(f"Could not load from MLflow: {e}")
        print("Training fallback model...")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        df    = pd.read_parquet("data/processed/jan2019_clean.parquet").sample(50000, random_state=42)
        X     = df[FEATURES]
        y     = df['duration_minutes']
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        MODEL = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        MODEL.fit(X_tr, y_tr)
        print("Fallback model ready!")

# ─────────────────────────────────────────
# Request schema
# ─────────────────────────────────────────
class TripRequest(BaseModel):
    trip_distance  : float
    fare_amount    : float
    PULocationID   : int
    DOLocationID   : int
    passenger_count: int

class PredictionResponse(BaseModel):
    predicted_duration_minutes: float
    drift_alert               : bool
    message                   : str

# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def root():
    return {"status": "running", "model": "NYC Taxi Duration Predictor"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(trip: TripRequest):
    start = time.time()

    input_df = pd.DataFrame([{
        'trip_distance'  : trip.trip_distance,
        'fare_amount'    : trip.fare_amount,
        'PULocationID'   : trip.PULocationID,
        'DOLocationID'   : trip.DOLocationID,
        'passenger_count': trip.passenger_count
    }])

    prediction = MODEL.predict(input_df)[0]

    # Simple drift alert: flag unusual predictions
    drift_alert = prediction > 60 or prediction < 1
    if drift_alert:
        DRIFT_ALERT_COUNT.inc()

    # Update Prometheus metrics
    latency = time.time() - start
    PREDICTION_COUNT.inc()
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_VALUE.observe(prediction)

    return PredictionResponse(
        predicted_duration_minutes=round(float(prediction), 2),
        drift_alert=drift_alert,
        message="Drift alert! Unusual prediction." if drift_alert else "Prediction normal."
    )

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)