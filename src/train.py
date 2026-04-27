import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Setup MLflow
mlflow.set_tracking_uri("mlflow")
mlflow.set_experiment("nyc-taxi-drift-detection")

FEATURES = ['trip_distance', 'fare_amount', 'PULocationID', 
            'DOLocationID', 'passenger_count']
TARGET = 'duration_minutes'

def train_model(data_path, run_name, period_label):
    print(f"\nTraining on: {period_label}")
    
    df = pd.read_parquet(data_path)
    
    # Sample to speed things up (100k rows is enough)
    if len(df) > 100000:
        df = df.sample(100000, random_state=42)
    
    X = df[FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("period", period_label)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae  = mean_absolute_error(y_test, predictions)
        r2   = r2_score(y_test, predictions)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"  RMSE : {rmse:.4f}")
        print(f"  MAE  : {mae:.4f}")
        print(f"  R2   : {r2:.4f}")
        
        return model, rmse, mae, r2

if __name__ == "__main__":
    results = {}
    
    datasets = {
        "baseline_jan2019" : ("data/processed/jan2019_clean.parquet", "Jan 2019 - Baseline"),
        "predrift_jan2020" : ("data/processed/jan2020_clean.parquet", "Jan 2020 - Pre COVID"),
        "drift_apr2020"    : ("data/processed/apr2020_clean.parquet", "Apr 2020 - Peak Drift"),
        "recovery_jan2022" : ("data/processed/jan2022_clean.parquet", "Jan 2022 - Recovery"),
    }
    
    for run_name, (path, label) in datasets.items():
        model, rmse, mae, r2 = train_model(path, run_name, label)
        results[label] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    
    print("\n========== RESULTS SUMMARY ==========")
    for period, metrics in results.items():
        print(f"{period}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")