import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from drift_detection import run_drift_report
import os
import json
from datetime import datetime

mlflow.set_tracking_uri("mlflow")
mlflow.set_experiment("automated-retraining")

FEATURES = ['trip_distance', 'fare_amount', 'PULocationID',
            'DOLocationID', 'passenger_count']
TARGET   = 'duration_minutes'

# ─────────────────────────────────────────
# RETRAIN FUNCTION
# ─────────────────────────────────────────
def retrain_model(data_path, trigger_reason, prev_rmse=None):
    print(f"\n🔄 Retraining triggered: {trigger_reason}")
    
    df = pd.read_parquet(data_path)
    if len(df) > 100000:
        df = df.sample(100000, random_state=42)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name=f"retrain_{trigger_reason}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("trigger_reason", trigger_reason)
        mlflow.log_param("retrain_data",   data_path)
        mlflow.log_param("train_size",     len(X_train))
        mlflow.log_param("n_estimators",   100)

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae  = mean_absolute_error(y_test, predictions)
        r2   = r2_score(y_test, predictions)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("r2",   r2)

        # Log improvement if we have previous RMSE
        if prev_rmse:
            improvement = prev_rmse - rmse
            mlflow.log_metric("rmse_improvement", improvement)
            print(f"  RMSE improvement : {improvement:+.4f}")

        mlflow.sklearn.log_model(model, "retrained_model")

        print(f"  New RMSE : {rmse:.4f}")
        print(f"  New MAE  : {mae:.4f}")
        print(f"  New R2   : {r2:.4f}")

        return model, rmse, mae, r2

# ─────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────
def run_pipeline():
    print("="*60)
    print("   AUTOMATED MLOPS DRIFT DETECTION PIPELINE")
    print("="*60)

    # Load datasets
    print("\n📂 Loading datasets...")
    jan2019 = pd.read_parquet("data/processed/jan2019_clean.parquet")
    jan2020 = pd.read_parquet("data/processed/jan2020_clean.parquet")
    apr2020 = pd.read_parquet("data/processed/apr2020_clean.parquet")
    jan2022 = pd.read_parquet("data/processed/jan2022_clean.parquet")

    def sample(df, n=50000):
        return df.sample(min(n, len(df)), random_state=42)

    baseline  = sample(jan2019)
    periods   = [
        ("Jan2020", sample(jan2020), "data/processed/jan2020_clean.parquet"),
        ("Apr2020", sample(apr2020), "data/processed/apr2020_clean.parquet"),
        ("Jan2022", sample(jan2022), "data/processed/jan2022_clean.parquet"),
    ]

    # Track state
    history_psi  = []
    current_rmse = 2.6505   # baseline RMSE from train.py
    retrain_log  = []
    pipeline_log = []

    print("\n🚀 Starting pipeline simulation...\n")

    for label, cur_df, data_path in periods:
        print(f"\n{'─'*60}")
        print(f"📅 Processing period: {label}")
        print(f"{'─'*60}")

        # Step 1 — Detect drift
        drift_result = run_drift_report(
            baseline, cur_df, label, history_psi
        )
        history_psi.append(drift_result['avg_psi'])

        # Step 2 — Trigger retraining if drift detected
        if drift_result['drift_detected']:
            print(f"\n  ⚠️  Drift detected! Triggering automated retraining...")
            model, new_rmse, mae, r2 = retrain_model(
                data_path,
                trigger_reason=label,
                prev_rmse=current_rmse
            )
            retrain_log.append({
                "period"       : label,
                "trigger_psi"  : drift_result['avg_psi'],
                "before_rmse"  : current_rmse,
                "after_rmse"   : new_rmse,
                "improvement"  : current_rmse - new_rmse
            })
            current_rmse = new_rmse
            action = "RETRAINED"
        else:
            print(f"\n  ✅  No drift detected. Model unchanged.")
            action = "STABLE"

        pipeline_log.append({
            "period"         : label,
            "avg_psi"        : drift_result['avg_psi'],
            "drift_detected" : bool(drift_result['drift_detected']),
            "action"         : action,
            "current_rmse"   : current_rmse
        })

    # Final summary
    print(f"\n\n{'='*60}")
    print("   PIPELINE EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Period':<15} {'PSI':<10} {'Drift':<10} {'Action':<12} {'RMSE'}")
    print(f"{'─'*60}")
    for entry in pipeline_log:
        drift_flag = "YES ⚠️" if entry['drift_detected'] else "NO ✅"
        print(f"{entry['period']:<15} {entry['avg_psi']:<10.4f} "
              f"{drift_flag:<10} {entry['action']:<12} {entry['current_rmse']:.4f}")

    if retrain_log:
        print(f"\n\n📊 Retraining Events:")
        print(f"{'─'*60}")
        for r in retrain_log:
            print(f"  Period    : {r['period']}")
            print(f"  RMSE before : {r['before_rmse']:.4f}")
            print(f"  RMSE after  : {r['after_rmse']:.4f}")
            print(f"  Improvement : {r['improvement']:+.4f}")
            print()

    # Save log to file
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/pipeline_log.json", "w") as f:
        json.dump(pipeline_log, f, indent=2)
    print("✅ Pipeline log saved to data/processed/pipeline_log.json")

if __name__ == "__main__":
    run_pipeline()