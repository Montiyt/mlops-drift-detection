import pandas as pd
import numpy as np
import shap
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("mlflow")

FEATURES = ['trip_distance', 'fare_amount', 'PULocationID',
            'DOLocationID', 'passenger_count']
TARGET   = 'duration_minutes'

def train_for_shap(data_path, n=5000):
    df = pd.read_parquet(data_path).sample(n, random_state=42)
    X  = df[FEATURES]
    y  = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X_test

def compute_shap(model, X_test, label):
    print(f"\n Computing SHAP values for: {label}")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_shap   = np.abs(shap_values).mean(axis=0)

    print(f"\n  Feature importance via SHAP ({label}):")
    for feat, val in sorted(zip(FEATURES, mean_shap), key=lambda x: -x[1]):
        bar = "█" * int(val * 10)
        print(f"    {feat:20s}: {val:.4f}  {bar}")

    return shap_values, mean_shap

def plot_shap_comparison(mean_shap_baseline, mean_shap_drift, label):
    os.makedirs("data/processed", exist_ok=True)

    x      = np.arange(len(FEATURES))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width/2, mean_shap_baseline, width,
                   label='Baseline (Jan 2019)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, mean_shap_drift,    width,
                   label=f'Drift ({label})',    color='tomato',    alpha=0.8)

    ax.set_xlabel('Features')
    ax.set_ylabel('Mean |SHAP value|')
    ax.set_title(f'Feature Importance Shift: Baseline vs {label}')
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    path = f"data/processed/shap_comparison_{label}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  Plot saved: {path}")
    return path

def run_shap_analysis():
    print("="*60)
    print("   SHAP EXPLAINABILITY ANALYSIS")
    print("="*60)

    # Train models on each period
    model_baseline, X_baseline = train_for_shap("data/processed/jan2019_clean.parquet")
    model_apr2020,  X_apr2020  = train_for_shap("data/processed/apr2020_clean.parquet")
    model_jan2022,  X_jan2022  = train_for_shap("data/processed/jan2022_clean.parquet")

    # Compute SHAP values
    _, mean_baseline = compute_shap(model_baseline, X_baseline, "Jan 2019 Baseline")
    _, mean_apr2020  = compute_shap(model_apr2020,  X_apr2020,  "Apr 2020 Drift")
    _, mean_jan2022  = compute_shap(model_jan2022,  X_jan2022,  "Jan 2022 Recovery")

    # Plot comparisons
    plot_shap_comparison(mean_baseline, mean_apr2020, "Apr2020")
    plot_shap_comparison(mean_baseline, mean_jan2022, "Jan2022")

    # Log to MLflow
    
    mlflow.set_experiment("drift-detection")
    with mlflow.start_run(run_name="shap_explainability"):
        
        for feat, b, d in zip(FEATURES, mean_baseline, mean_apr2020):
            mlflow.log_metric(f"shap_baseline_{feat}", round(float(b), 4))
            mlflow.log_metric(f"shap_drift_{feat}",    round(float(d), 4))
            mlflow.log_metric(f"shap_shift_{feat}",    round(float(abs(d-b)), 4))
        mlflow.log_artifact("data/processed/shap_comparison_Apr2020.png")
        mlflow.log_artifact("data/processed/shap_comparison_Jan2022.png")

    print("\n SHAP analysis complete and logged to MLflow!")

    # Print explainability summary
    print("\n" + "="*60)
    print("   DRIFT EXPLAINABILITY SUMMARY")
    print("="*60)
    print(f"\n{'Feature':<22} {'Baseline':>10} {'Apr2020':>10} {'Shift':>10}")
    print("─"*55)
    for feat, b, d in zip(FEATURES, mean_baseline, mean_apr2020):
        shift = d - b
        flag  = " ⚠️" if abs(shift) > 0.05 else " ✅"
        print(f"{feat:<22} {b:>10.4f} {d:>10.4f} {shift:>+10.4f}{flag}")

if __name__ == "__main__":
    run_shap_analysis()