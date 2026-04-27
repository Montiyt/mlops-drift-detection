import pandas as pd
import numpy as np
from scipy import stats
import mlflow
import json
import os

mlflow.set_tracking_uri("mlflow")
mlflow.set_experiment("drift-detection")

FEATURES = ['trip_distance', 'fare_amount', 'PULocationID',
            'DOLocationID', 'passenger_count']

# ─────────────────────────────────────────
# 1. KS TEST
# ─────────────────────────────────────────
def ks_test(ref, cur, feature):
    stat, p_value = stats.ks_2samp(ref[feature], cur[feature])
    return {"feature": feature, "statistic": round(stat, 4), "p_value": round(p_value, 6)}

# ─────────────────────────────────────────
# 2. PSI  (Population Stability Index)
# ─────────────────────────────────────────
def compute_psi(ref, cur, feature, bins=10):
    combined   = pd.concat([ref[feature], cur[feature]])
    breakpoints = np.linspace(combined.min(), combined.max(), bins + 1)

    def bucket(series):
        counts, _ = np.histogram(series, bins=breakpoints)
        pct = counts / len(series)
        pct = np.where(pct == 0, 1e-6, pct)   # avoid log(0)
        return pct

    ref_pct = bucket(ref[feature])
    cur_pct = bucket(cur[feature])

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return {"feature": feature, "psi": round(float(psi), 4)}

# ─────────────────────────────────────────
# 3. KL DIVERGENCE
# ─────────────────────────────────────────
def compute_kl(ref, cur, feature, bins=50):
    combined    = pd.concat([ref[feature], cur[feature]])
    breakpoints = np.linspace(combined.min(), combined.max(), bins + 1)

    def density(series):
        counts, _ = np.histogram(series, bins=breakpoints)
        pct = counts / len(series)
        return np.where(pct == 0, 1e-6, pct)

    ref_d = density(ref[feature])
    cur_d = density(cur[feature])

    kl = np.sum(ref_d * np.log(ref_d / cur_d))
    return {"feature": feature, "kl_divergence": round(float(kl), 4)}

# ─────────────────────────────────────────
# 4. ADAPTIVE THRESHOLD  ← your novel contribution
# ─────────────────────────────────────────
def adaptive_threshold(history_scores, z=2.0):
    """
    Instead of a fixed threshold, compute threshold dynamically
    from the rolling mean + z * std of historical PSI scores.
    This is what DRIFT-ACT was missing.
    """
    if len(history_scores) < 2:
        return 0.2   # fallback to standard PSI threshold
    mean = np.mean(history_scores)
    std  = np.std(history_scores)
    return round(mean + z * std, 4)

# ─────────────────────────────────────────
# 5. FULL DRIFT REPORT
# ─────────────────────────────────────────
def run_drift_report(ref_df, cur_df, period_label, history_psi=None):
    print(f"\n{'='*50}")
    print(f"Drift Report: {period_label}")
    print(f"{'='*50}")

    ks_results  = []
    psi_results = []
    kl_results  = []

    for feature in FEATURES:
        ks  = ks_test(ref_df, cur_df, feature)
        psi = compute_psi(ref_df, cur_df, feature)
        kl  = compute_kl(ref_df, cur_df, feature)

        ks_results.append(ks)
        psi_results.append(psi)
        kl_results.append(kl)

    # Average PSI across all features
    avg_psi = np.mean([r['psi'] for r in psi_results])
    avg_kl  = np.mean([r['kl_divergence'] for r in kl_results])
    ks_drifted = sum(1 for r in ks_results if r['p_value'] < 0.05)

    # Adaptive threshold
    history = history_psi if history_psi else []
    threshold = adaptive_threshold(history)
    
    # More sensitive: drift if ANY feature exceeds 0.1
    # OR average PSI exceeds adaptive threshold
    feature_drift = any(r['psi'] > 0.1 for r in psi_results)
    avg_drift = avg_psi > threshold
    drift_detected = feature_drift or avg_drift

    print(f"\n  Average PSI       : {avg_psi:.4f}")
    print(f"  Average KL Div    : {avg_kl:.4f}")
    print(f"  KS drifted features: {ks_drifted}/{len(FEATURES)}")
    print(f"  Adaptive threshold : {threshold:.4f}")
    print(f"  DRIFT DETECTED    : {'YES ⚠️' if drift_detected else 'NO ✅'}")

    print(f"\n  Per-feature PSI:")
    for r in psi_results:
        flag = "⚠️" if r['psi'] > 0.1 else "✅"
        print(f"    {r['feature']:20s}: {r['psi']:.4f} {flag}")

    # Log to MLflow
    with mlflow.start_run(run_name=f"drift_{period_label}"):
        mlflow.log_param("period", period_label)
        mlflow.log_param("adaptive_threshold", threshold)
        mlflow.log_metric("avg_psi", avg_psi)
        mlflow.log_metric("avg_kl_divergence", avg_kl)
        mlflow.log_metric("ks_drifted_features", ks_drifted)
        mlflow.log_metric("drift_detected", int(drift_detected))

        for r in psi_results:
            mlflow.log_metric(f"psi_{r['feature']}", r['psi'])
        for r in kl_results:
            mlflow.log_metric(f"kl_{r['feature']}", r['kl_divergence'])

    return {
        "period"          : period_label,
        "avg_psi"         : avg_psi,
        "avg_kl"          : avg_kl,
        "ks_drifted"      : ks_drifted,
        "threshold"       : threshold,
        "drift_detected"  : drift_detected
    }

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Loading cleaned datasets...")
    jan2019 = pd.read_parquet("data/processed/jan2019_clean.parquet")
    jan2020 = pd.read_parquet("data/processed/jan2020_clean.parquet")
    apr2020 = pd.read_parquet("data/processed/apr2020_clean.parquet")
    jan2022 = pd.read_parquet("data/processed/jan2022_clean.parquet")

    # Sample for speed
    def sample(df, n=50000):
        return df.sample(min(n, len(df)), random_state=42)

    ref     = sample(jan2019)   # baseline is always Jan 2019
    jan2020s = sample(jan2020)
    apr2020s = sample(apr2020)
    jan2022s = sample(jan2022)

    history_psi = []   # grows as we compare more periods

    r1 = run_drift_report(ref, jan2020s, "Jan2020_vs_Baseline", history_psi)
    history_psi.append(r1['avg_psi'])

    r2 = run_drift_report(ref, apr2020s, "Apr2020_vs_Baseline", history_psi)
    history_psi.append(r2['avg_psi'])

    r3 = run_drift_report(ref, jan2022s, "Jan2022_vs_Baseline", history_psi)
    history_psi.append(r3['avg_psi'])

    print("\n\n========== DRIFT SUMMARY ==========")
    for r in [r1, r2, r3]:
        status = "DRIFT ⚠️" if r['drift_detected'] else "STABLE ✅"
        print(f"{r['period']:35s} | PSI={r['avg_psi']:.4f} | {status}")