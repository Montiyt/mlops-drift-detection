import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from drift_detection import ks_test, compute_psi, compute_kl, adaptive_threshold

# ── Sample test data ───────────────────────────────────
def make_df(mean=5.0, std=2.0, n=1000):
    np.random.seed(42)
    return pd.DataFrame({
        'trip_distance'  : np.random.normal(mean, std, n).clip(0),
        'fare_amount'    : np.random.normal(mean * 2, std, n).clip(0),
        'PULocationID'   : np.random.randint(1, 263, n),
        'DOLocationID'   : np.random.randint(1, 263, n),
        'passenger_count': np.random.randint(1, 4, n),
    })

ref = make_df(mean=5.0)
cur_similar  = make_df(mean=5.1)   # almost no drift
cur_drifted  = make_df(mean=15.0)  # heavy drift

# ── Tests ──────────────────────────────────────────────
def test_ks_test_no_drift():
    result = ks_test(ref, cur_similar, 'trip_distance')
    assert 'statistic' in result
    assert 'p_value'   in result
    assert result['statistic'] < 0.5

def test_ks_test_drift():
    result = ks_test(ref, cur_drifted, 'trip_distance')
    assert result['statistic'] > 0.3

def test_psi_no_drift():
    result = compute_psi(ref, cur_similar, 'trip_distance')
    assert result['psi'] < 0.1

def test_psi_drift():
    result = compute_psi(ref, cur_drifted, 'trip_distance')
    assert result['psi'] > 0.1

def test_kl_no_drift():
    result = compute_kl(ref, cur_similar, 'trip_distance')
    assert result['kl_divergence'] < 0.5

def test_kl_drift():
    result = compute_kl(ref, cur_drifted, 'trip_distance')
    assert result['kl_divergence'] > 0.1

def test_adaptive_threshold_fallback():
    threshold = adaptive_threshold([])
    assert threshold == 0.2

def test_adaptive_threshold_with_history():
    history   = [0.05, 0.06, 0.07]
    threshold = adaptive_threshold(history)
    assert threshold > 0.05

def test_adaptive_threshold_detects_spike():
    history = [0.01, 0.02, 0.01]
    threshold = adaptive_threshold(history)
    assert 0.10 > threshold > 0.0