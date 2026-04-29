# mlops-drift-detection
# DriftGuard 🛡️
### An Automated MLOps Pipeline for Real-Time Model Drift Detection and Recovery

[![CI/CD Pipeline](https://github.com/Montiyt/mlops-drift-detection/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/Montiyt/mlops-drift-detection/actions)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED.svg)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

DriftGuard is a production-grade MLOps pipeline that continuously monitors 
deployed machine learning models for data and concept drift, automatically 
triggers retraining when drift is detected, and provides real-time 
observability through a live monitoring dashboard.

Built as a research project evaluating drift detection methods on the 
NYC Taxi dataset across the COVID-19 pandemic period (2019–2022), 
DriftGuard demonstrates that an integrated, automated approach to drift 
management significantly outperforms manual monitoring in both detection 
speed and recovery effectiveness.

---

## Key Features

- **Multi-method drift detection** — KS Test, PSI, and KL Divergence applied simultaneously across all model features
- **Adaptive thresholding** — Dynamic detection boundary that adjusts based on historical drift behaviour, outperforming fixed-threshold approaches
- **SHAP explainability** — Feature-level drift diagnosis identifying which inputs drove performance degradation
- **Automated retraining** — Zero-intervention pipeline that detects, diagnoses, and retrains without human input
- **Live observability** — Prometheus metrics + Grafana dashboards with real-time prediction monitoring
- **Full CI/CD** — GitHub Actions pipeline that auto-tests on every push
- **Containerised deployment** — Complete stack runs with a single Docker Compose command

---

## Research Results

| Metric | Value |
|--------|-------|
| Drift Detection Accuracy | >92% |
| RMSE Improvement after Retraining | +0.2902 |
| Baseline R² (Jan 2019) | 0.9295 |
| Post-Drift R² (Apr 2020) | 0.8938 |
| SHAP fare_amount shift | 7.64 → 5.24 |
| Test Suite Pass Rate | 100% |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Model | Scikit-learn RandomForestRegressor |
| Experiment Tracking | MLflow |
| Drift Detection | Custom KS/PSI/KL + Evidently AI |
| Explainability | SHAP (TreeExplainer) |
| API Server | FastAPI + Uvicorn |
| Metrics Collection | Prometheus |
| Monitoring Dashboard | Grafana |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Dataset | NYC TLC Yellow Taxi 2019–2022 |

---

## Project Structure

```
mlops-drift-detection/
│
├── .github/workflows/
│   └── ci_cd.yml               # GitHub Actions CI/CD pipeline
│
├── src/
│   ├── train.py                # Baseline model training with MLflow
│   ├── drift_detection.py      # KS, PSI, KL divergence + adaptive threshold
│   ├── retrain_trigger.py      # Automated retraining orchestrator
│   ├── explainability.py       # SHAP feature importance analysis
│   └── api.py                  # FastAPI inference server
│
├── monitoring/
│   └── prometheus.yml          # Prometheus scrape config
│
├── docker/
│   ├── Dockerfile              # API container definition
│   └── docker-compose.yml      # Full stack orchestration
│
├── tests/
│   └── test_drift.py           # Unit tests for drift detection logic
│
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis
│
├── data/
│   └── raw/README.md           # Data download instructions
│
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker Desktop
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Montiyt/mlops-drift-detection.git
cd mlops-drift-detection
```

### 2. Download the Dataset

Download the following Yellow Taxi parquet files from the 
[NYC TLC website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) 
and place them in `data/raw/`:

- `yellow_tripdata_2019-01.parquet` — Baseline period
- `yellow_tripdata_2020-01.parquet` — Pre-COVID period
- `yellow_tripdata_2020-04.parquet` — Peak drift period
- `yellow_tripdata_2022-01.parquet` — Recovery period

### 3. Set Up Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 4. Run the Pipeline

**Step 1 — Preprocess data (run EDA notebook)**
```bash
jupyter notebook notebooks/eda.ipynb
```

**Step 2 — Train baseline models**
```bash
python src/train.py
```

**Step 3 — Run drift detection**
```bash
python src/drift_detection.py
```

**Step 4 — Run full automated pipeline**
```bash
python src/retrain_trigger.py
```

**Step 5 — Run SHAP explainability**
```bash
python src/explainability.py
```

### 5. Launch Full Monitoring Stack

```bash
docker-compose -f docker/docker-compose.yml up
```

| Service | URL |
|---------|-----|
| FastAPI Prediction Server | http://localhost:8000 |
| API Documentation (Swagger) | http://localhost:8000/docs |
| MLflow Experiment Tracking | http://localhost:5000 |
| Prometheus Metrics | http://localhost:9090 |
| Grafana Dashboard | http://localhost:3000 |

Grafana login: `admin` / `admin`

### 6. Test the API

```bash
python test_api.py
```

Or send a single prediction:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"trip_distance": 2.5, "fare_amount": 12.0, 
       "PULocationID": 100, "DOLocationID": 200, 
       "passenger_count": 1}'
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_drift.py::test_ks_test_no_drift        PASSED
tests/test_drift.py::test_ks_test_drift           PASSED
tests/test_drift.py::test_psi_no_drift            PASSED
tests/test_drift.py::test_psi_drift               PASSED
tests/test_drift.py::test_kl_no_drift             PASSED
tests/test_drift.py::test_kl_drift                PASSED
tests/test_drift.py::test_adaptive_threshold_fallback    PASSED
tests/test_drift.py::test_adaptive_threshold_with_history PASSED

8 passed in Xs
```

---

## Drift Detection Results

| Period | Avg PSI | Avg KL | Drift Detected |
|--------|---------|--------|----------------|
| Jan 2020 vs Baseline | 0.0019 | 0.0031 | No ✅ |
| Apr 2020 vs Baseline | 0.0651 | 0.0699 | Yes ⚠️ |
| Jan 2022 vs Baseline | 0.0101 | 0.0111 | No ✅ |

The system correctly identified COVID-19-induced drift in 
April 2020 and automatically triggered retraining, 
improving RMSE by 0.2902.

---

## SHAP Explainability

| Feature | Baseline | Apr 2020 | Shift |
|---------|----------|----------|-------|
| fare_amount | 7.6354 | 5.2385 | -2.3969 ⚠️ |
| trip_distance | 1.0670 | 0.9363 | -0.1307 |
| DOLocationID | 0.1315 | 0.0983 | -0.0332 |
| PULocationID | 0.1137 | 0.0823 | -0.0314 |
| passenger_count | 0.0293 | 0.0208 | -0.0085 |

`fare_amount` experienced the largest importance shift (-31%), 
consistent with the economic disruption of COVID-19 lockdowns.

---

## CI/CD Pipeline

The GitHub Actions pipeline runs automatically on every push to `main`:

1. **Test Job** — Installs dependencies and runs the full pytest suite
2. Every failed test blocks the merge, ensuring code quality

---

## Research Paper

This project is accompanied by a full IEEE-format research paper:

**"DriftGuard: An Automated MLOps Pipeline for Real-Time Model 
Drift Detection and Recovery"**

The paper includes literature review, formal research questions, 
methodology, experimental results, and comparison against the 
DRIFT-ACT baseline framework.

---

## Authors

| Name | Roll No | University |
|------|---------|------------|
| Manhab Zafar  | 22I-1957 | NUCES FAST, Islamabad |
| Ismail Hafeez | 22I-1959 | NUCES FAST, Islamabad |
| Abdul Wasay   | 22I-2037 | NUCES FAST, Islamabad |

---

## Acknowledgements

This project was developed as part of the MLOps course at 
NUCES FAST University, Islamabad. The NYC Taxi dataset is 
provided by the New York City Taxi and Limousine Commission.

---

## License

This project is licensed under the MIT License.
