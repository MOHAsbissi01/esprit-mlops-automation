# ML Automation System — FastAPI + MLflow + Docker + n8n

## Overview

This project implements a **multi-actor Machine Learning automation system** integrating:

- **FastAPI** for model serving and REST API
- **MLflow** for experiment tracking, run comparison, and artifact storage
- **Docker + Docker Compose** for containerized deployment
- **n8n** for workflow orchestration (automated prediction + scheduled retraining)
- **Frontend Dashboard** (`dashboard.html`) for live browser-based predictions
- **3 ML pipelines** across 3 domain actors

The system is designed to be **modular, scalable, containerized, and fully automated**.

---

## Project Structure

```
ml_api/
├── actors/
│   ├── __init__.py
│   ├── actor1.py                          # Ecological Director logic
│   ├── actor2.py                          # Mobility Director logic
│   └── actor3.py                          # Security Manager logic
├── mlflow/
│   └── mlruns/                            # Persistent MLflow run storage (volume-mounted)
├── results/
│   └── predictions.json                   # Stored prediction history (45+ entries)
├── training/
│   ├── __init__.py
│   ├── train_actor1.py                    # MLflow-tracked training for actor1_ecologique
│   ├── train_actor2.py                    # MLflow-tracked training for actor2_mobilites
│   └── train_actor3.py                    # MLflow-tracked training for actor3_securite
├── utils/
│   └── __init__.py
├── main.py                                # FastAPI application entry point
├── dashboard.html                         # Browser frontend — calls /predict live
├── Dockerfile                             # Container definition (python:3.11-slim)
├── docker-compose.yml                     # Orchestrates api + mlflow services
├── requirements.txt                       # Python dependencies
├── n8n_prediction_workflow.json           # n8n basic prediction workflow
├── n8n_prediction_workflow_extended.json  # n8n extended workflow (alerts + sheets)
├── n8n_retraining_workflow.json           # n8n scheduled retraining workflow
├── README.md
└── Claude.md                              # This file
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                             │
│   dashboard.html (browser)   n8n webhook   curl/requests        │
└────────────────┬────────────────────┬───────────────────────────┘
                 │                    │
                 ▼                    ▼
        ┌─────────────────────────────────┐
        │   FastAPI  (port 8000)          │
        │   GET  /health                  │
        │   POST /predict                 │
        │   POST /retrain                 │
        │   POST /save-n8n-result         │
        └────────┬────────────────────────┘
                 │
        ┌────────▼────────────────────────┐
        │  Actor dispatch (actor1/2/3.py) │
        │  XGBoost · RF · KMeans · IF     │
        └────────┬────────────────────────┘
                 │
        ┌────────▼──────────┐   ┌───────────────────┐
        │ results/           │   │ MLflow (port 5000) │
        │ predictions.json   │   │ Experiments/Runs   │
        └────────────────────┘   └───────────────────┘

Training pipeline:
  python training/train_actorN.py --run vN
          → subprocess: actor*/main.py
          → MLflow: log params + metrics + .pkl artifacts
```

---

## Dependencies (`requirements.txt`)

```
fastapi
uvicorn
scikit-learn
xgboost
numpy
pandas
mlflow
```

---

## Docker Setup

### `Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker-compose.yml`

| Service | Image | Port | Notes |
|---|---|---|---|
| `api` | Built from `Dockerfile` | `8000` | Mounts actor data dirs; depends on mlflow |
| `mlflow` | `ghcr.io/mlflow/mlflow:latest` | `5000` | Persists runs to `./mlflow/mlruns` |

Key config:
- `MLFLOW_TRACKING_URI=http://mlflow:5000` injected into the API container
- Actor data folders mounted:
  - `../actor1_ecologique:/actor1_ecologique`
  - `../actor2_mobilites:/actor2_mobilites`
  - `../actor3_securite:/actor3_securite`

### Start everything

```bash
docker-compose up --build
```

> **Note:** First build downloads ~600MB+ (xgboost pulls `nvidia_nccl_cu12` at 300MB).
> Subsequent builds use Docker layer cache.

---

## FastAPI Endpoints

### `GET /health` — Verified response (2026-04-23)

```json
{
  "status": "ok",
  "version": "2.0.0",
  "actors": {
    "actor1": { "tasks": ["co2", "energy", "cluster"], "description": "Ecological Director" },
    "actor2": { "tasks": ["charge", "cancellation"],  "description": "Mobility Director" },
    "actor3": { "tasks": ["severity", "risk_cluster", "anomaly"], "description": "Security Manager" }
  },
  "predictions_stored": 45,
  "predictions_file": "/app/results/predictions.json",
  "timestamp": "2026-04-23T22:17:42.007507+00:00"
}
```

### `POST /predict` — Verified response (2026-04-23)

**Request:**
```json
{
  "actor": "actor1",
  "task": "co2",
  "features": {
    "zone_encoded": 2, "mode_encoded": 1, "mode_co2_mean": 45.3,
    "annee": 2023, "mois_sin": 0.5, "mois_cos": 0.866,
    "pm25": 18.4, "no2": 32.1, "aqi_index": 75.0,
    "co2_lag1": 42.0, "co2_lag3": 39.5,
    "energie_lag1": 120.0, "energie_lag3": 115.0,
    "aqi_lag1": 72.0, "pm25_lag1": 17.8,
    "co2_roll3": 40.8, "energie_roll3": 117.5, "pm25_roll3": 18.0
  }
}
```

**Response:**
```json
{
  "status": "success",
  "actor": "actor1",
  "task": "co2",
  "result": 2.266724109649658,
  "latency_ms": 3157.866,
  "timestamp": "2026-04-23T22:20:22.738379+00:00"
}
```

> First-call latency ~3s (cold model load). Subsequent calls near-instant.

### `POST /retrain` — Triggers retraining for all actors
### `POST /save-n8n-result` — Saves results forwarded from n8n
### `GET /docs` — Swagger UI at `http://localhost:8000/docs`

---

## Actor Responsibilities

### Actor 1 — Ecological (`actor1_ecologique`)

| Model file | Type | Output |
|---|---|---|
| `xgboost_co2.pkl` | XGBoost Regression | CO₂ (kg) |
| `xgboost_nrj.pkl` | XGBoost Regression | Energy (kWh) |
| `xgboost_charge.pkl` | XGBoost Regression | Charge per mode |
| `kmeans_pollution_zones.pkl` | K-Means | Pollution cluster |
| Prophet | Time Series | AQI forecast (720 rows) |

**Feature names** (from `xgboost_features.pkl`):
```
zone_encoded, mode_encoded, mode_co2_mean, annee, mois_sin, mois_cos,
pm25, no2, aqi_index, co2_lag1, co2_lag3, energie_lag1, energie_lag3,
aqi_lag1, pm25_lag1, co2_roll3, energie_roll3, pm25_roll3
```

Pipeline: **7/7 OK** — 1911 rows → 1871 ML rows

---

### Actor 2 — Mobility (`actor2_mobilites`)

| Model file | Type | Output |
|---|---|---|
| `xgboost_charge.pkl` | XGBoost Regression | Passenger load (RMSE=20.30) |
| `xgboost_cancellation.pkl` | XGBoost Classification | Cancellation risk (AUC=0.52) |
| Prophet (10 zones) | Time Series | Congestion forecast (MAE=0.857) |

**KPI:** Taux de Ponctualité = **98.22%** — Pipeline: **7/7 OK**

---

### Actor 3 — Security (`actor3_securite`)

| Model file | Type | Output |
|---|---|---|
| `rf_severity.pkl` | Random Forest | Accident severity (F1=1.0, AUC=1.0) |
| `kmeans_risk.pkl` | K-Means (k=3) | Zone risk cluster (Silhouette=0.242) |
| `isolation_forest.pkl` | Isolation Forest | Anomaly detection (96/1911 = 5%) |

**KPIs:** Densité accidents=19.9, Indice gravité=0.1407 — Pipeline: **7/7 OK**

---

## MLflow Tracking

### Local server (without Docker)

```bash
python -m mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri ./mlflow/mlruns \
  --default-artifact-root ./mlflow/mlruns
```

> Use `python -m mlflow` — the `mlflow` CLI is not on PATH with Microsoft Store Python.

### Training scripts structure

```python
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("actorN_<domain>")

with mlflow.start_run(run_name=args.run):         # --run v1 / v2
    mlflow.log_params({
        "n_estimators": 100, "max_depth": 6,
        "learning_rate": 0.1, "actor": "actorN", "models": "..."
    })
    result = subprocess.run(["python", "main.py"], cwd=ACTOR_DIR,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, ...)
    if result.returncode == 0:
        mlflow.log_metric("training_success", 1)
        for pkl in OUTPUTS_DIR.glob("*.pkl"):
            mlflow.log_artifact(pkl, artifact_path="models")
    else:
        mlflow.log_metric("training_success", 0)
        raise RuntimeError(result.stderr)
```

### Experiments created

| Experiment | Experiment ID | Runs | .pkl artifacts |
|---|---|---|---|
| `actor1_ecologique` | `864163389833378263` | v1, v2 | 11 |
| `actor2_mobilites` | `691914286126122040` | v1, v2 | 5 |
| `actor3_securite` | `614386804897203239` | v1, v2 | 8 |

### Run training

```bash
python training/train_actor1.py --run v1
python training/train_actor2.py --run v1
python training/train_actor3.py --run v1

# Second run for comparison
python training/train_actor1.py --run v2
python training/train_actor2.py --run v2
python training/train_actor3.py --run v2
```

---

## n8n Workflow Automation

Three workflow files are available in the project root.

---

### 1. `n8n_prediction_workflow.json` — Basic Prediction

**Trigger:** Webhook POST at `/webhook/ml-predict`

**Node flow:**
```
Webhook (POST /ml-predict)
    ↓
HTTP Request FastAPI → POST http://127.0.0.1:8000/predict
    ↓
IF Success (status == "success")
    ├── TRUE  → Save Prediction (POST /save-n8n-result) → Respond Success
    └── FALSE → Respond Error (HTTP 500)
```

**Key details:**
- Webhook ID: `ml-predict-webhook`
- FastAPI timeout: 30 000 ms
- Body forwarded dynamically: `$json.body.actor ?? $json.actor`
- Saved with source tag: `"n8n-prediction-workflow"`

---

### 2. `n8n_prediction_workflow_extended.json` — Extended with Alerts

**Trigger:** Same webhook POST at `/webhook/ml-predict`

**Full node flow:**
```
Webhook
    ↓
HTTP Request FastAPI → POST /predict
    ↓
IF Success
    ├── TRUE →
    │   Set — Enrich Fields  (adds risk_label + summary)
    │       ↓                          ↓                     ↓
    │   Switch — Actor Router    Save Prediction    Google Sheets Log
    │       ├── Ecological → IF High Risk → Telegram Eco Alert
    │       ├── Mobility   → IF High Risk → Telegram Mobility Alert
    │       └── Security   → IF High Risk → Telegram Security Alert
    │                               ↓
    │                       Gmail — High Risk Alert (HTML email)
    │                               ↓
    │                       IF Anomaly → Trigger Retraining?
    │                               ↓ (if yes)
    │                       POST /retrain  (timeout: 600 000 ms)
    │
    └── FALSE → Respond Error
```

**Enrichment fields added by `Set — Enrich Fields` node:**
- `risk_label`: `"HIGH_RISK"` or `"NORMAL"` — based on actor-specific conditions:
  - Actor 1: `result.cluster == 2`
  - Actor 2: `result.cancellation_risk == "High"`
  - Actor 3: `result.severity == "High"` OR `result.anomaly == true`
- `summary`: human-readable string with actor, task, risk, latency

**Alert integrations:**
- **Telegram** — per-actor bot messages (3 separate nodes, credentials: `YOUR_TELEGRAM_CREDENTIAL_ID`)
- **Gmail** — HTML email alert for any HIGH_RISK prediction (credentials: `YOUR_GMAIL_CREDENTIAL_ID`)
- **Google Sheets** — appends every prediction to Sheet1 (columns: timestamp, actor, task, risk_label, result_json, latency_ms, summary)
- **Auto-retrain** — triggers `POST /retrain` when a HIGH_RISK anomaly is detected

**Credentials to configure:**
| Integration | Credential ID placeholder |
|---|---|
| Telegram | `YOUR_TELEGRAM_CREDENTIAL_ID` |
| Gmail | `YOUR_GMAIL_CREDENTIAL_ID` |
| Google Sheets | `YOUR_SHEETS_CREDENTIAL_ID` |
| Telegram Chat | `YOUR_TELEGRAM_CHAT_ID` |
| Google Sheet | `YOUR_GOOGLE_SHEET_ID` |

---

### 3. `n8n_retraining_workflow.json` — Scheduled Retraining

**Trigger:** Cron — every **Sunday at 02:00** (`0 2 * * 0`)

**Node flow:**
```
Cron Weekly (0 2 * * 0)
    ↓
Health Check → GET http://127.0.0.1:8000/health
    ↓
IF API Healthy (status == "ok")
    ├── TRUE  → Retrain All Actors (POST /retrain, timeout: 600 000 ms)
    │               ↓
    │           IF Retrain Success (all_success == true)
    │               ├── TRUE  → Ping FastAPI Health (GET /health)
    │               └── FALSE → Log Partial Failure (noOp)
    │
    └── FALSE → Log API Down (noOp)
```

**How to import any workflow into n8n:**
1. Open n8n UI (default: `http://localhost:5678`)
2. **Workflows → Import from file**
3. Select the `.json` file
4. Configure credentials for Telegram / Gmail / Sheets
5. Activate the workflow

---

## Frontend Dashboard (`dashboard.html`)

A standalone HTML file that calls the prediction API directly from the browser.

**Open:** just double-click `dashboard.html` (requires Docker running on port 8000).

**Features:**
- **Health check on load** — green/red dot + API version + prediction count
- **3 actor cards** (teal / purple / red) with task selectors
- **Run button** per actor — calls `POST /predict` with real feature payloads
- **Live result display** — shows `data.result` with latency and timestamp
- **Request log** — bottom panel shows all calls with success/error coloring
- **Spinner + pulse animation** — visual feedback during and after calls

**Actor feature payloads used:**

| Actor | Features (count) |
|---|---|
| Actor 1 | 18 features from `xgboost_features.pkl` |
| Actor 2 | 9 mobility features (zone, line, mode, hour, congestion…) |
| Actor 3 | 7 security features (zone, accidents, crimes, taux…) |

---

## Key Known Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `mlflow: command not found` | Microsoft Store Python doesn't add scripts to PATH | Use `python -m mlflow` |
| `UnicodeEncodeError` on Windows | cp1252 terminal can't print emoji (✅❌🏃) | Set `PYTHONIOENCODING=utf-8` in subprocess env + reconfigure `sys.stdout/stderr` |
| `version` attribute warning in docker-compose | `version:` key is obsolete in Compose v2 | Safe to ignore |
| Cold start latency (~3s on first `/predict`) | Model loaded from disk on first request | Normal; subsequent calls use in-memory cache |
| `nvidia_nccl_cu12` pulling 300MB during build | xgboost pulls GPU lib automatically | Expected; only happens once — cached by Docker layers |

---

## Conclusion

This system demonstrates a **production-grade ML automation pipeline** combining:

- **Data science** — XGBoost, Random Forest, K-Means, Prophet, Isolation Forest
- **Backend engineering** — FastAPI, dynamic actor dispatch, REST API
- **Experiment tracking** — MLflow (params, metrics, artifact versioning, run comparison)
- **Containerized deployment** — Docker + Docker Compose (api + mlflow services)
- **Workflow automation** — n8n (webhook prediction, scheduled retraining, Telegram/Gmail/Sheets alerts)
- **Frontend** — Live browser dashboard with per-actor prediction UI

**Verified working end-to-end on 2026-04-23:**
- ✅ Docker build successful (`docker-compose up --build`)
- ✅ `/health` → `status: ok`, all 3 actors registered, v2.0.0
- ✅ `/predict` → `result: 2.2667 kg CO₂` (actor1, task=co2, HTTP 200)
- ✅ MLflow UI → 3 experiments × 2 runs (v1, v2) each with full `.pkl` artifacts
- ✅ n8n workflows imported: prediction (basic + extended) + retraining
- ✅ `dashboard.html` — live browser calls to `/predict` with animated results
