# 🤖 ESPRIT ML Automation System — n8n + FastAPI

> **University Project · Part 1: n8n ML Automation**
> Production-level multi-actor ML prediction and retraining system.

---

## 📐 Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │           n8n (port 5678)           │
                    │                                     │
                    │  ┌──────────────┐                  │
                    │  │ Prediction   │  Webhook trigger  │
                    │  │  Workflow    │◄──────────────────┤
                    │  └──────┬───────┘                  │
                    │         │ HTTP POST                 │
                    │  ┌──────▼───────┐                  │
                    │  │ Retraining   │  Cron (weekly)    │
                    │  │  Workflow    │◄──────────────────┤
                    │  └──────┬───────┘                  │
                    └─────────┼───────────────────────────┘
                              │
                    ┌─────────▼───────────────────────────┐
                    │       FastAPI (port 8000)           │
                    │                                     │
                    │  POST /predict                      │
                    │  ┌──────────────────────────────┐  │
                    │  │  Dynamic Actor Router        │  │
                    │  └──────┬──────┬──────┬─────────┘  │
                    │         │      │      │             │
                    │    actor1  actor2  actor3           │
                    │    (eco)  (mob)  (sec)             │
                    └─────────────────────────────────────┘
                              │
                    ┌─────────▼───────────────────────────┐
                    │       .pkl Model Files              │
                    │  actor1_ecologique/outputs/         │
                    │  actor2_mobilites/outputs/          │
                    │  actor3_securite/outputs/           │
                    └─────────────────────────────────────┘
                              │
                    ┌─────────▼───────────────────────────┐
                    │  results/predictions.json           │
                    │  (append-only, thread-safe)         │
                    └─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ml_api/
├── main.py                         # FastAPI application (unified /predict endpoint)
├── requirements.txt                # Python dependencies
├── ml_api.log                      # Runtime log file (auto-created)
├── n8n_prediction_workflow.json    # n8n Prediction Workflow (import-ready)
├── n8n_retraining_workflow.json    # n8n Retraining Workflow (import-ready)
├── README.md                       # ← You are here
│
├── actors/
│   ├── __init__.py
│   ├── actor1.py                   # Ecological: co2, energy, cluster
│   ├── actor2.py                   # Mobility: charge, cancellation
│   └── actor3.py                   # Security: severity, risk_cluster, anomaly
│
├── utils/
│   ├── __init__.py
│   └── loader.py                   # Shared thread-safe .pkl loader with cache
│
└── results/
    └── predictions.json            # Persistent prediction store (append mode)
```

**Training scripts (outside ml_api/):**

```
actor1_ecologique/main.py           # Retrain Actor1 models
actor2_mobilites/main.py            # Retrain Actor2 models
actor3_securite/main.py             # Retrain Actor3 models
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites

- Python 3.10+
- Node.js 18+ (for n8n)
- n8n (`npm install -g n8n`)

### 2. Install Python dependencies

```bash
cd "C:/Users/sbiss/OneDrive - ESPRIT/Desktop/ml_api"
pip install -r requirements.txt
```

### 3. Ensure trained models exist

The actor1/2/3 `outputs/` directories must contain `.pkl` files.
If not already trained, run each pipeline once:

```bash
# From the Desktop folder
cd actor1_ecologique && python main.py
cd ../actor2_mobilites && python main.py
cd ../actor3_securite  && python main.py
```

---

## 🚀 Running FastAPI

```bash
# From the Desktop folder (important — models are loaded relative to CWD)
cd "C:/Users/sbiss/OneDrive - ESPRIT/Desktop"
uvicorn ml_api.main:app --reload --host 0.0.0.0 --port 8000
```

Or from within `ml_api/`:

```bash
cd "C:/Users/sbiss/OneDrive - ESPRIT/Desktop/ml_api"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

> ⚠️ **Important**: Run from the **Desktop** directory so that relative paths like
> `actor1_ecologique/outputs/xgboost_co2.pkl` resolve correctly.

**Verify it's running:**

```
http://localhost:8000/health   → system status
http://localhost:8000/docs     → Swagger UI (interactive API docs)
http://localhost:8000/redoc    → ReDoc UI
```

---

## 🚀 Running n8n

```bash
n8n start
```

Then open: [http://localhost:5678](http://localhost:5678)

---

## 📥 Importing n8n Workflows

1. Open n8n → **Workflows** → **Import from File**
2. Import `ml_api/n8n_prediction_workflow.json`
3. Import `ml_api/n8n_retraining_workflow.json`
4. Activate both workflows

---

## 🔮 Making Predictions

### Via Swagger UI (quickest)

Open [http://localhost:8000/docs](http://localhost:8000/docs) → POST `/predict`

### Via curl / Postman

**Actor1 — CO2 Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "actor": "actor1",
    "task": "co2",
    "features": {"temp": 22.5, "humidity": 65.0}
  }'
```

**Actor2 — Passenger Load:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "actor": "actor2",
    "task": "charge",
    "features": {"heure": 8, "delay_minutes": 0, "temperature": 5.0}
  }'
```

**Actor2 — Cancellation Risk:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "actor": "actor2",
    "task": "cancellation",
    "features": {"heure": 18, "delay_minutes": 45}
  }'
```

**Actor3 — Accident Severity:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "actor": "actor3",
    "task": "severity",
    "features": {"nb_accidents": 3, "gravite_index": 1.4}
  }'
```

**Actor3 — Anomaly Detection:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "actor": "actor3",
    "task": "anomaly",
    "features": {"nb_accidents": 150, "gravite_index": 4.8}
  }'
```

### Via n8n Webhook

Send a POST to the webhook URL shown in n8n (after activating the prediction workflow):

```
POST http://localhost:5678/webhook/ml-predict
Content-Type: application/json

{
  "actor": "actor2",
  "task": "charge",
  "features": { "heure": 8, "delay_minutes": 12 }
}
```

---

## 📊 Response Format

Every prediction returns:

```json
{
  "status": "success",
  "actor": "actor2",
  "task": "cancellation",
  "result": {
    "cancellation_probability": 0.0312,
    "prediction": 0,
    "risk_level": "LOW",
    "threshold_used": 0.047
  },
  "latency_ms": 12.4,
  "timestamp": "2026-04-17T12:34:56+00:00"
}
```

---

## 💾 Prediction Storage

Every prediction is automatically appended to:

```
ml_api/results/predictions.json
```

The file is also accessible via:

```
GET http://localhost:8000/predictions?limit=50&actor=actor2
```

---

## 🔁 Retraining

### Manual (via CLI)

```bash
cd "C:/Users/sbiss/OneDrive - ESPRIT/Desktop/actor2_mobilites"
python main.py
```

### Automated (via n8n)

The retraining workflow runs every **Sunday at 02:00 AM**.
It executes all 3 actor pipelines sequentially and logs results.

To trigger manually in n8n: open the Retraining Workflow → **Execute Workflow**.

---

## 🎯 Actor & Task Reference

| Actor  | Task          | Model              | Output Type         |
|--------|---------------|--------------------|---------------------|
| actor1 | co2           | XGBoost Regressor  | `float` (CO2 value) |
| actor1 | energy        | XGBoost Regressor  | `float` (NRJ value) |
| actor1 | cluster       | K-Means            | `{"cluster": int}`  |
| actor2 | charge        | XGBoost Regressor  | `float` (0–1 load)  |
| actor2 | cancellation  | XGBoost Classifier | `{"cancellation_probability": float, "prediction": 0/1, "risk_level": str}` |
| actor3 | severity      | Random Forest      | `{"severity_class": int, "severity_label": str, "probabilities": dict}` |
| actor3 | risk_cluster  | K-Means            | `{"risk_cluster": int, "risk_label": str}` |
| actor3 | anomaly       | Isolation Forest   | `{"is_anomaly": 0/1, "anomaly_label": str, "anomaly_score": float}` |

---

## 🚦 Error Handling

| HTTP Code | Meaning                                    |
|-----------|-------------------------------------------|
| 200       | Prediction success                        |
| 422       | Invalid actor, task, or missing features  |
| 503       | Model .pkl not found (retrain first)      |
| 500       | Unexpected pipeline failure               |

---

## 🏗️ Key Design Decisions

1. **Single unified `/predict` endpoint** — cleaner than per-actor URLs, easier to route in n8n.
2. **Lazy model loading with in-memory caching** — first request pays IO cost, subsequent requests are fast.
3. **Thread-safe file writes** — `threading.Lock()` prevents race conditions when multiple n8n executions arrive simultaneously.
4. **Pydantic v2 validation** — actor and task are validated before touching any model file.
5. **Structured JSON logging** — all logs are machine-readable for future log-aggregation tooling.
6. **CORS enabled** — allows n8n, Swagger UI, and any browser-based client to call the API.

---

## 📋 Rubric Alignment

| Criterion                    | Implementation                                          |
|------------------------------|---------------------------------------------------------|
| FastAPI endpoint             | `POST /predict` with dynamic actor+task routing         |
| Dynamic model loading        | `_load()` with thread-safe cache in each actor module   |
| Structured JSON output       | `PredictResponse` Pydantic model                        |
| Error handling               | HTTP 422/503/500 with descriptive detail messages       |
| Prediction storage           | Append-only `results/predictions.json`                  |
| n8n Prediction Workflow      | Webhook → Set → HTTP → IF → Write → Respond             |
| n8n Retraining Workflow      | Cron → Execute ×3 → Aggregate → IF → Log               |
| Logging                      | Structured JSON, per-module loggers, file + stdout      |
| README / Documentation       | This file                                               |
| Modular code                 | `actors/`, `utils/` packages, clear separation          |
| Clean n8n node naming        | Emoji + descriptive names throughout workflows          |

---

*Generated for ESPRIT University — ML Automation Project (Part 1)*
