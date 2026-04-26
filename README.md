# ESPRIT MLOps Automation System
> University Project — ESPRIT Engineering School — Option ERP-BI
> Production-level multi-actor ML prediction and retraining system with full MLOps integration.

## 🧱 Stack
Table with: FastAPI, MLflow, n8n, Docker + Docker Compose, Google Gemini, Gmail

## 🎭 Actors
Table with actor1 (Ecological - XGBoost CO2/Energy + KMeans), actor2 (Mobility - XGBoost Charge/Cancellation), actor3 (Security - RF Severity + KMeans Risk + IsolationForest Anomaly)

## 📐 Architecture Overview
Copy the ASCII diagram exactly from Document 2

## 📁 Project Structure
ml_api_docker/
├── actors/              # actor1.py actor2.py actor3.py
├── training/            # train_actor1.py train_actor2.py train_actor3.py
├── mlflow/mlruns/       # MLflow persistent storage
├── results/             # predictions.json
├── main.py              # FastAPI application
├── Dockerfile
├── docker-compose.yml   # api + mlflow + n8n services
├── requirements.txt
├── dashboard.html       # Browser prediction UI
├── n8n_mlops_activation_workflow.json
├── n8n_prediction_workflow.json
├── n8n_prediction_workflow_extended.json
└── n8n_retraining_workflow.json

Training scripts (mounted via Docker volumes):
actor1_ecologique/main.py
actor2_mobilites/main.py
actor3_securite/main.py

## 🚀 Quick Start (Docker)

### Prerequisites
- Docker Desktop running
- Gmail OAuth2 credential (for n8n alerts)
- Google Gemini API key (for AI summaries)

### Start everything
cd ml_api_docker
docker-compose up --build

### Services
Table: FastAPI http://localhost:8000, Swagger http://localhost:8000/docs, MLflow http://localhost:5000, n8n http://localhost:5678

### Import n8n workflows
1. Open http://localhost:5678
2. Workflows → Import from file → import all 4 JSON files
3. Settings → Credentials → add Gmail OAuth2 and Google Gemini API

## ⚙️ Local Setup (Without Docker)
### Prerequisites
- Python 3.10+
- Node.js 18+
- n8n (npm install -g n8n)

### Start order (3 terminals)
Terminal 1 - MLflow:
cd ml_api_docker
python -m mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlflow/mlruns --default-artifact-root ./mlflow/mlruns --workers 1

Terminal 2 - FastAPI:
cd ml_api_docker
python -m uvicorn main:app --host 0.0.0.0 --port 8000

Terminal 3 - n8n:
n8n start

## 🔮 API Endpoints
Table: GET /health, POST /predict, POST /retrain, GET /predictions, GET /docs

## 🔮 Making Predictions
Copy all curl examples from Document 2 exactly

## 📊 Response Format
Copy the JSON response example from Document 2

## 🤖 MLOps Workflow (n8n)
Triggered manually — runs full pipeline:
1. Health check FastAPI
2. Retrain all 3 actors (overwrites .pkl files + logs to MLflow)
3. Check MLflow is running
4. Gemini generates AI summary
5. Gmail sends full report (success or failure)

## 📊 MLflow Experiments
Table: actor1_ecologique (v1,v2 - 11 pkls), actor2_mobilites (v1,v2 - 5 pkls), actor3_securite (v1,v2 - 8 pkls)

## 🎯 Actor & Task Reference
Copy the full table from Document 2 exactly

## 🚦 Error Handling
Copy the HTTP codes table from Document 2 exactly

## 🏗️ Key Design Decisions
Copy the 6 numbered points from Document 2 exactly

## 📋 Rubric Alignment
Copy the full rubric table from Document 2 and add these extra rows:
| MLflow experiment tracking | 3 experiments x 2 runs, params + metrics + artifacts |
| Docker containerization | docker-compose with api + mlflow + n8n on ml_network |
| MLOps activation workflow | n8n manual trigger → retrain → MLflow → Gemini → Gmail |
| AI-generated summaries | Google Gemini API via n8n HTTP Request node |

## ✅ Verified Working
- Docker build successful (docker-compose up --build)
- /health → status ok, all 3 actors registered, v2.0.0
- /predict → result 2.2667 kg CO2 (actor1, task=co2, HTTP 200)
- MLflow UI → 3 experiments x 2 runs with full .pkl artifacts
- n8n workflows → prediction + retraining + MLops activation
- Gmail → success email with Gemini AI summary
- dashboard.html → live browser predictions

---
ESPRIT Engineering School — ML Automation System — Week S12 MLOps Phase
