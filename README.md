 
# ESPRIT MLOps Automation System

> University Project — ESPRIT Engineering School
> Multi-actor Machine Learning automation pipeline with full MLOps integration.

## Stack
- **FastAPI** — REST API for model serving
- **MLflow** — Experiment tracking, run comparison, artifact versioning
- **n8n** — Workflow orchestration (prediction + retraining + alerts)
- **Docker + Docker Compose** — Full containerized deployment
- **Google Gemini** — AI-generated pipeline summaries in email reports
- **Gmail** — Automated success/failure notifications

## Actors
| Actor | Domain | Models |
|---|---|---|
| actor1 | Ecological | XGBoost CO2, XGBoost Energy, K-Means Clustering |
| actor2 | Mobility | XGBoost Passenger Load, XGBoost Cancellation Risk |
| actor3 | Security | Random Forest Severity, K-Means Risk Zones, Isolation Forest Anomaly |

## Project Structure
ml_api_docker/
├── actors/              # actor1.py actor2.py actor3.py
├── training/            # train_actor1.py train_actor2.py train_actor3.py
├── mlflow/mlruns/       # MLflow persistent storage
├── results/             # predictions.json
├── main.py              # FastAPI application
├── Dockerfile
├── docker-compose.yml   # api + mlflow + n8n
├── requirements.txt
├── dashboard.html       # Browser prediction UI
├── n8n_mlops_activation_workflow.json
├── n8n_prediction_workflow.json
├── n8n_prediction_workflow_extended.json
└── n8n_retraining_workflow.json

## Quick Start

### Prerequisites
- Docker Desktop running
- Gmail OAuth2 credential (for n8n)
- Google Gemini API key (for AI summaries)

### Start everything
docker-compose up --build

### Services
| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| n8n | http://localhost:5678 |

### Import n8n workflows
1. Open http://localhost:5678
2. Workflows → Import from file
3. Import all 4 workflow JSON files
4. Add Gmail and Gemini credentials in Settings

## API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | /health | System status |
| POST | /predict | Run prediction |
| POST | /retrain | Trigger retraining |
| GET | /predictions | Browse stored predictions |
| GET | /docs | Swagger UI |

## MLOps Workflow
Triggered manually in n8n — runs the full pipeline:
1. Health check FastAPI
2. Retrain all 3 actors (overwrites .pkl files)
3. Logs new run to MLflow
4. Gemini generates AI summary
5. Gmail sends full report with details

## MLflow Experiments
| Experiment | Runs | Artifacts |
|---|---|---|
| actor1_ecologique | v1, v2 | 11 .pkl files |
| actor2_mobilites | v1, v2 | 5 .pkl files |
| actor3_securite | v1, v2 | 8 .pkl files |

## Verified Working
- Docker build successful
- /health → status ok, all 3 actors registered
- /predict → result returned (actor1 co2: 2.2667 kg)
- MLflow UI → 3 experiments x 2 runs with full artifacts
- n8n workflows → prediction + retraining + MLops activation
- Gmail → success email with Gemini AI summary
- dashboard.html → live browser predictions

## Course
ESPRIT Engineering School — Option ERP-BI
ML Automation System — Week S12 MLOps Phase
ML Automation System — Week S13 Grafana + prometheus