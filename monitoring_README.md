# S13 Monitoring — ML API

## Stack

| Component | Role |
|---|---|
| **Prometheus** | Scrapes `/metrics` every 10 s, stores time-series in local TSDB |
| **Grafana** | Visualises Prometheus data — dashboards for latency, error rate, confidence, request counts |
| **drift_detector.py** | Reads `results/predictions.json`, runs KS-test + confidence-drop check per actor |
| **alerting.py** | Polls Prometheus live, runs drift checks, fires alerts to `ml_api.log` + n8n webhook |
| **simulate_scenarios.py** | Fires 3 rounds × 13 scenarios across all 3 actors to generate realistic metric traffic |

---

## Prerequisites

- **Python packages:** `prometheus-client`, `scipy`, `requests` — already in `requirements.txt`
- **Prometheus binary:** `C:\prometheus-3.11.3.windows-amd64\prometheus-3.11.3.windows-amd64\prometheus.exe`
- **Grafana:** running on `localhost:3000` (Windows service or manual start)

---

## Startup Order (6 terminals)

**Terminal 1 — MLflow**
```powershell
cd "C:\Users\sbiss\OneDrive - ESPRIT\Desktop\ml_api_2"
python -m mlflow server --host 0.0.0.0 --port 5000 `
  --backend-store-uri ./mlflow/mlruns `
  --default-artifact-root ./mlflow/mlruns `
  --workers 1
```

**Terminal 2 — FastAPI**
```powershell
cd "C:\Users\sbiss\OneDrive - ESPRIT\Desktop\ml_api_2"
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**Terminal 3 — n8n**
```powershell
n8n start
```

**Terminal 4 — Prometheus**
```powershell
"C:\prometheus-3.11.3.windows-amd64\prometheus-3.11.3.windows-amd64\prometheus.exe" `
  --config.file="C:\Users\sbiss\OneDrive - ESPRIT\Desktop\ml_api_2\prometheus.yml"
```

**Terminal 5 — Grafana**
```
Check http://localhost:3000  (Windows service auto-starts; no command needed)
```

**Terminal 6 — Alerting (continuous)**
```powershell
cd "C:\Users\sbiss\OneDrive - ESPRIT\Desktop\ml_api_2"
python alerting.py
```

---

## Grafana Setup (one-time)

1. Open `http://localhost:3000` → login `admin` / `admin`
2. **Configuration → Data Sources → Add data source → Prometheus**
   - URL: `http://localhost:9090`
   - Click **Save & Test** — expect green "Data source is working"
3. **Dashboards → Import → Upload `grafana_dashboard.json`**
   - Set `DS_PROMETHEUS` to the data source created above → **Import**

---

## Baseline Values

| Actor  | Confidence baseline | Max latency (p95) | Max error rate |
|--------|--------------------:|------------------:|---------------:|
| actor1 | 0.85                | 2.0 s             | 10 %           |
| actor2 | 0.82                | 2.0 s             | 10 %           |
| actor3 | 0.80                | 2.0 s             | 10 %           |

Thresholds are defined as constants at the top of `alerting.py` and can be changed there without touching any other file.

---

## Running Simulations

```powershell
python simulate_scenarios.py
# Runs 3 rounds × 13 scenarios across all 3 actors
# Watch Grafana live at http://localhost:3000
```

---

## Alerting

```powershell
python alerting.py
# Runs one full check immediately, then loops every 30 s
# Alerts written to ml_api.log
# Violations POSTed to n8n webhook → Telegram / Gmail
```

Four alert rules fired by `alerting.py`:

| Rule | Trigger |
|---|---|
| `high_latency_p95` | Prometheus p95 latency > 2.0 s |
| `high_error_rate` | `ml_api_error_rate` gauge > 10 % |
| `low_model_confidence` | `ml_api_model_confidence` gauge < 0.75 |
| `distribution_drift` | KS-test detects shift vs. historical baseline |
| `confidence_drop` | Mean prediction value drops > 5 % below baseline |

---

## Watch Logs Live (PowerShell)

```powershell
Get-Content "C:\Users\sbiss\OneDrive - ESPRIT\Desktop\ml_api_2\ml_api.log" -Wait -Tail 20
```

Alert lines follow this format:
```
[ALERT] 2026-05-03T14:05:00+00:00 | rule=high_latency_p95 | actor=actor1 | value=37.49 | details=p95=37.487s > threshold=2.0s
```

---

## Observability Explained

| Layer | Tool | Answers |
|---|---|---|
| **Metrics** | Prometheus + Grafana | *What* is happening — counters, gauges, histograms over time |
| **Logs** | `ml_api.log` | *Why* it happened — errors, drift triggers, alert details |

---

## Drift Detection

```powershell
python drift_detector.py
# Reads results/predictions.json
# KS-test per actor — recent 50 predictions vs. oldest 100 (baseline)
# Flags confidence drop > 5 % below per-actor baseline
# Output: {actor: {drift: bool, confidence_drop: bool}}
```

Drift is also run automatically inside every `alerting.py` cycle — no separate process needed during continuous monitoring.

---

## S13 Deliverables Checklist

- [ ] Prometheus scraping `/metrics` every 10 s (`health="up"` in `/api/v1/targets`)
- [ ] Grafana dashboard imported (5 panels — requests, latency, error rate, confidence, drift)
- [ ] `alerting.py` running continuously with all 4 alert rules active
- [ ] `drift_detector.py` passing for all 3 actors (output shows no false positives on clean data)
- [ ] `simulate_scenarios.py` demonstrated — 3 rounds × 13 scenarios visible in Grafana
- [ ] `ml_api.log` showing `[ALERT]` entries and structured JSON request logs
- [ ] `monitoring_README.md` complete ✓
