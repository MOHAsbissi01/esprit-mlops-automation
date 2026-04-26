"""
ESPRIT University Project — ML Automation System
=================================================
FastAPI multi-actor prediction backend.

Architecture:
  POST /predict  → dynamic dispatch to actor1 / actor2 / actor3
  GET  /health   → liveness probe
  GET  /predictions → browse stored predictions

Each actor handles its own model loading, feature validation and inference.
Every prediction is appended to results/predictions.json (thread-safe).

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── Import actor modules ───────────────────────────────────────────────────────
from actors import actor1, actor2, actor3

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
PREDICTIONS_FILE = RESULTS_DIR / "predictions.json"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Structured JSON Logging ────────────────────────────────────────────────────
LOG_FORMAT = (
    '{"time":"%(asctime)s","level":"%(levelname)s",'
    '"module":"%(module)s","message":%(message)s}'
)
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "ml_api.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ml_api")

# ── Thread-safe prediction storage ────────────────────────────────────────────
_file_lock = threading.Lock()

# ── Valid actors and their supported tasks ─────────────────────────────────────
ACTOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "actor1": {
        "module": actor1,
        "tasks": ["co2", "energy", "cluster"],
        "description": "Ecological Director (CO2, energy, pollution clustering)",
    },
    "actor2": {
        "module": actor2,
        "tasks": ["charge", "cancellation"],
        "description": "Mobility Director (passenger load, cancellation risk)",
    },
    "actor3": {
        "module": actor3,
        "tasks": ["severity", "risk_cluster", "anomaly"],
        "description": "Security Manager (severity classification, risk zones, anomaly detection)",
    },
}


# ── Pydantic Schemas ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """
    Unified prediction request.

    - actor:    which actor to invoke  (actor1 | actor2 | actor3)
    - task:     which ML model to use  (depends on actor)
    - features: dict of raw feature values
    """

    actor: str = Field(..., description="Actor identifier: actor1 | actor2 | actor3")
    task: str = Field(..., description="Task / model identifier")
    features: Dict[str, Any] = Field(..., description="Feature key-value pairs")

    @field_validator("actor")
    @classmethod
    def actor_must_be_valid(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ACTOR_REGISTRY:
            raise ValueError(
                f"Unknown actor '{v}'. Valid options: {list(ACTOR_REGISTRY.keys())}"
            )
        return v

    @field_validator("task")
    @classmethod
    def task_must_be_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("task must not be empty")
        return v.strip().lower()

    @field_validator("features")
    @classmethod
    def features_must_be_non_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError("features dict must not be empty")
        return v


class PredictResponse(BaseModel):
    status: str
    actor: str
    task: str
    result: Any
    latency_ms: float
    timestamp: str


class SaveResultRequest(BaseModel):
    """Payload sent by n8n to persist a prediction record."""

    timestamp: str
    actor: str
    task: str
    result: Any
    latency_ms: Optional[float] = None
    source: str = "n8n-workflow"


# ── Helpers ────────────────────────────────────────────────────────────────────
def save_prediction(record: dict) -> int:
    """Append a prediction record to predictions.json (thread-safe)."""
    with _file_lock:
        existing: list = []
        if PREDICTIONS_FILE.exists():
            try:
                with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, ValueError):
                logger.warning('"predictions.json was corrupt — resetting"')
                existing = []
        existing.append(record)
        with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    return len(existing)


# ── FastAPI Application ────────────────────────────────────────────────────────
app = FastAPI(
    title="ESPRIT ML Automation API",
    description=(
        "Production-level multi-actor ML prediction backend.\n\n"
        "Supports 3 actors (ecological, mobility, security) with "
        "dynamic model dispatch, structured logging, and persistent "
        "prediction storage."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Logging Middleware ─────────────────────────────────────
@app.middleware("http")
async def request_logger(request: Request, call_next):
    t0 = time.perf_counter()
    client = request.client.host if request.client else "unknown"
    logger.info(f'"Incoming: {request.method} {request.url.path} client={client}"')
    response = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        f'"Completed: {request.method} {request.url.path} '
        f'status={response.status_code} latency_ms={ms}"'
    )
    return response


# ── Global Exception Handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f'"Unhandled exception on {request.url.path}: {type(exc).__name__}: {exc}"')
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": f"Internal server error: {str(exc)}",
            "path": str(request.url.path),
        },
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    """Liveness probe — returns actor registry and stored prediction count."""
    total_predictions = 0
    if PREDICTIONS_FILE.exists():
        try:
            with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                total_predictions = len(json.load(f))
        except Exception:
            pass

    return {
        "status": "ok",
        "version": "2.0.0",
        "actors": {
            name: {
                "tasks": meta["tasks"],
                "description": meta["description"],
            }
            for name, meta in ACTOR_REGISTRY.items()
        },
        "predictions_stored": total_predictions,
        "predictions_file": str(PREDICTIONS_FILE),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/predictions", tags=["Storage"])
async def list_predictions(limit: int = 50, actor: Optional[str] = None):
    """
    Browse stored predictions.

    Query params:
      limit - max records to return (default 50)
      actor - filter by actor name (optional)
    """
    if not PREDICTIONS_FILE.exists():
        return {"total": 0, "predictions": []}

    with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        all_preds = json.load(f)

    if actor:
        all_preds = [p for p in all_preds if p.get("actor") == actor]

    return {
        "total": len(all_preds),
        "showing": min(limit, len(all_preds)),
        "predictions": all_preds[-limit:],
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(body: PredictRequest):
    """
    Unified ML prediction endpoint.

    Routes the request to the correct actor module, loads the appropriate
    .pkl model, runs inference, stores the result, and returns structured JSON.

    **Example — actor2 / cancellation risk:**
    ```json
    {
      "actor": "actor2",
      "task": "cancellation",
      "features": {
        "heure": 8,
        "delay_minutes": 12,
        "temperature": 5.0
      }
    }
    ```
    """
    t_start = time.perf_counter()

    # ── 1. Validate actor / task combination ──────────────────────────────────
    actor_meta = ACTOR_REGISTRY[body.actor]  # already validated by Pydantic
    if body.task not in actor_meta["tasks"]:
        logger.warning(
            f'"Invalid task — actor={body.actor} task={body.task} '
            f'valid_tasks={actor_meta["tasks"]}"'
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_task",
                "actor": body.actor,
                "task": body.task,
                "valid_tasks": actor_meta["tasks"],
            },
        )

    logger.info(
        f'"Predict request — actor={body.actor} task={body.task} '
        f'features_keys={list(body.features.keys())}"'
    )

    # ── 2. Dispatch to actor module ────────────────────────────────────────────
    try:
        actor_module = actor_meta["module"]
        result = actor_module.predict(body.task, {"features": body.features})
    except ValueError as exc:
        latency_ms = round((time.perf_counter() - t_start) * 1000, 3)
        logger.error(f'"ValueError — actor={body.actor} task={body.task} error={exc}"')
        raise HTTPException(
            status_code=422,
            detail={"error": "validation_error", "detail": str(exc)},
        )
    except FileNotFoundError as exc:
        latency_ms = round((time.perf_counter() - t_start) * 1000, 3)
        logger.error(f'"Model file not found — actor={body.actor} task={body.task} error={exc}"')
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_found",
                "detail": str(exc),
                "hint": "Run the actor training pipeline first to generate .pkl files.",
            },
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - t_start) * 1000, 3)
        logger.error(
            f'"Unexpected error — actor={body.actor} task={body.task} '
            f'error_type={type(exc).__name__} error={exc}"'
        )
        raise HTTPException(
            status_code=500,
            detail={"error": "pipeline_failure", "detail": str(exc)},
        )

    # ── 3. Record and return ───────────────────────────────────────────────────
    latency_ms = round((time.perf_counter() - t_start) * 1000, 3)
    timestamp = datetime.now(timezone.utc).isoformat()

    record = {
        "timestamp": timestamp,
        "actor": body.actor,
        "task": body.task,
        "features": body.features,
        "result": result,
        "latency_ms": latency_ms,
        "status": "success",
    }

    total = save_prediction(record)
    logger.info(
        f'"Prediction success — actor={body.actor} task={body.task} '
        f'latency_ms={latency_ms} total_stored={total}"'
    )

    return PredictResponse(
        status="success",
        actor=body.actor,
        task=body.task,
        result=result,
        latency_ms=latency_ms,
        timestamp=timestamp,
    )


# ── Save n8n Result Endpoint ──────────────────────────────────────────────────
@app.post("/save-n8n-result", tags=["Storage"])
async def save_n8n_result(body: SaveResultRequest):
    """
    Persist a prediction record sent by n8n.

    n8n Code nodes cannot use Node.js built-ins like `fs` because the
    sandboxed task runner disallows them. This endpoint lets n8n delegate
    all file I/O to FastAPI (Python) instead.
    """
    record = {
        "timestamp": body.timestamp,
        "actor": body.actor,
        "task": body.task,
        "result": body.result,
        "latency_ms": body.latency_ms,
        "source": body.source,
    }
    total = save_prediction(record)
    logger.info(
        f'"n8n save — actor={body.actor} task={body.task} total_stored={total}"'
    )
    return {"status": "saved", "total_stored": total, "record": record}


# ── Retraining Endpoint ───────────────────────────────────────────────────────
DESKTOP = BASE_DIR.parent  # ml_api/ → Desktop/

ACTOR_SCRIPTS = [
    {"name": "actor1", "script": str(DESKTOP / "actor1_ecologique" / "main.py")},
    {"name": "actor2", "script": str(DESKTOP / "actor2_mobilites" / "main.py")},
    {"name": "actor3", "script": str(DESKTOP / "actor3_securite" / "main.py")},
]


@app.post("/retrain", tags=["Retraining"])
def retrain():
    """
    Trigger retraining of all actor models.
    Runs each actor's main.py sequentially and returns per-actor status.
    Called by n8n retraining workflow via HTTP Request node.

    UTF-8 fix: Windows defaults subprocess stdout/stderr to cp1252 which
    cannot encode emoji characters used in actor training scripts.
    We force UTF-8 via both PYTHONUTF8=1 (Python 3.7+) and
    PYTHONIOENCODING=utf-8 (legacy fallback), plus explicit encoding= arg.
    """
    # Build a UTF-8-safe environment for all child processes
    utf8_env = os.environ.copy()
    utf8_env["PYTHONUTF8"] = "1"
    utf8_env["PYTHONIOENCODING"] = "utf-8"

    results: Dict[str, Any] = {}
    for actor in ACTOR_SCRIPTS:
        script_path = Path(actor["script"])
        if not script_path.exists():
            results[actor["name"]] = {
                "status": "SKIPPED",
                "reason": f"Script not found: {script_path}",
            }
            continue
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(script_path.parent),
                capture_output=True,
                text=True,
                encoding="utf-8",          # explicit UTF-8 for stdout/stderr
                errors="replace",          # replace any remaining un-encodable chars
                env=utf8_env,              # force PYTHONUTF8=1 in the child process
                timeout=600,
            )
            results[actor["name"]] = {
                "status": "SUCCESS" if proc.returncode == 0 else "FAILED",
                "exit_code": proc.returncode,
                "stdout": proc.stdout[-500:] if proc.stdout else "",
                "stderr": proc.stderr[-300:] if proc.stderr else "",
            }
            logger.info(f'"Retrain {actor["name"]}: exit={proc.returncode}"')
        except subprocess.TimeoutExpired:
            results[actor["name"]] = {"status": "TIMEOUT", "exit_code": -1}
        except Exception as exc:
            results[actor["name"]] = {"status": "ERROR", "detail": str(exc)}

    all_success = all(v.get("status") == "SUCCESS" for v in results.values())
    return {
        "status": "complete",
        "all_success": all_success,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actors": results,
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")