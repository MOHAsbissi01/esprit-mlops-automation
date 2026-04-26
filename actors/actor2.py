"""
Actor 2 — Directeur Mobilités (RATP / Île-de-France Mobilités)
===============================================================
Wraps inference for:
  - charge       : XGBoost regression → passenger load (charge_estimee 0–1)
  - cancellation : XGBoost classification → trip cancellation risk

Models are loaded lazily and cached.  Feature lists come from *_features.pkl
so the API is always in sync with whatever was trained.

PKL files (from actor2_mobilites/outputs/):
  xgboost_charge.pkl               — trained regressor
  xgboost_charge_features.pkl      — ordered feature names
  xgboost_cancellation.pkl         — trained classifier
  xgboost_cancellation_features.pkl
  charge_encoding.pkl              — encoding maps (line/zone means, etc.)
"""

import logging
import os
import pickle
import threading
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("ml_api.actor2")

# ── Paths ──────────────────────────────────────────────────────────────────────
# Absolute: actor2.py → actors/ → ml_api/ → Desktop/ → actor2_mobilites/outputs/
BASE = Path(__file__).resolve().parent.parent.parent / "actor2_mobilites" / "outputs"

# ── In-memory cache ────────────────────────────────────────────────────────────
_cache: Dict[str, Any] = {}
_lock = threading.Lock()


def _load(filename: str) -> Any:
    """Load a .pkl file from actor2 outputs, with in-memory caching."""
    key = str(filename)
    if key not in _cache:
        with _lock:
            if key not in _cache:
                path = BASE / filename
                if not path.exists():
                    raise FileNotFoundError(
                        f"Actor2 model file not found: {path}. "
                        "Run actor2_mobilites/main.py to train models."
                    )
                logger.info(f'"Actor2: loading {filename}"')
                with open(path, "rb") as f:
                    _cache[key] = pickle.load(f)
    return _cache[key]


# ── Public interface ───────────────────────────────────────────────────────────
def predict(task: str, data: dict) -> Any:
    """
    Run inference for actor2.

    Parameters
    ----------
    task : str
        One of: 'charge', 'cancellation'
    data : dict
        Must contain a 'features' dict.

    Returns
    -------
    float | dict
        charge      → float (load factor 0–1)
        cancellation → {"probability": float, "prediction": int, "risk_level": str}
    """
    features: dict = data.get("features", {})

    # ── Passenger-load Regression ──────────────────────────────────────────────
    if task == "charge":
        model = _load("xgboost_charge.pkl")
        feature_list = _load("xgboost_charge_features.pkl")

        _warn_missing(features, feature_list, task)
        X = [[features.get(f, 0) for f in feature_list]]
        result = float(model.predict(X)[0])

        logger.info(f'"Actor2 charge: result={result:.4f}"')
        return result

    # ── Trip-cancellation Classification ──────────────────────────────────────
    elif task == "cancellation":
        model = _load("xgboost_cancellation.pkl")
        feature_list = _load("xgboost_cancellation_features.pkl")

        _warn_missing(features, feature_list, task)
        X = [[features.get(f, 0) for f in feature_list]]

        # predict_proba returns [[prob_class0, prob_class1]]
        prob_cancel = float(model.predict_proba(X)[0][1])

        # Tuned threshold 0.047 accounts for extreme class imbalance (1.78% positives)
        THRESHOLD = 0.047
        prediction = int(prob_cancel > THRESHOLD)
        risk_level = _cancellation_risk_level(prob_cancel)

        result = {
            "cancellation_probability": round(prob_cancel, 4),
            "prediction": prediction,          # 1 = likely cancelled
            "risk_level": risk_level,          # LOW / MEDIUM / HIGH
            "threshold_used": THRESHOLD,
        }

        logger.info(
            f'"Actor2 cancellation: prob={prob_cancel:.4f} '
            f'pred={prediction} risk={risk_level}"'
        )
        return result

    else:
        raise ValueError(
            f"Unknown task '{task}' for actor2. "
            "Valid tasks: ['charge', 'cancellation']"
        )


def _cancellation_risk_level(prob: float) -> str:
    """Convert a cancellation probability into a readable risk label."""
    if prob < 0.10:
        return "LOW"
    elif prob < 0.40:
        return "MEDIUM"
    return "HIGH"


def _warn_missing(features: dict, required: list, task: str) -> None:
    """Log a warning for any features that will be imputed with 0."""
    missing = [f for f in required if f not in features]
    if missing:
        logger.warning(
            f'"Actor2 {task}: missing features {missing} — defaulting to 0"'
        )