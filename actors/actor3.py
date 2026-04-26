"""
Actor 3 — Responsable Sécurité des Transports Urbains
======================================================
Wraps inference for:
  - severity    : Random Forest classification → accident severity (0/1/2)
  - risk_cluster: K-Means clustering → zone risk profile (Low/Medium/High)
  - anomaly     : Isolation Forest → abnormal crime/accident spike detection

Models are loaded lazily and cached.

PKL files (from actor3_securite/outputs/):
  rf_severity.pkl          — trained Random Forest classifier
  rf_severity_features.pkl — ordered feature names for severity model
  kmeans_risk.pkl          — trained K-Means
  kmeans_scaler.pkl        — StandardScaler fitted for K-Means
  kmeans_features.pkl      — feature names used by K-Means
  isolation_forest.pkl     — trained Isolation Forest
  anomaly_scaler.pkl       — scaler for Isolation Forest
  anomaly_features.pkl     — feature names for Isolation Forest
"""

import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("ml_api.actor3")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent / "actor3_securite" / "outputs"

# ── In-memory cache ────────────────────────────────────────────────────────────
_cache: Dict[str, Any] = {}
_lock = threading.Lock()

# ── Severity label mapping ─────────────────────────────────────────────────────
SEVERITY_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# ── Risk cluster label mapping (based on k=3 clusters) ────────────────────────
RISK_CLUSTER_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


def _load(filename: str) -> Any:
    """Load a .pkl file from actor3 outputs, with in-memory caching."""
    key = str(filename)
    if key not in _cache:
        with _lock:
            if key not in _cache:
                path = BASE / filename
                if not path.exists():
                    raise FileNotFoundError(
                        f"Actor3 model file not found: {path}. "
                        "Run actor3_securite/main.py to train models."
                    )
                logger.info(f'"Actor3: loading {filename}"')
                with open(path, "rb") as f:
                    _cache[key] = pickle.load(f)
    return _cache[key]


# ── Public interface ───────────────────────────────────────────────────────────
def predict(task: str, data: dict) -> Any:
    """
    Run inference for actor3.

    Parameters
    ----------
    task : str
        One of: 'severity', 'risk_cluster', 'anomaly'
    data : dict
        Must contain a 'features' dict.

    Returns
    -------
    dict
        Enriched prediction with label / risk information.
    """
    features: dict = data.get("features", {})

    # ── Accident Severity Classification ──────────────────────────────────────
    if task == "severity":
        model = _load("rf_severity.pkl")
        feature_list = _load("rf_severity_features.pkl")

        _warn_missing(features, feature_list, task)
        X = [[features.get(f, 0) for f in feature_list]]
        pred_class = int(model.predict(X)[0])

        # Return probability for each class too
        proba = model.predict_proba(X)[0].tolist()

        result = {
            "severity_class": pred_class,
            "severity_label": SEVERITY_LABELS.get(pred_class, str(pred_class)),
            "probabilities": {
                str(i): round(p, 4) for i, p in enumerate(proba)
            },
        }

        logger.info(
            f'"Actor3 severity: class={pred_class} '
            f'label={result["severity_label"]}"'
        )
        return result

    # ── Zone Risk Clustering ───────────────────────────────────────────────────
    elif task == "risk_cluster":
        model = _load("kmeans_risk.pkl")
        scaler = _load("kmeans_scaler.pkl")
        feature_list = _load("kmeans_features.pkl")

        _warn_missing(features, feature_list, task)
        X = [[features.get(f, 0) for f in feature_list]]
        X_scaled = scaler.transform(X)
        cluster = int(model.predict(X_scaled)[0])

        result = {
            "risk_cluster": cluster,
            "risk_label": RISK_CLUSTER_LABELS.get(cluster, f"Cluster {cluster}"),
        }

        logger.info(
            f'"Actor3 risk_cluster: cluster={cluster} '
            f'label={result["risk_label"]}"'
        )
        return result

    # ── Anomaly Detection ─────────────────────────────────────────────────────
    elif task == "anomaly":
        model = _load("isolation_forest.pkl")
        scaler = _load("anomaly_scaler.pkl")
        feature_list = _load("anomaly_features.pkl")

        _warn_missing(features, feature_list, task)
        X = [[features.get(f, 0) for f in feature_list]]
        X_scaled = scaler.transform(X)

        # IsolationForest: -1 = anomaly, 1 = normal
        pred = int(model.predict(X_scaled)[0])
        score = float(model.score_samples(X_scaled)[0])  # lower = more anomalous

        result = {
            "is_anomaly": int(pred == -1),
            "anomaly_label": "ANOMALY" if pred == -1 else "NORMAL",
            "anomaly_score": round(score, 6),  # more negative → more anomalous
        }

        logger.info(
            f'"Actor3 anomaly: is_anomaly={result["is_anomaly"]} '
            f'score={score:.4f}"'
        )
        return result

    else:
        raise ValueError(
            f"Unknown task '{task}' for actor3. "
            "Valid tasks: ['severity', 'risk_cluster', 'anomaly']"
        )


def _warn_missing(features: dict, required: list, task: str) -> None:
    """Log a warning for any features that will be imputed with 0."""
    missing = [f for f in required if f not in features]
    if missing:
        logger.warning(
            f'"Actor3 {task}: missing features {missing} — defaulting to 0"'
        )