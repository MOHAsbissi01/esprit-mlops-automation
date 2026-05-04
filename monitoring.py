"""
monitoring.py — Pure Prometheus metrics for the ML Automation System.

No FastAPI imports. This module is safe to import in any context.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

REQUEST_COUNTER = Counter(
    name="ml_api_requests_total",
    documentation="Total number of prediction/retrain requests.",
    labelnames=["actor", "endpoint", "status"],
)

LATENCY_HISTOGRAM = Histogram(
    name="ml_api_request_duration_seconds",
    documentation="Request latency in seconds.",
    labelnames=["actor", "endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60],
)

ERROR_RATE = Gauge(
    name="ml_api_error_rate",
    documentation="Ratio of failed requests over total requests per actor (0.0–1.0).",
    labelnames=["actor"],
)

MODEL_CONFIDENCE = Gauge(
    name="ml_api_model_confidence",
    documentation="Last-observed model confidence / probability score per actor.",
    labelnames=["actor"],
)

# ---------------------------------------------------------------------------
# Per-actor request/error accumulators (used to compute rolling error rate)
# ---------------------------------------------------------------------------
_totals: dict[str, int] = {}
_errors: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def instrument_request(
    actor: str,
    endpoint: str,
    duration: float,
    success: bool,
    confidence: float | None = None,
    raw_result=None,
) -> None:
    """Record a single request into all relevant Prometheus metrics.

    Parameters
    ----------
    actor:      Actor identifier, e.g. ``"actor1"`` / ``"actor2"`` / ``"actor3"``.
    endpoint:   API path that was called, e.g. ``"/predict"`` or ``"/retrain"``.
    duration:   Wall-clock time of the request in **seconds**.
    success:    ``True`` if the request completed without error, ``False`` otherwise.
    confidence: Optional model probability / confidence score (0.0–1.0).
                When provided, it is recorded in the MODEL_CONFIDENCE gauge.
    """
    # Auto-extract confidence from result dict if not provided explicitly
    if confidence is None and isinstance(raw_result, dict):
        for key in (
            "confidence",
            "cancellation_probability",
            "anomaly_score",
            "risk_score",
            "severity_score",
        ):
            if key in raw_result:
                confidence = float(raw_result[key])
                break

    status = "success" if success else "error"

    # Counter — always increment
    REQUEST_COUNTER.labels(actor=actor, endpoint=endpoint, status=status).inc()

    # Latency — observe regardless of outcome
    LATENCY_HISTOGRAM.labels(actor=actor, endpoint=endpoint).observe(duration)

    # Rolling error rate per actor
    _totals[actor] = _totals.get(actor, 0) + 1
    if not success:
        _errors[actor] = _errors.get(actor, 0) + 1

    rate = _errors.get(actor, 0) / _totals[actor]
    ERROR_RATE.labels(actor=actor).set(rate)

    # Optional model confidence
    if confidence is not None:
        MODEL_CONFIDENCE.labels(actor=actor).set(confidence)


def get_metrics_response() -> tuple[bytes, str]:
    """Return ``(body, content_type)`` suitable for a raw HTTP metrics endpoint.

    Example usage in FastAPI::

        from monitoring import get_metrics_response
        from fastapi import Response

        @app.get("/metrics")
        def metrics():
            body, content_type = get_metrics_response()
            return Response(content=body, media_type=content_type)
    """
    return generate_latest(), CONTENT_TYPE_LATEST
