"""
alerting.py — Real-time alert engine for the ML Automation System.
==================================================================
Polls Prometheus for live metric values, runs drift detection, and fires
alerts to both the structured log file and the n8n webhook.

Usage
-----
One-shot check:
    python alerting.py

Continuous monitoring (every 30 s, runs after the initial check):
    python alerting.py          # starts continuous loop after first report
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests

from drift_detector import run_drift_check

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MAX_LATENCY_P95: float = 2.0    # seconds  — p95 latency above this = alert
MAX_ERROR_RATE:  float = 0.10   # ratio    — error rate above 10 % = alert
MIN_CONFIDENCE:  float = 0.75   # ratio    — model confidence below this = alert
DRIFT_ALERT:     bool  = True   # fire alert when drift / confidence-drop detected

# ---------------------------------------------------------------------------
# Integration endpoints
# ---------------------------------------------------------------------------

N8N_WEBHOOK: str = "http://localhost:5678/webhook/alert"
PROMETHEUS:  str = "http://localhost:9090/api/v1/query"

# ---------------------------------------------------------------------------
# Logging — appends to the existing ml_api.log
# ---------------------------------------------------------------------------

LOG_FILE: str = "ml_api.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",           # raw message; we build the full line ourselves
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("alerting")

ACTORS = ["actor1", "actor2", "actor3"]


# ---------------------------------------------------------------------------
# 1-4. Individual threshold checks
# ---------------------------------------------------------------------------

def check_latency(actor: str, value: float) -> bool:
    """Return True if p95 latency *value* (seconds) exceeds MAX_LATENCY_P95."""
    return value > MAX_LATENCY_P95


def check_error_rate(actor: str, value: float) -> bool:
    """Return True if *value* (0.0–1.0) exceeds MAX_ERROR_RATE."""
    return value > MAX_ERROR_RATE


def check_confidence(actor: str, value: float | None) -> bool:
    """Return True if *value* is present and below MIN_CONFIDENCE."""
    return value is not None and value < MIN_CONFIDENCE


def check_drift() -> dict[str, dict[str, bool]]:
    """Run drift detection for all actors.

    Returns
    -------
    dict mapping actor → {"drift": bool, "confidence_drop": bool}
    """
    return run_drift_check()


# ---------------------------------------------------------------------------
# Alert dispatcher
# ---------------------------------------------------------------------------

def fire_alert(
    rule: str,
    actor: str,
    value: Any,
    details: str = "",
) -> None:
    """Write an alert to ml_api.log and POST it to the n8n webhook.

    The log line format is:
        [ALERT] <ISO timestamp> | rule=<rule> | actor=<actor> | value=<value> | details=<details>

    The webhook POST is non-blocking (timeout=3 s) and fails silently so that
    a downed n8n instance never crashes the alert loop.
    """
    ts = datetime.now(timezone.utc).isoformat()

    # ── Structured log entry ─────────────────────────────────────────────────
    log_line = (
        f"[ALERT] {ts} | rule={rule} | actor={actor} | value={value} | details={details}"
    )
    logger.warning(log_line)

    # ── n8n webhook (fire-and-forget) ────────────────────────────────────────
    payload = {
        "rule":      rule,
        "actor":     actor,
        "value":     value,
        "details":   details,
        "timestamp": ts,
    }
    try:
        requests.post(N8N_WEBHOOK, json=payload, timeout=3)
    except Exception:
        # n8n may not be running — never let this crash the alert loop
        pass


# ---------------------------------------------------------------------------
# Prometheus query helpers
# ---------------------------------------------------------------------------

def _prom_query(promql: str) -> list[dict]:
    """Execute a PromQL instant query and return the result list.

    Returns an empty list on any error (server down, timeout, bad JSON).
    """
    try:
        resp = requests.get(PROMETHEUS, params={"query": promql}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {}).get("result", [])
    except Exception:
        return []


def _fetch_latency_p95(actor: str) -> float | None:
    """Return the p95 latency (seconds) for *actor*, or None if unavailable."""
    promql = (
        f'histogram_quantile(0.95, '
        f'rate(ml_api_request_duration_seconds_bucket{{actor="{actor}"}}[5m]))'
    )
    results = _prom_query(promql)
    for r in results:
        try:
            return float(r["value"][1])
        except (KeyError, IndexError, ValueError, TypeError):
            pass
    return None


def _fetch_error_rate(actor: str) -> float | None:
    """Return the current error rate gauge value for *actor*, or None."""
    promql = f'ml_api_error_rate{{actor="{actor}"}}'
    results = _prom_query(promql)
    for r in results:
        try:
            return float(r["value"][1])
        except (KeyError, IndexError, ValueError, TypeError):
            pass
    return None


def _fetch_confidence(actor: str) -> float | None:
    """Return the last-observed model confidence for *actor*, or None."""
    promql = f'ml_api_model_confidence{{actor="{actor}"}}'
    results = _prom_query(promql)
    for r in results:
        try:
            return float(r["value"][1])
        except (KeyError, IndexError, ValueError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def check_all_alerts() -> dict[str, dict[str, bool]]:
    """Run all 4 checks for every actor and fire alerts for violations.

    Metrics are read live from Prometheus; drift is read from predictions.json.

    Returns
    -------
    summary : dict
        {actor: {latency_ok, error_rate_ok, confidence_ok, drift_ok}}
    """
    summary: dict[str, dict[str, bool]] = {}

    # ── Drift check (one call covers all actors) ─────────────────────────────
    drift_results: dict[str, dict[str, bool]] = {}
    try:
        drift_results = check_drift()
    except Exception as exc:
        logger.error(f"[ALERTING] drift check failed: {exc}")

    # ── Per-actor checks ─────────────────────────────────────────────────────
    for actor in ACTORS:
        latency_ok    = True
        error_rate_ok = True
        confidence_ok = True
        drift_ok      = True

        # 1. Latency p95
        latency = _fetch_latency_p95(actor)
        if latency is not None:
            if check_latency(actor, latency):
                latency_ok = False
                fire_alert(
                    rule="high_latency_p95",
                    actor=actor,
                    value=round(latency, 4),
                    details=f"p95={latency:.3f}s > threshold={MAX_LATENCY_P95}s",
                )
        else:
            logger.info(f"[ALERTING] {actor}: latency p95 not available in Prometheus yet")

        # 2. Error rate
        error_rate = _fetch_error_rate(actor)
        if error_rate is not None:
            if check_error_rate(actor, error_rate):
                error_rate_ok = False
                fire_alert(
                    rule="high_error_rate",
                    actor=actor,
                    value=round(error_rate, 4),
                    details=f"error_rate={error_rate:.1%} > threshold={MAX_ERROR_RATE:.0%}",
                )
        else:
            logger.info(f"[ALERTING] {actor}: error_rate not available in Prometheus yet")

        # 3. Model confidence
        confidence = _fetch_confidence(actor)
        if check_confidence(actor, confidence):
            confidence_ok = False
            fire_alert(
                rule="low_model_confidence",
                actor=actor,
                value=round(confidence, 4),
                details=f"confidence={confidence:.4f} < threshold={MIN_CONFIDENCE}",
            )
        elif confidence is None:
            logger.info(f"[ALERTING] {actor}: model_confidence not available in Prometheus yet")

        # 4. Drift
        if DRIFT_ALERT:
            actor_drift = drift_results.get(actor, {})
            if actor_drift.get("drift"):
                drift_ok = False
                fire_alert(
                    rule="distribution_drift",
                    actor=actor,
                    value="detected",
                    details="KS-test detected distribution shift vs. historical baseline",
                )
            if actor_drift.get("confidence_drop"):
                drift_ok = False
                fire_alert(
                    rule="confidence_drop",
                    actor=actor,
                    value="detected",
                    details="Mean prediction value dropped below baseline - 0.05",
                )

        summary[actor] = {
            "latency_ok":    latency_ok,
            "error_rate_ok": error_rate_ok,
            "confidence_ok": confidence_ok,
            "drift_ok":      drift_ok,
        }

    return summary


def run_continuous(interval: int = 30) -> None:
    """Loop forever, calling check_all_alerts() every *interval* seconds.

    Prints a timestamped summary table to stdout after each cycle.
    Catches all exceptions so a transient error never kills the loop.
    """
    print(f"\n[ALERTING] Continuous mode started — interval={interval}s  (Ctrl-C to stop)\n")

    while True:
        try:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"{'─' * 60}")
            print(f"  Cycle @ {ts}")
            print(f"{'─' * 60}")

            summary = check_all_alerts()

            # Pretty-print summary table
            print(f"  {'Actor':<10} {'Latency':>10} {'ErrorRate':>10} {'Confidence':>12} {'Drift':>8}")
            print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*8}")
            for actor, checks in summary.items():
                def _flag(ok: bool) -> str:
                    return "  OK  " if ok else " ALERT"
                print(
                    f"  {actor:<10}"
                    f" {_flag(checks['latency_ok']):>10}"
                    f" {_flag(checks['error_rate_ok']):>10}"
                    f" {_flag(checks['confidence_ok']):>12}"
                    f" {_flag(checks['drift_ok']):>8}"
                )
            print()

        except Exception as exc:
            logger.error(f"[ALERTING] Unexpected error in monitoring cycle: {exc}")

        time.sleep(interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  ML Alerting Engine — Initial Check")
    print("=" * 60)

    initial_summary = check_all_alerts()

    print("\nFull summary:")
    print(json.dumps(initial_summary, indent=2))
    print()

    run_continuous(interval=30)
