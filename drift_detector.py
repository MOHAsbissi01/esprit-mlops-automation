"""
drift_detector.py — Data drift detection for the ML Automation System.
======================================================================
Reads results/predictions.json and checks per-actor prediction distributions
for two types of drift:

  1. Confidence drop  — average confidence falls below a known baseline.
  2. Distribution shift — KS test on recent prediction values vs. baseline
                          sample indicates the distribution has shifted.

Run directly:
    python drift_detector.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
PREDICTIONS_FILE = BASE_DIR / "results" / "predictions.json"

# ---------------------------------------------------------------------------
# Per-actor baseline confidence scores (established during initial evaluation)
# ---------------------------------------------------------------------------
BASELINES: dict[str, float] = {
    "actor1": 0.85,
    "actor2": 0.82,
    "actor3": 0.80,
}

ALL_ACTORS = list(BASELINES.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_numeric(result: Any) -> float | None:
    """Extract a single numeric value from a prediction result.

    Handles the three shapes seen in predictions.json:
      - float / int          → returned directly
      - dict                 → tries 'confidence', then the first numeric value
      - str                  → attempts float() parse
      - anything else        → returns None
    """
    if isinstance(result, (int, float)):
        return float(result)

    if isinstance(result, dict):
        # Prefer explicit confidence key
        if "confidence" in result:
            try:
                return float(result["confidence"])
            except (TypeError, ValueError):
                pass
        # Fall back to first numeric value found
        for v in result.values():
            if isinstance(v, (int, float)):
                return float(v)
        return None

    if isinstance(result, str):
        try:
            return float(result)
        except ValueError:
            return None

    return None


def _load_all() -> list[dict]:
    """Load predictions.json; returns an empty list if the file is missing or corrupt."""
    if not PREDICTIONS_FILE.exists():
        return []
    try:
        with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Tasks whose entries belong to actor3 when the 'actor' field is absent/wrong.
_ACTOR3_TASKS: frozenset[str] = frozenset({"severity", "risk_zone", "anomaly", "risk_cluster"})


def load_predictions(actor: str) -> list[float]:
    """Return a list of recent numeric prediction values for *actor*.

    Non-numeric results are silently skipped. Entries with actor == 'string'
    (test/dummy data) are ignored.

    actor3 entries may be stored with task names rather than 'actor3' in the
    actor field; those are matched via ``_ACTOR3_TASKS``.

    Parameters
    ----------
    actor:  e.g. ``"actor1"``, ``"actor2"``, ``"actor3"``
    """
    all_entries = _load_all()
    values: list[float] = []
    for entry in all_entries:
        entry_actor = entry.get("actor", "")
        entry_task  = entry.get("task", "")

        # Direct match
        if entry_actor == actor:
            pass
        # actor3 fallback: match by task name when actor field is unreliable
        elif actor == "actor3" and entry_task in _ACTOR3_TASKS:
            pass
        else:
            continue

        # Skip placeholder / test rows
        if entry_actor == "string":
            continue

        v = _extract_numeric(entry.get("result"))
        if v is not None:
            values.append(v)
    return values


def detect_confidence_drop(
    actor: str,
    baseline: float,
    threshold: float = 0.05,
) -> bool:
    """Return True if the mean prediction value for *actor* has dropped below
    ``baseline - threshold``.

    Parameters
    ----------
    actor:     Actor identifier.
    baseline:  Known-good average confidence / output value.
    threshold: Acceptable tolerance below the baseline (default 0.05).
    """
    values = load_predictions(actor)
    if not values:
        # No data → cannot confirm drop, return False conservatively
        return False
    avg = statistics.mean(values)
    return avg < (baseline - threshold)


def detect_distribution_shift(
    actor: str,
    baseline_values: list[float],
    window: int = 50,
    min_samples: int = 10,
) -> bool:
    """Return True if recent predictions have shifted significantly from
    *baseline_values* using a two-sample Kolmogorov-Smirnov test.

    Parameters
    ----------
    actor:            Actor identifier.
    baseline_values:  Reference distribution — should be real historical values,
                      NOT synthetic data.
    window:           Number of most-recent predictions to compare (default 50).
    min_samples:      Minimum records required in *both* samples before the KS
                      test is attempted (default 10).  Returns False with a
                      printed warning when the threshold is not met.
    """
    recent = load_predictions(actor)[-window:]
    if len(recent) < min_samples:
        print(
            f"  [WARN] {actor}: only {len(recent)} recent record(s) — "
            f"need >= {min_samples} to run KS test. Skipping drift check."
        )
        return False
    if len(baseline_values) < min_samples:
        print(
            f"  [WARN] {actor}: baseline has only {len(baseline_values)} record(s) — "
            f"need >= {min_samples}. Skipping drift check."
        )
        return False
    ks_stat, _p_value = stats.ks_2samp(baseline_values, recent)
    return ks_stat > 0.3


def run_drift_check() -> dict[str, dict[str, bool]]:
    """Run both drift checks for all three actors.

    Baseline is seeded from the **oldest 100 real prediction values** for each
    actor (i.e. everything except the most-recent 50).  The most-recent 50 are
    used as the "current" window passed to the KS test.

    If an actor has fewer than 10 total records the KS test is skipped and
    drift is reported as False (see detect_distribution_shift).

    Returns
    -------
    dict mapping actor -> {"drift": bool, "confidence_drop": bool}
    """
    results: dict[str, dict[str, bool]] = {}

    for actor in ALL_ACTORS:
        all_values = load_predictions(actor)

        # Split: oldest records form the baseline, newest 50 are the test window.
        # We take up to 100 values before the last 50 as the reference population.
        if len(all_values) > 50:
            baseline_sample = all_values[:-50][-100:]   # up to 100 historical
        else:
            baseline_sample = all_values                # will be caught by min_samples guard

        confidence_drop = detect_confidence_drop(
            actor=actor,
            baseline=BASELINES[actor],
            threshold=0.05,
        )
        drift = detect_distribution_shift(
            actor=actor,
            baseline_values=baseline_sample,
            window=50,
        )

        results[actor] = {
            "drift": drift,
            "confidence_drop": confidence_drop,
        }

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  ML Drift Detector — Prediction Distribution Check")
    print("=" * 55)

    all_entries = _load_all()
    print(f"\nLoaded {len(all_entries)} prediction records from {PREDICTIONS_FILE}\n")

    for actor in ALL_ACTORS:
        values = load_predictions(actor)
        baseline = BASELINES[actor]
        avg = statistics.mean(values) if values else None
        print(f"[{actor}]")
        print(f"  Records found : {len(values)}")
        print(f"  Baseline conf : {baseline}")
        if avg is not None:
            print(f"  Mean value    : {avg:.4f}")
            print(f"  Conf drop?    : {'[!!] YES' if avg < baseline - 0.05 else '[OK] NO'}")
        else:
            print("  Mean value    : N/A (no numeric results)")
            print("  Conf drop?    : N/A")
        print()

    print("-" * 55)
    print("Running full drift check...\n")

    report = run_drift_check()

    for actor, checks in report.items():
        drift_flag    = "[!!] DRIFT DETECTED"    if checks["drift"]            else "[OK] No shift"
        conf_flag     = "[!!] DROP DETECTED"     if checks["confidence_drop"]  else "[OK] Within baseline"
        print(f"  {actor}:")
        print(f"    Distribution shift : {drift_flag}")
        print(f"    Confidence drop    : {conf_flag}")

    print("\n" + "=" * 55)
