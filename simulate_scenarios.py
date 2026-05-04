"""
simulate_scenarios.py — Send realistic prediction requests to the ML API.
=========================================================================
Exercises all 3 actors across their tasks so that Prometheus metrics
(traffic, latency, error rate, confidence) populate in Grafana.

Feature structure per actor:
  actor1  co2 / energy : nb_voyageurs, distance_km, type_transport, heure, jour_semaine
  actor1  cluster      : pm25, no2, co2_kg, energie_kwh  (4-feature scaler)
  actor2  charge       : heure, delay_minutes, temperature
  actor2  cancellation : heure, delay_minutes, temperature
  actor3  severity     : nb_accidents, nb_victimes, heure, zone_type, meteo
  actor3  risk_cluster : nb_accidents, nb_victimes, heure, zone_type, meteo
  actor3  anomaly      : nb_accidents, nb_victimes, heure, zone_type, meteo

Run AFTER uvicorn is up on port 8000:
    python simulate_scenarios.py
"""

import random
import time
from datetime import datetime

try:
    import requests
except ImportError:
    raise SystemExit("requests not installed — run: pip install requests")

BASE_URL = "http://localhost:8000"
DELAY_BETWEEN_CALLS = 0.5   # seconds between each request

# ---------------------------------------------------------------------------
# Scenario payloads — exact feature structures per actor
# ---------------------------------------------------------------------------

SCENARIOS = [
    # ── Actor 1 — Écologique ──────────────────────────────────────────────
    # task: co2
    {
        "actor": "actor1",
        "task": "co2",
        "features": {
            "nb_voyageurs": 120,
            "distance_km": 45.5,
            "type_transport": 1,
            "heure": 8,
            "jour_semaine": 2,
        },
    },
    {
        "actor": "actor1",
        "task": "co2",
        "features": {
            "nb_voyageurs": 300,
            "distance_km": 12.0,
            "type_transport": 2,
            "heure": 17,
            "jour_semaine": 5,
        },
    },
    # task: energy
    {
        "actor": "actor1",
        "task": "energy",
        "features": {
            "nb_voyageurs": 80,
            "distance_km": 30.0,
            "type_transport": 1,
            "heure": 10,
            "jour_semaine": 3,
        },
    },
    {
        "actor": "actor1",
        "task": "energy",
        "features": {
            "nb_voyageurs": 250,
            "distance_km": 55.0,
            "type_transport": 3,
            "heure": 7,
            "jour_semaine": 1,
        },
    },
    # task: cluster — 4-feature scaler (pm25, no2, co2_kg, energie_kwh)
    {
        "actor": "actor1",
        "task": "cluster",
        "features": {
            "pm25": 25.0,
            "no2": 45.0,
            "co2_kg": 320.0,
            "energie_kwh": 130.0,
        },
    },

    # ── Actor 2 — Mobilités ───────────────────────────────────────────────
    # task: charge
    {
        "actor": "actor2",
        "task": "charge",
        "features": {
            "heure": 8,
            "delay_minutes": 0,
            "temperature": 5.0,
        },
    },
    {
        "actor": "actor2",
        "task": "charge",
        "features": {
            "heure": 18,
            "delay_minutes": 12,
            "temperature": 22.0,
        },
    },
    # task: cancellation
    {
        "actor": "actor2",
        "task": "cancellation",
        "features": {
            "heure": 14,
            "delay_minutes": 1000,
            "temperature": 1.0,
        },
    },
    {
        "actor": "actor2",
        "task": "cancellation",
        "features": {
            "heure": 7,
            "delay_minutes": 0,
            "temperature": 18.0,
        },
    },

    # ── Actor 3 — Sécurité ────────────────────────────────────────────────
    # task: severity
    {
        "actor": "actor3",
        "task": "severity",
        "features": {
            "nb_accidents": 5,
            "nb_victimes": 12,
            "heure": 23,
            "zone_type": 2,
            "meteo": 1,
        },
    },
    {
        "actor": "actor3",
        "task": "severity",
        "features": {
            "nb_accidents": 1,
            "nb_victimes": 2,
            "heure": 10,
            "zone_type": 0,
            "meteo": 0,
        },
    },
    # task: risk_cluster
    {
        "actor": "actor3",
        "task": "risk_cluster",
        "features": {
            "nb_accidents": 8,
            "nb_victimes": 20,
            "heure": 23,
            "zone_type": 2,
            "meteo": 1,
        },
    },
    # task: anomaly
    {
        "actor": "actor3",
        "task": "anomaly",
        "features": {
            "nb_accidents": 50,
            "nb_victimes": 100,
            "heure": 3,
            "zone_type": 2,
            "meteo": 1,
        },
    },
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_simulation(rounds: int = 3) -> None:
    print("=" * 60)
    print("  ML API Scenario Simulator")
    print(f"  Target : {BASE_URL}")
    print(f"  Scenarios: {len(SCENARIOS)}  |  Rounds: {rounds}")
    print("=" * 60)

    # Confirm API is reachable
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        health = r.json()
        print(f"\n[OK] API health: {health.get('status')} | "
              f"version={health.get('version')} | "
              f"stored={health.get('predictions_stored')}\n")
    except Exception as exc:
        raise SystemExit(f"[ERROR] Cannot reach {BASE_URL}/health — is uvicorn running?\n{exc}")

    success_count = 0
    error_count = 0

    for rnd in range(1, rounds + 1):
        print(f"--- Round {rnd}/{rounds} ---")
        batch = SCENARIOS.copy()
        random.shuffle(batch)

        for scenario in batch:
            actor = scenario["actor"]
            task  = scenario["task"]

            try:
                resp = requests.post(
                    f"{BASE_URL}/predict",
                    json=scenario,
                    timeout=30,
                )
                data = resp.json()
                if resp.status_code == 200:
                    success_count += 1
                    latency = data.get("latency_ms", "?")
                    result  = data.get("result", "?")
                    print(f"  [OK]  {actor}/{task:<15}  "
                          f"latency={latency:>8.1f}ms  result={str(result)[:45]}")
                else:
                    error_count += 1
                    print(f"  [ERR] {actor}/{task:<15}  "
                          f"HTTP {resp.status_code}  detail={data.get('detail', '')}")

            except requests.exceptions.Timeout:
                error_count += 1
                print(f"  [ERR] {actor}/{task:<15}  TIMEOUT")
            except Exception as exc:
                error_count += 1
                print(f"  [ERR] {actor}/{task:<15}  {exc}")

            time.sleep(DELAY_BETWEEN_CALLS + random.uniform(0, 0.3))

        print()

    print("=" * 60)
    print(f"  Done.  Success={success_count}  Errors={error_count}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)


if __name__ == "__main__":
    run_simulation(rounds=3)
