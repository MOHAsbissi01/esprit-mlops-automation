"""
Training script for Actor 3 - Securite.
Triggers the actor's own main.py, tracks the run in MLflow,
and logs resulting .pkl model artifacts.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 output on Windows terminals (avoids emoji UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import mlflow

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ACTOR_DIR = Path(__file__).resolve().parent.parent.parent / "actor3_securite"
OUTPUTS_DIR = ACTOR_DIR / "outputs"

# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "actor3_securite"

# ---------------------------------------------------------------------------
# Default hyper-parameter values (logged as params for traceability)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "actor": "actor3",
    "models": "severity, risk_cluster, anomaly",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Actor 3 - Securite")
    parser.add_argument(
        "--run",
        type=str,
        default="v1",
        help="MLflow run name (default: v1)",
    )
    return parser.parse_args()


def run_training() -> None:
    args = parse_args()

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=args.run):

        # -- Log parameters --------------------------------------------------
        mlflow.log_params(DEFAULT_PARAMS)

        # -- Execute actor training ------------------------------------------
        print(f"[train_actor3] Launching training in: {ACTOR_DIR}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=str(ACTOR_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        print(result.stdout)

        if result.returncode == 0:
            # -- Success: log metric -----------------------------------------
            mlflow.log_metric("training_success", 1)
            print("[train_actor3] Training succeeded.")

            # -- Log .pkl artifacts ------------------------------------------
            if OUTPUTS_DIR.exists():
                pkl_files = list(OUTPUTS_DIR.glob("*.pkl"))
                if pkl_files:
                    for pkl_path in pkl_files:
                        mlflow.log_artifact(str(pkl_path), artifact_path="models")
                        print(f"[train_actor3] Logged artifact: {pkl_path.name}")
                else:
                    print(f"[train_actor3] Warning: no .pkl files found in {OUTPUTS_DIR}")
            else:
                print(f"[train_actor3] Warning: outputs directory not found: {OUTPUTS_DIR}")

        else:
            # -- Failure: log metric and raise --------------------------------
            mlflow.log_metric("training_success", 0)
            print(f"[train_actor3] Training failed.\nSTDERR:\n{result.stderr}", file=sys.stderr)
            raise RuntimeError(
                f"Actor 3 training failed (exit code {result.returncode}):\n{result.stderr}"
            )


if __name__ == "__main__":
    run_training()
