from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "netflix_customer_churn.csv"


def run(label: str, script: str):
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    if args.refresh_data or not RAW.exists():
        run("Ingest Kaggle churn dataset", "src/data/ingest_churn.py")

    run("Prepare and validate model data", "src/data/prepare_churn_data.py")

    if not args.skip_train:
        run("Train, tune and track models", "src/models/train.py")

    steps = [
        ("Run fairness analysis", "src/models/fairness.py"),
        ("Run threshold sensitivity", "src/models/sensitivity.py"),
        ("Run feature-ablation test", "src/models/feature_ablation.py"),
        ("Run watch-time sensitivity", "src/models/watch_time_sensitivity.py"),
        ("Calibrate model probabilities", "src/models/probability_calibration.py"),
        ("Generate SHAP explanations", "src/models/explain.py"),
        ("Generate counterfactual example", "src/models/counterfactual.py"),
    ]
    for label, script in steps:
        run(label, script)

    print("\nPipeline complete.")
    print("Model: models/churn_model_calibrated.joblib")
    print("Reports: reports/")
    print("MLflow: mlflow/mlflow.db or MLFLOW_TRACKING_URI")


if __name__ == "__main__":
    main()
