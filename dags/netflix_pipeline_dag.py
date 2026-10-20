from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path(__file__).resolve().parents[1]

with DAG(
    dag_id="netflix_customer_churn_pipeline",
    description="End-to-end churn data, model, fairness and validation pipeline",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
    tags=["ban6800", "netflix", "churn"],
) as dag:
    prepare = BashOperator(
        task_id="prepare_and_validate_data",
        bash_command=f"cd {PROJECT_ROOT} && python src/data/prepare_churn_data.py",
    )
    train = BashOperator(
        task_id="train_and_track_models",
        bash_command=f"cd {PROJECT_ROOT} && python src/models/train.py",
    )
    fairness = BashOperator(
        task_id="fairness_audit",
        bash_command=f"cd {PROJECT_ROOT} && python src/models/fairness.py",
    )
    validate = BashOperator(
        task_id="robustness_and_calibration",
        bash_command=f"cd {PROJECT_ROOT} && python run_pre_presentation_validations.py",
    )
    explain = BashOperator(
        task_id="generate_explanations",
        bash_command=f"cd {PROJECT_ROOT} && python src/models/explain.py",
    )

    prepare >> train >> fairness >> validate >> explain
