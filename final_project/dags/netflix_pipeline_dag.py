from __future__ import annotations

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="netflix_customer_segmentation_pipeline",
    description="ETL pipeline for Netflix customer segmentation analytics",
    start_date=datetime(2026, 9, 1),
    schedule="@daily",
    catchup=False,
    tags=["ban6800", "netflix", "segmentation"],
) as dag:

    generate_sample = BashOperator(
        task_id="generate_sample_data",
        bash_command="python src/data/generate_sample_data.py",
    )

    ingest = BashOperator(
        task_id="ingest_raw_ratings",
        bash_command="python src/data/ingest.py",
    )

    clean = BashOperator(
        task_id="clean_and_standardize",
        bash_command="python src/data/clean.py",
    )

    validate = BashOperator(
        task_id="great_expectations_validation",
        bash_command="python src/validation/validate_data.py",
    )

    build_features = BashOperator(
        task_id="build_customer_features",
        bash_command="python src/features/build_features.py",
    )

    bias_check = BashOperator(
        task_id="representation_bias_check",
        bash_command="python src/bias/bias_detection.py",
    )

    generate_sample >> ingest >> clean >> validate >> build_features >> bias_check
