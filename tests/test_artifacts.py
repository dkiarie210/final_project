from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_final_project_artifacts_exist():
    required = [
        ROOT / "models" / "churn_model_calibrated.joblib",
        ROOT / "reports" / "model_comparison.csv",
        ROOT / "reports" / "fairness_metrics.csv",
        ROOT / "reports" / "probability_calibration_results.csv",
        ROOT / "reports" / "figures" / "shap_summary.png",
        ROOT / "docker-compose.yml",
        ROOT / ".github" / "workflows" / "ci.yml",
    ]
    assert all(path.exists() for path in required)
