# Final Project Requirement Coverage

| Requirement | Artifact |
|---|---|
| Complete repository | repository root |
| README / setup | `README.md` |
| Data pipeline | `src/data/`, `dags/` |
| Predictive model | `src/models/train.py`, `models/` |
| MLflow / registry | `src/models/train.py`, `mlflow/` |
| Global/local XAI | `src/models/explain.py`, `reports/figures/shap_*` |
| Counterfactual | `src/models/counterfactual.py`, `reports/counterfactual_example.csv` |
| Fairness | `src/models/fairness.py`, `reports/fairness_metrics.csv` |
| Model card | `models/model_card.md` |
| Explanation store | `reports/explanations/`, `governance/explanation_store.md` |
| Live API code | `api/main.py`, `render.yaml` |
| Stakeholder/XAI dashboard | `dashboard/streamlit_app.py` |
| Monitoring dashboard | `monitoring/monitoring_app.py` |
| CI/CD | `.github/workflows/ci.yml` |
| Docker Compose | `docker-compose.yml`, `docker/` |
| Retraining | `run_model_pipeline.py`, Compose `pipeline` profile |
| Ethical AI / AIA / compliance | `governance/` |
| Incident / decommissioning | `governance/incident_response_and_decommissioning.md` |
| AI disclosure | official NXU form still must be completed by the student |
