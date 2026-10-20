# Final Project Integration Guide

This repository is the canonical BAN6800 final project.

## Run locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-final.txt
pytest -q
uvicorn api.main:app --reload
```

Dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

Monitoring:

```bash
streamlit run monitoring/monitoring_app.py --server.port 8502
```

## Full model rebuild

```bash
python run_model_pipeline.py
```

## Full Docker stack

```bash
docker compose up --build -d
```

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Monitoring: http://localhost:8502
- MLflow: http://localhost:5000

Retrain in Docker:

```bash
docker compose --profile training run --rm pipeline
```

## Submission evidence

- GitHub Actions run: `.github/workflows/ci.yml`
- Model Registry / experiment history: MLflow
- Stakeholder dashboard: `dashboard/streamlit_app.py`
- Monitoring dashboard: `monitoring/monitoring_app.py`
- XAI: `reports/figures/shap_*`
- Fairness: `reports/fairness_metrics.csv`
- Counterfactual: `reports/counterfactual_example.csv`
- Governance: `governance/`
