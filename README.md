# Netflix Customer Churn Analytics — BAN6800 Final Project

Integrated analytics system covering data preparation, model training, MLflow tracking, SHAP explainability, fairness auditing, calibrated churn probabilities, FastAPI inference, Streamlit stakeholder exploration, monitoring, Docker, and GitHub Actions.

## Quick start

### 1. Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-final.txt
```

The bundled model was created with `scikit-learn 1.8.0`; the dependency file pins that version.

### 2. Use the existing trained artifacts

The repository already contains the calibrated model and report outputs. Run locally:

```bash
uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs`.

In another terminal:

```bash
source .venv/bin/activate
streamlit run dashboard/streamlit_app.py
```

Open `http://localhost:8501`.

Monitoring:

```bash
source .venv/bin/activate
streamlit run monitoring/monitoring_app.py --server.port 8502
```

Open `http://localhost:8502`.

### 3. Rebuild the model and all validation outputs

```bash
python run_model_pipeline.py
```

Use `--refresh-data` only when you want to download the Kaggle dataset again:

```bash
python run_model_pipeline.py --refresh-data
```

## Docker

Start the full serving stack:

```bash
docker compose up --build -d
```

Services:

- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`
- Monitoring: `http://localhost:8502`
- MLflow: `http://localhost:5000`

Check status/logs:

```bash
docker compose ps
docker compose logs -f
```

Stop:

```bash
docker compose down
```

Retrain inside Docker:

```bash
docker compose --profile training run --rm pipeline
```

## Tests

```bash
pytest -q
```

## CI/CD

`.github/workflows/ci.yml` runs on pushes and pull requests to `main`:

1. installs Python dependencies
2. compiles source files
3. runs pytest
4. validates Docker Compose
5. builds API, dashboard, and monitoring images
6. optionally triggers a Render API deployment when `RENDER_DEPLOY_HOOK_URL` is configured as a GitHub secret

Streamlit Cloud can deploy `dashboard/streamlit_app.py` directly from the same GitHub repository. Set `API_URL` to the deployed FastAPI URL when available; otherwise the dashboard falls back to the bundled calibrated model.

## Main project artifacts

```text
api/                         FastAPI /predict service
dashboard/                   Stakeholder + XAI + fairness dashboard
monitoring/                  Performance, fairness and drift dashboard
src/data/                    Ingestion and data preparation
src/models/                  Training, XAI, fairness, calibration and robustness
dags/                        Airflow orchestration definition
models/                      Original and calibrated model artifacts
reports/                     Metrics, figures, explanations and validation outputs
governance/                  Ethical AI and compliance documentation
.github/workflows/ci.yml      CI/CD workflow
docker-compose.yml           Multi-container runtime
render.yaml                  API deployment blueprint
```

## Final project evidence

- Model comparison and hold-out metrics: `reports/model_comparison.csv`
- Fairness: `reports/fairness_metrics.csv`
- Calibration: `reports/probability_calibration_results.csv`
- Robustness: `reports/feature_ablation_results.csv`, `reports/watch_time_treatment_sensitivity.csv`
- SHAP: `reports/figures/shap_summary.png`, `reports/figures/shap_local.png`
- Counterfactual example: `reports/counterfactual_example.csv`
- Model card: `models/model_card.md`
- Ethical governance: `governance/`
