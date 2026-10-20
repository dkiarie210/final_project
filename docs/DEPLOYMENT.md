# Deployment

## FastAPI on Render

1. Push this repository to GitHub.
2. In Render, create a Blueprint from the repository using `render.yaml`.
3. After deployment, verify:
   - `/health`
   - `/docs`
   - `/predict`
4. Copy the live API URL into the final report.
5. Optional CI/CD: add the Render deploy hook as the GitHub secret `RENDER_DEPLOY_HOOK_URL`.

## Stakeholder dashboard on Streamlit Cloud

1. Create a Streamlit Cloud app from the same GitHub repository.
2. Main file: `dashboard/streamlit_app.py`.
3. The app can score with the bundled model. If a public API URL is configured as `API_URL`, it will use the API first.
4. Copy the live dashboard URL into the final report.

## Monitoring dashboard on Streamlit Cloud

Create a second app from the same repository:

`monitoring/monitoring_app.py`

Copy the live monitoring URL into the final report.
