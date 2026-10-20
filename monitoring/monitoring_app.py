from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = Path(os.getenv("PROJECT_ROOT", APP_DIR.parent))
REPORTS = Path(os.getenv("REPORTS_DIR", ROOT / "reports"))
DATA = Path(os.getenv("DATA_DIR", ROOT / "data"))

st.set_page_config(page_title="Netflix Churn Monitoring", layout="wide")
st.title("Netflix Churn Model Monitoring")
st.caption("Performance, fairness, data drift and governance")


def read_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return float("nan")
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = pd.cut(expected, cuts, include_lowest=True).value_counts(sort=False, normalize=True)
    a = pd.cut(actual, cuts, include_lowest=True).value_counts(sort=False, normalize=True)
    e = np.clip(e.to_numpy(), 1e-6, None)
    a = np.clip(a.to_numpy(), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


model_cmp = read_csv(REPORTS / "model_comparison.csv")
fairness = read_csv(REPORTS / "fairness_metrics.csv")
calibration = read_csv(REPORTS / "probability_calibration_results.csv")

perf_tab, fair_tab, drift_tab, incident_tab = st.tabs(["Performance", "Fairness", "Data Drift", "Governance / Incidents"])

with perf_tab:
    if calibration is not None and not calibration.empty:
        best = calibration.sort_values("brier_score").iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{best['accuracy']:.1%}")
        c2.metric("Recall", f"{best['recall']:.1%}")
        c3.metric("ROC-AUC", f"{best['roc_auc']:.4f}")
        c4.metric("Brier score", f"{best['brier_score']:.4f}")
        st.dataframe(calibration, hide_index=True, use_container_width=True)
    if model_cmp is not None:
        st.plotly_chart(px.bar(model_cmp, x="model", y="roc_auc", title="ROC-AUC by Model"), use_container_width=True)

with fair_tab:
    if fairness is None:
        st.info("Fairness metrics are not available yet.")
    else:
        st.dataframe(fairness, hide_index=True, use_container_width=True)
        gender = fairness[fairness["protected_attribute"] == "gender"].dropna(subset=["gender"])
        age = fairness[fairness["protected_attribute"] == "age_group"].dropna(subset=["age_group"])
        if not gender.empty:
            st.plotly_chart(px.bar(gender, x="gender", y="selection_rate", title="Selection Rate by Gender"), use_container_width=True)
        if not age.empty:
            st.plotly_chart(px.bar(age, x="age_group", y="selection_rate", title="Selection Rate by Age Group"), use_container_width=True)
        st.caption("Fairness metrics are monitoring signals, not legal conclusions.")

with drift_tab:
    reference = read_csv(DATA / "processed" / "netflix_churn_model_input.csv")
    uploaded = st.file_uploader("Upload a new scoring batch CSV", type=["csv"])
    if reference is None:
        st.info("Reference dataset not found.")
    elif uploaded is not None:
        current = pd.read_csv(uploaded)
        numeric = [c for c in ["watch_hours", "last_login_days", "monthly_fee", "number_of_profiles", "avg_watch_time_per_day"] if c in current.columns]
        rows = []
        for col in numeric:
            value = psi(reference[col], current[col])
            rows.append({"feature": col, "psi": value, "status": "Stable" if value < 0.10 else ("Watch" if value < 0.25 else "Investigate")})
        drift = pd.DataFrame(rows)
        st.dataframe(drift, hide_index=True, use_container_width=True)
        if not drift.empty:
            st.plotly_chart(px.bar(drift, x="feature", y="psi", color="status", title="Population Stability Index"), use_container_width=True)
    else:
        st.info("Upload a new batch to compare with the model reference data.")

with incident_tab:
    st.markdown("""
**Escalation triggers**
- material decline in recall, ROC-AUC or calibration
- fairness deterioration
- schema/data-quality failure
- privacy/security incident
- unexpected explanation behavior

**Response**
1. Restrict use if customer impact may be material.
2. Preserve the input batch, model version, prediction and explanation evidence.
3. Notify model, data, privacy/risk and security owners.
4. Diagnose data, model, threshold and deployment causes.
5. Approve remediation, rollback, retraining or decommissioning.
""")
