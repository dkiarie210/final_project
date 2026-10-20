from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Netflix Churn Model Monitoring", layout="wide")
st.title("Netflix Churn Model Monitoring")
st.caption("Performance, data quality, drift and fairness monitoring for the BAN6800 final project")

REPORTS = Path("reports")
DATA = Path("data")


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
predictions = read_csv(REPORTS / "test_predictions.csv")
calibration = read_csv(REPORTS / "probability_calibration_results.csv")

perf_tab, fair_tab, drift_tab, incident_tab = st.tabs([
    "Performance", "Fairness", "Data Drift", "Governance / Incidents"
])

with perf_tab:
    st.subheader("Model performance")
    if calibration is not None and not calibration.empty:
        best = calibration.sort_values("brier_score").iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{best['accuracy']:.1%}")
        c2.metric("Recall", f"{best['recall']:.1%}")
        c3.metric("ROC-AUC", f"{best['roc_auc']:.4f}")
        c4.metric("Brier score", f"{best['brier_score']:.4f}")
        st.dataframe(calibration, use_container_width=True)
    elif model_cmp is not None:
        st.dataframe(model_cmp, use_container_width=True)
    else:
        st.info("Add reports/model_comparison.csv or reports/probability_calibration_results.csv.")

    if model_cmp is not None:
        fig = px.bar(model_cmp, x="model", y="roc_auc", title="ROC-AUC by model")
        st.plotly_chart(fig, use_container_width=True)

with fair_tab:
    st.subheader("Continuous fairness monitoring")
    if fairness is None:
        st.info("Add reports/fairness_metrics.csv to display fairness metrics.")
    else:
        st.dataframe(fairness, use_container_width=True)
        if "selection_rate" in fairness.columns:
            group_col = "gender" if "gender" in fairness.columns else ("age_group" if "age_group" in fairness.columns else None)
            if group_col:
                clean = fairness[fairness[group_col].notna()].copy()
                fig = px.bar(clean, x=group_col, y="selection_rate", title="Selection rate by audited group")
                st.plotly_chart(fig, use_container_width=True)
        st.caption("Fairness metrics are monitoring signals, not legal conclusions. Escalate material changes for human review.")

with drift_tab:
    st.subheader("Reference vs. new-batch drift check")
    reference_path = DATA / "processed" / "netflix_churn_model_input.csv"
    if not reference_path.exists():
        reference_path = DATA / "netflix_churn_model_input.csv"
    reference = read_csv(reference_path)
    uploaded = st.file_uploader("Upload a new scoring batch CSV to check drift", type=["csv"])
    if reference is None:
        st.info("Reference model dataset not found under data/processed/netflix_churn_model_input.csv.")
    elif uploaded is not None:
        current = pd.read_csv(uploaded)
        numeric = [c for c in ["watch_hours", "last_login_days", "monthly_fee", "number_of_profiles", "avg_watch_time_per_day"] if c in reference.columns and c in current.columns]
        rows = []
        for col in numeric:
            value = psi(reference[col], current[col])
            status = "Stable" if value < 0.10 else ("Watch" if value < 0.25 else "Investigate")
            rows.append({"feature": col, "psi": value, "status": status})
        drift = pd.DataFrame(rows)
        st.dataframe(drift, use_container_width=True)
        if not drift.empty:
            st.plotly_chart(px.bar(drift, x="feature", y="psi", color="status", title="Population Stability Index"), use_container_width=True)
        st.caption("Suggested interpretation: PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 investigate. Thresholds are governance triggers, not universal laws.")
    else:
        st.info("Upload a new batch to calculate drift against the training reference data.")

with incident_tab:
    st.subheader("Ethical incident response")
    st.markdown("""
**Trigger examples**
- material drop in recall/ROC-AUC
- fairness metric deterioration
- schema or data-quality failure
- unexpected model behavior or explanation mismatch
- privacy/security incident

**Response sequence**
1. Pause or restrict automated use where needed.
2. Preserve logs, model version, input batch and explanation evidence.
3. Notify Model Owner and Privacy/Risk reviewers.
4. Diagnose data, drift, threshold and feature issues.
5. Approve remediation, rollback or retraining through governance review.
6. Document closure and lessons learned.
""")
