from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = Path(os.getenv("PROJECT_ROOT", APP_DIR.parent))
REPORTS = Path(os.getenv("REPORTS_DIR", ROOT / "reports"))
FIGURES = REPORTS / "figures"
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "churn_model_calibrated.joblib"))
API_URL = os.getenv("API_URL", "").rstrip("/")
try:
    API_URL = str(st.secrets.get("API_URL", API_URL)).rstrip("/")
except Exception:
    pass

st.set_page_config(page_title="Netflix Churn Insights", layout="wide")
st.title("Netflix Customer Churn Insights")
st.caption("BAN6800 Final Project · Stakeholder Dashboard")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name)


@st.cache_data
def load_json(name: str):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


model_comparison = load_csv("model_comparison.csv")
fairness = load_csv("fairness_metrics.csv")
ablation = load_csv("feature_ablation_results.csv")
watch_time = load_csv("watch_time_treatment_sensitivity.csv")
calibration = load_csv("probability_calibration_results.csv")
thresholds = load_csv("sensitivity_thresholds.csv")
readiness = load_json("data_readiness_summary.json")
model = load_model()


def score(payload: dict) -> dict:
    if API_URL:
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=8)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            pass
    if model is None:
        raise RuntimeError("Prediction service and local model are unavailable.")
    row = pd.DataFrame([payload])
    probability = float(model.predict_proba(row)[0, 1])
    return {
        "prediction": int(probability >= 0.50),
        "prediction_label": "churned" if probability >= 0.50 else "retained",
        "churn_probability": probability,
    }


tabs = st.tabs(["Overview", "Prediction Explorer", "What-If", "Explainability", "Fairness & Transparency"])

with tabs[0]:
    calibrated = calibration[calibration["model"] == "sigmoid_calibrated"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{calibrated['accuracy']:.1%}")
    c2.metric("Recall", f"{calibrated['recall']:.1%}")
    c3.metric("ROC-AUC", f"{calibrated['roc_auc']:.4f}")
    c4.metric("Brier Score", f"{calibrated['brier_score']:.4f}")

    left, right = st.columns([1.2, 1])
    with left:
        st.image(str(FIGURES / "roc_curve.png"), caption="Hold-out ROC comparison")
    with right:
        fig = px.bar(model_comparison, x="model", y="roc_auc", text="roc_auc", title="ROC-AUC by Model", range_y=[0, 1.05])
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Readiness")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rows", f"{readiness['input_rows']:,}")
    d2.metric("Duplicates", readiness["duplicates_removed"])
    d3.metric("Missing values", sum(readiness["missing_values"].values()))
    d4.metric(">24h watch-time flags", readiness["avg_watch_over_24h"])

with tabs[1]:
    st.subheader("Prediction Explorer")
    a, b, c = st.columns(3)
    payload = {
        "watch_hours": a.number_input("Watch hours", min_value=0.0, value=8.0, step=1.0),
        "last_login_days": a.number_input("Days since last login", min_value=0.0, value=30.0, step=1.0),
        "monthly_fee": a.number_input("Monthly fee", min_value=0.0, value=14.99, step=0.5),
        "number_of_profiles": b.number_input("Number of profiles", min_value=1.0, value=3.0, step=1.0),
        "avg_watch_time_per_day": b.number_input("Average watch time per day", min_value=0.0, value=0.3, step=0.1),
        "subscription_type": b.selectbox("Subscription type", ["Basic", "Standard", "Premium"]),
        "region": c.selectbox("Region", ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]),
        "device": c.selectbox("Device", ["Desktop", "Laptop", "Mobile", "TV", "Tablet"]),
        "payment_method": c.selectbox("Payment method", ["Credit Card", "Crypto", "Debit Card", "Gift Card", "PayPal"]),
        "favorite_genre": st.selectbox("Favorite genre", ["Action", "Comedy", "Documentary", "Drama", "Horror", "Romance", "Sci-Fi"]),
    }

    try:
        result = score(payload)
        st.metric("Predicted churn probability", f"{result['churn_probability']:.1%}")
        st.write("Prediction label:", f"**{result['prediction_label'].title()}**")
    except RuntimeError as exc:
        st.warning(str(exc))
    st.caption("Academic decision-support prototype. Not an automatic customer decision system.")

with tabs[2]:
    st.subheader("Threshold What-If")
    threshold = st.slider("Decision threshold", 0.30, 0.70, 0.50, 0.05)
    selected = thresholds.iloc[(thresholds["threshold"] - threshold).abs().argsort()[:1]].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{selected['accuracy']:.1%}")
    c2.metric("Precision", f"{selected['precision']:.1%}")
    c3.metric("Recall", f"{selected['recall']:.1%}")
    c4.metric("F1", f"{selected['f1']:.1%}")
    st.plotly_chart(px.line(thresholds, x="threshold", y=["precision", "recall", "f1"], markers=True, title="Threshold Trade-Off"), use_container_width=True)

    st.subheader("Robustness")
    left, right = st.columns(2)
    left.dataframe(ablation[["scenario", "roc_auc", "roc_auc_drop_vs_all_features"]], hide_index=True, use_container_width=True)
    right.dataframe(watch_time[["scenario", "accuracy", "recall", "roc_auc"]], hide_index=True, use_container_width=True)

with tabs[3]:
    st.subheader("Explainability")
    left, right = st.columns(2)
    left.image(str(FIGURES / "shap_summary.png"), caption="Global SHAP explanation")
    right.image(str(FIGURES / "shap_local.png"), caption="Local SHAP example")
    st.write("Recent activity and viewing behavior carry most of the predictive signal. Payment and subscription characteristics add secondary context.")
    counterfactual_path = REPORTS / "counterfactual_example.csv"
    if counterfactual_path.exists():
        st.subheader("Counterfactual Example")
        st.dataframe(pd.read_csv(counterfactual_path), hide_index=True, use_container_width=True)

with tabs[4]:
    st.subheader("Fairness Summary")
    gender = fairness[fairness["protected_attribute"] == "gender"].copy()
    age = fairness[fairness["protected_attribute"] == "age_group"].copy()
    gdp = gender["demographic_parity_difference_overall"].iloc[0]
    geo = gender["equalized_odds_difference_overall"].iloc[0]
    adp = age["demographic_parity_difference_overall"].iloc[0]
    aeo = age["equalized_odds_difference_overall"].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gender DP diff", f"{gdp:.3f}")
    m2.metric("Gender EO diff", f"{geo:.3f}")
    m3.metric("Age DP diff", f"{adp:.3f}")
    m4.metric("Age EO diff", f"{aeo:.3f}")
    st.dataframe(gender[["gender", "accuracy", "recall", "selection_rate"]], hide_index=True, use_container_width=True)
    st.dataframe(age[["age_group", "accuracy", "recall", "selection_rate"]], hide_index=True, use_container_width=True)

    st.subheader("Transparency Statement")
    st.info("This is an academic churn-status classifier built on a public dataset. It identifies statistical patterns associated with the provided churn label, does not explain why a customer churned, and has not been validated on Netflix production data.")
