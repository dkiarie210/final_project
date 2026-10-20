from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "netflix_churn_model_input.csv"
MODEL = ROOT / "models" / "churn_model.joblib"
FIGS = ROOT / "reports" / "figures"
STORE = ROOT / "reports" / "explanations" / "explanation_log.jsonl"
FEATURES = ["watch_hours", "last_login_days", "monthly_fee", "number_of_profiles", "avg_watch_time_per_day", "subscription_type", "region", "device", "payment_method", "favorite_genre"]


def main():
    pipeline = joblib.load(MODEL)
    df = pd.read_csv(DATA)
    sample = df[FEATURES].sample(n=min(200, len(df)), random_state=42)

    prep = pipeline.named_steps["prep"]
    classifier = pipeline.named_steps["model"]
    transformed = prep.transform(sample)
    feature_names = prep.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names, index=sample.index)

    explainer = shap.TreeExplainer(classifier)
    values = explainer(transformed_df)

    FIGS.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(values, transformed_df, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIGS / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    shap.plots.waterfall(values[0], max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(FIGS / "shap_local.png", dpi=180, bbox_inches="tight")
    plt.close()

    contributions = pd.Series(values[0].values, index=feature_names).sort_values()
    probability = float(pipeline.predict_proba(sample.iloc[[0]])[0, 1])
    record = {
        "prediction_id": f"sample-{sample.index[0]}",
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": "gradient-boosting-v1",
        "decision_threshold": 0.50,
        "predicted_class": int(probability >= 0.50),
        "churn_probability": probability,
        "explanation_method": "SHAP TreeExplainer",
        "top_positive_contributors": contributions.tail(5).to_dict(),
        "top_negative_contributors": contributions.head(5).to_dict(),
    }
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(record) + "\n", encoding="utf-8")
    print(f"Saved SHAP figures to {FIGS}")
    print(f"Saved explanation record to {STORE}")


if __name__ == "__main__":
    main()
