from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "netflix_churn_model_input.csv"
MODEL = ROOT / "models" / "churn_model_calibrated.joblib"
OUTPUT = ROOT / "reports" / "counterfactual_example.csv"
FEATURES = ["watch_hours", "last_login_days", "monthly_fee", "number_of_profiles", "avg_watch_time_per_day", "subscription_type", "region", "device", "payment_method", "favorite_genre"]


def main():
    df = pd.read_csv(DATA)
    model = joblib.load(MODEL)
    X = df[FEATURES].copy()
    probabilities = model.predict_proba(X)[:, 1]
    retained = df[df["churned"] == 0]

    target_values = {
        "last_login_days": float(retained["last_login_days"].quantile(0.25)),
        "watch_hours": float(retained["watch_hours"].quantile(0.75)),
        "avg_watch_time_per_day": float(retained["avg_watch_time_per_day"].quantile(0.75)),
    }

    chosen = None
    for idx in pd.Series(probabilities, index=df.index).sort_values(ascending=False).index:
        original = X.loc[idx].copy()
        if probabilities[idx] < 0.50:
            continue
        alternative = original.copy()
        for col, value in target_values.items():
            alternative[col] = value
        alt_prob = float(model.predict_proba(pd.DataFrame([alternative]))[0, 1])
        if alt_prob < 0.50:
            chosen = (idx, original, alternative, float(probabilities[idx]), alt_prob)
            break

    if chosen is None:
        idx = int(pd.Series(probabilities, index=df.index).idxmax())
        original = X.loc[idx].copy()
        alternative = original.copy()
        for col, value in target_values.items():
            alternative[col] = value
        chosen = (idx, original, alternative, float(probabilities[idx]), float(model.predict_proba(pd.DataFrame([alternative]))[0, 1]))

    idx, original, alternative, original_prob, alt_prob = chosen
    rows = [
        {"scenario": "original", "customer_row": idx, "churn_probability": original_prob, "predicted_label": "churned" if original_prob >= 0.50 else "retained", **{k: original[k] for k in target_values}},
        {"scenario": "behavioral_what_if", "customer_row": idx, "churn_probability": alt_prob, "predicted_label": "churned" if alt_prob >= 0.50 else "retained", **{k: alternative[k] for k in target_values}},
    ]
    out = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(out.to_string(index=False))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
