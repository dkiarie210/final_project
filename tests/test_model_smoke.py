from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "churn_model_calibrated.joblib"
DATA = ROOT / "data" / "processed" / "netflix_churn_model_input.csv"
FEATURES = ["watch_hours", "last_login_days", "monthly_fee", "number_of_profiles", "avg_watch_time_per_day", "subscription_type", "region", "device", "payment_method", "favorite_genre"]


def test_calibrated_model_predicts_probability():
    model = joblib.load(MODEL)
    row = pd.read_csv(DATA, nrows=1)[FEATURES]
    probability = float(model.predict_proba(row)[0, 1])
    assert 0.0 <= probability <= 1.0
