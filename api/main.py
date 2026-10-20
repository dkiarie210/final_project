from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "churn_model_calibrated.joblib"))
INDEX_PATH = Path(__file__).resolve().parent / "static" / "index.html"
MODEL_VERSION = "2.0.0-calibrated"

app = FastAPI(title="Netflix Customer Churn API", version=MODEL_VERSION)
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
    return _model


class ChurnRequest(BaseModel):
    watch_hours: float = Field(ge=0)
    last_login_days: float = Field(ge=0)
    monthly_fee: float = Field(ge=0)
    number_of_profiles: float = Field(ge=1)
    avg_watch_time_per_day: float = Field(ge=0)
    subscription_type: str
    region: str
    device: str
    payment_method: str
    favorite_genre: str


@app.get("/", include_in_schema=False)
def home():
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    return {"service": "Netflix Customer Churn API", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "model_version": MODEL_VERSION,
    }


@app.get("/metadata")
def metadata():
    return {
        "model": "calibrated_gradient_boosting",
        "version": MODEL_VERSION,
        "threshold": 0.50,
        "purpose": "academic churn-status classification",
    }


@app.post("/predict")
def predict(request: ChurnRequest):
    try:
        model = get_model()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not found. Run the training pipeline first.")

    row = pd.DataFrame([request.model_dump()])
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.50)
    return {
        "prediction": prediction,
        "prediction_label": "churned" if prediction else "retained",
        "churn_probability": round(probability, 6),
        "decision_threshold": 0.50,
        "model_version": MODEL_VERSION,
    }
