from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "netflix_churn_model_input.csv"
MODEL_OUT = ROOT / "models" / "churn_model.joblib"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
MLFLOW_DIR = ROOT / "mlflow"

NUMERIC = ["watch_hours", "last_login_days", "monthly_fee", "number_of_profiles", "avg_watch_time_per_day"]
CATEGORICAL = ["subscription_type", "region", "device", "payment_method", "favorite_genre"]
TARGET = "churned"


def preprocessor():
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "pred": pred,
        "prob": prob,
    }


def main():
    df = pd.read_csv(DATA)
    X = df[NUMERIC + CATEGORICAL].copy()
    y = df[TARGET].astype(int)
    train_idx, test_idx = train_test_split(df.index, test_size=0.20, stratify=y, random_state=42)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "dummy_baseline": (Pipeline([("prep", preprocessor()), ("model", DummyClassifier(strategy="most_frequent"))]), {}),
        "logistic_regression": (Pipeline([("prep", preprocessor()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))]), {"model__C": [0.1, 1.0, 10.0]}),
        "random_forest": (Pipeline([("prep", preprocessor()), ("model", RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1))]), {"model__n_estimators": [200, 400], "model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 3]}),
        "gradient_boosting": (Pipeline([("prep", preprocessor()), ("model", GradientBoostingClassifier(random_state=42))]), {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]}),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

    mlflow = None
    if os.getenv("DISABLE_MLFLOW", "0") != "1":
        try:
            import mlflow as _mlflow
            import mlflow.sklearn  # noqa: F401
            mlflow = _mlflow
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("netflix_customer_churn_prediction")
        except ImportError:
            print("MLflow not installed; continuing without experiment logging.")

    rows, fitted = [], {}
    for name, (pipe, params) in candidates.items():
        run_ctx = mlflow.start_run(run_name=name) if mlflow else nullcontext()
        with run_ctx:
            if params:
                search = GridSearchCV(pipe, params, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
                search.fit(X_train, y_train)
                model = search.best_estimator_
                if mlflow:
                    mlflow.log_params(search.best_params_)
                    mlflow.log_metric("best_cv_roc_auc", float(search.best_score_))
            else:
                model = pipe.fit(X_train, y_train)

            result = evaluate(model, X_test, y_test)
            metrics = {k: float(v) for k, v in result.items() if k not in {"pred", "prob"}}
            if mlflow:
                mlflow.log_metrics(metrics)
                mlflow.log_param("protected_attributes_in_training", False)
            rows.append({"model": name, **metrics})
            fitted[name] = model

    comparison = pd.DataFrame(rows).sort_values(["roc_auc", "f1"], ascending=False)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)

    best_name = comparison[comparison.model != "dummy_baseline"].iloc[0].model
    best_model = fitted[best_name]
    best = evaluate(best_model, X_test, y_test)
    joblib.dump(best_model, MODEL_OUT)

    pd.DataFrame({
        "customer_id": df.loc[test_idx, "customer_id"].astype(str).values,
        "y_true": y_test.values,
        "y_pred": best["pred"],
        "y_prob": best["prob"],
        "gender": df.loc[test_idx, "gender"].astype(str).values,
        "age": df.loc[test_idx, "age"].values,
    }).to_csv(REPORTS / "test_predictions.csv", index=False)

    ConfusionMatrixDisplay.from_predictions(y_test, best["pred"], display_labels=["Retained", "Churned"], values_format="d")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(FIGS / "confusion_matrix.png", dpi=180)
    plt.close()

    plt.figure()
    for name, model in fitted.items():
        result = evaluate(model, X_test, y_test)
        fpr, tpr, _ = roc_curve(y_test, result["prob"])
        plt.plot(fpr, tpr, label=f"{name} AUC={result['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Netflix Churn Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "roc_curve.png", dpi=180)
    plt.close()

    metadata = {"best_model": best_name, "training_features": NUMERIC + CATEGORICAL, "excluded": ["customer_id", "age", "gender"]}
    (REPORTS / "best_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if mlflow:
        with mlflow.start_run(run_name=f"BEST_{best_name}"):
            mlflow.log_param("selected_model", best_name)
            mlflow.log_metrics({k: float(v) for k, v in best.items() if k not in {"pred", "prob"}})
            mlflow.log_artifact(str(MODEL_OUT))
            mlflow.log_artifact(str(REPORTS / "model_comparison.csv"))
            mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name="NetflixChurnModel")

    print(comparison.to_string(index=False))
    print(f"\nSelected model: {best_name}\nSaved: {MODEL_OUT}")


if __name__ == "__main__":
    main()
