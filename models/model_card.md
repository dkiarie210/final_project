# Model Card — Netflix Customer Churn Model

## Model
Serving candidate: **Sigmoid-calibrated Gradient Boosting classifier**.

## Purpose
Classify the public dataset's `churned` label and provide an auditable probability score for academic decision-support demonstrations.

## Inputs
`watch_hours`, `last_login_days`, `monthly_fee`, `number_of_profiles`, `avg_watch_time_per_day`, `subscription_type`, `region`, `device`, `payment_method`, `favorite_genre`.

## Excluded from prediction
- `customer_id`: identifier only
- `age`: fairness audit only
- `gender`: fairness audit only

## Validation
80/20 stratified hold-out, 5-fold cross-validation, baseline comparison, ROC-AUC, accuracy, precision, recall, F1, confusion matrix, threshold sensitivity, feature ablation, anomalous watch-time sensitivity, and probability calibration.

## Explainability
Global and local SHAP outputs are stored under `reports/figures/`. Auditable explanation records are stored under `reports/explanations/`.

## Fairness
Audited by gender and age group using group performance, selection rate, demographic parity difference, and equalized odds difference.

## Limitations
This is a public academic dataset, not Netflix production data. It does not define a verified future churn horizon. Very high measured performance may reflect dataset-specific or synthetic structure and must not be assumed to generalize to production populations.

## Intended use
Academic analytics, model validation, explainability, fairness monitoring, API/dashboard demonstration, and portfolio evidence.

## Prohibited use
Automatic account restrictions, individualized pricing, sensitive-trait inference, or other high-impact customer decisions without production validation and governance approval.
