# Algorithmic Impact Assessment

## System
Calibrated Gradient Boosting classifier for churn-status prediction using a public Netflix customer churn dataset.

## Potential benefits
- consistent identification of behavioral patterns associated with the churn label
- transparent comparison of model outputs through SHAP and stakeholder dashboards
- reusable analytics/MLOps workflow for experimentation and monitoring

## Main risks
- dataset may be synthetic or structurally easier than production data
- public dataset does not define a future churn horizon
- strong dependence on engagement variables may not generalize to future populations
- fairness results may drift when demographics, data or threshold change
- probability scores may be over-trusted by non-technical users
- privacy risk if future deployments introduce identifiable production data

## Impact level
**Moderate academic / potentially high operational impact if deployed.** The current project is a prototype; production use would require additional validation and organizational approvals.

## Controls
- customer ID excluded from prediction
- age and gender excluded from predictive features and used only for fairness auditing
- stratified hold-out testing and cross-validation
- SHAP explanations, model card, fairness metrics and threshold sensitivity
- feature-ablation, anomalous-watch-time sensitivity and probability calibration tests
- monitoring, incident response, retraining and decommissioning plans
