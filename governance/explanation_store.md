# Explanation Store Design

For auditable predictions, store an explanation record containing:

- prediction_id
- prediction_timestamp
- model_version
- decision_threshold
- predicted_class
- churn_probability
- input schema/version (not unnecessary raw identifiers)
- top positive SHAP contributors
- top negative SHAP contributors
- explanation method/version
- API/dashboard source
- reviewer or incident reference when applicable

Recommended path:
`reports/explanations/explanation_log.jsonl`

Retention should follow the same approved retention schedule as model-audit evidence. Access should be restricted because explanation records can reveal behavioral attributes even when direct identifiers are omitted.
