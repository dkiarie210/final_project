# Ethical Incident Response and Decommissioning

## Incident triggers
- material decline in recall, ROC-AUC or calibration
- fairness or bias metrics exceeding approved thresholds
- data drift or schema failures
- privacy/security incident
- explanation outputs inconsistent with model behavior
- evidence of harmful or prohibited use

## Response
1. Restrict or pause use when customer impact may be material.
2. Preserve input, prediction, model-version, explanation and audit logs.
3. Escalate to Model Owner, Data Owner, Privacy/Legal, Security and Risk.
4. Diagnose data, model, threshold and deployment causes.
5. Approve rollback, remediation or retraining before restoration.
6. Record corrective action and governance sign-off.

## Decommissioning
Retire the model when it is no longer fit for purpose, replaced by a materially better governed model, cannot meet performance/fairness thresholds, or the underlying business purpose ends. Archive model card, registry metadata, evaluation evidence and explanation/audit logs; revoke endpoints and credentials; apply data-retention/disposal rules; update dashboards and documentation so users cannot continue using a retired model.
