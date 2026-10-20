# Ethical AI Framework — Netflix Customer Churn Analytics

## Purpose
Use customer analytics as decision support while protecting privacy, limiting harmful use, maintaining explainability and enabling human oversight.

## Principles
1. **Purpose limitation** — use data only for approved churn analytics and model validation.
2. **Data minimization** — use only variables needed for the defined analytical purpose.
3. **Privacy by design** — protect identifiers, restrict access, avoid unnecessary joins and retain data only as justified.
4. **Fairness** — monitor performance and error rates across audited groups; do not claim fairness from one test sample alone.
5. **Explainability** — maintain global and local explanations for important model decisions.
6. **Human oversight** — predictions support review and analysis; they are not automatic customer decisions.
7. **Traceability** — version code, data, models, metrics and explanation outputs.
8. **Accountability** — assign model, data, privacy and business owners with clear escalation paths.

## Approved use
- analytical investigation of churn patterns
- controlled stakeholder dashboards
- model research, validation and monitoring

## Prohibited use
- automatic account restriction or denial of service
- sensitive-trait inference
- individualized pricing solely from churn score
- claims that public academic data represents current Netflix production customers
- deployment without current legal, privacy, security and model-risk review
