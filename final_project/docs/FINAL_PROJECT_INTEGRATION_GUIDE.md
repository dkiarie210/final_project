# Final Project Integration Guide

## Canonical repository
Use one repository for the final submission. The earlier Module 3 artifact used:
`https://github.com/dkiarie210/business_analytics`

Merge the Module 4 modeling/API code and Module 5 dashboard code into that repository (or choose another single canonical repository and update every link consistently).

## Recommended final structure

```text
business_analytics/
├── .github/workflows/ci.yml
├── api/
├── dashboard/
├── monitoring/
├── governance/
├── dags/
├── src/
├── data/
├── models/
├── reports/
├── tests/
├── docker/
├── docker-compose.yml
├── requirements-final.txt
└── README.md
```

## Final checks
- GitHub project board screenshot included
- MLflow experiment + Model Registry screenshots included
- calibrated model file available to API/dashboard
- live `/predict` endpoint works
- live Streamlit stakeholder dashboard works
- monitoring dashboard works
- fairness metrics match Module 4/5 report values
- CI workflow passes
- `docker compose config` succeeds
- AI Disclosure Form completed truthfully
