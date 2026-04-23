# Explainability Notes

The project exposes local explanations through `POST /explain` and
`GET /explain/sample`. The report generator is:

```bash
python -m src.explain --data data/cs-training.csv --output-dir reports
```

When `shap` is installed, the project uses `shap.TreeExplainer` for the
GradientBoosting model. If SHAP is unavailable in the local environment, the
code falls back to model feature importances so the endpoint and artifacts stay
demoable, but the README calls out that this fallback is less faithful than
true SHAP values.

Artifacts:

- `reports/shap_summary.png`
- `reports/shap_global_importance.csv`
- `reports/shap_local_sample.json`
