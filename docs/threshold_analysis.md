# Calibration and Threshold Analysis

The deployed threshold is `0.23`, selected as a practical balance between recall
on defaulters and the operational cost of declining good borrowers.

Artifacts:

- `reports/evaluation_metrics.json` stores ROC-AUC, PR-AUC, Brier score,
  precision, recall and F1 at the deployed threshold.
- `reports/calibration_curve.csv` and `reports/calibration_curve.png` show
  predicted probability reliability versus observed default rate.
- `reports/threshold_analysis.csv` compares thresholds `0.10`, `0.23`, `0.30`
  and `0.50` with confusion-matrix counts, decline rate and a simple business
  cost where false negatives cost five times more than false positives.

The business cost is illustrative, not a universal lending policy. In a real
deployment, the weights should be set with credit policy, expected loss, APR,
capital constraints and fairness review.
