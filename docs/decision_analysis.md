# Decision Analysis

The deployed threshold is `0.23`. It is a middle ground between a very sensitive
policy (`0.10`) and a stricter policy (`0.40`).

Artifacts:

- `reports/decision_threshold_analysis.csv`
- `reports/decision_threshold_analysis.json`
- `reports/decision_threshold_analysis.png`

Business assumption: missing a future defaulter is more expensive than declining
a good borrower, so `FN cost = 5 * FP cost`.

At lower thresholds, recall improves but false positives increase. At higher
thresholds, precision improves but more actual defaulters are missed. The `0.23`
threshold keeps recall meaningfully higher than stricter cutoffs while avoiding
the larger false-positive volume of `0.10`.
