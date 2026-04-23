from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.core.config import DEFAULT_THRESHOLD, MODEL_PATH, REPORTS_DIR
from src.features import extract_target, load_training_data, prepare_features
from src.model_io import load_model


def evaluate_model(
    data_path: str,
    model_path: Path = MODEL_PATH,
    output_dir: Path = REPORTS_DIR,
    threshold: float = DEFAULT_THRESHOLD,
    random_state: int = 42,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_training_data(data_path)
    X = prepare_features(data)
    y = extract_target(data)
    _, X_valid, _, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = load_model(model_path)
    probabilities = model.predict_proba(X_valid)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "pr_auc": float(average_precision_score(y_valid, probabilities)),
        "brier_score": float(brier_score_loss(y_valid, probabilities)),
        "f1": float(f1_score(y_valid, predictions)),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "threshold": float(threshold),
    }
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    write_calibration_artifacts(y_valid, probabilities, output_dir)
    write_threshold_artifacts(y_valid, probabilities, output_dir)
    return metrics


def write_calibration_artifacts(y_true, probabilities, output_dir: Path) -> None:
    prob_true, prob_pred = calibration_curve(
        y_true, probabilities, n_bins=10, strategy="quantile"
    )
    pd.DataFrame(
        {"mean_predicted_probability": prob_pred, "observed_default_rate": prob_true}
    ).to_csv(output_dir / "calibration_curve.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed default rate")
    plt.title("Calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=160)
    plt.close()


def write_threshold_artifacts(y_true, probabilities, output_dir: Path) -> None:
    rows = []
    false_negative_cost = 5
    false_positive_cost = 1
    for threshold in [0.10, 0.23, 0.30, 0.50]:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        rows.append(
            {
                "threshold": threshold,
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
                "decline_rate": float(predictions.mean()),
                "business_cost": int(fp * false_positive_cost + fn * false_negative_cost),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "threshold_analysis.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(table["threshold"], table["business_cost"], marker="o", label="Cost")
    plt.plot(table["threshold"], table["recall"], marker="o", label="Recall")
    plt.plot(table["threshold"], table["precision"], marker="o", label="Precision")
    plt.xlabel("Decision threshold")
    plt.title("Threshold trade-off analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_analysis.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate credit scoring artifacts.")
    parser.add_argument("--data", required=True, help="Path to cs-training.csv")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--output-dir", default=str(REPORTS_DIR))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_model(
        args.data,
        model_path=Path(args.model),
        output_dir=Path(args.output_dir),
        threshold=args.threshold,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
