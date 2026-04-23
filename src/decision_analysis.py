from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.core.config import DEFAULT_THRESHOLD, MODEL_PATH, REPORTS_DIR
from src.features import extract_target, load_training_data, prepare_features
from src.model_io import load_model


DEFAULT_THRESHOLDS = [0.10, DEFAULT_THRESHOLD, 0.40]
FALSE_POSITIVE_COST = 1
FALSE_NEGATIVE_COST = 5


def analyze_decision_thresholds(
    data_path: str,
    model_path: Path = MODEL_PATH,
    output_dir: Path = REPORTS_DIR,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = thresholds or DEFAULT_THRESHOLDS

    data = load_training_data(data_path)
    X = prepare_features(data)
    y = extract_target(data)
    _, X_valid, _, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    probabilities = load_model(model_path).predict_proba(X_valid)[:, 1]
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_valid, predictions).ravel()
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_valid, predictions, zero_division=0),
                "recall": recall_score(y_valid, predictions, zero_division=0),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "cost": int(fp * FALSE_POSITIVE_COST + fn * FALSE_NEGATIVE_COST),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "decision_threshold_analysis.csv", index=False)
    (output_dir / "decision_threshold_analysis.json").write_text(
        json.dumps(result.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    write_decision_plot(result, output_dir)
    return result


def write_decision_plot(result: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(result["threshold"], result["cost"], marker="o", label="Cost")
    plt.plot(result["threshold"], result["recall"], marker="o", label="Recall")
    plt.plot(result["threshold"], result["precision"], marker="o", label="Precision")
    plt.xlabel("Threshold")
    plt.title("Business decision threshold trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "decision_threshold_analysis.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze business decision thresholds.")
    parser.add_argument("--data", required=True, help="Path to cs-training.csv")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--output-dir", default=str(REPORTS_DIR))
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Decision thresholds to compare.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_decision_thresholds(
        args.data,
        model_path=Path(args.model),
        output_dir=Path(args.output_dir),
        thresholds=args.thresholds,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
