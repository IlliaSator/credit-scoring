from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.core.config import DEFAULT_THRESHOLD, FEATURE_NAMES, MODELS_DIR, TARGET_COLUMN
from src.features import extract_target, load_training_data, prepare_features
from src.model_io import save_model_bundle


def train_model(
    data_path: str,
    output_dir: Path = MODELS_DIR,
    threshold: float = DEFAULT_THRESHOLD,
    random_state: int = 42,
) -> dict[str, float]:
    data = load_training_data(data_path)
    X = prepare_features(data)
    y = extract_target(data)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    model = GradientBoostingClassifier(max_depth=4, random_state=random_state)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_valid)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "pr_auc": float(average_precision_score(y_valid, probabilities)),
        "f1_at_threshold": float(f1_score(y_valid, predictions)),
        "threshold": float(threshold),
        "positive_class_rate": float(y.mean()),
        "validation_rows": int(len(y_valid)),
    }

    save_model_bundle(
        model,
        FEATURE_NAMES,
        output_dir,
        {
            "model_name": "GradientBoostingClassifier",
            "model_version": "training-pipeline-v1",
            "threshold": threshold,
            "positive_class_rate": metrics["positive_class_rate"],
            "feature_count": len(FEATURE_NAMES),
            "random_state": random_state,
        },
    )
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the credit scoring model.")
    parser.add_argument("--data", required=True, help="Path to cs-training.csv")
    parser.add_argument("--output-dir", default=str(MODELS_DIR))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model(
        args.data,
        output_dir=Path(args.output_dir),
        threshold=args.threshold,
        random_state=args.random_state,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
