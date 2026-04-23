from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.config import MODEL_PATH, REPORTS_DIR
from src.features import load_training_data, prepare_features
from src.model_io import load_model
from src.services.explainability import write_shap_artifacts


def generate_explainability_report(
    data_path: str,
    model_path: Path = MODEL_PATH,
    output_dir: Path = REPORTS_DIR,
    max_rows: int = 1000,
) -> dict[str, str]:
    data = load_training_data(data_path)
    features = prepare_features(data)
    model = load_model(model_path)
    return write_shap_artifacts(model, features, output_dir, max_rows=max_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP explainability artifacts.")
    parser.add_argument("--data", required=True, help="Path to cs-training.csv")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--output-dir", default=str(REPORTS_DIR))
    parser.add_argument("--max-rows", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = generate_explainability_report(
        args.data,
        model_path=Path(args.model),
        output_dir=Path(args.output_dir),
        max_rows=args.max_rows,
    )
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
