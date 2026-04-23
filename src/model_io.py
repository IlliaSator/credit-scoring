from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.config import FEATURE_NAMES_PATH, MODEL_METADATA_PATH, MODEL_PATH


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def load_model(path: Path = MODEL_PATH) -> Any:
    return load_pickle(path)


def load_feature_names(path: Path = FEATURE_NAMES_PATH) -> list[str]:
    return list(load_pickle(path))


def save_model_bundle(
    model: Any,
    feature_names: list[str],
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(model, output_dir / "model.pkl")
    save_pickle(feature_names, output_dir / "feature_names.pkl")
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def load_model_metadata(path: Path = MODEL_METADATA_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
