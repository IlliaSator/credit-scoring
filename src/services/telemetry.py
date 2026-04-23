from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from src.core.config import LOGS_DIR


class ServiceMetrics:
    def __init__(self) -> None:
        self.request_count = 0
        self.prediction_count = 0
        self.errors_count = 0
        self.total_latency_seconds = 0.0
        self._lock = Lock()

    def record_request(self, latency_seconds: float, is_error: bool = False) -> None:
        with self._lock:
            self.request_count += 1
            self.total_latency_seconds += latency_seconds
            if is_error:
                self.errors_count += 1

    def record_prediction(self) -> None:
        with self._lock:
            self.prediction_count += 1

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            avg_latency = (
                self.total_latency_seconds / self.request_count
                if self.request_count
                else 0.0
            )
            return {
                "request_count": self.request_count,
                "prediction_count": self.prediction_count,
                "errors_count": self.errors_count,
                "request_latency_avg_seconds": round(avg_latency, 6),
                "request_latency_total_seconds": round(self.total_latency_seconds, 6),
            }


service_metrics = ServiceMetrics()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_request_id() -> str:
    return str(uuid4())


def log_prediction(event: dict[str, Any], path: Path | None = None) -> None:
    path = path or LOGS_DIR / "inference_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def monotonic_time() -> float:
    return time.perf_counter()
