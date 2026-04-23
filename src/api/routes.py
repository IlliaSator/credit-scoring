from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from src.core.config import DEFAULT_THRESHOLD, PROJECT_ROOT
from src.model_io import load_model_metadata
from src.schemas.credit import ClientFeatures
from src.services.explainability import explain_prediction
from src.services.prediction import predict_default
from src.services.telemetry import (
    log_prediction,
    new_request_id,
    now_utc,
    service_metrics,
)

router = APIRouter()

SAMPLE_CLIENT = {
    "RevolvingUtilizationOfUnsecuredLines": 0.95,
    "age": 32,
    "NumberOfTime30_59DaysPastDueNotWorse": 2,
    "DebtRatio": 0.65,
    "MonthlyIncome": 2800,
    "NumberOfOpenCreditLinesAndLoans": 9,
    "NumberOfTimes90DaysLate": 1,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60_89DaysPastDueNotWorse": 1,
    "NumberOfDependents": 2,
}


@router.get("/")
def frontend() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "src" / "frontend.html")


@router.post("/predict")
def predict(client: ClientFeatures, request: Request):
    payload = client.model_dump()
    result = predict_default(request.app.state.model, payload, DEFAULT_THRESHOLD)
    service_metrics.record_prediction()
    request_id = new_request_id()
    log_prediction(
        {
            "request_id": request_id,
            "timestamp": now_utc(),
            "probability": result["default_probability"],
            "risk_level": result["risk_level"],
            "decision": result["decision"],
        }
    )
    return result


@router.post("/explain")
def explain(client: ClientFeatures, request: Request):
    return explain_prediction(
        request.app.state.model,
        client.model_dump(),
        request.app.state.feature_names,
    )


@router.get("/explain/sample")
def explain_sample(request: Request):
    return explain_prediction(
        request.app.state.model,
        SAMPLE_CLIENT,
        request.app.state.feature_names,
    )


@router.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model": "GradientBoosting",
        "threshold": DEFAULT_THRESHOLD,
        "features": len(request.app.state.feature_names),
    }


@router.get("/metrics")
def metrics():
    return service_metrics.snapshot()


@router.get("/model/info")
def model_info(request: Request):
    metadata = load_model_metadata()
    return {
        "model_name": metadata.get("model_name", "GradientBoostingClassifier"),
        "model_version": metadata.get("model_version", "manual-notebook-artifact"),
        "threshold": metadata.get("threshold", DEFAULT_THRESHOLD),
        "feature_count": len(request.app.state.feature_names),
        "trained_at": metadata.get("trained_at"),
        "positive_class_rate": metadata.get("positive_class_rate"),
    }
