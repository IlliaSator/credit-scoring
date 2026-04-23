import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src.api.routes import router
from src.core.config import FEATURE_NAMES_PATH, MODEL_PATH
from src.model_io import load_feature_names, load_model
from src.services.telemetry import monotonic_time, service_metrics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts...")
    app.state.model = load_model(MODEL_PATH)
    app.state.feature_names = load_feature_names(FEATURE_NAMES_PATH)
    logger.info("Model loaded with %s features", len(app.state.feature_names))
    yield
    logger.info("Application stopped")


app = FastAPI(
    title="Credit Scoring API",
    description=(
        "Default probability scoring for consumer credit applications. "
        "Model: GradientBoosting. Dataset: Give Me Some Credit."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def collect_service_metrics(request: Request, call_next):
    start = monotonic_time()
    is_error = False
    try:
        response = await call_next(request)
        is_error = response.status_code >= 500
        return response
    except Exception:
        is_error = True
        raise
    finally:
        service_metrics.record_request(monotonic_time() - start, is_error=is_error)


app.include_router(router)
