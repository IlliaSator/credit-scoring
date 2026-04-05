import pickle
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os


# ЛОГИРОВАНИЕ

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# КОНФИГ
# Параметры не хардкодим в коде — выносим в одно место
CONFIG = {
    "threshold": 0.23,        # оптимальный порог
    "model_path": "models/model.pkl",
    "features_path": "models/feature_names.pkl",
    "risk_thresholds": {
        "low": 0.10,
        "medium": 0.30
    }
}

#  ЗАГРУЗКА МОДЕЛИ


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Загрузка модели...")
    try:
        with open(CONFIG["model_path"], "rb") as f:
            app.state.model = pickle.load(f)
        with open(CONFIG["features_path"], "rb") as f:
            app.state.feature_names = pickle.load(f)
        logger.info(
            f"Модель загружена. Признаков: {len(app.state.feature_names)}")
    except FileNotFoundError as e:
        logger.error(f"Файл модели не найден: {e}")
        raise
    yield

    logger.info("Приложение остановлено")

# ПРИЛОЖЕНИЕ
app = FastAPI(
    title="Credit Scoring API",
    description="Предсказание вероятности дефолта по кредиту. Модель: GradientBoosting. Данные: Give Me Some Credit (Kaggle)",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def frontend():
    return FileResponse("src/frontend.html")

# СХЕМА ВХОДНЫХ ДАННЫХ


class ClientFeatures(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: int

    # Валидация входных данных
    @field_validator('age')
    def age_must_be_valid(cls, v):
        if v < 18 or v > 100:
            raise ValueError('Возраст должен быть от 18 до 100 лет')
        return v

    @field_validator('MonthlyIncome')
    def income_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Доход не может быть отрицательным')
        return v

    @field_validator('RevolvingUtilizationOfUnsecuredLines')
    def utilization_must_be_valid(cls, v):
        # Обрезаем выбросы
        return min(v, 1.0)

# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ


def build_features(client: ClientFeatures) -> np.ndarray:
    """Строит вектор признаков включая производные"""
    total_late = (client.NumberOfTime30_59DaysPastDueNotWorse +
                  client.NumberOfTimes90DaysLate +
                  client.NumberOfTime60_89DaysPastDueNotWorse)
    has_late = int(total_late > 0)
    debt_per_dep = client.DebtRatio / (client.NumberOfDependents + 1)
    is_young = int(client.age < 35)
    loans_per_inc = client.NumberOfOpenCreditLinesAndLoans / \
        (client.MonthlyIncome + 1)

    return np.array([[
        client.RevolvingUtilizationOfUnsecuredLines,
        client.age,
        client.NumberOfTime30_59DaysPastDueNotWorse,
        client.DebtRatio,
        client.MonthlyIncome,
        client.NumberOfOpenCreditLinesAndLoans,
        client.NumberOfTimes90DaysLate,
        client.NumberRealEstateLoansOrLines,
        client.NumberOfTime60_89DaysPastDueNotWorse,
        client.NumberOfDependents,
        total_late,
        has_late,
        debt_per_dep,
        is_young,
        loans_per_inc
    ]])

# ЭНДПОИНТЫ


@app.post("/predict")
def predict(client: ClientFeatures):
    try:
        #  признаки
        X = build_features(client)

        # Предсказание
        probability = app.state.model.predict_proba(X)[0][1]

        # Уровень риска
        t = CONFIG["risk_thresholds"]
        if probability < t["low"]:
            risk_level = "LOW"
        elif probability < t["medium"]:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # Решение по оптимальному порогу из ноутбука
        decision = "Отказать" if probability >= CONFIG["threshold"] else "Одобрить"

        logger.info(
            f"Предсказание: prob={probability:.4f} "
            f"risk={risk_level} decision={decision} "
            f"age={client.age} income={client.MonthlyIncome}"
        )

        return {
            "default_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "decision": decision,
            "threshold_used": CONFIG["threshold"]
        }

    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "GradientBoosting",
        "threshold": CONFIG["threshold"],
        "features": len(app.state.feature_names)
    }
