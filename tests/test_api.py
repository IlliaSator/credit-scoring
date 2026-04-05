import pytest
from fastapi.testclient import TestClient
from src.main import app

# Используем with — это запускает lifespan (загрузку модели)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

# ── HEALTH CHECK ──────────────────────────────────────────────────


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "GradientBoosting"
    assert data["features"] == 15

# ── ВАЛИДНЫЕ ЗАПРОСЫ ──────────────────────────────────────────────


def test_predict_low_risk(client):
    """Надёжный клиент должен получить LOW риск"""
    response = client.post("/predict", json={
        "RevolvingUtilizationOfUnsecuredLines": 0.10,
        "age": 55,
        "NumberOfTime30_59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.20,
        "MonthlyIncome": 8000,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1
    })
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "risk_level" in data
    assert "decision" in data
    assert "threshold_used" in data
    assert data["default_probability"] < 0.23
    assert data["risk_level"] == "LOW"
    assert data["decision"] == "Одобрить"


def test_predict_high_risk(client):
    """Клиент с просрочками должен получить HIGH риск"""
    response = client.post("/predict", json={
        "RevolvingUtilizationOfUnsecuredLines": 0.95,
        "age": 25,
        "NumberOfTime30_59DaysPastDueNotWorse": 3,
        "DebtRatio": 0.80,
        "MonthlyIncome": 2000,
        "NumberOfOpenCreditLinesAndLoans": 12,
        "NumberOfTimes90DaysLate": 2,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60_89DaysPastDueNotWorse": 2,
        "NumberOfDependents": 3
    })
    assert response.status_code == 200
    data = response.json()
    assert data["default_probability"] >= 0.23
    assert data["risk_level"] == "HIGH"
    assert data["decision"] == "Отказать"


def test_predict_returns_probability_range(client):
    """Вероятность должна быть от 0 до 1"""
    response = client.post("/predict", json={
        "RevolvingUtilizationOfUnsecuredLines": 0.50,
        "age": 40,
        "NumberOfTime30_59DaysPastDueNotWorse": 1,
        "DebtRatio": 0.40,
        "MonthlyIncome": 5000,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2
    })
    assert response.status_code == 200
    prob = response.json()["default_probability"]
    assert 0.0 <= prob <= 1.0

# ── ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ ──────────────────────────────────────


def test_predict_invalid_age_too_young(client):
    response = client.post("/predict", json={
        "RevolvingUtilizationOfUnsecuredLines": 0.30,
        "age": 15,
        "NumberOfTime30_59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.35,
        "MonthlyIncome": 5000,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0
    })
    assert response.status_code == 422


def test_predict_invalid_age_too_old(client):
    response = client.post("/predict", json={
        "RevolvingUtilizationOfUnsecuredLines": 0.30,
        "age": 150,
        "NumberOfTime30_59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.35,
        "MonthlyIncome": 5000,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0
    })
    assert response.status_code == 422


def test_predict_negative_income(client):
    response = client.post("/predict", json={
        "RevolvingUtilizationOfUnsecuredLines": 0.30,
        "age": 40,
        "NumberOfTime30_59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.35,
        "MonthlyIncome": -1000,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0
    })
    assert response.status_code == 422


def test_predict_missing_field(client):
    response = client.post("/predict", json={
        "age": 40,
        "MonthlyIncome": 5000
    })
    assert response.status_code == 422


def test_predict_wrong_type(client):
    response = client.post("/predict", json={
        "RevolvingUtilizationOfUnsecuredLines": "много",
        "age": 40,
        "NumberOfTime30_59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.35,
        "MonthlyIncome": 5000,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0
    })
    assert response.status_code == 422
