# 🏦 Credit Scoring API

Система предсказания вероятности дефолта по кредиту на основе реальных банковских данных.

## 📊 Результаты

| Модель | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.862 | 0.389 | 0.216 |
| **GradientBoosting** | **0.868** | **0.400** | **0.448** |

> PR-AUC выбрана как основная метрика из-за дисбаланса классов (6.7% дефолтов).
> Оптимальный порог 0.23 подобран перебором по F1 — на 60% лучше стандартного 0.5.

## 🔍 Ключевые находки

- **TotalLatePayments** — созданный признак стал важнейшим (importance=0.59)
- Молодые заёмщики (18-35 лет) имеют вдвое больший риск дефолта
- 19.8% пропусков в MonthlyIncome обработаны медианной импутацией

## 🗂️ Структура проекта
```
credit-scoring/
├── notebooks/
│   └── 01_eda_and_modeling.ipynb  # EDA, очистка, обучение, метрики
├── src/
│   └── main.py                    # FastAPI с логированием и валидацией
├── models/
│   ├── model.pkl                  # Обученная модель
│   └── feature_names.pkl          # Список признаков
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🚀 Быстрый старт

### Через Docker (рекомендуется)
```bash
git clone https://github.com/твой-профиль/credit-scoring.git
cd credit-scoring
docker-compose up -d
```

API доступен на `http://localhost:8000/docs`

### Локально
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn src.main:app --reload
```

## 📡 Пример запроса
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.85,
    "age": 28,
    "NumberOfTime30_59DaysPastDueNotWorse": 2,
    "DebtRatio": 0.6,
    "MonthlyIncome": 3000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 1,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60_89DaysPastDueNotWorse": 1,
    "NumberOfDependents": 1
  }'
```

Ответ:
```json
{
  "default_probability": 0.5667,
  "risk_level": "HIGH",
  "decision": "Отказать",
  "threshold_used": 0.23
}
```

## 🛠️ Стек

`Python` `scikit-learn` `FastAPI` `Docker` `MLflow` `pandas` `seaborn`

## 📈 Данные

Датасет: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) (Kaggle)
- 150,000 реальных записей банковских клиентов
- Скачайте `cs-training.csv` и положите в `data/`