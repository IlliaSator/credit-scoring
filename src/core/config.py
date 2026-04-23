from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

DEFAULT_THRESHOLD = 0.23
RISK_THRESHOLDS = {
    "low": 0.10,
    "medium": 0.30,
}

TARGET_COLUMN = "SeriousDlqin2yrs"
ID_COLUMN = "Unnamed: 0"

RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

ENGINEERED_FEATURES = [
    "TotalLatePayments",
    "HasAnyLatePayment",
    "DebtPerDependent",
    "IsYoung",
    "LoansPerIncome",
]

FEATURE_NAMES = RAW_FEATURES + ENGINEERED_FEATURES
