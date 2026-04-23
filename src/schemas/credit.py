from pydantic import BaseModel, field_validator


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

    @field_validator("age")
    @classmethod
    def age_must_be_valid(cls, value: int) -> int:
        if value < 18 or value > 100:
            raise ValueError("Age must be between 18 and 100")
        return value

    @field_validator("MonthlyIncome")
    @classmethod
    def income_must_be_positive(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Monthly income cannot be negative")
        return value

    @field_validator("RevolvingUtilizationOfUnsecuredLines")
    @classmethod
    def utilization_must_be_valid(cls, value: float) -> float:
        return min(value, 1.0)


class PredictionResponse(BaseModel):
    default_probability: float
    risk_level: str
    decision: str
    threshold_used: float
