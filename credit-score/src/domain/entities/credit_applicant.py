# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

# Nomes das features na ordem esperada pelo modelo
FEATURE_NAMES = [
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
    # features derivadas
    "total_late_payments",
    "debt_to_income",
    "utilization_category",
]


# Representa um solicitante de credito com suas features financeiras brutas
@dataclass(frozen=True)
class CreditApplicant:
    """
    Entidade central do dominio de credit scoring.
    Contem as features financeiras brutas de um solicitante.
    """
    applicant_id: str
    revolving_utilization: float    # utilizacao de credito rotativo [0, 1]
    age: int                        # idade em anos
    late_30_59_days: int            # atrasos de 30-59 dias
    debt_ratio: float               # razao divida/renda
    monthly_income: float           # renda mensal em USD
    open_credit_lines: int          # linhas de credito abertas
    late_90_days: int               # atrasos > 90 dias
    real_estate_loans: int          # emprestimos imobiliarios
    late_60_89_days: int            # atrasos de 60-89 dias
    dependents: int                 # numero de dependentes

    # Valida que os campos obrigatorios sao validos e nao negativos
    def __post_init__(self) -> None:
        if self.age <= 0:
            raise ValueError(f"age must be > 0, got {self.age}")
        if self.monthly_income < 0:
            raise ValueError(f"monthly_income must be >= 0, got {self.monthly_income}")
        if not (0 <= self.revolving_utilization <= 1):
            raise ValueError(f"revolving_utilization must be in [0,1], got {self.revolving_utilization}")

    # Retorna array numpy com features originais + features derivadas
    def to_numpy(self) -> np.ndarray:
        total_late = self.late_30_59_days + self.late_60_89_days + self.late_90_days
        debt_to_income = self.debt_ratio / (self.monthly_income + 1e-9)
        utilization_cat = min(int(self.revolving_utilization * 5), 4)  # 0-4
        return np.array([
            self.revolving_utilization,
            self.age,
            self.late_30_59_days,
            self.debt_ratio,
            self.monthly_income,
            self.open_credit_lines,
            self.late_90_days,
            self.real_estate_loans,
            self.late_60_89_days,
            self.dependents,
            total_late,
            debt_to_income,
            utilization_cat,
        ], dtype=np.float64)

    # Constroi um CreditApplicant a partir de um dicionario (ex: request da API)
    @classmethod
    def from_dict(cls, data: dict, applicant_id: str = "unknown") -> "CreditApplicant":
        return cls(
            applicant_id=applicant_id,
            revolving_utilization=float(data["RevolvingUtilizationOfUnsecuredLines"]),
            age=int(data["age"]),
            late_30_59_days=int(data["NumberOfTime30-59DaysPastDueNotWorse"]),
            debt_ratio=float(data["DebtRatio"]),
            monthly_income=float(data["MonthlyIncome"]),
            open_credit_lines=int(data["NumberOfOpenCreditLinesAndLoans"]),
            late_90_days=int(data["NumberOfTimes90DaysLate"]),
            real_estate_loans=int(data["NumberRealEstateLoansOrLines"]),
            late_60_89_days=int(data["NumberOfTime60-89DaysPastDueNotWorse"]),
            dependents=int(data["NumberOfDependents"]),
        )


# Representa o resultado do scoring de credito com score, risco e explicabilidade
@dataclass(frozen=True)
class CreditScore:
    """
    Resultado do credit scoring para um solicitante.
    score         : escala 0-1000 (quanto maior, melhor o credito)
    pd            : probability of default [0, 1]
    risk_grade    : A, B, C, D, E (padrao de mercado)
    top_factors   : lista dos fatores que mais impactaram o score (SHAP)
    """
    applicant_id: str
    score: int                      # 0-1000
    pd: float                       # probability of default
    risk_grade: str                 # A, B, C, D, E
    recommendation: str             # APPROVE, REVIEW, DENY
    top_factors: list               # top 3 features SHAP
    latency_ms: float

    # Retorna True se a recomendacao e aprovar o credito
    @property
    def is_approved(self) -> bool:
        return self.recommendation == "APPROVE"

    # Mapeia probability of default para score 0-1000 (invertido: menor PD = maior score)
    @staticmethod
    def pd_to_score(pd: float) -> int:
        score = int((1 - pd) * 1000)
        return max(0, min(1000, score))

    # Mapeia score para risk grade (A=excelente, E=alto risco)
    @staticmethod
    def score_to_grade(score: int) -> str:
        if score >= 800: return "A"
        if score >= 650: return "B"
        if score >= 500: return "C"
        if score >= 350: return "D"
        return "E"

    # Mapeia score para recomendacao de negocio
    @staticmethod
    def score_to_recommendation(score: int) -> str:
        if score >= 650: return "APPROVE"
        if score >= 400: return "REVIEW"
        return "DENY"
