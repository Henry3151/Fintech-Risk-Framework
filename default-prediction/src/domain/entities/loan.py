# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

# Nomes das features na ordem esperada pelo modelo Cox
FEATURE_NAMES = [
    "loan_amnt",
    "int_rate",
    "grade_encoded",
    "emp_length_years",
    "annual_inc",
    "dti",
    "fico_range_low",
    "open_acc",
    "revol_util",
    "total_acc",
    "inq_last_6mths",
    "pub_rec",
    "term_months",
]

# Mapeamento de grade para valor numerico (A=melhor, G=pior)
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}


# Representa um emprestimo com features para predicao de default
@dataclass(frozen=True)
class Loan:
    """
    Entidade central do dominio de default prediction.
    Contem as features financeiras e temporais de um emprestimo.
    """
    loan_id: str
    loan_amnt: float          # valor do emprestimo em USD
    int_rate: float           # taxa de juros anual em %
    grade: str                # grade do emprestimo (A-G)
    emp_length_years: float   # tempo de emprego em anos
    annual_inc: float         # renda anual em USD
    dti: float                # debt-to-income ratio
    fico_range_low: float     # score FICO minimo
    open_acc: int             # contas abertas
    revol_util: float         # utilizacao de credito rotativo %
    total_acc: int            # total de contas
    inq_last_6mths: int       # consultas de credito nos ultimos 6 meses
    pub_rec: int              # registros publicos negativos
    term_months: int          # prazo do emprestimo (36 ou 60 meses)

    # Valida que os campos criticos sao validos
    def __post_init__(self) -> None:
        if self.loan_amnt <= 0:
            raise ValueError(f"loan_amnt must be > 0, got {self.loan_amnt}")
        if not (0 < self.int_rate < 100):
            raise ValueError(f"int_rate must be in (0, 100), got {self.int_rate}")
        if self.grade not in GRADE_MAP:
            raise ValueError(f"grade must be A-G, got {self.grade}")
        if self.annual_inc < 0:
            raise ValueError(f"annual_inc must be >= 0, got {self.annual_inc}")
        if self.term_months not in (36, 60):
            raise ValueError(f"term_months must be 36 or 60, got {self.term_months}")

    # Retorna array numpy com todas as features numericas
    def to_numpy(self) -> np.ndarray:
        return np.array([
            self.loan_amnt,
            self.int_rate,
            GRADE_MAP[self.grade],
            self.emp_length_years,
            self.annual_inc,
            self.dti,
            self.fico_range_low,
            self.open_acc,
            self.revol_util,
            self.total_acc,
            self.inq_last_6mths,
            self.pub_rec,
            self.term_months,
        ], dtype=np.float64)

    # Constroi um Loan a partir de um dicionario (ex: request da API)
    @classmethod
    def from_dict(cls, data: dict, loan_id: str = "unknown") -> "Loan":
        return cls(
            loan_id=loan_id,
            loan_amnt=float(data["loan_amnt"]),
            int_rate=float(data["int_rate"]),
            grade=str(data["grade"]).upper(),
            emp_length_years=float(data["emp_length_years"]),
            annual_inc=float(data["annual_inc"]),
            dti=float(data["dti"]),
            fico_range_low=float(data["fico_range_low"]),
            open_acc=int(data["open_acc"]),
            revol_util=float(data["revol_util"]),
            total_acc=int(data["total_acc"]),
            inq_last_6mths=int(data["inq_last_6mths"]),
            pub_rec=int(data["pub_rec"]),
            term_months=int(data["term_months"]),
        )


# Representa o resultado da predicao de default com curva de sobrevivencia
@dataclass(frozen=True)
class DefaultPrediction:
    """
    Resultado da predicao de default para um emprestimo.
    survival_at_12m : probabilidade de nao defaultar em 12 meses
    survival_at_24m : probabilidade de nao defaultar em 24 meses
    survival_at_36m : probabilidade de nao defaultar em 36 meses
    median_survival : tempo mediano ate default em meses
    hazard_ratio    : risco relativo vs. emprestimo base (Cox)
    risk_tier       : LOW, MEDIUM, HIGH, VERY_HIGH
    """
    loan_id: str
    survival_at_12m: float
    survival_at_24m: float
    survival_at_36m: float
    median_survival_months: Optional[float]
    hazard_ratio: float
    risk_tier: str
    pd_12m: float   # probability of default em 12 meses
    latency_ms: float

    # Retorna True se o emprestimo e considerado de baixo risco
    @property
    def is_low_risk(self) -> bool:
        return self.risk_tier in {"LOW"}

    # Classifica o risco baseado na probabilidade de default em 12 meses
    @staticmethod
    def classify_risk(pd_12m: float) -> str:
        if pd_12m < 0.05:  return "LOW"
        if pd_12m < 0.15:  return "MEDIUM"
        if pd_12m < 0.30:  return "HIGH"
        return "VERY_HIGH"
