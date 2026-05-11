# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np

from domain.entities.credit_applicant import CreditApplicant


# Processa o CSV bruto do Give Me Some Credit aplicando todas as regras de limpeza
class CreditDataProcessor:
    """
    Pipeline de feature engineering para o dataset Give Me Some Credit.
    Trata nulls, outliers, valores sentinela e cria features derivadas.
    """

    # Define os limites de clip para outliers baseados na analise exploratoria
    CLIP_RULES = {
        "RevolvingUtilizationOfUnsecuredLines": (0.0, 1.0),
        "DebtRatio": (0.0, 10.0),
        "MonthlyIncome": (0.0, 50000.0),
        "NumberOfTime30-59DaysPastDueNotWorse": (0, 10),
        "NumberOfTime60-89DaysPastDueNotWorse": (0, 10),
        "NumberOfTimes90DaysLate": (0, 10),
    }

    # Carrega o CSV, aplica todas as transformacoes e retorna X, y prontos para treino
    def load_and_process(self, path: str) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(path)
        df = self._clean(df)
        df = self._engineer_features(df)
        X  = df.drop(columns=["SeriousDlqin2yrs"])
        y  = df["SeriousDlqin2yrs"]
        return X, y

    # Remove coluna de indice, registros invalidos e aplica todas as limpezas
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Remove coluna de indice gerada pelo pandas
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # Remove registros com age invalido
        df = df[df["age"] > 0]
        df = df[df["age"] <= 100]

        # Trata valor sentinela 98 nas colunas de atraso como outlier
        for col in ["NumberOfTime30-59DaysPastDueNotWorse",
                    "NumberOfTime60-89DaysPastDueNotWorse",
                    "NumberOfTimes90DaysLate"]:
            df[col] = df[col].replace(98, np.nan)

        # Imputa MonthlyIncome pela mediana da faixa etaria (grupos de 10 anos)
        df["age_group"] = (df["age"] // 10) * 10
        df["MonthlyIncome"] = df.groupby("age_group")["MonthlyIncome"].transform(
            lambda x: x.fillna(x.median())
        )
        # Fallback: mediana global para grupos sem dados suficientes
        df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

        # Imputa NumberOfDependents pela mediana global
        df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
            df["NumberOfDependents"].median()
        )

        # Imputa colunas de atraso com 0 (ausencia de atraso e o valor mais plausivel)
        for col in ["NumberOfTime30-59DaysPastDueNotWorse",
                    "NumberOfTime60-89DaysPastDueNotWorse",
                    "NumberOfTimes90DaysLate"]:
            df[col] = df[col].fillna(0)

        # Aplica clip nos outliers conforme regras definidas
        for col, (low, high) in self.CLIP_RULES.items():
            if col in df.columns:
                df[col] = df[col].clip(low, high)

        df = df.drop(columns=["age_group"])
        return df

    # Cria features derivadas que aumentam o poder preditivo do modelo
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Total de pagamentos em atraso (soma das tres janelas temporais)
        df["total_late_payments"] = (
            df["NumberOfTime30-59DaysPastDueNotWorse"] +
            df["NumberOfTime60-89DaysPastDueNotWorse"] +
            df["NumberOfTimes90DaysLate"]
        )

        # Razao divida/renda (debt-to-income ratio)
        df["debt_to_income"] = df["DebtRatio"] / (df["MonthlyIncome"] + 1e-9)
        df["debt_to_income"] = df["debt_to_income"].clip(0, 100)

        # Categorizacao da utilizacao de credito (0=baixa, 4=maxima)
        df["utilization_category"] = (df["RevolvingUtilizationOfUnsecuredLines"] * 5).clip(0, 4).astype(int)

        return df

    # Converte um DataFrame processado em lista de entidades CreditApplicant
    def to_applicants(self, df: pd.DataFrame) -> List[CreditApplicant]:
        applicants = []
        for idx, row in df.iterrows():
            try:
                applicants.append(CreditApplicant(
                    applicant_id=str(idx),
                    revolving_utilization=float(row["RevolvingUtilizationOfUnsecuredLines"]),
                    age=int(row["age"]),
                    late_30_59_days=int(row["NumberOfTime30-59DaysPastDueNotWorse"]),
                    debt_ratio=float(row["DebtRatio"]),
                    monthly_income=float(row["MonthlyIncome"]),
                    open_credit_lines=int(row["NumberOfOpenCreditLinesAndLoans"]),
                    late_90_days=int(row["NumberOfTimes90DaysLate"]),
                    real_estate_loans=int(row["NumberRealEstateLoansOrLines"]),
                    late_60_89_days=int(row["NumberOfTime60-89DaysPastDueNotWorse"]),
                    dependents=int(row["NumberOfDependents"]),
                ))
            except ValueError:
                continue
        return applicants
