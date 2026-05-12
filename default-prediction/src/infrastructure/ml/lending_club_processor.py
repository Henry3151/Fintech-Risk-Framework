# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple
import pandas as pd
import numpy as np

from domain.entities.loan import GRADE_MAP


# Processa o CSV bruto do Lending Club para Survival Analysis
class LendingClubProcessor:
    """
    Pipeline de preparacao de dados do Lending Club para Cox PH.

    Colunas necessarias:
      loan_status, issue_d, last_pymnt_d, term,
      loan_amnt, int_rate, grade, emp_length,
      annual_inc, dti, fico_range_low,
      open_acc, revol_util, total_acc,
      inq_last_6mths, pub_rec

    Saida para lifelines.CoxPHFitter:
      duration_col : 'duration_months' (tempo ate evento ou censura)
      event_col    : 'event' (1=default, 0=censurado)
    """

    # Statuses que representam default (evento)
    DEFAULT_STATUSES = {
        "Charged Off",
        "Default",
        "Late (31-120 days)",
    }

    # Colunas necessarias para o modelo
    REQUIRED_COLS = [
        "loan_status", "issue_d", "last_pymnt_d", "term",
        "loan_amnt", "int_rate", "grade", "emp_length",
        "annual_inc", "dti", "fico_range_low",
        "open_acc", "revol_util", "total_acc",
        "inq_last_6mths", "pub_rec",
    ]

    # Carrega o CSV, processa e retorna DataFrame pronto para lifelines
    def load_and_process(
        self,
        path: str,
        sample_n: int = 200_000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        print(f"    Carregando {path}...")
        df = pd.read_csv(
            path,
            usecols=self.REQUIRED_COLS,
            low_memory=False,
        )
        print(f"    Shape bruto: {df.shape}")

        # Amostra para treino rapido se necessario
        if sample_n and len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=random_state)
            print(f"    Amostra: {df.shape}")

        df = self._clean(df)
        df = self._compute_duration(df)
        df = self._engineer_features(df)
        df = self._select_features(df)
        print(f"    Shape final: {df.shape}")
        print(f"    Eventos (default): {df['event'].sum()} ({df['event'].mean():.2%})")
        print(f"    Censurados: {(df['event'] == 0).sum()}")
        return df

    # Remove nulls, filtra grades invalidas e converte tipos
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Remove linhas sem datas ou grade invalida
        df = df.dropna(subset=["issue_d", "loan_status"])
        df = df[df["grade"].isin(GRADE_MAP.keys())]

        # Converte datas
        df["issue_d"]     = pd.to_datetime(df["issue_d"],     format="%b-%Y")
        df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")

        # Converte int_rate de string "13.99%" para float 13.99
        df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "").astype(float)

        # Converte revol_util de string para float
        df["revol_util"] = df["revol_util"].astype(str).str.replace("%", "")
        df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")

        # Converte term de "36 months" para 36
        df["term_months"] = df["term"].str.extract(r"(\d+)").astype(int)

        # Converte emp_length para anos numericos
        df["emp_length_years"] = self._parse_emp_length(df["emp_length"])

        # Imputa nulls
        df["revol_util"]       = df["revol_util"].fillna(df["revol_util"].median())
        df["emp_length_years"] = df["emp_length_years"].fillna(df["emp_length_years"].median())
        df["dti"]              = df["dti"].fillna(df["dti"].median())
        df["inq_last_6mths"]   = df["inq_last_6mths"].fillna(0)
        df["pub_rec"]          = df["pub_rec"].fillna(0)
        df["open_acc"]         = df["open_acc"].fillna(df["open_acc"].median())
        df["total_acc"]        = df["total_acc"].fillna(df["total_acc"].median())

        # Remove outliers extremos de annual_inc
        p99 = df["annual_inc"].quantile(0.99)
        df  = df[df["annual_inc"] <= p99]

        return df

    # Calcula duration_months (tempo ate evento ou censura) e event (0/1)
    def _compute_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Define evento: 1 se default, 0 se censurado
        df["event"] = df["loan_status"].isin(self.DEFAULT_STATUSES).astype(int)

        # Para emprestimos sem last_pymnt_d, usa issue_d + term como censura
        df["end_date"] = df["last_pymnt_d"]
        mask_no_pymnt = df["end_date"].isna()
        df.loc[mask_no_pymnt, "end_date"] = (
            df.loc[mask_no_pymnt, "issue_d"] +
            pd.to_timedelta(df.loc[mask_no_pymnt, "term_months"] * 30, unit="D")
        )

        # Calcula duracao em meses
        df["duration_months"] = (
            (df["end_date"] - df["issue_d"]) / np.timedelta64(30, "D")
        ).round(1)

        # Remove duracoes invalidas
        df = df[df["duration_months"] > 0]
        df = df[df["duration_months"] <= 72]  # max 6 anos

        return df

    # Cria grade_encoded numerico a partir da coluna grade
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["grade_encoded"] = df["grade"].map(GRADE_MAP)
        return df

    # Seleciona apenas as colunas necessarias para o modelo Cox
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            "duration_months", "event",
            "loan_amnt", "int_rate", "grade_encoded",
            "emp_length_years", "annual_inc", "dti",
            "fico_range_low", "open_acc", "revol_util",
            "total_acc", "inq_last_6mths", "pub_rec", "term_months",
        ]
        return df[feature_cols].dropna()

    # Converte a coluna emp_length para anos numericos
    @staticmethod
    def _parse_emp_length(series: pd.Series) -> pd.Series:
        def parse(val):
            if pd.isna(val): return np.nan
            val = str(val).lower().strip()
            if "10+" in val: return 10.0
            if "< 1" in val: return 0.5
            digits = "".join(c for c in val if c.isdigit())
            return float(digits) if digits else np.nan
        return series.apply(parse)
