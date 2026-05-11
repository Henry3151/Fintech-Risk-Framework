# -*- coding: utf-8 -*-
from __future__ import annotations
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np

from domain.entities.customer import Customer


# Constroi features RFM a partir do DataFrame bruto do Online Retail UCI
class RFMBuilder:
    """
    Calcula Recency, Frequency e Monetary a partir do dataset Online Retail.
    Colunas esperadas: CustomerID, InvoiceDate, InvoiceNo, Quantity, UnitPrice.
    """

    # Recebe a data de referencia para calculo de recency (padrao: ultima data do dataset)
    def __init__(self, reference_date: datetime | None = None) -> None:
        self._reference_date = reference_date

    # Carrega o CSV, limpa os dados e retorna uma lista de entidades Customer com RFM
    def build_from_csv(self, path: str) -> List[Customer]:
        df = self._load_and_clean(path)
        return self._compute_rfm(df)

    # Recebe um DataFrame ja carregado, limpa e retorna lista de Customer
    def build_from_dataframe(self, df: pd.DataFrame) -> List[Customer]:
        df = self._clean(df)
        return self._compute_rfm(df)

    # Carrega o CSV com encoding latin-1 (padrao do Online Retail UCI)
    def _load_and_clean(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, encoding="latin-1")
        return self._clean(df)

    # Remove cancelamentos, nulls, quantidades negativas e precos zerados
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Remove linhas sem CustomerID
        df = df.dropna(subset=["Customer ID"])
        df["Customer ID"] = df["Customer ID"].astype(str).str.strip()

        # Remove cancelamentos (InvoiceNo comecando com C)
        df = df[~df["Invoice"].astype(str).str.startswith("C")]

        # Remove quantidades e precos invalidos
        df = df[df["Quantity"] > 0]
        df = df[df["Price"] > 0]

        # Converte InvoiceDate para datetime
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        # Calcula valor total por linha
        df["TotalPrice"] = df["Quantity"] * df["Price"]

        return df

    # Agrega por CustomerID calculando Recency, Frequency e Monetary
    def _compute_rfm(self, df: pd.DataFrame) -> List[Customer]:
        ref = self._reference_date or df["InvoiceDate"].max() + pd.Timedelta(days=1)

        rfm = df.groupby("Customer ID").agg(
            recency=("InvoiceDate",  lambda x: (ref - x.max()).days),
            frequency=("Invoice",  "nunique"),
            monetary=("TotalPrice",  "sum"),
        ).reset_index()

        # Remove outliers extremos (monetary > percentil 99)
        p99 = rfm["monetary"].quantile(0.99)
        rfm = rfm[rfm["monetary"] <= p99]

        return [
            Customer(
                customer_id=row["Customer ID"],
                recency=float(row["recency"]),
                frequency=float(row["frequency"]),
                monetary=float(row["monetary"]),
            )
            for _, row in rfm.iterrows()
        ]
