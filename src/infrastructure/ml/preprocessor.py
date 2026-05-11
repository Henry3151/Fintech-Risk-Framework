from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

AMOUNT_IDX = 29
TIME_IDX   = 0

class FraudPreprocessor:
    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray) -> 'FraudPreprocessor':
        df = pd.DataFrame(X[:, [TIME_IDX, AMOUNT_IDX]], columns=['Time', 'Amount'])
        self._scaler.fit(df)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_out = X.copy().astype(float)
        df = pd.DataFrame(X_out[:, [TIME_IDX, AMOUNT_IDX]], columns=['Time', 'Amount'])
        X_out[:, [TIME_IDX, AMOUNT_IDX]] = self._scaler.transform(df)
        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> 'FraudPreprocessor':
        return joblib.load(path)
