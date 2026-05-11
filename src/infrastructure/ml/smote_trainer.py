"""
SmoteTrainer — handles class-imbalance oversampling.
Mirrors the original features/build_features.py (SMOTE sampling_strategy=0.1).
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE


class SmoteTrainer:
    """
    Applies SMOTE oversampling to the training set.

    Parameters
    ----------
    sampling_strategy : ratio of minority to majority after resampling.
                        0.1 means 10 % of legit transactions = fraud transactions.
    random_state      : reproducibility seed.
    """

    def __init__(
        self,
        sampling_strategy: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self._smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

    def resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Oversample the minority class.

        Returns
        -------
        X_resampled, y_resampled
        """
        return self._smote.fit_resample(X, y)
