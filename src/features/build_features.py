# -*- coding: utf-8 -*-
"""
build_features.py - SMOTE para balancear o treino.
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

def load_splits():
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    X_train_legit = np.load(PROCESSED_DIR / "X_train_legit.npy")
    return X_train, X_test, y_train, y_test, X_train_legit

def apply_smote(X_train, y_train, sampling_strategy=0.1, random_state=42):
    print(f"  Antes do SMOTE - fraudes: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    fraud_count = y_res.sum()
    print(f"  Apos SMOTE    - fraudes: {fraud_count:,} ({fraud_count/len(y_res)*100:.2f}%)")
    print(f"  Total amostras: {len(y_res):,}")
    return X_res, y_res

def get_feature_names(n_pca=28):
    return [f"V{i}" for i in range(1, n_pca + 1)] + ["Amount", "Time"]
