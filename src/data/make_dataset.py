"""
make_dataset.py — Carrega o CSV bruto e gera os splits de treino/teste.
Uso: python src/data/make_dataset.py
"""
import os, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

def load_raw(path=RAW_PATH):
    print(f"[1/5] Carregando dados de {path} ...")
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {path}\n"
            "Baixe em: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "e coloque creditcard.csv em data/raw/"
        )
    df = pd.read_csv(path)
    missing = {"Time", "Amount", "Class"} - set(df.columns)
    if missing:
        raise ValueError(f"Colunas faltando: {missing}")
    print(f"    Shape: {df.shape}")
    print(f"    Fraudes: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    return df

def preprocess(df):
    print("[2/5] Pre-processando features ...")
    scaler = StandardScaler()
    df = df.copy()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])
    return df, scaler

def split_data(df, test_size=0.2, random_state=42):
    print("[3/5] Dividindo treino/teste ...")
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values
    y = df["Class"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_train_legit = X_train[y_train == 0]
    print(f"    Treino: {X_train.shape[0]:,} | Legítimas: {X_train_legit.shape[0]:,} | Fraudes: {y_train.sum():,}")
    print(f"    Teste : {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test, X_train_legit

def save_processed(X_train, X_test, y_train, y_test, X_train_legit, scaler):
    print("[4/5] Salvando dados processados ...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    np.save(PROCESSED_DIR / "X_train_legit.npy", X_train_legit)
    joblib.dump(scaler, MODELS_DIR / "preprocessor.joblib")
    print(f"    Salvo em: {PROCESSED_DIR}")

def run():
    print("=" * 55)
    print("  Pipeline de preparacao de dados - Fraude em Cartao")
    print("=" * 55)
    df = load_raw()
    df, scaler = preprocess(df)
    X_train, X_test, y_train, y_test, X_train_legit = split_data(df)
    save_processed(X_train, X_test, y_train, y_test, X_train_legit, scaler)
    print("[5/5] Concluido com sucesso.")
    print("=" * 55)

if __name__ == "__main__":
    run()
