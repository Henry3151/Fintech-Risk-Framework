# -*- coding: utf-8 -*-
"""
train_classifier.py - XGBoost supervisionado com SMOTE + anomaly score.
Uso: python src/models/train_classifier.py
"""
import json, joblib, numpy as np, torch, xgboost as xgb, sys, mlflow, mlflow.xgboost
from pathlib import Path
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
sys.path.insert(0, str(ROOT / "src"))
from features.build_features import apply_smote
from models.train_autoencoder import FraudAutoencoder

def load_autoencoder(input_dim=30, device="cpu"):
    model = FraudAutoencoder(input_dim=input_dim)
    model.load_state_dict(torch.load(MODELS_DIR / "autoencoder.pt", map_location=device))
    model.eval()
    return model

def add_reconstruction_error(X, model, device="cpu"):
    with torch.no_grad():
        errors = model.reconstruction_error(torch.tensor(X.astype(np.float32)).to(device)).numpy()
    return np.column_stack([X, errors])

def find_best_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1[:-1])
    print(f"  Melhor threshold: {thresholds[best_idx]:.4f} | F1: {f1[best_idx]:.4f}")
    return float(thresholds[best_idx])

def train(device="cpu"):
    print("=" * 55)
    print("  Treinando XGBoost Classifier")
    print("=" * 55)

    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    X_test  = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")

    print("\n[1/4] Adicionando reconstruction_error como feature ...")
    ae = load_autoencoder(input_dim=X_train.shape[1], device=device)
    X_train_enr = add_reconstruction_error(X_train, ae, device)
    X_test_enr  = add_reconstruction_error(X_test,  ae, device)
    print(f"  Shape apos enriquecimento: {X_train_enr.shape}")

    print("\n[2/4] Aplicando SMOTE ...")
    X_res, y_res = apply_smote(X_train_enr, y_train)

    print("\n[3/4] Treinando XGBoost ...")
    neg, pos = (y_res == 0).sum(), (y_res == 1).sum()
    params = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg/pos,
                  eval_metric="aucpr", early_stopping_rounds=30,
                  random_state=42, n_jobs=-1)
    clf = xgb.XGBClassifier(**params)

    val_size = int(0.15 * len(X_res))
    clf.fit(X_res[val_size:], y_res[val_size:],
            eval_set=[(X_res[:val_size], y_res[:val_size])], verbose=50)

    print("\n[4/4] Avaliando e calibrando threshold ...")
    y_proba   = clf.predict_proba(X_test_enr)[:, 1]
    pr_auc    = average_precision_score(y_test, y_proba)
    print(f"  PR-AUC: {pr_auc:.4f}")
    threshold = find_best_threshold(y_test, y_proba)
    print(classification_report(y_test, (y_proba >= threshold).astype(int),
                                target_names=["Legitima", "Fraude"]))

    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name="xgboost-classifier"):

        mlflow.log_params({
            "model":               "XGBoostClassifier",
            "n_estimators":        params["n_estimators"],
            "max_depth":           params["max_depth"],
            "learning_rate":       params["learning_rate"],
            "subsample":           params["subsample"],
            "colsample_bytree":    params["colsample_bytree"],
            "early_stopping_rounds": params["early_stopping_rounds"],
            "smote_sampling_strategy": 0.1,
        })

        mlflow.log_metrics({
            "pr_auc":            round(pr_auc, 4),
            "best_threshold":    round(threshold, 4),
            "best_iteration":    clf.best_iteration,
            "n_estimators_best": clf.best_iteration,
        })

        joblib.dump(clf, MODELS_DIR / "classifier.joblib")
        mlflow.log_artifact(str(MODELS_DIR / "classifier.joblib"), artifact_path="model")

        ae_meta = json.load(open(MODELS_DIR / "autoencoder_metadata.json"))
        metadata = {"model": "XGBoostClassifier", "pr_auc": round(pr_auc, 4),
                    "best_threshold": threshold,
                    "autoencoder_threshold": ae_meta["threshold"],
                    "n_estimators_best": clf.best_iteration,
                    "feature_dim": X_train_enr.shape[1],
                    "model_version": "autoencoder_v1"}
        json.dump(metadata, open(MODELS_DIR / "classifier_metadata.json", "w"), indent=2)
        mlflow.log_artifact(str(MODELS_DIR / "classifier_metadata.json"), artifact_path="model")

    print(f"\n  Modelo salvo: {MODELS_DIR / 'classifier.joblib'}")
    print("=" * 55)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(device=device)
