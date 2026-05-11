# -*- coding: utf-8 -*-
"""
Script de treinamento do modelo de credit scoring.

Uso:
    python scripts/train_credit_score.py --data data/raw/cs-training.csv

Dataset:
    https://www.kaggle.com/competitions/GiveMeSomeCredit/data
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, roc_curve
)

from infrastructure.ml.credit_data_processor import CreditDataProcessor
from infrastructure.repositories.file_scoring_repository import FileScoringRepository


# Calcula a estatistica KS (Kolmogorov-Smirnov) entre adimplentes e inadimplentes
def ks_statistic(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


# Analisa vies algoritmico por faixa etaria comparando taxas de aprovacao
def fairness_analysis(y_true, y_prob, ages, threshold=0.35):
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "age": ages})
    df["predicted_default"] = (df["y_prob"] >= threshold).astype(int)
    df["age_group"] = pd.cut(df["age"], bins=[18, 30, 40, 50, 60, 70, 100],
                              labels=["18-30", "31-40", "41-50", "51-60", "61-70", "71+"])
    result = df.groupby("age_group", observed=True).agg(
        approval_rate=("predicted_default", lambda x: 1 - x.mean()),
        default_rate=("y_true", "mean"),
        n=("y_true", "count")
    ).round(4)
    return result


# Ponto de entrada do script de treinamento
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--figures-dir", default="reports/figures")
    args = parser.parse_args()

    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    print(">>> Carregando e processando dados...")
    processor = CreditDataProcessor()
    X, y = processor.load_and_process(args.data)
    print(f"    Shape apos limpeza: {X.shape}")
    print(f"    Taxa de inadimplencia: {y.mean():.4f}")

    # Guarda coluna de idade para analise de fairness antes de remover do X
    ages = X["age"].values

    X_train, X_val, y_train, y_val, ages_train, ages_val = train_test_split(
        X, y, ages, test_size=0.2, random_state=42, stratify=y
    )

    print(">>> Treinando LightGBM...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    print(">>> Calibrando probabilidades (Platt scaling)...")
    calibrator = CalibratedClassifierCV(lgbm, method="sigmoid", cv=5)
    calibrator.fit(X_val, y_val)

    print(">>> Calculando metricas...")
    y_prob_cal = calibrator.predict_proba(X_val)[:, 1]
    auc    = roc_auc_score(y_val, y_prob_cal)
    pr_auc = average_precision_score(y_val, y_prob_cal)
    ks     = ks_statistic(y_val, y_prob_cal)
    brier  = brier_score_loss(y_val, y_prob_cal)

    print(f"    AUC-ROC   : {auc:.4f}")
    print(f"    PR-AUC    : {pr_auc:.4f}")
    print(f"    KS        : {ks:.4f}")
    print(f"    Brier     : {brier:.4f}")

    print(">>> Analisando vies por faixa etaria...")
    fairness = fairness_analysis(y_val.values, y_prob_cal, ages_val)
    print(fairness.to_string())

    print(">>> Calculando SHAP values...")
    explainer = shap.TreeExplainer(lgbm)
    shap_vals = explainer.shap_values(X_val.iloc[:500])
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    print(">>> Gerando graficos...")
    _plot_shap_summary(shap_vals, X_val.iloc[:500], args.figures_dir)
    _plot_calibration(y_val, y_prob_cal, args.figures_dir)
    _plot_score_distribution(y_val, y_prob_cal, args.figures_dir)
    _plot_fairness(fairness, args.figures_dir)
    _plot_dashboard(auc, pr_auc, ks, brier, shap_vals, X_val.iloc[:500],
                    y_val, y_prob_cal, fairness, args.figures_dir)

    print(">>> Salvando artefatos...")
    metadata = {
        "auc_roc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "ks_statistic": round(ks, 4),
        "brier_score": round(brier, 4),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": X_train.shape[1],
        "feature_names": X_train.columns.tolist(),
        "best_iteration": lgbm.best_iteration_,
    }
    repo = FileScoringRepository(models_dir=args.models_dir)
    repo.save_artifacts(lgbm, calibrator, explainer, metadata)
    print(f">>> Artefatos salvos em {args.models_dir}/")


# Gera o grafico de SHAP summary (feature importance global)
def _plot_shap_summary(shap_vals, X, figures_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    mean_abs = np.abs(shap_vals).mean(axis=0)
    indices  = np.argsort(mean_abs)[::-1][:10]
    names    = [X.columns[i] for i in indices]
    vals     = mean_abs[indices]
    bars = ax.barh(range(len(names)), vals[::-1], color="#3498db", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], color="white", fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", color="white")
    ax.set_title("Feature Importance (SHAP)", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/shap_summary.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()


# Gera o grafico de calibracao de probabilidade (reliability diagram)
def _plot_calibration(y_true, y_prob, figures_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    fraction_of_positives, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(mean_predicted, fraction_of_positives, "s-", color="#2ecc71", label="LightGBM calibrado")
    ax.plot([0, 1], [0, 1], "--", color="#95a5a6", label="Calibracao perfeita")
    ax.set_xlabel("Probabilidade prevista", color="white")
    ax.set_ylabel("Fracao de positivos", color="white")
    ax.set_title("Calibration Curve (Reliability Diagram)", color="white", fontsize=12)
    ax.legend(facecolor="#0d1117", labelcolor="white", edgecolor="#30363d")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/calibration_curve.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()


# Gera a distribuicao de scores separada por adimplentes e inadimplentes
def _plot_score_distribution(y_true, y_prob, figures_dir):
    scores = np.array([CreditScoreHelper.pd_to_score(p) for p in y_prob])
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.hist(scores[y_true == 0], bins=50, alpha=0.6, color="#2ecc71", label="Adimplente", density=True)
    ax.hist(scores[y_true == 1], bins=50, alpha=0.6, color="#e74c3c", label="Inadimplente", density=True)
    ax.axvline(650, color="#f39c12", linestyle="--", linewidth=1.5, label="Threshold APPROVE (650)")
    ax.axvline(400, color="#e67e22", linestyle="--", linewidth=1.5, label="Threshold REVIEW (400)")
    ax.set_xlabel("Credit Score (0-1000)", color="white")
    ax.set_ylabel("Densidade", color="white")
    ax.set_title("Distribuicao de Credit Scores", color="white", fontsize=12)
    ax.legend(facecolor="#0d1117", labelcolor="white", edgecolor="#30363d")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/score_distribution.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()


# Gera o grafico de taxa de aprovacao por faixa etaria (analise de fairness)
def _plot_fairness(fairness_df, figures_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    groups = fairness_df.index.astype(str)
    x = range(len(groups))
    bars = ax.bar(x, fairness_df["approval_rate"], color="#3498db", alpha=0.8, label="Taxa de aprovacao")
    ax.plot(x, fairness_df["default_rate"], "o-", color="#e74c3c", linewidth=2, label="Taxa de inadimplencia real")
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, color="white")
    ax.set_xlabel("Faixa etaria", color="white")
    ax.set_ylabel("Taxa", color="white")
    ax.set_title("Fairness Analysis â€” Aprovacao e Inadimplencia por Faixa Etaria", color="white", fontsize=11)
    ax.legend(facecolor="#0d1117", labelcolor="white", edgecolor="#30363d")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/fairness_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()


# Gera o dashboard consolidado com as 4 visualizacoes principais
def _plot_dashboard(auc, pr_auc, ks, brier, shap_vals, X_val,
                    y_true, y_prob, fairness_df, figures_dir):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Credit Score Model â€” Dashboard", color="white", fontsize=16, y=1.01)

    axes = [fig.add_subplot(2, 2, i+1) for i in range(4)]
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # 1. SHAP importance
    mean_abs = np.abs(shap_vals).mean(axis=0)
    indices  = np.argsort(mean_abs)[::-1][:8]
    names    = [X_val.columns[i] for i in indices]
    axes[0].barh(range(len(names)), mean_abs[indices][::-1], color="#3498db", alpha=0.8)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names[::-1], fontsize=8)
    axes[0].set_title("SHAP Feature Importance")

    # 2. Score distribution
    scores = np.array([int((1 - p) * 1000) for p in y_prob])
    y_arr  = np.array(y_true)
    axes[1].hist(scores[y_arr == 0], bins=40, alpha=0.6, color="#2ecc71", label="Adimplente", density=True)
    axes[1].hist(scores[y_arr == 1], bins=40, alpha=0.6, color="#e74c3c", label="Inadimplente", density=True)
    axes[1].axvline(650, color="#f39c12", linestyle="--", linewidth=1.5)
    axes[1].set_title("Score Distribution")
    axes[1].legend(facecolor="#0d1117", labelcolor="white", edgecolor="#30363d", fontsize=8)

    # 3. Calibration
    fraction, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    axes[2].plot(mean_pred, fraction, "s-", color="#2ecc71", label="Modelo")
    axes[2].plot([0, 1], [0, 1], "--", color="#95a5a6", label="Perfeito")
    axes[2].set_title("Calibration Curve")
    axes[2].legend(facecolor="#0d1117", labelcolor="white", edgecolor="#30363d", fontsize=8)

    # 4. Fairness
    groups = fairness_df.index.astype(str)
    x = range(len(groups))
    axes[3].bar(x, fairness_df["approval_rate"], color="#3498db", alpha=0.7, label="Aprovacao")
    axes[3].plot(x, fairness_df["default_rate"], "o-", color="#e74c3c", linewidth=2, label="Inadimplencia")
    axes[3].set_xticks(list(x))
    axes[3].set_xticklabels(groups, fontsize=8)
    axes[3].set_title("Fairness por Faixa Etaria")
    axes[3].legend(facecolor="#0d1117", labelcolor="white", edgecolor="#30363d", fontsize=8)

    # Metricas no titulo
    metrics_text = f"AUC-ROC: {auc:.4f}  |  PR-AUC: {pr_auc:.4f}  |  KS: {ks:.4f}  |  Brier: {brier:.4f}"
    fig.text(0.5, 0.98, metrics_text, ha="center", color="#95a5a6", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/dashboard.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"    Dashboard salvo em {figures_dir}/dashboard.png")


# Helper para converter PD em score sem importar o dominio no script
class CreditScoreHelper:
    @staticmethod
    def pd_to_score(pd: float) -> int:
        return max(0, min(1000, int((1 - pd) * 1000)))


if __name__ == "__main__":
    main()
