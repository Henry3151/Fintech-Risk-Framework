# -*- coding: utf-8 -*-
"""
generate_reports.py
-------------------
Gera graficos de avaliacao dos modelos e loga tudo no MLflow.
Tambem salva os graficos em reports/figures/ para o README.

Uso:
    python src/models/generate_reports.py

Graficos gerados:
    1. Curva Precision-Recall (PR Curve)
    2. Distribuicao do Reconstruction Error (Autoencoder)
    3. Feature Importance (XGBoost top-20)
    4. Matriz de Confusao
    5. Distribuicao de Probabilidade de Fraude
"""

import json
import sys
import joblib
import numpy as np
import torch
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
FIGURES_DIR   = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from models.train_autoencoder import FraudAutoencoder

# ── Paleta do projeto ──────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
BORDER    = "#30363d"
GREEN     = "#3fb950"
RED       = "#f85149"
BLUE      = "#58a6ff"
ORANGE    = "#d29922"
PURPLE    = "#bc8cff"
TEXT      = "#e6edf3"
TEXT_MUTED= "#8b949e"

STYLE = {
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       TEXT_MUTED,
    "ytick.color":       TEXT_MUTED,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  CARD_BG,
    "legend.edgecolor":  BORDER,
    "font.family":       "monospace",
}
plt.rcParams.update(STYLE)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_data():
    X_test  = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")
    return X_test, y_test


def load_autoencoder(input_dim=30):
    meta  = json.load(open(MODELS_DIR / "autoencoder_metadata.json"))
    model = FraudAutoencoder(input_dim=meta["input_dim"])
    model.load_state_dict(torch.load(MODELS_DIR / "autoencoder.pt", map_location="cpu"))
    model.eval()
    return model, meta


def load_classifier():
    clf  = joblib.load(MODELS_DIR / "classifier.joblib")
    meta = json.load(open(MODELS_DIR / "classifier_metadata.json"))
    return clf, meta


def get_reconstruction_errors(X, model):
    with torch.no_grad():
        return model.reconstruction_error(torch.tensor(X)).numpy()


def enrich(X, errors):
    return np.column_stack([X, errors])


# ── Plot 1: Curva PR ──────────────────────────────────────────────────────────

def plot_pr_curve(y_test, y_proba_clf, recon_errors, clf_meta):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Classifier
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba_clf)
    ap_c = average_precision_score(y_test, y_proba_clf)
    ax.plot(rec_c, prec_c, color=BLUE,   lw=2.5, label=f"Autoencoder + XGBoost  PR-AUC={ap_c:.4f}")

    # Autoencoder sozinho
    prec_a, rec_a, _ = precision_recall_curve(y_test, recon_errors)
    ap_a = average_precision_score(y_test, recon_errors)
    ax.plot(rec_a, prec_a, color=PURPLE, lw=2,   linestyle="--",
            label=f"Autoencoder (anomaly) PR-AUC={ap_a:.4f}")

    # Baseline aleatorio
    baseline = y_test.mean()
    ax.axhline(baseline, color=TEXT_MUTED, lw=1.5, linestyle=":", label=f"Baseline aleatorio  PR-AUC={baseline:.4f}")

    # Threshold atual
    thresh = clf_meta["best_threshold"]
    ax.axvline(0.88, color=ORANGE, lw=1, linestyle="--", alpha=0.7, label=f"Threshold calibrado = {thresh:.4f}")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precisao", fontsize=12)
    ax.set_title("Curva Precision-Recall", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True)

    path = FIGURES_DIR / "pr_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ── Plot 2: Distribuicao Reconstruction Error ─────────────────────────────────

def plot_reconstruction_error(y_test, recon_errors, ae_meta):
    fig, ax = plt.subplots(figsize=(9, 5))

    legit  = recon_errors[y_test == 0]
    fraud  = recon_errors[y_test == 1]
    thresh = ae_meta["threshold"]

    bins = np.linspace(0, min(recon_errors.max(), np.percentile(recon_errors, 99.5)), 80)

    ax.hist(legit, bins=bins, color=GREEN, alpha=0.7, label=f"Legitimas  (n={len(legit):,})", density=True)
    ax.hist(fraud, bins=bins, color=RED,   alpha=0.8, label=f"Fraudes    (n={len(fraud):,})", density=True)
    ax.axvline(thresh, color=ORANGE, lw=2.5, linestyle="--",
               label=f"Threshold p95 = {thresh:.4f}")

    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Densidade", fontsize=12)
    ax.set_title("Distribuicao do Erro de Reconstrucao — Autoencoder", fontsize=14,
                 fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(True)

    # Anotacao
    ax.annotate("Fraudes geram\nerro alto →",
                xy=(thresh * 1.8, ax.get_ylim()[1] * 0.6),
                color=RED, fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED),
                xytext=(thresh * 3, ax.get_ylim()[1] * 0.75))

    path = FIGURES_DIR / "reconstruction_error_dist.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ── Plot 3: Feature Importance ────────────────────────────────────────────────

def plot_feature_importance(clf):
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "Recon_Error"]
    importances   = clf.feature_importances_

    # Top 15
    idx  = np.argsort(importances)[::-1][:15]
    top_names = [feature_names[i] for i in idx]
    top_vals  = importances[idx]

    # Cores: Recon_Error em destaque
    colors = [ORANGE if n == "Recon_Error" else BLUE for n in top_names]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(len(top_names)), top_vals[::-1], color=colors[::-1],
                   height=0.7, edgecolor=BORDER)

    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=11)
    ax.set_xlabel("Importancia (gain)", fontsize=12)
    ax.set_title("Top 15 Features — XGBoost", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, axis="x")

    # Legenda manual
    patch_recon = mpatches.Patch(color=ORANGE, label="Recon_Error (anomaly score)")
    patch_pca   = mpatches.Patch(color=BLUE,   label="Features PCA (V1-V28) / Amount / Time")
    ax.legend(handles=[patch_recon, patch_pca], fontsize=9, loc="lower right")

    path = FIGURES_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ── Plot 4: Matriz de Confusao ────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred):
    cm   = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    classes = ["Legitima", "Fraude"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Previsto", fontsize=12)
    ax.set_ylabel("Real",     fontsize=12)
    ax.set_title("Matriz de Confusao", fontsize=14, fontweight="bold", pad=15)

    thresh_color = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = TEXT if cm[i, j] < thresh_color else DARK_BG
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    # Labels nos quadrantes
    labels = [["TN", "FP"], ["FN", "TP"]]
    colors_q = [[GREEN, RED], [RED, GREEN]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.35, labels[i][j], ha="center", va="center",
                    fontsize=9, color=colors_q[i][j], alpha=0.9)

    path = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ── Plot 5: Score Distribution ────────────────────────────────────────────────

def plot_score_distribution(y_test, y_proba, threshold):
    fig, ax = plt.subplots(figsize=(9, 5))

    legit = y_proba[y_test == 0]
    fraud = y_proba[y_test == 1]
    bins  = np.linspace(0, 1, 60)

    ax.hist(legit, bins=bins, color=GREEN, alpha=0.75,
            label=f"Legitimas (n={len(legit):,})", density=True)
    ax.hist(fraud, bins=bins, color=RED,   alpha=0.85,
            label=f"Fraudes   (n={len(fraud):,})", density=True)
    ax.axvline(threshold, color=ORANGE, lw=2.5, linestyle="--",
               label=f"Threshold = {threshold:.4f}")

    ax.set_xlabel("Probabilidade de Fraude", fontsize=12)
    ax.set_ylabel("Densidade", fontsize=12)
    ax.set_title("Distribuicao do Score de Fraude — XGBoost", fontsize=14,
                 fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(True)

    path = FIGURES_DIR / "score_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ── Dashboard (figura combinada para README) ──────────────────────────────────

def plot_dashboard(y_test, y_proba, recon_errors, clf, ae_meta, clf_meta):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Titulo ──
    fig.text(0.5, 0.97,
             "Credit Card Fraud Detection — Model Evaluation Dashboard",
             ha="center", va="top", fontsize=16, fontweight="bold",
             color=TEXT, fontfamily="monospace")
    fig.text(0.5, 0.935,
             f"PR-AUC: {clf_meta['pr_auc']:.4f}  |  Threshold: {clf_meta['best_threshold']:.4f}  |  Dataset: 284,807 transacoes  |  Fraudes: 0.17%",
             ha="center", va="top", fontsize=10, color=TEXT_MUTED, fontfamily="monospace")

    threshold = clf_meta["best_threshold"]
    y_pred    = (y_proba >= threshold).astype(int)

    # ── 1. PR Curve ──
    ax1 = fig.add_subplot(gs[0, 0])
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba)
    ap_c = average_precision_score(y_test, y_proba)
    prec_a, rec_a, _ = precision_recall_curve(y_test, recon_errors)
    ap_a = average_precision_score(y_test, recon_errors)
    ax1.plot(rec_c, prec_c, color=BLUE,   lw=2, label=f"Hibrido  {ap_c:.4f}")
    ax1.plot(rec_a, prec_a, color=PURPLE, lw=1.5, linestyle="--", label=f"AE solo  {ap_a:.4f}")
    ax1.axhline(y_test.mean(), color=TEXT_MUTED, lw=1, linestyle=":", label="Baseline")
    ax1.set_title("Curva PR", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Recall", fontsize=9); ax1.set_ylabel("Precisao", fontsize=9)
    ax1.legend(fontsize=7); ax1.grid(True); ax1.set_xlim([0,1]); ax1.set_ylim([0,1.05])

    # ── 2. Reconstruction Error ──
    ax2 = fig.add_subplot(gs[0, 1])
    legit_e = recon_errors[y_test == 0]
    fraud_e = recon_errors[y_test == 1]
    bins_e  = np.linspace(0, np.percentile(recon_errors, 99), 60)
    ax2.hist(legit_e, bins=bins_e, color=GREEN, alpha=0.7, density=True, label="Legitimas")
    ax2.hist(fraud_e, bins=bins_e, color=RED,   alpha=0.8, density=True, label="Fraudes")
    ax2.axvline(ae_meta["threshold"], color=ORANGE, lw=2, linestyle="--",
                label=f"p95={ae_meta['threshold']:.3f}")
    ax2.set_title("Erro de Reconstrucao", fontsize=11, fontweight="bold")
    ax2.set_xlabel("MSE", fontsize=9); ax2.legend(fontsize=7); ax2.grid(True)

    # ── 3. Feature Importance ──
    ax3 = fig.add_subplot(gs[0, 2])
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "Recon_Error"]
    imp  = clf.feature_importances_
    idx  = np.argsort(imp)[::-1][:10]
    top_n = [feature_names[i] for i in idx]
    top_v = imp[idx]
    colors_fi = [ORANGE if n == "Recon_Error" else BLUE for n in top_n]
    ax3.barh(range(10), top_v[::-1], color=colors_fi[::-1], height=0.65, edgecolor=BORDER)
    ax3.set_yticks(range(10)); ax3.set_yticklabels(top_n[::-1], fontsize=8)
    ax3.set_title("Top 10 Features", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Importancia", fontsize=9); ax3.grid(True, axis="x")

    # ── 4. Score Distribution ──
    ax4 = fig.add_subplot(gs[1, 0])
    legit_s = y_proba[y_test == 0]
    fraud_s = y_proba[y_test == 1]
    bins_s  = np.linspace(0, 1, 50)
    ax4.hist(legit_s, bins=bins_s, color=GREEN, alpha=0.75, density=True, label="Legitimas")
    ax4.hist(fraud_s, bins=bins_s, color=RED,   alpha=0.85, density=True, label="Fraudes")
    ax4.axvline(threshold, color=ORANGE, lw=2, linestyle="--", label=f"thresh={threshold:.2f}")
    ax4.set_title("Score Distribution", fontsize=11, fontweight="bold")
    ax4.set_xlabel("P(fraude)", fontsize=9); ax4.legend(fontsize=7); ax4.grid(True)

    # ── 5. Confusion Matrix ──
    ax5 = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(y_test, y_pred)
    im = ax5.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax5.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                     fontsize=12, fontweight="bold",
                     color=DARK_BG if cm[i,j] > cm.max()/2 else TEXT)
    ax5.set_xticks([0,1]); ax5.set_xticklabels(["Legitima","Fraude"], fontsize=8)
    ax5.set_yticks([0,1]); ax5.set_yticklabels(["Legitima","Fraude"], fontsize=8)
    ax5.set_title("Matriz de Confusao", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Previsto", fontsize=9); ax5.set_ylabel("Real", fontsize=9)

    # ── 6. Metricas Resumo ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    ax6.set_facecolor(CARD_BG)

    from sklearn.metrics import recall_score, precision_score, f1_score
    recall_v    = recall_score(y_test, y_pred)
    precision_v = precision_score(y_test, y_pred)
    f1_v        = f1_score(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = [
        ("PR-AUC",         f"{clf_meta['pr_auc']:.4f}", BLUE),
        ("Recall",         f"{recall_v:.4f}",            GREEN),
        ("Precisao",       f"{precision_v:.4f}",         GREEN),
        ("F1-Score",       f"{f1_v:.4f}",                BLUE),
        ("TP (fraudes OK)",f"{tp:,}",                    GREEN),
        ("FP (falso alarm)",f"{fp:,}",                   ORANGE),
        ("FN (fraude miss)",f"{fn:,}",                   RED),
        ("TN (legit OK)",  f"{tn:,}",                    TEXT_MUTED),
    ]
    ax6.set_title("Metricas Finais", fontsize=11, fontweight="bold", color=TEXT, pad=10)
    for i, (label, value, color) in enumerate(metrics):
        y_pos = 0.88 - i * 0.115
        ax6.text(0.02, y_pos, label, transform=ax6.transAxes,
                 fontsize=9, color=TEXT_MUTED, fontfamily="monospace")
        ax6.text(0.98, y_pos, value, transform=ax6.transAxes,
                 fontsize=10, color=color, fontweight="bold",
                 fontfamily="monospace", ha="right")
        if i < len(metrics) - 1:
            ax6.plot([0.02, 0.98], [y_pos - 0.04, y_pos - 0.04],
         color=BORDER, lw=0.5, transform=ax6.transAxes)

    path = FIGURES_DIR / "dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 55)
    print("  Gerando relatorios e graficos")
    print("=" * 55)

    X_test, y_test = load_data()
    ae_model, ae_meta = load_autoencoder()
    clf, clf_meta     = load_classifier()

    recon_errors = get_reconstruction_errors(X_test, ae_model)
    X_test_enr   = enrich(X_test, recon_errors)
    y_proba      = clf.predict_proba(X_test_enr)[:, 1]
    threshold    = clf_meta["best_threshold"]
    y_pred       = (y_proba >= threshold).astype(int)

    print("\n[1/6] PR Curve ...")
    p1 = plot_pr_curve(y_test, y_proba, recon_errors, clf_meta)

    print("[2/6] Reconstruction Error Distribution ...")
    p2 = plot_reconstruction_error(y_test, recon_errors, ae_meta)

    print("[3/6] Feature Importance ...")
    p3 = plot_feature_importance(clf)

    print("[4/6] Confusion Matrix ...")
    p4 = plot_confusion_matrix(y_test, y_pred)

    print("[5/6] Score Distribution ...")
    p5 = plot_score_distribution(y_test, y_proba, threshold)

    print("[6/6] Dashboard (README) ...")
    p6 = plot_dashboard(y_test, y_proba, recon_errors, clf, ae_meta, clf_meta)

    # ── MLflow ──
    print("\n  Logando no MLflow ...")
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name="model-evaluation"):
        from sklearn.metrics import recall_score, precision_score, f1_score
        mlflow.log_metrics({
            "pr_auc":     clf_meta["pr_auc"],
            "recall":     round(recall_score(y_test, y_pred), 4),
            "precision":  round(precision_score(y_test, y_pred), 4),
            "f1_score":   round(f1_score(y_test, y_pred), 4),
            "threshold":  round(threshold, 4),
            "ae_pr_auc":  ae_meta["pr_auc"],
        })
        for p in [p1, p2, p3, p4, p5, p6]:
            mlflow.log_artifact(str(p), artifact_path="figures")

    print("\n  Graficos salvos em: reports/figures/")
    print("  Abra o MLflow: mlflow ui --port 5000")
    print("=" * 55)


if __name__ == "__main__":
    run()
