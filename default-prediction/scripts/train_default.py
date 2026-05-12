# -*- coding: utf-8 -*-
"""
Script de treinamento do modelo Cox Proportional Hazards.

Uso:
    python scripts/train_default.py --data data/raw/lending_club.csv

Dataset:
    https://www.kaggle.com/datasets/wordsforthewise/lending-club
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter

from infrastructure.ml.lending_club_processor import LendingClubProcessor
from infrastructure.repositories.file_default_repository import FileDefaultRepository


# Ponto de entrada do script de treinamento
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        required=True)
    parser.add_argument("--models-dir",  default="models")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--sample",      type=int, default=200000)
    args = parser.parse_args()

    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    print(">>> Carregando e processando dados...")
    processor = LendingClubProcessor()
    df = processor.load_and_process(args.data, sample_n=args.sample)

    feature_cols = [
        "loan_amnt", "int_rate", "grade_encoded", "emp_length_years",
        "annual_inc", "dti", "fico_range_low", "open_acc",
        "revol_util", "total_acc", "inq_last_6mths", "pub_rec", "term_months",
    ]

    # Split treino/teste mantendo proporcao de eventos
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                          stratify=df["event"])
    print(f"    Treino: {len(train_df)} | Teste: {len(test_df)}")

    # Escala features (Cox e sensivel a escala)
    print(">>> Escalando features...")
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    print(">>> Treinando Cox Proportional Hazards...")
    cox = CoxPHFitter(penalizer=0.1)
    cox.fit(
        train_df[feature_cols + ["duration_months", "event"]],
        duration_col="duration_months",
        event_col="event",
        show_progress=True,
    )

    print()
    cox.print_summary()

    print(">>> Calculando metricas...")
    # C-index no conjunto de teste
    pred_hr = cox.predict_partial_hazard(test_df[feature_cols])
    c_idx = concordance_index(
        test_df["duration_months"],
        -pred_hr,
        test_df["event"]
    )
    print(f"    C-index (teste): {c_idx:.4f}")

    # Brier score em 12 meses
    brier_12m = _brier_score(cox, test_df, feature_cols, t=12)
    print(f"    Brier score 12m: {brier_12m:.4f}")

    print(">>> Gerando graficos...")
    _plot_kaplan_meier(df, args.figures_dir)
    _plot_survival_by_grade(cox, train_df, feature_cols, args.figures_dir)
    _plot_cox_coefficients(cox, args.figures_dir)
    _plot_dashboard(cox, df, test_df, feature_cols, c_idx, brier_12m, args.figures_dir)

    print(">>> Salvando artefatos...")
    metadata = {
        "c_index": round(c_idx, 4),
        "brier_score_12m": round(brier_12m, 4),
        "n_events": int(df["event"].sum()),
        "n_censored": int((df["event"] == 0).sum()),
        "n_train": len(train_df),
        "feature_cols": feature_cols,
        "penalizer": 0.1,
    }
    repo = FileDefaultRepository(models_dir=args.models_dir)
    repo.save_artifacts(cox, scaler, metadata)
    print(f">>> Artefatos salvos em {args.models_dir}/")


# Calcula o Brier score em um tempo t para avaliar calibracao da curva de sobrevivencia
def _brier_score(cox, test_df, feature_cols, t):
    try:
        sf = cox.predict_survival_function(test_df[feature_cols], times=[t])
        s_t = sf.values[0]
        y = (test_df["duration_months"] <= t) & (test_df["event"] == 1)
        return float(np.mean((s_t - (1 - y.values.astype(float))) ** 2))
    except Exception:
        return 0.0


# Gera a curva Kaplan-Meier geral do dataset
def _plot_kaplan_meier(df, figures_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    kmf = KaplanMeierFitter()
    kmf.fit(df["duration_months"], event_observed=df["event"])
    kmf.plot_survival_function(ax=ax, color="#3498db", linewidth=2,
                                label="Todos os emprestimos", ci_show=True)

    ax.set_xlabel("Meses", color="white")
    ax.set_ylabel("Probabilidade de Sobrevivencia", color="white")
    ax.set_title("Curva Kaplan-Meier — Sobrevivencia ate Default", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#0d1117", labelcolor="white", edgecolor="#30363d")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/kaplan_meier.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("    kaplan_meier.png salvo")


# Gera curvas de sobrevivencia Cox separadas por grade do emprestimo
def _plot_survival_by_grade(cox, train_df, feature_cols, figures_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    colors = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c", "#9b59b6", "#1abc9c"]
    grade_names = ["A", "B", "C", "D", "E", "F", "G"]

    median_row = train_df[feature_cols].median().to_frame().T

    for grade_val, (grade_name, color) in enumerate(zip(grade_names, colors), start=1):
        row = median_row.copy()
        row["grade_encoded"] = grade_val
        try:
            sf = cox.predict_survival_function(row)
            ax.plot(sf.index, sf.iloc[:, 0], color=color, linewidth=2,
                    label=f"Grade {grade_name}")
        except Exception:
            continue

    ax.set_xlabel("Meses", color="white")
    ax.set_ylabel("P(Sobrevivencia)", color="white")
    ax.set_title("Curvas de Sobrevivencia Cox por Grade", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#0d1117", labelcolor="white", edgecolor="#30363d", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/survival_by_grade.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("    survival_by_grade.png salvo")


# Gera o grafico de coeficientes Cox (hazard ratios) com intervalos de confianca
def _plot_cox_coefficients(cox, figures_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    summary = cox.summary
    coefs   = summary["coef"].sort_values()
    ci_low  = summary["coef lower 95%"].reindex(coefs.index)
    ci_high = summary["coef upper 95%"].reindex(coefs.index)

    y_pos = range(len(coefs))
    ax.barh(list(y_pos), coefs.values,
            xerr=[coefs.values - ci_low.values, ci_high.values - coefs.values],
            color=["#e74c3c" if v > 0 else "#2ecc71" for v in coefs.values],
            alpha=0.8, capsize=4, error_kw={"color": "white", "linewidth": 1})
    ax.axvline(0, color="white", linestyle="--", linewidth=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(coefs.index.tolist(), color="white", fontsize=9)
    ax.set_xlabel("Coeficiente Cox (log hazard ratio)", color="white")
    ax.set_title("Coeficientes Cox — Fatores de Risco de Default", color="white", fontsize=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/cox_coefficients.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("    cox_coefficients.png salvo")


# Gera o dashboard consolidado com as 4 visualizacoes principais
def _plot_dashboard(cox, df, test_df, feature_cols, c_idx, brier_12m, figures_dir):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Default Prediction — Cox PH Dashboard", color="white", fontsize=16, y=1.01)

    axes = [fig.add_subplot(2, 2, i+1) for i in range(4)]
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # 1. Kaplan-Meier
    kmf = KaplanMeierFitter()
    kmf.fit(df["duration_months"], event_observed=df["event"])
    kmf.plot_survival_function(ax=axes[0], color="#3498db", linewidth=2,
                                label="KM curve", ci_show=False)
    axes[0].set_title("Kaplan-Meier Global")
    axes[0].legend(facecolor="#0d1117", labelcolor="white", edgecolor="#30363d", fontsize=8)

    # 2. Sobrevivencia por grade
    colors_g = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]
    median_row = test_df[feature_cols].median().to_frame().T
    for grade_val, (gname, color) in enumerate(zip(["A","B","C","D","E"], colors_g), start=1):
        row = median_row.copy()
        row["grade_encoded"] = grade_val
        try:
            sf = cox.predict_survival_function(row)
            axes[1].plot(sf.index, sf.iloc[:, 0], color=color, linewidth=1.5, label=f"Grade {gname}")
        except Exception:
            pass
    axes[1].set_title("Sobrevivencia por Grade")
    axes[1].legend(facecolor="#0d1117", labelcolor="white", edgecolor="#30363d", fontsize=7)

    # 3. Coeficientes Cox
    summary = cox.summary
    coefs   = summary["coef"].sort_values()
    y_pos   = range(len(coefs))
    axes[2].barh(list(y_pos), coefs.values,
                 color=["#e74c3c" if v > 0 else "#2ecc71" for v in coefs.values],
                 alpha=0.8)
    axes[2].axvline(0, color="white", linestyle="--", linewidth=1)
    axes[2].set_yticks(list(y_pos))
    axes[2].set_yticklabels(coefs.index.tolist(), fontsize=7)
    axes[2].set_title("Coeficientes Cox")

    # 4. Distribuicao de hazard ratios no conjunto de teste
    hr = cox.predict_partial_hazard(test_df[feature_cols])
    axes[3].hist(hr[test_df["event"] == 0], bins=50, alpha=0.6,
                 color="#2ecc71", label="Censurado", density=True)
    axes[3].hist(hr[test_df["event"] == 1], bins=50, alpha=0.6,
                 color="#e74c3c", label="Default", density=True)
    axes[3].set_title("Distribuicao Hazard Ratio")
    axes[3].legend(facecolor="#0d1117", labelcolor="white", edgecolor="#30363d", fontsize=8)

    metrics_text = f"C-index: {c_idx:.4f}  |  Brier Score 12m: {brier_12m:.4f}  |  Eventos: {int(df['event'].sum())} ({df['event'].mean():.1%})"
    fig.text(0.5, 0.98, metrics_text, ha="center", color="#95a5a6", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/dashboard.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"    Dashboard salvo em {figures_dir}/dashboard.png")


if __name__ == "__main__":
    main()
