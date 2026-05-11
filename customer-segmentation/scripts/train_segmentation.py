# -*- coding: utf-8 -*-
"""
Script de treinamento do modelo de segmentacao.

Uso:
    python scripts/train_segmentation.py --data data/raw/online_retail.csv

Download do dataset:
    https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
    ou: https://archive.ics.uci.edu/dataset/352/online+retail
"""
import argparse
import sys
from pathlib import Path

# Adiciona src/ ao path para imports funcionarem
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from infrastructure.ml.rfm_builder import RFMBuilder
from infrastructure.models.kmeans_umap_model import KMeansUMAPModel
from infrastructure.repositories.file_segmentation_repository import FileSegmentationRepository


# Ponto de entrada do script de treinamento
def main():
    parser = argparse.ArgumentParser(description="Train customer segmentation model")
    parser.add_argument("--data",       required=True,  help="Path to online_retail.csv")
    parser.add_argument("--models-dir", default="models", help="Where to save artifacts")
    parser.add_argument("--n-clusters", type=int, default=5)
    args = parser.parse_args()

    print(">>> Construindo features RFM...")
    builder   = RFMBuilder()
    customers = builder.build_from_csv(args.data)
    print(f"    {len(customers)} clientes com RFM calculado")

    print(">>> Treinando KMeans + UMAP...")
    repository = FileSegmentationRepository(models_dir=args.models_dir)
    model      = KMeansUMAPModel(repository=repository, n_clusters=args.n_clusters)
    model.fit(customers)

    metrics = model.get_metrics()
    print(f"    Silhouette score : {metrics['silhouette_score']}")
    print(f"    Inertia          : {metrics['inertia']}")
    print(f"    Clusters         : {metrics['n_clusters']}")
    print(f">>> Artefatos salvos em {args.models_dir}/")


if __name__ == "__main__":
    main()
