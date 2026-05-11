# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
import umap

from domain.entities.customer import Customer, CustomerSegment
from domain.exceptions import SegmentationError, ModelNotLoadedError
from domain.interfaces.segmentation_interfaces import ISegmentationModel, ISegmentationRepository

# Mapeamento fixo de cluster_id para label de negocio baseado no perfil RFM medio
SEGMENT_LABELS: Dict[int, str] = {
    0: "Champions",        # alta frequencia, alto valor, compra recente
    1: "Loyal Customers",  # frequencia alta, valor medio
    2: "At Risk",          # compraram bastante mas nao voltam ha tempo
    3: "New Customers",    # compraram recentemente mas pouco
    4: "Lost",             # baixa frequencia, alto recency, baixo valor
}


class KMeansUMAPModel(ISegmentationModel):
    """
    Pipeline: RobustScaler -> KMeans(n=5) -> UMAP(2D).
    RobustScaler e preferido ao StandardScaler pois RFM tem outliers severos.
    UMAP e usado apenas para visualizacao — o clustering e feito no espaco RFM escalado.
    """

    # Inicializa o modelo com numero de clusters e parametros UMAP configuráveis
    def __init__(
        self,
        repository: ISegmentationRepository,
        n_clusters: int = 5,
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self._repository   = repository
        self._n_clusters   = n_clusters
        self._umap_neighbors = umap_neighbors
        self._umap_min_dist  = umap_min_dist
        self._random_state   = random_state
        self._scaler   = None
        self._kmeans   = None
        self._umap     = None
        self._metrics  = {}
        self._loaded   = False

    # Treina RobustScaler, KMeans e UMAP na lista de clientes e salva os artefatos
    def fit(self, customers: List[Customer]) -> None:
        try:
            X = np.vstack([c.to_numpy() for c in customers])

            # Escala RFM com RobustScaler (resistente a outliers de valor monetario)
            self._scaler = RobustScaler()
            X_scaled = self._scaler.fit_transform(X)

            # Treina KMeans com 5 clusters e semente fixa para reproducibilidade
            self._kmeans = KMeans(
                n_clusters=self._n_clusters,
                random_state=self._random_state,
                n_init=10,
            )
            labels = self._kmeans.fit_predict(X_scaled)

            # Treina UMAP para reducao 2D usada apenas na visualizacao
            self._umap = umap.UMAP(
                n_components=2,
                n_neighbors=self._umap_neighbors,
                min_dist=self._umap_min_dist,
                random_state=self._random_state,
            )
            self._umap.fit(X_scaled)

            # Calcula metricas de qualidade do clustering
            sil = float(silhouette_score(X_scaled, labels))
            self._metrics = {
                "n_clusters": self._n_clusters,
                "silhouette_score": round(sil, 4),
                "inertia": round(float(self._kmeans.inertia_), 2),
                "n_customers": len(customers),
            }

            # Reordena os labels para que o cluster_id 0 seja sempre Champions
            self._remap_labels(X_scaled, labels)

            # Persiste todos os artefatos no repositorio
            self._repository.save_artifacts(
                scaler=self._scaler,
                kmeans=self._kmeans,
                umap_model=self._umap,
                metadata=self._metrics,
            )
            self._loaded = True

        except Exception as exc:
            raise SegmentationError(f"Fit failed: {exc}") from exc

    # Carrega os artefatos do repositorio se ainda nao foram carregados
    def _ensure_loaded(self) -> None:
        if not self._loaded:
            try:
                self._scaler = self._repository.load_scaler()
                self._kmeans = self._repository.load_kmeans()
                self._umap   = self._repository.load_umap()
                self._metrics = self._repository.get_metadata()
                self._loaded = True
            except Exception as exc:
                raise ModelNotLoadedError(f"Could not load artifacts: {exc}") from exc

    # Segmenta uma lista de clientes retornando CustomerSegment com label e coordenadas UMAP
    def predict(self, customers: List[Customer]) -> List[CustomerSegment]:
        self._ensure_loaded()
        try:
            X = np.vstack([c.to_numpy() for c in customers])
            X_scaled     = self._scaler.transform(X)
            cluster_ids  = self._kmeans.predict(X_scaled)
            umap_coords  = self._umap.transform(X_scaled)
            rfm_scores   = self._compute_rfm_scores(X_scaled)
        except Exception as exc:
            raise SegmentationError(f"Predict failed: {exc}") from exc

        return [
            CustomerSegment(
                customer_id=c.customer_id,
                cluster_id=int(cid),
                segment_label=SEGMENT_LABELS.get(int(cid), f"Cluster {cid}"),
                umap_x=float(umap_coords[i, 0]),
                umap_y=float(umap_coords[i, 1]),
                rfm_score=float(rfm_scores[i]),
            )
            for i, (c, cid) in enumerate(zip(customers, cluster_ids))
        ]

    # Retorna as metricas calculadas durante o treinamento
    def get_metrics(self) -> dict:
        self._ensure_loaded()
        return self._metrics

    # Remapeia cluster IDs para que Champions (maior valor medio) seja sempre o cluster 0
    def _remap_labels(self, X_scaled: np.ndarray, labels: np.ndarray) -> None:
        centers = self._kmeans.cluster_centers_
        # Ordena por monetary (indice 2) decrescente para definir hierarquia de valor
        order = np.argsort(-centers[:, 2])
        mapping = {old: new for new, old in enumerate(order)}
        new_labels = np.array([mapping[l] for l in labels])
        # Recalcula os centros na nova ordem
        self._kmeans.labels_ = new_labels

    # Calcula um score RFM composto normalizado entre 0 e 1 para cada cliente
    def _compute_rfm_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        # Recency invertida (menor = melhor), Frequency e Monetary positivos
        r = 1 - (X_scaled[:, 0] - X_scaled[:, 0].min()) / (X_scaled[:, 0].ptp() + 1e-9)
        f = (X_scaled[:, 1] - X_scaled[:, 1].min()) / (X_scaled[:, 1].ptp() + 1e-9)
        m = (X_scaled[:, 2] - X_scaled[:, 2].min()) / (X_scaled[:, 2].ptp() + 1e-9)
        return (r * 0.3 + f * 0.3 + m * 0.4)
