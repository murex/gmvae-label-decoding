# src/weak_supervision_labeling/embedding.py

from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def project_2d(
    Z: np.ndarray,
    method: str,
    *,
    random_state: int = 0,
    perplexity: int = 30,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.0,
    umap_metric: str = "euclidean",
) -> np.ndarray:
    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError(f"Z must be (N,D), got {Z.shape}")

    if Z.shape[1] == 2 and method == "pca":
        return Z

    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(Z)

    if method == "tsne":
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(Z)

    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError("UMAP requested but umap-learn not installed.") from e
        return umap.UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=random_state,
        ).fit_transform(Z)

    raise ValueError("method must be 'pca', 'tsne' or 'umap'")
