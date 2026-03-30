# src/weak_supervision_labeling/plotting/lattent_scatter.py

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import string

from weak_supervision_labeling.embedding import project_2d
from .base import save_close, Titles
from .colors import label_palette


def plot_latent_discrete_labels(
    Z: np.ndarray,
    labels: np.ndarray,
    *,
    method: str = "pca",
    titles: Titles = Titles(),
    legend_title: str | None = None,
    perplexity: int = 30,
    random_state: int = 0,
    savepath=None,
    s: float = 6,
    letters: bool = False,
    label_formatter=None,
    umap_n_neighbors=30,
    umap_min_dist=0.0,
    umap_metric="euclidean",
):
    """
    2D latent scatter with colors = true labels (shared palette across figures).
    """

    Z = np.asarray(Z)
    labels = np.asarray(labels)

    Z2 = project_2d(
        Z,
        method,
        random_state=random_state,
        perplexity=perplexity,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
    )

    if labels.ndim > 1:
        labels = labels.argmax(axis=1)

    uniq = np.unique(labels, return_counts=False)

    # default label formatter
    if label_formatter is None:
        if letters:
            alphabet = list(string.ascii_lowercase)
            label_formatter = lambda y: alphabet[int(y)] if 0 <= int(y) < 26 else str(y)
        else:
            label_formatter = lambda y: str(int(y))

    # palette based on max label
    n_classes = int(np.max(labels) + 1)
    palette = label_palette(n_classes)

    def _color_of(y):
        if y < 0:
            return (0.7, 0.7, 0.7, 1.0)
        return palette[int(y) % len(palette)]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for y in uniq:
        idx = labels == y
        ax.scatter(
            Z2[idx, 0],
            Z2[idx, 1],
            s=s,
            color=_color_of(y),
            alpha=0.65,
            edgecolors="white",
            linewidths=0.25,
            label=label_formatter(y),
            rasterized=True,
        )

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if titles.title:
        ax.set_title(titles.title, fontsize=11)
    if titles.suptitle:
        fig.suptitle(titles.suptitle, fontsize=13)

    if legend_title is not None:
        handles, labels_leg = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels_leg,
            title=legend_title,
            fontsize=8,
            title_fontsize=9,
            loc="best",
            frameon=True,
            framealpha=0.9,
            markerscale=1.6,
        )

    fig.tight_layout()
    save_close(fig, savepath, dpi=300)