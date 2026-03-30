# src/weak_supervision_labeling/plotting/colors.py

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def distinct_colors(n: int, base_cmap: str = "tab20"):
    """
    Return a list of n visually distinct RGBA colors.
    """
    if n <= 0:
        return []

    cat = plt.get_cmap(base_cmap)
    if hasattr(cat, "N") and n <= cat.N:
        xs = np.linspace(0, 1, n, endpoint=False)
        return [cat(x) for x in xs]

    hsv = plt.get_cmap("hsv")
    xs = np.linspace(0, 1, n, endpoint=False)
    return [hsv(x) for x in xs]


def label_palette(n_classes: int):
    """
    Return a palette with >= n_classes distinct colors.
    """
    cmaps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    colors = []
    for cmap in cmaps:
        colors.extend([cmap(i) for i in range(cmap.N)])
    if n_classes <= len(colors):
        return colors[:n_classes]
    return [colors[i % len(colors)] for i in range(n_classes)]