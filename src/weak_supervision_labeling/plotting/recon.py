# src/weak_supervision_labeling/plotting/recon.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_dataset_reconstructions_grid(
    *,
    X: np.ndarray,
    X_rec: np.ndarray,
    savepath: Path,
    side: int = 28,
    n: int = 9,
    y_true: np.ndarray | None = None,
    dataset: str | None = None, 
    show_y_on_original: bool = True,
    y_prefix: str = "y=",
    label_loc: str = "tl",                # "tl" | "tr" | "bl" | "br"
    label_fontsize: int = 12,
    label_alpha: float = 0.65,            # box alpha
):
    """
    Plot real images vs reconstructions side-by-side.
    """
    assert n == 9, "(3x3)"

    X = np.asarray(X)
    X_rec = np.asarray(X_rec)

    n = int(min(n, len(X), len(X_rec)))
    if n <= 0:
        raise ValueError("Empty X/X_rec")

    if y_true is not None:
        y_true = np.asarray(y_true).reshape(-1)
        if y_true.shape[0] < n:
            raise ValueError(f"y_true must have at least n={n} elements, got {y_true.shape[0]}")

    def _lab_to_txt(y: int) -> str:
        if dataset is None:
            return str(int(y))
        ds = str(dataset).lower()
        if ds == "emnist":
            yi = int(y)
            if 0 <= yi <= 25:
                return chr(ord("a") + yi)
            return str(yi)
        return str(int(y))

    def _label_xy(loc: str):
        loc = (loc or "tl").lower()
        if loc == "tl":
            return (0.03, 0.97, "left", "top")
        if loc == "tr":
            return (0.97, 0.97, "right", "top")
        if loc == "bl":
            return (0.03, 0.03, "left", "bottom")
        if loc == "br":
            return (0.97, 0.03, "right", "bottom")
        return (0.03, 0.97, "left", "top")

    lx, ly, ha, va = _label_xy(label_loc)

    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(
        3, 7,
        width_ratios=[1, 1, 1, 0.25, 1, 1, 1],
        wspace=0.05,
        hspace=0.05,
    )

    # left: original data
    for i in range(n):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(X[i].reshape(side, side), cmap="gray")
        ax.axis("off")

        if show_y_on_original and (y_true is not None):
            txt = f"{y_prefix}{_lab_to_txt(int(y_true[i]))}"
            ax.text(
                lx, ly, txt,
                transform=ax.transAxes,
                ha=ha, va=va,
                fontsize=label_fontsize,
                color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=label_alpha, edgecolor="none"),
            )

    # right: reconstructions
    for i in range(n):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs[r, c + 4])
        ax.imshow(X_rec[i].reshape(side, side), cmap="gray")
        ax.axis("off")

    # titles
    fig.suptitle(
        "encode → decode on real samples",
        fontsize=16,
        y=0.99,
        fontweight="semibold",
    )
    fig.text(0.27, 0.89, "Data (original)", ha="center", fontsize=13)
    fig.text(0.73, 0.89, "Reconstruction", ha="center", fontsize=13)

    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=220, bbox_inches="tight")
    plt.close(fig)
