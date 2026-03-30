# src/weak_supervision_labeling/plotting/gmvae.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .colors import label_palette
from .base import Titles, save_close
from weak_supervision_labeling.helpers import as_1d_labels
from weak_supervision_labeling.gmvae_analysis import dataset_key, n_classes_for_palette, component_majority_stats, soft_purity


def plot_component_purity_soft(
    *,
    qc: np.ndarray,
    y_true: np.ndarray,
    title: str = "GMVAE — component purity and weight",
    savepath: str | Path | None = None,
    dataset: str | None = None,
    min_count_to_show: int = 0,
    sort_by: str = "count",  # "count" | "soft" | "mass"
    max_components: int | None = None,
    annotate_values: bool = True,
    legend_max_items: int = 26,
    titles_plot: bool = True,
):
    qc = np.asarray(qc)
    if qc.ndim != 2:
        raise ValueError(f"qc must be (N,K), got {qc.shape}")
    K = int(qc.shape[1])

    maj, counts, mean_qc, purity_hard = component_majority_stats(qc=qc, y_true=y_true, K=K)
    purity_soft = soft_purity(qc=qc, y_true=y_true, K=K, majority_label=maj)

    keep = np.ones(K, dtype=bool)
    if min_count_to_show > 0:
        keep &= (counts >= int(min_count_to_show))
    idx = np.where(keep)[0]
    if idx.size == 0:
        return

    if sort_by == "count":
        order = np.argsort(-counts[idx])
    elif sort_by == "soft":
        order = np.argsort(-purity_soft[idx])
    elif sort_by == "mass":
        order = np.argsort(-mean_qc[idx])
    else:
        raise ValueError("sort_by must be one of: 'count','soft','mass'")

    idx = idx[order]
    if max_components is not None:
        idx = idx[: int(max_components)]

    M = int(idx.size)
    x = np.arange(M)

    dataset_l = dataset_key(dataset)

    def _lab_display(lab: int) -> str:
        if lab < 0:
            return "?"
        if dataset_l == "emnist" and 0 <= lab <= 25:
            return chr(ord("a") + int(lab))
        return str(int(lab))

    maj_labels = np.array([int(maj[c]) for c in idx], dtype=int)
    comp_ids = [str(int(c)) for c in idx]

    y1 = as_1d_labels(y_true)
    n_classes = n_classes_for_palette(dataset, y1)
    pal = label_palette(n_classes)

    bar_colors = [
        (0.7, 0.7, 0.7, 1.0) if l < 0 else pal[int(l) % len(pal)]
        for l in maj_labels
    ]

    fig_w = min(20.0, max(10.0, 0.45 * M + 6.0))
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))

    bars = ax.bar(x, purity_soft[idx], color=bar_colors)

    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)

    ax.set_ylabel("purity")
    ax.set_xlabel("component id (sorted by purity)")

    ax.set_xticks(x)
    step = max(1, M // 18)
    labels = [cid if (i % step == 0) else "" for i, cid in enumerate(comp_ids)]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    ax_mass = ax.twinx()
    ax_mass.set_ylabel(r"$\mathbb{E}_x[q(c\mid x)]$", fontsize=11)

    mvals = np.asarray(mean_qc[idx], dtype=float)
    mmax = float(np.nanmax(mvals)) if mvals.size > 0 else 1.0
    ax_mass.set_ylim(0.0, max(1e-6, 1.15 * mmax))
    ax_mass.grid(False)

    line_mass, = ax_mass.plot(
        x,
        mvals,
        color="black",
        linewidth=1.8,
        marker="o",
        markersize=4.5,
        label=r"$\mathbb{E}_x[q(c\mid x)]$",
        zorder=6,
    )

    if annotate_values:
        for j, v in enumerate(mvals):
            ax_mass.text(
                j,
                v + 0.02 * max(mmax, 1e-6),
                f"{v:.02f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="black",
                clip_on=False,
            )

    uniq = sorted(u for u in np.unique(maj_labels) if u >= 0)
    if len(uniq) > 0:
        uniq_show = uniq[: int(legend_max_items)]
        handles = [
            Patch(facecolor=pal[u % len(pal)], edgecolor="none", label=_lab_display(int(u)))
            for u in uniq_show
        ]
        handles = handles + [line_mass]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=min(len(handles), 13),
            frameon=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.94),
        )

    if titles_plot and title:
        fig.suptitle(title, fontsize=18, fontweight="bold", family="serif")
        fig.text(
            0.5,
            0.02,
            f"Sorted by: {sort_by} | showing {M}/{K} components | "
            r"bar colors = majority true label | black curve = $E[q(c\mid x)]$",
            ha="center",
            va="bottom",
            fontsize=9,
            alpha=0.9,
        )

    fig.tight_layout(rect=[0, 0.06, 1, 0.86])
    save_close(fig, savepath, dpi=200)


@torch.no_grad()
def plot_generated_by_cluster(
    *,
    model=None,
    decode_fn=None,
    cluster_stats=None,
    n_per_cluster=10,
    device="cuda",
    figsize=None,
    cmap="gray",
    savepath=None,
    use_mu_only=False,
    dpi=300,
    img_shape=(28, 28),
    vmin=0.0,
    vmax=1.0,
    show_label_on_left: bool = True,
    left_fontsize: int = 11,
    left_pad_frac: float = 0.03,
    n_blocks: int = 2,              # 2 columns of clusters
    rows_per_block: int = 10,
    max_clusters: int | None = 26,  # show at most this many clusters
    pick: str = "coverage",         # "first" | "random" | "coverage"
    seed: int = 0,
    label_col: bool = True,
    tight_pad: float = 0.10,
):
    
    titles=Titles(suptitle="Generated samples per component", title="")

    def _row_label(c: int) -> str:
        return f"c={int(c)}"

    # ---- build samplers and decoder ----
    if model is not None:
        model.eval()
        K = int(model.K)
        z_dim = int(model.z_dim)

        def sample_z(c, n):
            mu_c = model.pz_mu[c].to(device)
            logvar_c = model.pz_logvar[c].to(device)
            mu_ = mu_c.unsqueeze(0).expand(n, z_dim)
            if use_mu_only:
                return mu_
            logvar_ = logvar_c.unsqueeze(0).expand(n, z_dim)
            return model.reparam(mu_, logvar_)

        def decode(z):
            return model.decode(z)

    else:
        if decode_fn is None or cluster_stats is None:
            raise ValueError("Empirical mode requires decode_fn and cluster_stats.")
        K = len(cluster_stats)
        z_dim = next(iter(cluster_stats.values()))["mu"].shape[0]

        def sample_z(c, n):
            mu_ = torch.tensor(cluster_stats[c]["mu"], device=device, dtype=torch.float32)
            var_ = torch.tensor(cluster_stats[c]["var"], device=device, dtype=torch.float32)
            std_ = torch.sqrt(var_ + 1e-6)
            mu_ = mu_.unsqueeze(0).expand(n, z_dim)
            std_ = std_.unsqueeze(0).expand(n, z_dim)
            if use_mu_only:
                return mu_
            return mu_ + std_ * torch.randn_like(std_)

        def decode(z):
            return decode_fn(z)

    # ---- choose which clusters to show ----
    n_blocks = int(max(1, n_blocks))
    rows_per_block = int(max(1, rows_per_block))
    capacity = n_blocks * rows_per_block

    m = K if max_clusters is None else int(min(K, max_clusters))
    m = int(min(m, capacity))

    if m <= 0:
        raise ValueError("Nothing to plot: m<=0")

    if pick == "first":
        clusters = np.arange(m, dtype=int)
    elif pick == "random":
        rng = np.random.default_rng(int(seed))
        clusters = rng.choice(K, size=m, replace=False).astype(int)
        clusters.sort()
    elif pick == "coverage":
        clusters = np.unique(np.round(np.linspace(0, K - 1, m)).astype(int))
        if len(clusters) < m:
            missing = [i for i in range(K) if i not in set(clusters)]
            clusters = np.concatenate([clusters, np.array(missing[: (m - len(clusters))], dtype=int)])
        clusters = clusters[:m]
    else:
        raise ValueError("pick must be in {'first','random','coverage'}")

    cols_block = (1 if (label_col and show_label_on_left) else 0) + int(n_per_cluster)
    total_cols = n_blocks * cols_block + (n_blocks - 1) 
    total_rows = rows_per_block

    if figsize is None:
        cell_w = 0.55 
        cell_h = 0.55
        W = max(11.0, cell_w * total_cols)
        H = max(7.0, cell_h * total_rows)
        W = min(W, 26.0)
        H = min(H, 16.0)
        figsize = (W, H)

    fig, axes = plt.subplots(total_rows, total_cols, figsize=figsize, dpi=dpi, squeeze=False)
    for r in range(total_rows):
        for c in range(total_cols):
            axes[r, c].axis("off")

    for i, c_global in enumerate(clusters.tolist()):
        block_id = i // rows_per_block
        r = i % rows_per_block
        if block_id >= n_blocks:
            break

        c0 = block_id * cols_block + block_id

        img_start = c0
        if label_col and show_label_on_left:
            ax_lab = axes[r, c0]
            ax_lab.axis("off")
            ax_lab.text(
                0.98,
                0.50,
                _row_label(c_global),
                ha="right",
                va="center",
                fontsize=left_fontsize,
                fontweight="bold",
                color="black",
                transform=ax_lab.transAxes,
            )
            img_start = c0 + 1

        z = sample_z(c_global, n_per_cluster)
        x_hat = decode(z)

        for j in range(n_per_cluster):
            ax = axes[r, img_start + j]
            img = x_hat[j].view(*img_shape).detach().cpu()
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.axis("off")

        if block_id < (n_blocks - 1):
            sep_col = c0 + cols_block
            axes[r, sep_col].axis("off")

    top = 0.98
    if titles.suptitle:
        fig.suptitle(titles.suptitle, fontsize=18, fontweight="bold")
        top = 0.93
    if titles.title:
        fig.text(0.5, top, titles.title, ha="center", va="top", fontsize=13, fontweight="semibold")
        top -= 0.035

    fig.tight_layout(pad=tight_pad, rect=[left_pad_frac, 0, 1, top])
    save_close(fig, savepath, dpi=dpi)
    return fig



def plot_gain_vs_margin(
    *,
    qc: np.ndarray,
    y_true: np.ndarray,
    gain_per_label: dict[int, float],
    fig_path: str | None = None,
    dataset: str | None = None,
    titles: Titles = Titles(),
    titles_plot: bool = True,
):
    """
    Scatter: per-label soft advantage vs posterior margin.
    """
    qc = np.asarray(qc)
    y_true = np.asarray(y_true)

    # --- margin = q1 - q2
    top2 = np.partition(qc, -2, axis=1)[:, -2:]
    margins = top2[:, 1] - top2[:, 0]

    labels = np.unique(y_true)

    mean_margin = []
    gains = []
    label_names = []

    for l in labels:
        mask = y_true == l
        mean_margin.append(margins[mask].mean())
        gains.append(gain_per_label.get(int(l), 0.0))

        if dataset == "emnist":
            label_names.append(chr(ord("a") + int(l)))
        else:
            label_names.append(str(l))

    mean_margin = np.array(mean_margin)
    gains = np.array(gains)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_margin, gains, s=80)

    for x0, y0, n in zip(mean_margin, gains, label_names):
        ax.text(x0, y0, n)

    ax.axhline(0, color="k", lw=1)

    r = np.corrcoef(mean_margin, gains)[0, 1]

    if titles_plot:
        ax.set_title(titles.title or f"Soft advantage vs posterior margin\nPearson r={r:.2f}")

    ax.set_xlabel("Mean posterior margin  $E[q_{(1)}-q_{(2)}]$")
    ax.set_ylabel(r"$\Delta Acc_\ell = Acc_\ell^{soft} - Acc_\ell^{hard}$")
    ax.grid(alpha=0.3)

    save_close(fig, fig_path)


