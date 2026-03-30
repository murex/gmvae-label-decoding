# src/weak_supervision_labeling/plotting/label_map_per_class.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .base import Titles, save_close
from weak_supervision_labeling.per_class import per_label_accuracy, label_names_from_labels, PerLabelDeltaMultiSeedResult, DeltaPerLabelStats, GainEntropyPerLabel


def plot_delta_acc_per_label(
    *,
    y_true_U: np.ndarray,
    y_pred_soft_U: np.ndarray,
    y_pred_hard_U: np.ndarray,
    dataset: str | None,
    savepath: Path,
    titles: Titles = Titles(),
    dpi: int = 220,
    sort_by_gain: bool = True,
):
    labels, acc_soft = per_label_accuracy(y_true_U, y_pred_soft_U)
    labels2, acc_hard = per_label_accuracy(y_true_U, y_pred_hard_U)

    if not np.array_equal(labels, labels2):
        raise ValueError("soft/hard labels mismatch in per_label_accuracy")

    delta = acc_soft - acc_hard

    order = np.argsort(-delta) if sort_by_gain else np.arange(len(labels))
    labels = labels[order]
    delta = delta[order]

    xticks = label_names_from_labels(labels, dataset)

    fig = plt.figure(figsize=(12, 4), dpi=dpi)
    ax = plt.gca()
    ax.bar(np.arange(len(labels)), delta)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(xticks)
    ax.set_ylabel(r"$\Delta$ accuracy (soft − hard)")
    ax.set_xlabel("True label")
    if titles.suptitle:
        fig.suptitle(titles.suptitle)
    if titles.title:
        ax.set_title(titles.title)

    fig.tight_layout()
    save_close(fig, savepath)


def plot_delta_acc_per_label_multi_seed(
    res: PerLabelDeltaMultiSeedResult,
    *,
    dataset: str | None,
    savepath: Path,
    titles: Titles = Titles(),
    dpi: int = 220,
    sort_by_gain: bool = True,
    show_error: bool = True,
    ci_level: float = 0.95,
):
    labels = np.asarray(res.labels, dtype=int)
    delta = np.asarray(res.delta, dtype=float)  # shape: (n_seeds, n_labels)

    if delta.ndim != 2:
        raise ValueError(f"res.delta must be 2D (n_seeds, n_labels), got {delta.shape}")
    if delta.shape[1] != len(labels):
        raise ValueError(
            f"res.delta has {delta.shape[1]} labels but res.labels has length {len(labels)}"
        )

    n_seeds = delta.shape[0]
    mean = np.nanmean(delta, axis=0)

    if n_seeds >= 2:
        std = np.nanstd(delta, axis=0, ddof=1)
        se = std / np.sqrt(n_seeds)

        alpha = 1.0 - float(ci_level)
        try:
            from scipy.stats import t
            tcrit = float(t.ppf(1.0 - alpha / 2.0, df=n_seeds - 1))
        except Exception:
            tcrit = 1.96 if abs(ci_level - 0.95) < 1e-12 else 1.96

        err = tcrit * se
    else:
        err = np.zeros_like(mean)

    order = np.argsort(-mean) if sort_by_gain else np.arange(len(labels))
    labels = labels[order]
    mean = mean[order]
    err = err[order]

    xticks = label_names_from_labels(labels, dataset)

    fig = plt.figure(figsize=(12, 4), dpi=dpi)
    ax = plt.gca()

    x = np.arange(len(labels))
    ax.bar(x, mean)

    if show_error and n_seeds >= 2:
        ax.errorbar(x, mean, yerr=err, fmt="none", capsize=2)

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.set_ylabel(r"$\Delta$ accuracy (soft − hard)")
    ax.set_xlabel("True label")

    if titles.suptitle:
        fig.suptitle(titles.suptitle)
    if titles.title:
        ax.set_title(titles.title)

    fig.tight_layout()
    save_close(fig, savepath)
    return fig


def plot_delta_acc_per_label_meanstd(
    *,
    labels: np.ndarray,                 # (C,)
    delta_mean: np.ndarray,             # (C,)
    delta_std: np.ndarray,              # (C,)
    dataset: str | None,
    savepath: Path,
    titles: Titles = Titles(),
    dpi: int = 240,
    keep: str = "robust_gain",          # "all" | "gain" | "robust_gain"
    robust_k: float = 1.0,              # robust if mean - k*std > 0
    top_k: int = 18,                    # show top_k after filtering
):
    labels = np.asarray(labels).astype(int).reshape(-1)
    mu = np.asarray(delta_mean, dtype=float).reshape(-1)
    sd = np.asarray(delta_std, dtype=float).reshape(-1)

    eps = 1e-12
    snr = mu / (sd + eps)

    ok = np.isfinite(mu) & np.isfinite(sd)
    labels, mu, sd, snr = labels[ok], mu[ok], sd[ok], snr[ok]

    if keep == "gain":
        mask = mu > 0
        labels, mu, sd, snr = labels[mask], mu[mask], sd[mask], snr[mask]
    elif keep == "robust_gain":
        mask = (mu - robust_k * sd) > 0
        labels, mu, sd, snr = labels[mask], mu[mask], sd[mask], snr[mask]

    # sort by mean gain
    order = np.argsort(-mu)
    labels, mu, sd, snr = labels[order], mu[order], sd[order], snr[order]

    if top_k is not None and len(labels) > int(top_k):
        labels, mu, sd, snr = labels[:top_k], mu[:top_k], sd[:top_k], snr[:top_k]

    xticks = label_names_from_labels(labels, dataset)

    fig = plt.figure(figsize=(12.5, 4.5), dpi=dpi)
    ax = plt.gca()

    x = np.arange(len(labels))
    ax.bar(x, mu)
    ax.errorbar(x, mu, yerr=sd, fmt="none", capsize=3, linewidth=1.2)

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.set_ylabel(r"$\Delta$ accuracy (soft − hard)")
    ax.set_xlabel("True label")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    if titles.suptitle:
        fig.suptitle(titles.suptitle)
    if titles.title:
        ax.set_title(titles.title)

    # annotate SNR lightly
    for i, (m, s) in enumerate(zip(mu, snr)):
        ax.annotate(f"{s:.1f}×", (i, m), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, alpha=0.8)

    fig.tight_layout()
    save_close(fig, savepath)
    return DeltaPerLabelStats(labels=labels, delta_mean=mu, delta_std=sd, snr=snr)


def plot_gain_vs_std(
    *, labels, delta_mean, delta_std, dataset, savepath: Path, titles=Titles(), dpi=240
):
    labels = np.asarray(labels).astype(int).reshape(-1)
    mu = np.asarray(delta_mean, float).reshape(-1)
    sd = np.asarray(delta_std, float).reshape(-1)

    ok = np.isfinite(mu) & np.isfinite(sd)
    labels, mu, sd = labels[ok], mu[ok], sd[ok]

    fig = plt.figure(figsize=(7.2, 5.2), dpi=dpi)
    ax = plt.gca()

    ax.scatter(sd, mu)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("std over seeds")
    ax.set_ylabel("mean gain (soft − hard)")
    ax.grid(True, linestyle="--", alpha=0.25)

    # label points
    names = label_names_from_labels(labels, dataset)
    for x, y, t in zip(sd, mu, names):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(4, 2), fontsize=8, alpha=0.9)

    if titles.suptitle:
        fig.suptitle(titles.suptitle)
    if titles.title:
        ax.set_title(titles.title)

    fig.tight_layout()
    save_close(fig, savepath)






















def plot_gain_vs_entropy_multi_seed(
    res: GainEntropyPerLabel,
    *,
    dataset: str | None,
    savepath: Path,
    titles: Titles = Titles(),
    dpi: int = 220,
    show_error_x: bool = True,
    show_error_y: bool = True,
    robust_k: float | None = None,   # if set, keep points with gain_mean - k*gain_std > 0
    annotate: bool = True,
):
    labels = res.labels
    x = res.H_mean
    y = res.gain_mean
    xerr = res.H_std
    yerr = res.gain_std

    keep = np.isfinite(x) & np.isfinite(y)
    if robust_k is not None:
        keep = keep & ((y - float(robust_k) * yerr) > 0.0)

    labels_k = labels[keep]
    x_k = x[keep]
    y_k = y[keep]
    xerr_k = xerr[keep]
    yerr_k = yerr[keep]

    fig = plt.figure(figsize=(10, 6), dpi=dpi)
    ax = plt.gca()

    # scatter
    ax.scatter(x_k, y_k)

    # error bars
    if show_error_x or show_error_y:
        ax.errorbar(
            x_k,
            y_k,
            xerr=xerr_k if show_error_x else None,
            yerr=yerr_k if show_error_y else None,
            fmt="none",
            capsize=2,
            linewidth=1,
        )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(r"mean entropy on U:  $\mathbb{E}[H(q(c\mid x)) \mid y=\ell]$")
    ax.set_ylabel(r"mean gain on U:  $\Delta$ accuracy (soft − hard)")
    ax.set_title(titles.title or "Per-label gain vs entropy")
    if titles.suptitle:
        fig.suptitle(titles.suptitle)

    if annotate:
        names = label_names_from_labels(labels_k, dataset)
        for xi, yi, name in zip(x_k, y_k, names):
            ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(4, 2))

    fig.tight_layout()
    save_close(fig, savepath)