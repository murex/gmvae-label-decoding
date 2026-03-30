# src/weak_supervision_labeling/plotting/label_map_sweep.py

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullLocator, LogLocator
from weak_supervision_labeling.io import ensure_dir
from weak_supervision_labeling.sweeps import run_label_map_sweep_on_U_multi_seed


def plot_label_map_sweep_on_U_multi_seed(
    method,
    X: np.ndarray,
    y: np.ndarray,
    fracs: Sequence[float],
    seeds: Sequence[int],
    stratified: bool,
    supervised_baselines: Sequence[str],
    classes: np.ndarray | None = None,
    verbose: bool = True,
    *,
    savepath: str | Path | None = None,
    title: str | None = None,
    dpi: int = 240,
    close: bool = True,
    show_band: bool = True,
    show_baselines: bool = True,
    n_total: int | None = None,
    k_reference: int | None = None,
    show_top_axis_counts: bool = True,
    top_axis_every: int = 1,
    show_target_line: bool = True,
    target_y: float = 0.80,
    annotate_k_line: bool = True,
    legend_loc: str = "upper left",
    ci_level: float = 0.95,
):
    """
    Plot accuracy on unlabeled data versus labeled fraction (log-x), aggregated over seeds.
    """

    res = run_label_map_sweep_on_U_multi_seed(
        method=method,
        X=X,
        y=y,
        fracs=fracs,
        seeds=seeds,
        stratified=stratified,
        supervised_baselines=supervised_baselines,
        classes=classes,
        verbose=verbose,
        )

    fracs = np.asarray(res.fracs, dtype=float)
    x_pct = fracs * 100.0
    n_seeds = len(res.seeds)

    # Raw per-seed arrays: shape expected = (n_seeds, n_fracs)
    acc_soft = np.asarray(res.acc_soft_U, dtype=float)
    acc_hard = np.asarray(res.acc_hard_U, dtype=float)

    if acc_soft.ndim != 2 or acc_soft.shape[1] != len(fracs):
        raise ValueError(f"res.acc_soft_U must be (n_seeds, n_fracs), got {acc_soft.shape}")
    if acc_hard.ndim != 2 or acc_hard.shape[1] != len(fracs):
        raise ValueError(f"res.acc_hard_U must be (n_seeds, n_fracs), got {acc_hard.shape}")

    def _mean_and_ci_band(a: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(a, dtype=float)
        if a.ndim != 2:
            raise ValueError(f"Expected 2D array (n_seeds, n_points), got {a.shape}")

        mean = np.nanmean(a, axis=0)

        n = a.shape[0]
        if n <= 1:
            return mean, np.zeros_like(mean)

        # sample std across seeds
        std = np.nanstd(a, axis=0, ddof=1)
        se = std / np.sqrt(n)

        alpha = 1.0 - float(level)

        try:
            from scipy.stats import t
            tcrit = float(t.ppf(1.0 - alpha / 2.0, df=n - 1))
        except Exception:
            # fallback normal approx if scipy is unavailable
            tcrit = 1.96 if abs(level - 0.95) < 1e-12 else 1.96

        half = tcrit * se
        return mean, half

    m_soft, ci_soft = _mean_and_ci_band(acc_soft, level=ci_level)
    m_hard, ci_hard = _mean_and_ci_band(acc_hard, level=ci_level)

    fig, ax = plt.subplots(figsize=(10.0, 5.6), dpi=dpi)

    line_soft, = ax.plot(
        x_pct,
        m_soft,
        marker="o",
        markersize=7.8,
        linewidth=2.8,
        linestyle="-",
        label="Soft decoding (GMVAE)",
        zorder=5,
    )

    line_hard, = ax.plot(
        x_pct,
        m_hard,
        marker="o",
        markersize=7.8,
        linewidth=2.8,
        linestyle="-",
        label="Hard decoding (GMVAE)",
        zorder=4,
    )

    if show_band and n_seeds >= 2:
        ax.fill_between(
            x_pct,
            np.clip(m_soft - ci_soft, 0.0, 1.0),
            np.clip(m_soft + ci_soft, 0.0, 1.0),
            color=line_soft.get_color(),
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )

        ax.fill_between(
            x_pct,
            np.clip(m_hard - ci_hard, 0.0, 1.0),
            np.clip(m_hard + ci_hard, 0.0, 1.0),
            color=line_hard.get_color(),
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )

    pretty_baseline_names = {
        "logreg": "Logistic regression",
        "mlp": "MLP",
        "xgboost": "XGBoost",
    }

    baseline_order = ["logreg", "mlp", "xgboost"]
    baseline_names = list(res.acc_baselines_U.keys())
    ordered_names = [b for b in baseline_order if b in baseline_names]
    ordered_names += [b for b in baseline_names if b not in ordered_names]

    if show_baselines:
        for name in ordered_names:
            acc_base = np.asarray(res.acc_baselines_U[name], dtype=float)
            if acc_base.ndim != 2 or acc_base.shape[1] != len(fracs):
                raise ValueError(
                    f"res.acc_baselines_unlabeled['{name}'] must be (n_seeds, n_fracs), got {acc_base.shape}"
                )

            m_base, ci_base = _mean_and_ci_band(acc_base, level=ci_level)
            label = pretty_baseline_names.get(name, name)

            line_base, = ax.plot(
                x_pct,
                m_base,
                marker="s",
                markersize=6.6,
                linewidth=2.2,
                linestyle="--",
                label=label,
                zorder=3,
            )

            if show_band and n_seeds >= 2:
                ax.fill_between(
                    x_pct,
                    np.clip(m_base - ci_base, 0.0, 1.0),
                    np.clip(m_base + ci_base, 0.0, 1.0),
                    color=line_base.get_color(),
                    alpha=0.10,
                    linewidth=0,
                    zorder=1,
                )

    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)

    ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.28)
    ax.grid(False, which="minor", axis="both")

    ax.set_xlabel("Labeled data (%) — log scale")
    ax.set_ylabel("Accuracy on unlabeled data")

    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10) * 0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}%"))

    ax.tick_params(axis="x", which="major", length=6)
    ax.tick_params(axis="x", which="minor", length=3)

    if show_top_axis_counts and (n_total is not None):
        ax_top = ax.twiny()
        ax_top.set_xscale("log")
        ax_top.set_xlim(ax.get_xlim())

        idx = np.arange(len(x_pct))[::max(1, int(top_axis_every))]
        top_ticks = x_pct[idx]
        top_counts = np.rint(fracs[idx] * n_total).astype(int)

        ax_top.set_xticks(top_ticks)
        ax_top.set_xticklabels([f"{n:,}".replace(",", "\u202f") for n in top_counts])
        ax_top.set_xlabel("Number of labeled examples")
        ax_top.xaxis.set_minor_locator(NullLocator())
        ax_top.tick_params(axis="x", pad=4)

    if show_target_line:
        ax.axhline(
            target_y,
            color="0.45",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            zorder=0,
        )

        ax.text(
            1.01,
            target_y,
            f"{int(round(100 * target_y))}%",
            transform=ax.get_yaxis_transform(),
            color="0.35",
            fontsize=10,
            va="center",
            ha="left",
        )

    if (n_total is not None) and (k_reference is not None) and (k_reference > 0):
        x_k_pct = 100.0 * float(k_reference) / float(n_total)
        xmin, xmax = float(np.min(x_pct)), float(np.max(x_pct))

        if xmin <= x_k_pct <= xmax:
            ax.axvline(
                x_k_pct,
                color="0.35",
                linestyle=":",
                linewidth=2.0,
                alpha=0.95,
                zorder=0,
            )

            if annotate_k_line:
                ax.text(
                    x_k_pct,
                    0.03,
                    rf"$n_{{\mathrm{{lab}}}} = K = {k_reference}$",
                    rotation=90,
                    va="bottom",
                    ha="right",
                    color="0.30",
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        fc="white",
                        ec="none",
                        alpha=0.75,
                    ),
                )

    if title is None:
        ci_pct = int(round(100 * ci_level))
        title = f"Accuracy on unlabeled data vs. labeled fraction\n({n_seeds} seeds, mean ± {ci_pct}% CI)"
    ax.set_title(title)

    try:
        leg = ax.legend(loc=legend_loc, frameon=True, framealpha=0.92)
        handles = getattr(leg, "legendHandles", None)
        if handles is None:
            handles = getattr(leg, "legend_handles", None)
        if handles is not None:
            for h in handles:
                try:
                    h.set_markersize(7.5)
                except Exception:
                    pass
    except Exception:
        ax.legend(loc=legend_loc, frameon=True, framealpha=0.92)

    fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        ensure_dir(savepath.parent)
        fig.savefig(savepath, bbox_inches="tight")

    if close:
        plt.close(fig)

    return fig