# src/weak_supervision_labeling/plotting/generative.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import torch

from .base import Titles


def _label_id_to_emnist_letter(c: int) -> str:
    if 0 <= c < 26:
        return chr(ord("a") + c)
    return "?"


def _is_gmvae_like(model) -> bool:
    return (
        model is not None
        and all(hasattr(model, k) for k in ("pz_mu", "pz_logvar", "K"))
    )


def _reshape_flat_to_img(x_flat: np.ndarray, side: int) -> np.ndarray:
    return x_flat.reshape(side, side)


def _emnist_label_to_index(raw) -> int:
    """
    Accepts:
      - int 0..25
      - np.int
      - str digit "0".."25"
      - str letter "a".."z"
    Returns: 0..25 or 10**9 if unknown.
    """
    if raw is None:
        return 10**9
    if isinstance(raw, (np.integer,)):
        raw = int(raw)
    if isinstance(raw, int):
        return raw if 0 <= raw <= 25 else 10**9
    if isinstance(raw, str):
        s = raw.strip().lower()
        if len(s) == 1 and ("a" <= s <= "z"):
            return ord(s) - ord("a")
        if s.isdigit():
            k = int(s)
            return k if 0 <= k <= 25 else 10**9
    return 10**9


def _format_label_for_display(raw, *, ds: Optional[str]) -> str:
    if raw is None:
        return "?"
    if isinstance(raw, (np.integer,)):
        raw = int(raw)
    if ds == "emnist":
        idx = _emnist_label_to_index(raw)
        if 0 <= idx <= 25:
            return f"{idx} ({_label_id_to_emnist_letter(idx)})"
        if isinstance(raw, str) and len(raw.strip()) == 1:
            s = raw.strip().lower()
            if "a" <= s <= "z":
                return s
    return str(raw)


def _sample_gmvae_fixed_component(
    method,
    c: int,
    n: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample z ~ N(mu_c, diag(var_c)), decode -> X.
    """
    model = getattr(method, "model", None)
    if not _is_gmvae_like(model) or not callable(getattr(model, "decode", None)):
        raise ValueError("GMVAE sampling requires model with pz_mu/pz_logvar/K and decode")

    device = getattr(method, "device", torch.device("cpu"))
    mu_all = model.pz_mu.detach().to(device)          # (K, z_dim)
    logvar_all = model.pz_logvar.detach().to(device)  # (K, z_dim)

    K = int(getattr(model, "K"))
    c = int(c)
    if not (0 <= c < K):
        raise ValueError(f"component id out of range: c={c}, K={K}")

    mu_c = mu_all[c]          # (z_dim,)
    logvar_c = logvar_all[c]  # (z_dim,)
    z_dim = mu_c.shape[0]

    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        eps = torch.randn(n, z_dim, generator=g, device=device)
    else:
        eps = torch.randn(n, z_dim, device=device)

    z = mu_c.unsqueeze(0) + torch.exp(0.5 * logvar_c).unsqueeze(0) * eps

    model.eval()
    with torch.no_grad():
        x = model.decode(z)
    return x.detach().cpu().numpy()


def plot_generated_per_component_grid(
    *,
    method,
    savepath: Path,
    n_per_component: int = 10,
    side: int = 28,
    titles: Titles = Titles(),
    seed: int | None = None,
    sort_rows_by_label: bool = True,
    annotate_left: bool = True,
    rows_per_fig: int = 15,
    cell_w: float = 0.9,
    cell_h: float = 0.6,
    dpi: int = 250,
):
    model = getattr(method, "model", None)
    if not _is_gmvae_like(model):
        raise ValueError("plot_generated_per_component_grid is GMVAE-only (needs pz_mu/pz_logvar/K).")


    cfg = getattr(method, "cfg", {}) or {}
    dataset = cfg.get("dataset", None)
    ds = str(dataset).lower().strip() if dataset is not None else None
    K = int(getattr(model, "K"))

    comp2lab = getattr(method, "component_to_label_", None)
    if not isinstance(comp2lab, dict):
        comp2lab = {}

    # Sorting helpers
    def _label_to_index(raw_lab, ds: str | None) -> int | None:
        if raw_lab is None:
            return None

        if isinstance(raw_lab, str):
            s = raw_lab.strip().lower()
            if s == "" or s == "?":
                return None

            if ds == "emnist":
                if len(s) == 1 and ("a" <= s <= "z"):
                    return ord(s) - ord("a")
                if s.isdigit():
                    v = int(s)
                    if 0 <= v <= 25:
                        return v
                    if 1 <= v <= 26:
                        return v - 1
                return None

            if s.isdigit():
                v = int(s)
                return v if 0 <= v <= 9 else None
            return None

        if isinstance(raw_lab, (int, np.integer)):
            v = int(raw_lab)
            if ds == "emnist":
                if 0 <= v <= 25:
                    return v
                if 1 <= v <= 26:
                    return v - 1
                return None
            return v if 0 <= v <= 9 else None

        return None

    def _comp_sort_key(cid: int):
        """
        Sort by mapped true label index first, then by component id.
        Unknown labels are sent to the end.
        """
        idx = _label_to_index(comp2lab.get(int(cid), None), ds)
        if idx is None:
            return (1, 10**9, int(cid))
        return (0, int(idx), int(cid))

    comps = list(range(K))
    if sort_rows_by_label:
        comps = sorted(comps, key=_comp_sort_key)

    rows_per_fig = int(max(1, rows_per_fig))
    n_parts = int(np.ceil(len(comps) / rows_per_fig))

    for part in range(n_parts):
        start = part * rows_per_fig
        end = min(len(comps), (part + 1) * rows_per_fig)
        comps_part = comps[start:end]

        X_all = []
        for cid in comps_part:
            local_seed = None if seed is None else (int(seed) + 100_000 * int(cid))
            Xc = _sample_gmvae_fixed_component(method, cid, n_per_component, seed=local_seed)
            X_all.append(Xc)
        X_all = np.asarray(X_all)  # (rows, n_per_component, dim)

        nrows = len(comps_part)
        ncols = n_per_component

        fig_w = 3.0 + cell_w * ncols
        fig_h = 1.6 + cell_h * nrows
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fig_w, fig_h),
            squeeze=False
        )

        for r, cid in enumerate(comps_part):
            raw_lab = comp2lab.get(int(cid), None)

            for j in range(ncols):
                ax = axes[r, j]
                ax.axis("off")
                ax.imshow(
                    _reshape_flat_to_img(X_all[r, j], side),
                    cmap="gray",
                    interpolation="nearest"
                )

                if annotate_left and j == 0:
                    if raw_lab is None:
                        txt = f"c={cid}"
                    else:
                        txt = f"c={cid}\ny={_format_label_for_display(raw_lab, ds=ds)}"
                    ax.text(
                        -0.35, 0.5, txt,
                        transform=ax.transAxes,
                        ha="right", va="center",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="0.90", edgecolor="0.65")
                    )

        top = 1.0
        if titles.suptitle:
            fig.suptitle(titles.suptitle, y=0.995)
            top = 0.95
        if titles.title:
            fig.text(0.5, top, titles.title, ha="center", va="top")
            top -= 0.04

        fig.tight_layout(rect=[0.0, 0.0, 1.0, top])

        out = savepath
        if n_parts > 1:
            out_dir = savepath.parent / savepath.stem
            secondary_fig_dir = out_dir / "additional_figures"
            secondary_fig_dir.mkdir(parents=True, exist_ok=True)
            out = secondary_fig_dir / f"generated_examples_from_components/part{part:02d}.png"

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi)
        plt.close(fig)