# src/weak_supervision_labeling/plotting/pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from weak_supervision_labeling.io import ensure_dir
from weak_supervision_labeling.weak_supervision import (
    build_M_hard,
    build_M_soft,
    decode_hard,
    decode_soft,
    split_weak_supervision_indices,
)
from weak_supervision_labeling.per_class import delta_acc_per_label

from .generative import plot_generated_per_component_grid
from weak_supervision_labeling.gmvae_block import run_gmvae_block
from .label_map_per_class import (
    PerLabelDeltaMultiSeedResult,
    plot_delta_acc_per_label_multi_seed,
)

from .label_map_sweep import plot_label_map_sweep_on_U_multi_seed

from .latent_scatter import plot_latent_discrete_labels
from .recon import plot_dataset_reconstructions_grid
from .base import Titles
from .example import plot_single_soft_hard_example

from weak_supervision_labeling.helpers import as_1d_labels

from .label_map_per_class import plot_gain_vs_std







# -------------------------
# Config
# -------------------------
@dataclass
class PlotConfig:
    seed: int = 0
    titles_plot: bool = True
    skip_umap: bool = False
    skip_tsne: bool = False

    n_embed_plot: int = 10_000
    n_tsne: int = 1_000
    n_qc: int = 5_000

    label_map_frac_eval: float | None = None
    label_map_stratified: bool = True
    label_map_fracs: list[float] | None = None
    supervised_baselines: Sequence[str] | None = ("logreg",)

    do_label_map_delta_per_class_on_U: bool = True
    label_map_seeds: list[int] | None = None
    label_map_n_seeds: int = 5
    label_map_fragmentation_unit: str = "bits"

    do_label_map_gain_vs_fragmentation_on_U: bool = True
    do_label_map_soft_hard_examples_on_U: bool = True
    label_map_examples_maxN: int = 5000
    label_map_example_minentropy_bits: float = 1.0

    mode_name: str | None = None
    labels_ref: np.ndarray | None = None


# -------------------------
# Runner
# -------------------------
class PlotRunner:
    def __init__(self, dataset: str):
        self.dataset = dataset

    def _ttl(self, cfg: PlotConfig, title: str, suptitle: str | None = None) -> Titles:
        if not bool(getattr(cfg, "titles_plot", True)):
            return Titles()
        return Titles(title=title or "", suptitle=suptitle or "")

    def _maybe_title_str(self, cfg: PlotConfig, title: str | None) -> str:
        if not bool(getattr(cfg, "titles_plot", True)):
            return ""
        return str(title or "")

    def _load_or_embed(self, method, ckpt_dir: Path, X: np.ndarray):
        z_path = ckpt_dir / "Z_train.npy"
        if z_path.exists():
            try:
                return np.load(z_path)
            except Exception as e:
                print(f"  could not load cached Z_train.npy: {e}")
        if callable(getattr(method, "embed", None)):
            return method.embed(X)
        return None

    def _class_names_for_dataset(self, y: np.ndarray) -> list[str] | None:
        if self.dataset != "emnist":
            return None
        y = np.asarray(y).reshape(-1)
        classes = np.unique(y)
        if len(classes) == 26 and classes.min() == 0 and classes.max() == 25:
            return [chr(ord("a") + i) for i in range(26)]
        return None

    def _subsample_indices(self, n_total: int, n_wanted: int, *, rng: np.random.Generator) -> np.ndarray:
        n = int(min(max(n_wanted, 0), n_total))
        if n <= 0:
            return np.zeros((0,), dtype=int)
        if n == n_total:
            return np.arange(n_total, dtype=int)
        return rng.choice(n_total, size=n, replace=False).astype(int)

    def _pct_str(self, n_used: int, n_total: int) -> str:
        if n_total <= 0:
            return "0%"
        pct = 100.0 * float(n_used) / float(n_total)
        return f"{pct:.3f}%" if pct < 1.0 else f"{pct:.2f}%"

    @staticmethod
    def _sanitize_fracs(fracs) -> list[float]:
        if fracs is None:
            return []
        out: list[float] = []
        for f in fracs:
            try:
                f = float(f)
            except Exception:
                continue
            if 0.0 < f < 1.0:
                out.append(f)
        return sorted(set(out))

    @staticmethod
    def _label_map_eval_seeds(cfg: PlotConfig) -> list[int]:
        seeds = getattr(cfg, "label_map_seeds", None)
        if seeds is None:
            n = int(getattr(cfg, "label_map_n_seeds", 5))
            base = int(getattr(cfg, "seed", 0))
            seeds = [base + i for i in range(max(1, n))]
        return [int(s) for s in seeds]




    def run(
        self,
        *,
        method,
        method_dir: Path,
        ckpt_dir: Path,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
        cfg: PlotConfig,
        decoding_cache: dict | None = None,
    ):
        out_dir = ensure_dir(method_dir)
        secondary_fig_dir = out_dir / "additional_figures"
        secondary_fig_dir.mkdir(parents=True, exist_ok=True)

        print_labels = self.dataset == "emnist"
        mode_name = str(getattr(cfg, "mode_name", "") or "").lower().strip()
        rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))

        X = np.asarray(X)
        y = np.asarray(y)

        try:
            Ztr = self._load_or_embed(method, ckpt_dir, X)
            if Ztr is not None:
                Z = Ztr[: cfg.n_embed_plot]
                y_plot = y[: len(Z)]

                plot_latent_discrete_labels(
                    Z,
                    y_plot,
                    method="pca",
                    legend_title="True y",
                    titles=self._ttl(cfg, "Latent Space (PCA)", getattr(method, "plot_suptitle_pca", None)),
                    savepath=secondary_fig_dir / "latent_space_pca.png",
                    letters=print_labels,
                    random_state=cfg.seed,
                )

                if (not cfg.skip_umap) and (getattr(method, "name", "") not in {"hdbscan"}):
                    plot_latent_discrete_labels(
                        Z,
                        y_plot,
                        method="umap",
                        legend_title="True y",
                        titles=self._ttl(cfg, "Latent Space (UMAP)", "UMAP projection colored by true $y$"),
                        savepath=secondary_fig_dir / "latent_space_umap.png",
                        letters=print_labels,
                        random_state=cfg.seed,
                        umap_n_neighbors=getattr(method, "umap_n_neighbors", 30),
                        umap_min_dist=getattr(method, "umap_min_dist", 0.0),
                        umap_metric=getattr(method, "umap_metric", "euclidean"),
                    )

                if not cfg.skip_tsne:
                    n_sub = min(cfg.n_tsne, len(Z))
                    Zs, ys = Z[:n_sub], y_plot[:n_sub]
                    perplexity = min(30, max(5, (n_sub - 1) // 3))
                    plot_latent_discrete_labels(
                        Zs,
                        ys,
                        method="tsne",
                        perplexity=perplexity,
                        random_state=cfg.seed,
                        legend_title="True y",
                        titles=self._ttl(cfg, "Latent Space (t-SNE)", getattr(method, "plot_suptitle_tsne", None)),
                        savepath=secondary_fig_dir / "latent_space_tsne.png",
                        letters=print_labels,
                    )
        except Exception as e:
            print(f"  plot embed(true) failed: {e}")


        try:
            n = min(9, len(X))
            idx = rng.choice(len(X), size=n, replace=False)
            X_real = X[idx]
            X_rec = method.reconstruct(X_real, sample=False)

            plot_dataset_reconstructions_grid(
                X=X_real,
                X_rec=X_rec,
                y_true=y[idx],
                dataset=self.dataset,
                savepath=secondary_fig_dir / "reconstruction_real_vs_gmvae.png",
                label_loc="tl",
            )
        except Exception as e:
            import traceback
            print(f"  plot recon failed: {e}")
            traceback.print_exc()


        if mode_name in {"decoding_soft", "decoding_hard"}:
            try:
                frac = float(getattr(cfg, "label_map_frac_eval", 0.1) or 0.1)
                strat = bool(getattr(cfg, "label_map_stratified", True))
                y_1d = as_1d_labels(y)

                qc_tr_full = np.asarray(decoding_cache["qc_full"], dtype=float)
                idx_sup = np.asarray(decoding_cache["idx_sup"], dtype=int).reshape(-1)
                idx_unsup = np.asarray(decoding_cache["idx_unsup"], dtype=int).reshape(-1)
                all_classes = np.asarray(decoding_cache["classes"])
                yhat_tr_soft = as_1d_labels(decoding_cache["y_pred_soft"])
                yhat_tr_hard = as_1d_labels(decoding_cache["y_pred_hard"])

                pct = self._pct_str(int(idx_sup.size), len(X))

                # --------------------------
                # Sweep on unlabeled data
                # --------------------------
                try:
                    fracs_sweep = self._sanitize_fracs(getattr(cfg, "label_map_fracs", None))
                    if not fracs_sweep:
                        fracs_sweep = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]

                    if len(fracs_sweep) < 2:
                        print("  label_map_sweep_on_U: need >=2 valid fracs in (0,1) -> skip")
                    else:
                        seeds_eval = self._label_map_eval_seeds(cfg)
                        supervised_baselines = getattr(cfg, "supervised_baselines", None)
                        k_ref = getattr(getattr(method, "model", None), "K", None)

                        supervised_baselines = (
                            tuple(supervised_baselines)
                            if supervised_baselines is not None
                            else ("logreg",)
                        )

                        plot_label_map_sweep_on_U_multi_seed(
                            method=method,
                            X=X,
                            y=y_1d,
                            fracs=fracs_sweep,
                            seeds=seeds_eval,
                            stratified=strat,
                            supervised_baselines=supervised_baselines,
                            classes=all_classes,
                            verbose=False,
                            savepath=out_dir / "figure_3.png",
                            title=None,
                            n_total=int(len(X)),
                            k_reference=(int(k_ref) if k_ref is not None else None),
                            target_y=0.80,
                            dpi=240,
                            close=True,
                            show_band=True,
                            show_baselines=True,
                        )
                except Exception as e:
                    print(f"  label_map_sweep_on_U (multi-seed) failed: {e}")


                if bool(getattr(cfg, "do_label_map_delta_per_class_on_U", True)):
                    try:
                        seeds_eval = self._label_map_eval_seeds(cfg)
                        deltas: list[np.ndarray] = []
                        labels_ref: np.ndarray | None = None
                        seeds_used: list[int] = []

                        frag_unit = str(getattr(cfg, "label_map_fragmentation_unit", "bits") or "bits").lower().strip()
                        if frag_unit not in {"bits", "nats"}:
                            frag_unit = "bits"

                        for sd_i in seeds_eval:
                            idx_sup_s, idx_unsup_s = split_weak_supervision_indices(
                                y_1d,
                                frac=frac,
                                seed=int(sd_i),
                                stratified=strat,
                            )
                            idx_sup_s = np.asarray(idx_sup_s, dtype=int).reshape(-1)
                            idx_unsup_s = np.asarray(idx_unsup_s, dtype=int).reshape(-1)
                            if idx_sup_s.size == 0 or idx_unsup_s.size == 0:
                                continue

                            cls_s, M_s = build_M_soft(qc_tr_full, y_1d, idx_sup_s, classes=all_classes)
                            cls_h, M_h = build_M_hard(
                                qc_tr_full,
                                y_1d,
                                idx_sup_s,
                                classes=all_classes,
                                K=qc_tr_full.shape[1],
                            )

                            yhat_soft = as_1d_labels(decode_soft(qc_tr_full, cls_s, M_s))
                            yhat_hard = as_1d_labels(decode_hard(qc_tr_full, cls_h, M_h))

                            qcU = qc_tr_full[idx_unsup_s]
                            yU = y_1d[idx_unsup_s]
                            yU_soft = yhat_soft[idx_unsup_s]
                            yU_hard = yhat_hard[idx_unsup_s]

                            labels, delta = delta_acc_per_label(
                                y_true_U=yU,
                                y_pred_soft_U=yU_soft,
                                y_pred_hard_U=yU_hard,
                            )
                            if labels_ref is None:
                                labels_ref = labels
                            elif not np.array_equal(labels_ref, labels):
                                raise RuntimeError("per-label labels order mismatch across seeds")

                            deltas.append(delta)
                            seeds_used.append(int(sd_i))

                        if labels_ref is None or len(deltas) == 0:
                            print("  per-class delta (multi-seed): no valid seed produced a split -> skip")
                        else:
                            delta_mat = np.stack(deltas, axis=0)
                            mu = delta_mat.mean(axis=0)
                            sd = delta_mat.std(axis=0)

                            res_lbl = PerLabelDeltaMultiSeedResult(
                                labels=labels_ref,
                                seeds=np.asarray(seeds_used, dtype=int),
                                delta=delta_mat,
                            )

                            plot_delta_acc_per_label_multi_seed(
                                res_lbl,
                                dataset=self.dataset,
                                savepath=secondary_fig_dir / f"delta_acc_soft_minus_hard_on_U_frac{frac:g}.png",
                                titles=self._ttl(
                                    cfg,
                                    f"Per-label accuracy gain: soft − hard ({delta_mat.shape[0]} seeds, mean ± 95% CI)",
                                    f"Weak decoding on unlabeled data (labeled fraction: {pct})",
                                ),
                                dpi=240,
                                sort_by_gain=True,
                                show_error=True,
                            )

                            plot_gain_vs_std(
                                labels=labels_ref,
                                delta_mean=mu,
                                delta_std=sd,
                                dataset=self.dataset,
                                savepath=secondary_fig_dir / f"gain_vs_std_soft_minus_hard_frac{frac:g}.png",
                                titles=self._ttl(
                                    cfg,
                                    f"Per-label gain vs instability ({delta_mat.shape[0]} seeds)",
                                    f"Weak decoding on unlabeled data (labeled fraction: {pct})",
                                ),
                                dpi=240,
                            )
                    except Exception as e:
                        print(f"  label_map delta per class on U (multi-seed) failed: {e}")


                if bool(getattr(cfg, "do_label_map_soft_hard_examples_on_U", True)):
                    try:
                        Nshow = int(min(getattr(cfg, "label_map_examples_maxN", 5000), len(idx_unsup)))

                        XU = X[idx_unsup][:Nshow]
                        yU = y_1d[idx_unsup][:Nshow]
                        yU_hard = yhat_tr_hard[idx_unsup][:Nshow]
                        yU_soft = yhat_tr_soft[idx_unsup][:Nshow]
                        qcU = qc_tr_full[idx_unsup][:Nshow]

                        qc_sup = qc_tr_full[idx_sup]
                        c_sup = np.argmax(qc_sup, axis=1).astype(int)
                        visible_hard_clusters = np.unique(c_sup).astype(int)

                        plot_single_soft_hard_example(
                            X=XU,
                            y_true=yU,
                            y_hard=yU_hard,
                            y_soft=yU_soft,
                            dataset=self.dataset,
                            savepath=out_dir / "figure_2.png",
                            titles=self._ttl(
                                cfg,
                                "Example where soft decoding corrects a hard decoding error",
                                f"Weak decoding on unlabeled data (labeled fraction: {pct})",
                            ),
                            qc=qcU,
                            prefer="hard_wrong_soft_right",
                            dpi=240,
                            minentropy_bits=float(getattr(cfg, "label_map_example_minentropy_bits", 1.0)),
                            visible_hard_clusters=visible_hard_clusters,
                        )
                    except Exception as e:
                        print(f"  single example hard vs soft (U) failed: {e}")

            except Exception as e:
                print(f"  decoding_*: weak mapping + decoding failed: {e}")

        # --------------------------
        # qc plots
        # --------------------------
        # try:
        #     if callable(getattr(method, "predict_proba", None)):
        #         self._plots_qc(method, X, cfg, rng=rng)
        # except Exception as e:
        #     print(f"  plot qc failed: {e}")

        # --------------------------
        # GMVAE block
        # --------------------------
        mdl = getattr(method, "model", None)
        if mdl is not None and all(hasattr(mdl, k) for k in ("pz_mu", "pz_logvar", "K")):
            try:
                run_gmvae_block(
                    method=method,
                    fig_dir=out_dir,
                    X=X,
                    y=y,
                    cfg=cfg,
                    dataset=self.dataset,
                )
            except Exception as e:
                print(f"  gmvae block failed: {e}")


        try:
            mdl = getattr(method, "model", None)
            if mdl is not None and callable(getattr(mdl, "decode", None)):
                comp_to_label = getattr(method, "component_to_label_", None)
                can_sort = isinstance(comp_to_label, dict) and len(comp_to_label) > 0

                plot_generated_per_component_grid(
                    method=method,
                    savepath=out_dir,
                    n_per_component=10,
                    rows_per_fig=15,
                    cell_h=0.7,
                    dpi=250,
                    seed=0,
                    sort_rows_by_label=can_sort,
                    titles=self._ttl(
                        cfg,
                        "Samples from $p(z|c)$ decoded",
                        f"Generated samples per cluster from GMVAE (K={mdl.K})",
                    ),
                )
        except Exception as e:
            print(f"  plot generated grid failed: {e}")

    def _plots_qc(
        self,
        method,
        X,
        cfg: PlotConfig,
        *,
        rng: np.random.Generator,
    ):
        N_wanted = int(min(cfg.n_qc, len(X)))
        idx = self._subsample_indices(len(X), N_wanted, rng=rng)
        N = int(len(idx))
        if N == 0:
            print("  qc: empty subsample -> skip qc plots")
            return

        qc_raw = method.predict_proba(np.asarray(X)[idx])
        if qc_raw is None:
            print("  qc: predict_proba returned None -> skip qc plots")
            return

        qc_tr = np.asarray(qc_raw, dtype=float)
        if qc_tr.ndim != 2:
            print(f"  qc: expected (N,K) but got shape {qc_tr.shape} -> skip qc plots")
            return
        if qc_tr.shape[0] != N:
            print(f"  qc: expected N={N} rows but got {qc_tr.shape[0]} -> skip qc plots")
            return