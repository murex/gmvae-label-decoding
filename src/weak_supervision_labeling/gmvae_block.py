import numpy as np
from weak_supervision_labeling.helpers import as_1d_labels
from weak_supervision_labeling.plotting.gmvae import plot_component_purity_soft, plot_generated_by_cluster
from pathlib import Path
from weak_supervision_labeling.plotting.base import Titles


def run_gmvae_block(
    *,
    method,
    fig_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    cfg,
    dataset: str | None = None,
):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    secondary_fig_dir = fig_dir / "additional_figures"
    secondary_fig_dir.mkdir(parents=True, exist_ok=True)

    titles_plot = bool(getattr(cfg, "titles_plot", True))

    mdl = method.model
    device = getattr(method, "device", None)

    # Prior params (K, D)
    prior_mus = mdl.pz_mu.detach().cpu().numpy()
    prior_vars = np.exp(mdl.pz_logvar.detach().cpu().numpy())

    rng = np.random.RandomState(int(getattr(cfg, "seed", 0)))
    Nmax = min(500, int(len(X)))

    # Ensure y is (N,)
    y_all = as_1d_labels(np.asarray(y))
    if y_all is None:
        raise ValueError("y is required for gmvae block diagnostics (per-class plots).")

    classes = np.unique(y_all)
    if classes.size == 0:
        raise ValueError("y has no labels (empty unique classes).")


    per_class_target = max(1, Nmax // len(classes))
    idx_parts: list[np.ndarray] = []

    for c in classes:
        ids_c = np.flatnonzero(y_all == c)
        if ids_c.size == 0:
            continue
        take = min(per_class_target, ids_c.size)
        idx_parts.append(rng.choice(ids_c, size=take, replace=False))

    if len(idx_parts) == 0:
        raise ValueError("Could not sample any points (all classes empty?).")

    idx = np.concatenate(idx_parts)

    # complete to Nmax if needed
    if idx.size < Nmax:
        remaining = np.setdiff1d(np.arange(y_all.shape[0]), idx, assume_unique=False)
        if remaining.size > 0:
            extra = rng.choice(
                remaining, size=min(Nmax - idx.size, remaining.size), replace=False
            )
            idx = np.concatenate([idx, extra])

    # Final arrays
    X_plot = np.asarray(X)[idx]
    y_plot = y_all[idx]

    # check: expected number of classes
    expected = (
        getattr(cfg, "n_classes", None)
        or getattr(cfg, "n_classes_expected", None)
        or getattr(method, "n_classes", None)
    )
    if expected is not None:
        expected = int(expected)
        nuniq = len(np.unique(y_plot))
        if nuniq != expected:
            print(
                f"[gmvae:run_gmvae_block] WARNING: y has {nuniq} unique labels, expected {expected}. "
                "This usually means the dataset labels passed to gmvae_block are not what you think."
            )

    # Posterior params
    fn = getattr(method, "posterior_params", None)
    if not callable(fn):
        raise RuntimeError("method.posterior_params(X) not available.")

    post = fn(X_plot, batch_size=256)
    if isinstance(post, dict):
        post_pis = np.asarray(post["qc"])
        post_mus = np.asarray(post["mus"])
        post_vars = np.asarray(post["vars"])
    else:
        post_pis, post_mus, post_vars = post
        post_pis = np.asarray(post_pis)
        post_mus = np.asarray(post_mus)
        post_vars = np.asarray(post_vars)

    plot_component_purity_soft(
        qc=post_pis,
        y_true=y_plot,
        dataset=dataset,
        savepath=secondary_fig_dir / "component_purity_and_weight.png",
        sort_by="soft",
        min_count_to_show=0,
        max_components=None,
        titles_plot=titles_plot,
    )

    n_blocks=2
    rows_per_block=10
    n_ex = int(n_blocks * rows_per_block)

    plot_generated_by_cluster(
        model=mdl,
        n_per_cluster=8,
        n_blocks=n_blocks,
        rows_per_block=rows_per_block,
        max_clusters=26,
        pick="coverage",
        dpi=300,
        savepath=fig_dir / "figure_1.png",
    )

    try:
        z_mean = np.sum(post_pis[:, :, None] * post_mus, axis=1)  # (N,D)
    except Exception:
        z_mean = None

    return {
        "idx": idx,
        "X_plot": X_plot,
        "y_plot": y_plot,
        "post_pis": post_pis,
        "post_mus": post_mus,
        "post_vars": post_vars,
        "prior_mus": prior_mus,
        "prior_vars": prior_vars,
        "z_mean": z_mean,
    }