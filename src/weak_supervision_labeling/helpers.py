# scr/weak_supervision_labeling/helpers.py

from __future__ import annotations

from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_completeness_v_measure
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from weak_supervision_labeling.metrics import clustering_metrics
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from pathlib import Path
import re
from typing import Any


def entropy(p: np.ndarray, *, unit: str = "bits") -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    p = p / max(float(p.sum()), 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    logp = np.log(p) if unit == "nats" else np.log2(p)
    return float(-(p * logp).sum())


def infer_img_shape_from_flat(X: np.ndarray, default=(28, 28)) -> tuple[int, int]:
    X = np.asarray(X)
    if X.ndim >= 2:
        d = int(X.shape[-1])
        s = int(np.sqrt(d))
        if s * s == d:
            return (s, s)
    return default


def _to_numpy(x: Any):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def as_1d_labels(y: Any) -> np.ndarray | None:
    if y is None:
        return None

    y = _to_numpy(y)

    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    elif y.ndim == 2 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    elif y.ndim > 2:
        y = np.asarray(y).reshape(y.shape[0], -1)
        y = np.argmax(y, axis=1)
    else:
        y = np.asarray(y).reshape(-1)

    if np.issubdtype(y.dtype, np.floating):
        y_round = np.round(y)
        if np.all(np.abs(y - y_round) < 1e-6):
            y = y_round

    return y.astype(int)



# TensorBoard + figures dirs
def make_run_dirs(method_dir: Path, *, use_tb: bool):
    """
    method_dir = runs/<dataset>/<timestamp>/<tag>
    """
    ts_dir = method_dir.parent
    dataset_dir = ts_dir.parent

    dataset_name = dataset_dir.name
    timestamp = ts_dir.name
    tag = method_dir.name

    tb_dir = Path("tb_logs") / dataset_name / timestamp / tag
    fig_dir = Path("figures") / dataset_name / timestamp / tag

    tb_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(logdir=str(tb_dir)) if use_tb else None
    return writer, tb_dir, fig_dir


def _pretty_run_title(title: str) -> tuple[str, str]:
    raw = title.strip()

    m = re.search(r"^\s*([a-zA-Z0-9_+-]+)\s*\(([^)]+)\)", raw)
    if m:
        method = m.group(1).upper()
        dataset = m.group(2).upper()
    else:
        method = raw.split("—")[0].strip().upper()
        dataset = ""

    z = re.search(r"z(?:_dim)?(\d+)", raw)
    K = re.search(r"(?:^|_)K(\d+)(?:_|$)", raw)
    epochs = re.search(r"epochs(\d+)", raw)
    seed = re.search(r"seed(\d+)", raw)
    beta = re.search(r"beta([0-9]*\.?[0-9]+)", raw)

    parts = []
    if z:
        parts.append(f"z={z.group(1)}")
    if K:
        parts.append(f"K={K.group(1)}")
    if beta:
        parts.append(f"β={beta.group(1)}")
    if epochs:
        parts.append(f"epochs={epochs.group(1)}")
    if seed:
        parts.append(f"seed={seed.group(1)}")

    main = f"{method}" + (f" — {dataset}" if dataset else "")
    sub = " · ".join(parts) if parts else raw
    return main, sub


def save_loss_figures_from_history(history: dict, fig_dir: Path, title: str):
    if not isinstance(history, dict) or not history:
        return
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    hist = {}
    for k, v in history.items():
        if v is None or not isinstance(v, (list, tuple)) or len(v) == 0:
            continue
        kk = k
        if k in ("loss", "total", "total_loss"):
            kk = "loss/total"
        elif not k.startswith("loss/"):
            kk = f"loss/{k}"
        hist[kk] = [float(x) for x in v]

    if not hist:
        return

    main_title, sub_title = _pretty_run_title(title)

    def _apply_titles(fig, main_title: str, sub_title: str, right_note: str):
        fig.suptitle(main_title, fontsize=14, y=0.99)
        fig.text(0.5, 0.925, sub_title, ha="center", va="center", fontsize=13, alpha=0.85)
        fig.text(0.985, 0.935, right_note, ha="right", va="center", fontsize=12, alpha=0.75)

    if "loss/total" in hist:
        fig, ax = plt.subplots(figsize=(11.5, 5.2))
        ax.plot(hist["loss/total"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        _apply_titles(fig, main_title, sub_title, "Total loss")
        fig.subplots_adjust(top=0.84)
        fig.savefig(fig_dir / "loss_total.png", dpi=180)
        plt.close(fig)

    keys = sorted(hist.keys(), key=lambda x: (x != "loss/total", x))
    pretty = {
        "loss/total": "total",
        "loss/reconstruction": "reconstruction",
        "loss/recon": "reconstruction",
        "loss/KL": "KL",
        "loss/KL_z": "KL(z)",
        "loss/kl_z": "KL(z)",
        "loss/KL_c": "KL(c)",
        "loss/kl_c": "KL(c)",
        "loss/H(C|X)": "H(C|X)",
        "loss/h_c_given_x": "H(C|X)",
    }

    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    for k in keys:
        ax.plot(hist[k], label=pretty.get(k, k.replace("loss/", "")))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True, fontsize=11)
    _apply_titles(fig, main_title, sub_title, "Loss terms")
    fig.subplots_adjust(top=0.84)
    fig.savefig(fig_dir / "loss_terms.png", dpi=180)
    plt.close(fig)


def safe_predict(method, X: np.ndarray):
    try:
        return method.predict(X)
    except Exception as e:
        print(f"[{getattr(method, 'name', type(method).__name__)}] predict failed: {e}")
        return None


def print_metrics(tag: str, y_true: np.ndarray, y_pred: np.ndarray):
    m = clustering_metrics(y_true, y_pred)
    print(f"  {tag}: " + ", ".join([f"{k}={v:.4f}" for k, v in m.items()]))
    return m


@torch.no_grad()
def collect_gmvae_latents(model, loader, device, mode="map_mu", max_batches=None):
    """
    Récupère un latent par x et les labels y.

    mode:
      - "map_mu" : z = mu(x, argmax q(c|x))
      - "exp_mu" : z = sum_c q(c|x) * mu(x,c)
      - "sample" : sample c~q(c|x), z~q(z|x,c)
    """
    model.eval()
    Z, Y = [], []

    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break

        x = x.to(device, non_blocking=True).view(x.size(0), -1)

        qc_logits, qc, mu_all, logvar_all = model.encode(x)  # qc:(B,K), mu_all:(B,K,D)

        if mode == "map_mu":
            c_star = qc.argmax(dim=-1)                       # (B,)
            z = mu_all[torch.arange(x.size(0), device=device), c_star, :]  # (B,D)

        elif mode == "exp_mu":
            z = (qc.unsqueeze(-1) * mu_all).sum(dim=1)        # (B,D)

        elif mode == "sample":
            c_s = torch.multinomial(qc, num_samples=1).squeeze(1)  # (B,)
            mu = mu_all[torch.arange(x.size(0), device=device), c_s, :]
            logvar = logvar_all[torch.arange(x.size(0), device=device), c_s, :]
            z = model.reparam(mu, logvar)

        else:
            raise ValueError("mode must be one of: 'map_mu', 'exp_mu', 'sample'")

        Z.append(z.detach().cpu())
        Y.append(y.detach().cpu())

    Z = torch.cat(Z, dim=0).numpy()
    Y = torch.cat(Y, dim=0).numpy()
    return Z, Y


@torch.no_grad()
def predict_pi_clusters(model, loader, device, use="argmax", max_batches=200):
    """
    Equivalent de predict_pi_clusters mais pour le GMVAE simple.

    Ici, le "posterior" discret est explicitement q(c|x).

    use:
      - "argmax" : c_pred = argmax_c q(c|x)
      - "sample" : c_pred ~ Cat(q(c|x))  (stochastique)
    """
    model.eval()
    C, Y = [], []

    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break

        x = x.to(device, non_blocking=True).view(x.size(0), -1)

        qc_logits, qc, mu_all, logvar_all = model.encode(x)  # qc: (B,K)

        if use == "argmax":
            c_pred = qc.argmax(dim=-1)  # (B,)
        elif use == "sample":
            c_pred = torch.multinomial(qc, num_samples=1).squeeze(1)  # (B,)
        else:
            raise ValueError("use must be 'argmax' or 'sample'")

        C.append(c_pred.detach().cpu())
        Y.append(y.detach().cpu())

    C = torch.cat(C, dim=0).numpy()
    Y = torch.cat(Y, dim=0).numpy()
    return C, Y



def clustering_purity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)  # (C,K)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)


def clustering_metrics_kfree(y_true, y_pred):
    h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    return {"Homogeneity": h, "Completeness": c, "V-measure": v, "AMI": ami}


def kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    """
    KL( N(mu_q, diag(var_q)) || N(mu_p, diag(var_p)) )
    All tensors shape (..., D)
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    return 0.5 * torch.sum(
        logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0,
        dim=-1
    )



def make_loader_from_numpy(
    X: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    ds = TensorDataset(X_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)



def posterior_component_params(post_pis, post_mus, post_vars, mode: str = "argmax"):
    """
    Return (mu_x, var_x, c_x) for each x, picking a single component of q(z|x,c).
    - mode="argmax": c_x = argmax q(c|x)
    - mode="sample": sample c_x ~ Cat(q(c|x)) (stochastic)

    post_pis: (N,K), post_mus/post_vars: (N,K,D)
    """
    post_pis = np.asarray(post_pis)
    post_mus = np.asarray(post_mus)
    post_vars = np.asarray(post_vars)
    N, K = post_pis.shape

    if mode == "argmax":
        c = post_pis.argmax(axis=1)
    elif mode == "sample":
        u = np.random.rand(N, 1)
        cdf = np.cumsum(post_pis / (post_pis.sum(axis=1, keepdims=True) + 1e-12), axis=1)
        c = (u <= cdf).argmax(axis=1)
    else:
        raise ValueError(f"Unknown mode={mode}")

    mu = post_mus[np.arange(N), c]     # (N,D)
    var = post_vars[np.arange(N), c]   # (N,D)
    return mu, var, c


def build_component_to_label_map(qc: np.ndarray, y: np.ndarray, K: int) -> dict[int, int]:
    """
    qc: (N, K) = q(c|x)
    y: (N,) true labels (0..25 for EMNIST letters)
    returns: {k: y_mode_for_component_k}
    """
    c_hat = qc.argmax(axis=1)
    mapping: dict[int, int] = {}
    for k in range(K):
        mask = (c_hat == k)
        if not np.any(mask):
            continue
        ys = y[mask]
        vals, cnts = np.unique(ys, return_counts=True)
        mapping[k] = int(vals[np.argmax(cnts)])
    return mapping