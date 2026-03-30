import numpy as np
from weak_supervision_labeling.helpers import as_1d_labels


def dataset_key(dataset: str | None) -> str:
    return str(dataset).strip().lower() if dataset is not None else ""


def n_classes_for_palette(dataset: str | None, y: np.ndarray | None) -> int:
    """
    Canonical palette size:
      - mnist  -> 10
      - emnist -> 26
    Fallback: max(y)+1 (if y is available), else 10.
    """
    ds = dataset_key(dataset)
    if ds == "mnist":
        return 10
    if ds == "emnist":
        return 26
    if y is None or len(y) == 0:
        return 10
    return int(np.max(y) + 1)


def component_majority_stats(
    *,
    qc: np.ndarray | None,
    y_true: np.ndarray | None,
    K: int,
):
    majority_label = np.full(K, -1, dtype=int)
    counts = np.zeros(K, dtype=int)
    mean_qc = np.full(K, np.nan, dtype=float)
    purity = np.full(K, np.nan, dtype=float)

    if qc is None:
        return majority_label, counts, mean_qc, purity

    qc = np.asarray(qc)
    if qc.ndim != 2 or qc.shape[1] != K:
        raise ValueError(f"qc must be (N,K) with K={K}, got {qc.shape}")
    mean_qc = qc.mean(axis=0)

    if y_true is None:
        return majority_label, counts, mean_qc, purity

    y_true = as_1d_labels(y_true)
    if y_true.shape[0] != qc.shape[0]:
        raise ValueError(f"y_true and qc must match N, got {y_true.shape[0]} vs {qc.shape[0]}")

    c_hat = qc.argmax(axis=1)
    for c in range(K):
        idx = np.where(c_hat == c)[0]
        counts[c] = int(idx.size)
        if idx.size == 0:
            continue
        vals, cnts = np.unique(y_true[idx], return_counts=True)
        m = int(cnts.argmax())
        majority_label[c] = int(vals[m])
        purity[c] = float(cnts[m] / max(1, idx.size))

    return majority_label, counts, mean_qc, purity


def soft_purity(
    *,
    qc: np.ndarray,
    y_true: np.ndarray,
    K: int,
    majority_label: np.ndarray | None = None,
) -> np.ndarray:
    """
    Soft purity per component:
      purity_soft[c] = sum_i qc[i,c] * 1(y_i == maj[c]) / sum_i qc[i,c]
    """
    qc = np.asarray(qc, dtype=float)
    y_true = as_1d_labels(y_true)
    if qc.ndim != 2 or qc.shape[1] != K:
        raise ValueError(f"qc must be (N,K) with K={K}, got {qc.shape}")
    if y_true.shape[0] != qc.shape[0]:
        raise ValueError(f"y_true and qc must match N, got {y_true.shape[0]} vs {qc.shape[0]}")

    if majority_label is None:
        maj, _, _, _ = component_majority_stats(qc=qc, y_true=y_true, K=K)
        majority_label = maj
    majority_label = np.asarray(majority_label, dtype=int)

    denom = qc.sum(axis=0) + 1e-12
    out = np.full(K, np.nan, dtype=float)
    for c in range(K):
        lab = int(majority_label[c])
        if lab < 0:
            continue
        num = qc[y_true == lab, c].sum()
        out[c] = float(num / denom[c])
    return out