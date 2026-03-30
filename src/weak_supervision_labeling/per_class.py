# src/weak_supervision_labeling/per_class.py

from weak_supervision_labeling.helpers import as_1d_labels
import numpy as np
from dataclasses import dataclass


_EPS = 1e-12

def per_label_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      labels: (C,)
      acc:    (C,)
    """
    y_true = as_1d_labels(y_true)
    y_pred = as_1d_labels(y_pred)

    labels = np.unique(y_true)
    acc = np.zeros_like(labels, dtype=float)

    for i, lab in enumerate(labels):
        m = (y_true == lab)
        acc[i] = float((y_pred[m] == lab).mean()) if m.any() else np.nan

    return labels, acc


def delta_acc_per_label(
    *,
    y_true_U: np.ndarray,
    y_pred_soft_U: np.ndarray,
    y_pred_hard_U: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels, acc_soft = per_label_accuracy(y_true_U, y_pred_soft_U)
    labels2, acc_hard = per_label_accuracy(y_true_U, y_pred_hard_U)
    if not np.array_equal(labels, labels2):
        raise ValueError("soft/hard labels mismatch in per_label_accuracy")
    return labels, (acc_soft - acc_hard)


@dataclass
class PerLabelDeltaMultiSeedResult:
    labels: np.ndarray         # (L,)
    seeds: np.ndarray          # (S,)
    delta: np.ndarray          # (S,L)

    @property
    def mean(self) -> np.ndarray:
        return np.nanmean(self.delta, axis=0)

    @property
    def std(self) -> np.ndarray:
        return np.nanstd(self.delta, axis=0)
    

@dataclass
class DeltaPerLabelStats:
    labels: np.ndarray          # (C,)
    delta_mean: np.ndarray      # (C,)
    delta_std: np.ndarray       # (C,)
    snr: np.ndarray             # (C,) = mean/std


def entropy_qc(qc: np.ndarray, *, eps: float = _EPS) -> np.ndarray:
    """
    qc: (N,K) probabilities.
    returns: (N,) entropy in nats.
    """
    qc = np.asarray(qc, dtype=float)
    qc = np.clip(qc, eps, 1.0)
    qc = qc / qc.sum(axis=1, keepdims=True)
    return -np.sum(qc * np.log(qc), axis=1)


def per_label_mean(x: np.ndarray, y: np.ndarray):
    """
    x: (N,), y: (N,) int labels
    returns: labels (L,), mean (L,), count (L,)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    labels = np.unique(y)
    mean = np.zeros(len(labels), dtype=float)
    cnt = np.zeros(len(labels), dtype=int)
    for i, lab in enumerate(labels):
        m = (y == lab)
        cnt[i] = int(m.sum())
        mean[i] = float(x[m].mean()) if cnt[i] > 0 else np.nan
    return labels, mean, cnt


@dataclass
class GainEntropyPerLabel:
    labels: np.ndarray          # (L,)
    gain_mean: np.ndarray       # (L,)
    gain_std: np.ndarray        # (L,)
    H_mean: np.ndarray          # (L,)
    H_std: np.ndarray           # (L,)
    count_mean: np.ndarray      # (L,) average count in U (or just from first seed)
    seeds: np.ndarray           # (S,)



def aggregate_gain_entropy_over_seeds(
    *,
    labels_ref: np.ndarray,
    gains: list[np.ndarray],      # list of (L,)
    entropies: list[np.ndarray],  # list of (L,)
    counts: list[np.ndarray],     # list of (L,)
    seeds: list[int],
) -> GainEntropyPerLabel:
    G = np.stack(gains, axis=0)       # (S,L)
    H = np.stack(entropies, axis=0)   # (S,L)
    C = np.stack(counts, axis=0)      # (S,L)

    return GainEntropyPerLabel(
        labels=np.asarray(labels_ref, dtype=int),
        gain_mean=G.mean(axis=0),
        gain_std=G.std(axis=0, ddof=0),
        H_mean=H.mean(axis=0),
        H_std=H.std(axis=0, ddof=0),
        count_mean=C.mean(axis=0),
        seeds=np.asarray(seeds, dtype=int),
    )

def label_names_from_labels(labels: np.ndarray, dataset: str | None) -> list[str]:
    labels = np.asarray(labels).astype(int)
    if dataset == "emnist" and labels.min() >= 0 and labels.max() <= 25:
        return [chr(ord("a") + int(i)) for i in labels]
    return [str(int(i)) for i in labels]