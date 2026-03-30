# scr/weak_supervision_labeling/weak_supervision.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from weak_supervision_labeling.helpers import as_1d_labels


@dataclass
class WeakSupMatrices:
    classes: np.ndarray        # (L,)
    M_soft: np.ndarray         # (L, K)
    M_hard: np.ndarray         # (L, K)


def split_weak_supervision_indices(
    y: np.ndarray,
    *,
    frac: float = 0.1,
    seed: int = 0,
    stratified: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (idx_sup, idx_unsup).
    If stratified is True: global budget n_sup=round(frac*N) allocated across classes.
    """
    y = as_1d_labels(y)
    N = len(y)
    if not (0.0 < float(frac) < 1.0):
        raise ValueError(f"frac must be in (0,1), got {frac}")

    rng = np.random.default_rng(int(seed))

    n_sup = int(np.round(float(frac) * N))
    n_sup = int(np.clip(n_sup, 1, N - 1)) 

    if not stratified:
        idx = np.arange(N)
        rng.shuffle(idx)
        idx_sup = np.sort(idx[:n_sup])
        idx_unsup = np.sort(idx[n_sup:])
    else:
        classes = np.unique(y)
        L = int(classes.size)

        # indices per class
        per_class = {int(c): np.where(y == c)[0] for c in classes}
        for c in classes:
            rng.shuffle(per_class[int(c)])

        idx_sup_list: list[np.ndarray] = []

        if n_sup <= L:
            chosen = rng.choice(classes, size=n_sup, replace=False)
            for c in chosen:
                ids = per_class[int(c)]
                if ids.size > 0:
                    idx_sup_list.append(ids[:1])
        else:
            alloc = {int(c): (1 if per_class[int(c)].size > 0 else 0) for c in classes}
            used = int(sum(alloc.values()))
            remaining = n_sup - used

            sizes = np.array([per_class[int(c)].size for c in classes], dtype=float)
            w = sizes / max(1.0, sizes.sum())

            add = rng.multinomial(remaining, w) if remaining > 0 else np.zeros_like(sizes, dtype=int)

            for k, c in enumerate(classes):
                take = int(alloc[int(c)] + add[k])
                ids = per_class[int(c)]
                take = min(take, int(ids.size))
                if take > 0:
                    idx_sup_list.append(ids[:take])

        idx_sup = np.unique(np.concatenate(idx_sup_list)) if idx_sup_list else np.array([], dtype=int)

        if idx_sup.size < n_sup:
            mask = np.ones(N, dtype=bool)
            mask[idx_sup] = False
            remaining_ids = np.where(mask)[0]
            if remaining_ids.size > 0:
                need = min(int(n_sup - idx_sup.size), int(remaining_ids.size))
                extra = rng.choice(remaining_ids, size=need, replace=False)
                idx_sup = np.unique(np.concatenate([idx_sup, extra]))

        mask = np.ones(N, dtype=bool)
        mask[idx_sup] = False
        idx_unsup = np.where(mask)[0]

        idx_sup = np.sort(idx_sup)
        idx_unsup = np.sort(idx_unsup)
    return idx_sup, idx_unsup


def build_M_soft(
    qc: np.ndarray,
    y: np.ndarray,
    idx_sup: np.ndarray,
    *,
    classes: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    M_soft[l, c] = mean_{i in S_l} q(c|x_i)   then row-normalized.
    Returns (classes, M_soft) where classes has length L.
    """
    qc = np.asarray(qc, dtype=float)
    y = as_1d_labels(y)

    if qc.ndim != 2:
        raise ValueError(f"qc must be (N,K), got {qc.shape}")
    if qc.shape[0] != y.shape[0]:
        raise ValueError("qc and y length mismatch")
    if idx_sup.size == 0:
        raise ValueError("idx_sup is empty")

    yS = y[idx_sup]
    qcS = qc[idx_sup]

    if classes is None:
        classes = np.unique(y)
    classes = np.asarray(classes, dtype=int).reshape(-1)

    L = int(classes.size)
    K = int(qc.shape[1])

    row_of = {int(c): i for i, c in enumerate(classes)}
    rows = np.array([row_of.get(int(yi), -1) for yi in yS], dtype=int)
    keep = rows >= 0
    if not np.any(keep):
        raise ValueError("No labeled samples match provided classes")

    rows = rows[keep]
    qcS = qcS[keep]

    M = np.zeros((L, K), dtype=float)
    counts = np.zeros(L, dtype=float)

    np.add.at(M, rows, qcS)
    np.add.at(counts, rows, 1.0)

    counts = np.maximum(counts, 1.0)
    M = M / counts[:, None]

    # Row-normalize for safety
    row_sums = np.maximum(M.sum(axis=1, keepdims=True), eps)
    M = M / row_sums
    return classes, M


def build_M_hard(
    qc: np.ndarray,
    y: np.ndarray,
    idx_sup: np.ndarray,
    *,
    classes: np.ndarray | None = None,
    K: int | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hard mapping estimated from labeled subset S:
      c_i = argmax q(c|x_i)
      M_hard[l,c] ≈ P(y=l | c) estimated on S (column-normalized)
    """
    qc = np.asarray(qc, dtype=float)
    y = as_1d_labels(y)

    if qc.ndim != 2:
        raise ValueError(f"qc must be (N,K), got {qc.shape}")
    if qc.shape[0] != y.shape[0]:
        raise ValueError("qc and y length mismatch")
    if idx_sup.size == 0:
        raise ValueError("idx_sup is empty")

    K = int(qc.shape[1] if K is None else K)

    yS = y[idx_sup]
    qcS = qc[idx_sup]
    cS = np.argmax(qcS, axis=1).astype(int)

    if classes is None:
        classes = np.unique(y)
    classes = np.asarray(classes, dtype=int).reshape(-1)

    L = int(classes.size)

    row_of = {int(c): i for i, c in enumerate(classes)}
    rows = np.array([row_of.get(int(yi), -1) for yi in yS], dtype=int)
    keep = rows >= 0
    if not np.any(keep):
        raise ValueError("No labeled samples match provided classes")

    rows = rows[keep]
    cS = cS[keep]

    M = np.zeros((L, K), dtype=float)
    counts_c = np.zeros(K, dtype=float)

    np.add.at(M, (rows, cS), 1.0)
    np.add.at(counts_c, cS, 1.0)

    counts_c = np.maximum(counts_c, eps)
    M = M / counts_c[None, :]
    return classes, M


def decode_soft(qc: np.ndarray, classes: np.ndarray, M_soft: np.ndarray) -> np.ndarray:
    qc = np.asarray(qc, dtype=float)
    scores = qc @ M_soft.T  # (N,L)
    idx = np.argmax(scores, axis=1)
    return np.asarray(classes, dtype=int)[idx]


def decode_hard(qc: np.ndarray, classes: np.ndarray, M_hard: np.ndarray) -> np.ndarray:
    qc = np.asarray(qc, dtype=float)
    c = np.argmax(qc, axis=1).astype(int)
    col = M_hard[:, c].T  # (N,L)
    idx = np.argmax(col, axis=1)
    return np.asarray(classes, dtype=int)[idx]


def build_weak_sup_matrices(
    qc: np.ndarray,
    y: np.ndarray,
    *,
    frac: float = 0.1,
    seed: int = 0,
    stratified: bool = True,
    classes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, WeakSupMatrices]:
    """
      - split (idx_sup, idx_unsup)
      - build M_soft and M_hard on idx_sup only
      - by default, classes = all classes in y
    """
    y = as_1d_labels(y)
    idx_sup, idx_unsup = split_weak_supervision_indices(y, frac=frac, seed=seed, stratified=stratified)

    if classes is None:
        classes = np.unique(y)

    cls_soft, M_soft = build_M_soft(qc, y, idx_sup, classes=classes)
    cls_hard, M_hard = build_M_hard(qc, y, idx_sup, classes=classes, K=qc.shape[1])

    if not np.array_equal(cls_soft, cls_hard):
        order = {int(c): i for i, c in enumerate(cls_hard)}
        rows = [order[int(c)] for c in cls_soft]
        M_hard = M_hard[rows, :]
        cls_hard = cls_soft

    mats = WeakSupMatrices(classes=cls_soft, M_soft=M_soft, M_hard=M_hard)
    return idx_sup, idx_unsup, mats


def prepare_decoding(method, label_map_frac, label_map_stratified, seed: int,  *, X: np.ndarray, y: np.ndarray):
    qc_full = np.asarray(method.predict_proba(np.asarray(X)), dtype=float)
    if qc_full.ndim != 2 or qc_full.shape[0] != len(X):
        raise RuntimeError(f"predict_proba returned {qc_full.shape}, expected (N,K)")

    idx_sup, idx_unsup = split_weak_supervision_indices(
        np.asarray(y),
        frac=float(label_map_frac),
        seed=int(seed),
        stratified=bool(label_map_stratified),
    )

    classes = np.unique(np.asarray(y).reshape(-1).astype(int))

    cls_soft, M_soft = build_M_soft(
        qc_full,
        np.asarray(y),
        idx_sup,
        classes=classes,
    )
    cls_hard, M_hard = build_M_hard(
        qc_full,
        np.asarray(y),
        idx_sup,
        classes=classes,
        K=qc_full.shape[1],
    )

    if not np.array_equal(cls_soft, cls_hard):
        raise RuntimeError("weak decoding: class vectors mismatch")

    y_pred_soft = decode_soft(qc_full, cls_soft, M_soft)
    y_pred_hard = decode_hard(qc_full, cls_hard, M_hard)

    return {
        "qc_full": qc_full,
        "idx_sup": np.asarray(idx_sup, dtype=int),
        "idx_unsup": np.asarray(idx_unsup, dtype=int),
        "classes": classes,
        "cls_soft": cls_soft,
        "M_soft": M_soft,
        "cls_hard": cls_hard,
        "M_hard": M_hard,
        "y_pred_soft": np.asarray(y_pred_soft),
        "y_pred_hard": np.asarray(y_pred_hard),
    }