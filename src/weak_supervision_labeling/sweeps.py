# src/weak_supervision_labeling/sweeps.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Sequence
from tqdm import tqdm
from weak_supervision_labeling.helpers import as_1d_labels
from weak_supervision_labeling.weak_supervision import (
    build_M_hard,
    build_M_soft,
    decode_hard,
    decode_soft,
    split_weak_supervision_indices,
)
try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    _HAS_XGBOOST = False


@dataclass
class LabelMapSweepResult:
    fracs: np.ndarray  # (T,)
    n_sup: np.ndarray  # (T,)
    n_unsup: np.ndarray  # (T,)
    acc_soft_U: np.ndarray  # (T,)
    acc_hard_U: np.ndarray  # (T,)
    acc_baselines_U: dict[str, np.ndarray]  # name -> (T,)
    seed: int
    stratified: bool


@dataclass
class LabelMapSweepMultiSeedResult:
    fracs: np.ndarray  # (T,)
    seeds: np.ndarray  # (S,)
    n_sup: np.ndarray  # (S,T)
    n_unsup: np.ndarray  # (S,T)
    acc_soft_U: np.ndarray  # (S,T)
    acc_hard_U: np.ndarray  # (S,T)
    acc_baselines_U: dict[str, np.ndarray]  # name -> (S,T)
    stratified: bool

    @property
    def mean_soft(self) -> np.ndarray:
        return np.nanmean(self.acc_soft_U, axis=0)

    @property
    def mean_hard(self) -> np.ndarray:
        return np.nanmean(self.acc_hard_U, axis=0)

    @property
    def std_soft(self) -> np.ndarray:
        return np.nanstd(self.acc_soft_U, axis=0)

    @property
    def std_hard(self) -> np.ndarray:
        return np.nanstd(self.acc_hard_U, axis=0)

    def mean_baseline(self, name: str) -> np.ndarray:
        return np.nanmean(self.acc_baselines_U[name], axis=0)

    def std_baseline(self, name: str) -> np.ndarray:
        return np.nanstd(self.acc_baselines_U[name], axis=0)



def _fit_predict_supervised_baseline(
    *,
    X_sup: np.ndarray,
    y_sup: np.ndarray,
    X_unsup: np.ndarray,
    baseline: str,
    seed: int,
) -> np.ndarray:
    """
    Train a purely supervised baseline on S and predict on unlabeled set.

    Supported baselines:
      - "logreg"
      - "mlp"
      - "xgboost"
    """
    X_sup = np.asarray(X_sup)
    X_unsup = np.asarray(X_unsup)
    y_sup = as_1d_labels(y_sup)

    if X_sup.ndim > 2:
        X_sup = X_sup.reshape(len(X_sup), -1)
    if X_unsup.ndim > 2:
        X_unsup = X_unsup.reshape(len(X_unsup), -1)

    baseline = str(baseline).lower()

    if baseline == "logreg":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                random_state=int(seed),
            ),
        )
        clf.fit(X_sup, y_sup)
        return clf.predict(X_unsup)

    if baseline == "mlp":
        n_sup = len(y_sup)
        if n_sup == 0:
            raise ValueError("MLP baseline received no labeled samples.")

        use_early_stopping = n_sup >= 50

        if n_sup < 100:
            hidden = (64,)
        elif n_sup < 500:
            hidden = (128,)
        else:
            hidden = (256, 128)

        if use_early_stopping:
            val_frac = 0.1
            n_train_eff = max(1, int((1.0 - val_frac) * n_sup))
        else:
            val_frac = 0.0
            n_train_eff = n_sup

        batch_size = max(1, min(128, n_train_eff))

        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=batch_size,
                learning_rate_init=1e-3,
                max_iter=120,
                early_stopping=use_early_stopping,
                validation_fraction=val_frac,
                n_iter_no_change=8 if use_early_stopping else 20,
                random_state=int(seed),
            ),
        )
        clf.fit(X_sup, y_sup)
        return clf.predict(X_unsup)

    if baseline == "xgboost":
        if not _HAS_XGBOOST:
            raise ImportError(
                "xgboost is not installed, but baseline='xgboost' was requested."
            )

        classes = np.unique(y_sup)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_class = np.asarray(classes, dtype=int)
        y_sup_idx = np.asarray([class_to_idx[y] for y in y_sup], dtype=np.int64)

        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            num_class=len(classes),
            eval_metric="mlogloss",
            random_state=int(seed),
            tree_method="hist",
            n_jobs=-1,
        )
        clf.fit(X_sup, y_sup_idx)
        y_pred_idx = clf.predict(X_unsup)
        return idx_to_class[np.asarray(y_pred_idx, dtype=int)]

    raise ValueError(
        f"Unknown supervised baseline '{baseline}'. "
        f"Supported: 'logreg', 'mlp', 'xgboost'."
    )


def run_label_map_sweep_on_U(
    *,
    method,
    X: np.ndarray,
    y: np.ndarray,
    fracs: Sequence[float],
    seed: int = 0,
    stratified: bool = True,
    supervised_baselines: Sequence[str] = ("logreg",),
    classes: np.ndarray | None = None,
    verbose: bool = True,
) -> LabelMapSweepResult:
    """
    Sweep label_map_frac and evaluate weak-decoding accuracy only on unlabeled data
    (the unsupervised remainder of train).

    For each frac:
      - Split train into labeled data (supervised) and unlabeled data (remainder)
      - Build M_soft / M_hard using only labeled data
      - Decode on full train, but score only on unlabeled data
      - Train each supervised baseline only on labeled data, evaluate only on unlabeled data
    """
    X = np.asarray(X)
    y_1d = as_1d_labels(y)

    if classes is None:
        classes = np.unique(y_1d)
    classes = np.asarray(classes).astype(int).reshape(-1)

    if not callable(getattr(method, "predict_proba", None)):
        raise TypeError("method must implement predict_proba(X) -> (N,K)")

    qc_tr_full = np.asarray(method.predict_proba(X), dtype=float)
    if qc_tr_full.ndim != 2 or qc_tr_full.shape[0] != X.shape[0]:
        raise ValueError(f"predict_proba returned {qc_tr_full.shape}, expected (N,K)")

    fracs_arr = np.asarray(list(fracs), dtype=float).reshape(-1)
    if fracs_arr.size == 0:
        raise ValueError("fracs is empty")

    baseline_names = [str(b).lower() for b in supervised_baselines]
    baseline_names = list(dict.fromkeys(baseline_names))

    n_sup = np.zeros(fracs_arr.size, dtype=int)
    n_unsup = np.zeros(fracs_arr.size, dtype=int)
    acc_soft_U = np.zeros(fracs_arr.size, dtype=float)
    acc_hard_U = np.zeros(fracs_arr.size, dtype=float)
    acc_baselines_U = {
        name: np.zeros(fracs_arr.size, dtype=float) for name in baseline_names
    }

    N = int(len(y_1d))

    for t, frac in enumerate(fracs_arr):
        frac = float(frac)
        if frac <= 0.0 or frac >= 1.0:
            raise ValueError(f"Each frac must be in (0,1), got {frac}")

        idx_sup, idx_unsup = split_weak_supervision_indices(
            y_1d,
            frac=frac,
            seed=int(seed),
            stratified=bool(stratified),
        )

        idx_sup = np.asarray(idx_sup, dtype=int).reshape(-1)
        idx_unsup = np.asarray(idx_unsup, dtype=int).reshape(-1)

        n_sup[t] = int(idx_sup.size)
        n_unsup[t] = int(idx_unsup.size)

        if idx_sup.size == 0 or idx_unsup.size == 0:
            acc_soft_U[t] = np.nan
            acc_hard_U[t] = np.nan
            for name in baseline_names:
                acc_baselines_U[name][t] = np.nan
            if verbose:
                print(
                    f"[frac={frac:g}] empty split -> skip "
                    f"(n_sup={idx_sup.size}, n_unsup={idx_unsup.size})"
                )
            continue

        cls_soft, M_soft = build_M_soft(
            qc_tr_full,
            y_1d,
            idx_sup,
            classes=classes,
        )
        cls_hard, M_hard = build_M_hard(
            qc_tr_full,
            y_1d,
            idx_sup,
            classes=classes,
            K=qc_tr_full.shape[1],
        )
        if not np.array_equal(cls_soft, cls_hard):
            raise RuntimeError(
                "build_M_soft/build_M_hard returned different class orderings"
            )

        yhat_soft = as_1d_labels(decode_soft(qc_tr_full, cls_soft, M_soft))
        yhat_hard = as_1d_labels(decode_hard(qc_tr_full, cls_hard, M_hard))

        yU = y_1d[idx_unsup]
        acc_soft_U[t] = float(np.mean(yhat_soft[idx_unsup] == yU))
        acc_hard_U[t] = float(np.mean(yhat_hard[idx_unsup] == yU))

        for name in baseline_names:
            yhat_baseline_U = _fit_predict_supervised_baseline(
                X_sup=X[idx_sup],
                y_sup=y_1d[idx_sup],
                X_unsup=X[idx_unsup],
                baseline=name,
                seed=int(seed),
            )
            acc_baselines_U[name][t] = float(np.mean(yhat_baseline_U == yU))

        if verbose:
            pct = 100.0 * n_sup[t] / max(1, N)
            baseline_msg = "  ".join(
                f"{name}={acc_baselines_U[name][t]:.4f}" for name in baseline_names
            )
            print(
                f"[{pct:.3f}%] "
                f"Accuracy on unlabeld examples: soft={acc_soft_U[t]:.4f}  "
                f"hard={acc_hard_U[t]:.4f}  "
                f"{baseline_msg}"
            )

    return LabelMapSweepResult(
        fracs=fracs_arr,
        n_sup=n_sup,
        n_unsup=n_unsup,
        acc_soft_U=acc_soft_U,
        acc_hard_U=acc_hard_U,
        acc_baselines_U=acc_baselines_U,
        seed=int(seed),
        stratified=bool(stratified),
    )


def run_label_map_sweep_on_U_multi_seed(
    *,
    method,
    X: np.ndarray,
    y: np.ndarray,
    fracs: Sequence[float],
    seeds: Sequence[int],
    stratified: bool = True,
    supervised_baselines: Sequence[str] = ("logreg",),
    classes: np.ndarray | None = None,
    verbose: bool = True,
) -> LabelMapSweepMultiSeedResult:
    
    fracs_arr = np.asarray(list(fracs), dtype=float).reshape(-1)
    if fracs_arr.size == 0:
        raise ValueError("fracs is empty")

    seeds_arr = np.asarray(list(seeds), dtype=int).reshape(-1)
    if seeds_arr.size == 0:
        raise ValueError("seeds is empty")

    baseline_names = [str(b).lower() for b in supervised_baselines]
    baseline_names = list(dict.fromkeys(baseline_names))

    S = int(seeds_arr.size)
    T = int(fracs_arr.size)

    n_sup = np.zeros((S, T), dtype=int)
    n_unsup = np.zeros((S, T), dtype=int)
    acc_soft = np.zeros((S, T), dtype=float)
    acc_hard = np.zeros((S, T), dtype=float)
    acc_baselines = {
        name: np.zeros((S, T), dtype=float) for name in baseline_names
    }

    print()
    print("Comparing soft/hard decoding with semi-supervised baselines...")
    it = tqdm(enumerate(seeds_arr), total=len(seeds_arr), desc="SEEDS", leave=True)

    for i, sd in it:
        it.set_description(f"SEED {i}/{len(seeds_arr)}")
        r = run_label_map_sweep_on_U(
            method=method,
            X=X,
            y=y,
            fracs=fracs_arr,
            seed=int(sd),
            stratified=bool(stratified),
            supervised_baselines=baseline_names,
            classes=classes,
            verbose=verbose,
        )

        n_sup[i] = r.n_sup
        n_unsup[i] = r.n_unsup
        acc_soft[i] = r.acc_soft_U
        acc_hard[i] = r.acc_hard_U
        for name in baseline_names:
            acc_baselines[name][i] = r.acc_baselines_U[name]
    
    print("  -> done")


    return LabelMapSweepMultiSeedResult(
        fracs=fracs_arr,
        seeds=seeds_arr,
        n_sup=n_sup,
        n_unsup=n_unsup,
        acc_soft_U=acc_soft,
        acc_hard_U=acc_hard,
        acc_baselines_U=acc_baselines,
        stratified=bool(stratified),
    )