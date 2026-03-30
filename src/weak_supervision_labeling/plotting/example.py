
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .base import Titles, save_close
from weak_supervision_labeling.helpers import as_1d_labels, entropy, infer_img_shape_from_flat
from matplotlib.gridspec import GridSpec


def plot_single_soft_hard_example(
    *,
    X,
    y_true,
    y_hard,
    y_soft,
    dataset,
    savepath,
    titles: Titles = Titles(),
    qc: np.ndarray | None = None,
    class_names: list[str] | None = None,
    topk_qc: int = 5,
    prefer: str = "hard_wrong_soft_right",
    dpi: int = 220,
    minentropy_bits: float = 1.0,
    visible_hard_clusters: np.ndarray | None = None,
):
    

    X = np.asarray(X)
    y_true = as_1d_labels(y_true)
    y_hard = as_1d_labels(y_hard)
    y_soft = as_1d_labels(y_soft)

    if qc is None:
        print(f"  single example: qc is None -> skip {savepath.name}")
        return

    qc = np.asarray(qc, dtype=float)

    if prefer == "hard_wrong_soft_right":
        mask = (y_hard != y_true) & (y_soft == y_true)
        subtitle = "Hard wrong, soft right"
    else:
        mask = (y_soft != y_true) & (y_hard == y_true)
        subtitle = "Soft wrong, hard right"

    idx_all = np.where(mask)[0]
    if idx_all.size == 0:
        print(f"  single example: no samples for '{prefer}' -> skip {savepath.name}")
        return

    def _safe_norm_prob(v):
        v = np.asarray(v, float)
        s = v.sum()
        if s <= 0:
            return np.ones_like(v) / len(v)
        return v / s

    visible_hard_clusters_set = None
    if visible_hard_clusters is not None:
        visible_hard_clusters_set = set(np.asarray(visible_hard_clusters).astype(int).tolist())

    cand = []
    for j in idx_all:
        q = _safe_norm_prob(qc[j])
        q = np.clip(q, 1e-12, 1.0)
        order = np.argsort(-q)
        c_star = int(order[0])

        if visible_hard_clusters_set is not None and c_star not in visible_hard_clusters_set:
            continue

        q1 = float(q[order[0]])
        q2 = float(q[order[1]]) if len(order) > 1 else 0.0
        margin = q1 - q2
        Hq = entropy(q, unit="bits")
        cand.append((j, Hq, margin, q1))

    cand_hi = [c for c in cand if c[1] >= minentropy_bits]
    if len(cand_hi) > 0:
        cand = cand_hi

    if len(cand) == 0:
        print(f"  single example: no candidate after entropy/visibility filters -> skip {savepath.name}")
        return

    cand.sort(key=lambda x: (-x[1], x[2], x[3]))
    j = cand[0][0]

    if X.ndim == 2:
        H, W = infer_img_shape_from_flat(X)
        img = X[j].reshape(H, W)
    else:
        img = X[j]

    q = _safe_norm_prob(qc[j])
    order = np.argsort(-q)
    top = order[:topk_qc]

    q1 = q[top[0]]
    q2 = q[top[1]] if len(top) > 1 else 0
    margin = q1 - q2
    c_star = int(np.argmax(q))

    t = int(y_true[j])
    h = int(y_hard[j])
    s = int(y_soft[j])

    def _name(i):
        if class_names is not None and 0 <= int(i) < len(class_names):
            return str(class_names[int(i)])
        if dataset == "emnist":
            return chr(ord("a") + i)
        return str(i)

    fig = plt.figure(figsize=(12.5, 3.0), dpi=dpi)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 2.0, 2.2], wspace=0.35)

    if titles.title:
        fig.suptitle(titles.title + " — " + subtitle, fontsize=13)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(f"True: {_name(t)}", fontsize=12)

    ax_q = fig.add_subplot(gs[0, 1])
    ax_q.bar(range(len(top)), q[top])
    ax_q.set_ylim(0, 1)
    ax_q.set_xticks(range(len(top)))
    ax_q.set_xticklabels([f"$c_{{{c}}}$" for c in top])
    ax_q.set_title("Cluster posterior $q(c|x)$")
    ax_q.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.axis("off")

    hard_mark = r"$\checkmark$" if (h == t) else r"$\times$"
    soft_mark = r"$\checkmark$" if (s == t) else r"$\times$"

    lines = [
        f"Hard: {_name(h)} {hard_mark}",
        f"Soft: {_name(s)} {soft_mark}",
        "",
        f"Top cluster: $c_{{{c_star}}}$ ({q1:.2f})",
        f"Second best: {q2:.2f}",
        f"Margin: {margin:.2f}",
        "",
        "Soft succeeds because the posterior",
        "is spread across several plausible clusters.",
    ]

    ax_txt.text(0, 0.98, "\n".join(lines), va="top", fontsize=12)
    fig.subplots_adjust(top=0.82)
    save_close(fig, savepath)