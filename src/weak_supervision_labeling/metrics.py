# scr/weak_supervision_labeling/metrics.py

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, adjusted_mutual_info_score, homogeneity_completeness_v_measure


def cluster_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)  # (C,K)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)

def cluster_accuracy_soft(qc, y_true):
    """
    qc: (N,K)
    returns expected many-to-one accuracy
    """
    y_true = np.asarray(y_true).astype(int)
    qc = np.asarray(qc, dtype=float)
    N, K = qc.shape
    C = int(y_true.max() + 1)

    cm = np.zeros((C, K))
    for y in range(C):
        cm[y] = qc[y_true == y].sum(axis=0)

    return np.sum(np.max(cm, axis=0)) / np.sum(cm)


# ACC, NMI, AMI, Homogeneity, Completeness, V-measure, ARI
def clustering_metrics(y_true, y_pred):
    h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)

    return {
        "ACC": cluster_accuracy(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "AMI": adjusted_mutual_info_score(y_true, y_pred),
        "Homogeneity": h,
        "Completeness": c, 
        "V-measure": v,
        "ARI": adjusted_rand_score(y_true, y_pred),
    }

