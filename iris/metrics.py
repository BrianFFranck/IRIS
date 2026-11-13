"""How we score Iris.

QWK (quadratic weighted kappa) is the official PANDA metric. It measures
agreement between two graders and punishes big mistakes (ISUP 1 vs 5) much more
than small ones (2 vs 3). 1.0 = perfect, 0 = chance. Top PANDA teams: ~0.94 on
the private test set.

gg1_vs_gg2 gives the boundary Iris is built around its own numbers.
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def qwk(y_true, y_pred) -> float:
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def confusion(y_true, y_pred, n_classes: int = 6) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def gg1_vs_gg2(y_true, y_pred) -> dict:
    """The GG1 vs GG2 boundary (Gleason 3+3 vs 3+4), which often decides
    between monitoring a patient and treating them."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    gg1 = y_true == 1
    gg2 = y_true == 2
    return {
        "n_gg2": int(gg2.sum()),
        "gg2_called_gg2": int((y_pred[gg2] == 2).sum()),
        "gg2_called_gg1": int((y_pred[gg2] == 1).sum()),
        "n_gg1": int(gg1.sum()),
        "gg1_called_gg2": int((y_pred[gg1] == 2).sum()),
    }
