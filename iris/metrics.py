"""How we score Iris.

QWK (quadratic weighted kappa) is the official PANDA metric. It measures
agreement between two graders and punishes big mistakes (ISUP 1 vs 5) much more
than small ones (2 vs 3). 1.0 = perfect, 0 = chance. Top PANDA teams: ~0.94 on
the private test set. A first honest baseline at level 1 is often ~0.75-0.85.
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def qwk(y_true, y_pred) -> float:
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def confusion(y_true, y_pred, n_classes: int = 6) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def isup2_vs_3(y_true, y_pred) -> dict:
    """The clinically painful boundary (3+4 vs 4+3).

    TODO: restrict to slides whose TRUE grade is 2 or 3, and report:
      - accuracy on just those slides
      - how many true-2 were called 3 and how many true-3 were called 2
      - (bonus) the same thing split by data_provider
    This number is the story of your write-up, so make it clear and honest.
    """
    raise NotImplementedError
