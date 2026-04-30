"""Metric helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """ROC-AUC with OOD as the positive class.

    Returns 0.5 if either set is empty (degenerate)."""
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return 0.5
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y_true, y_score))


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    return float((preds == labels).mean())
