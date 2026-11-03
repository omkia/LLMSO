"""Youden's J and the detector performance rows printed in the manuscript."""

from __future__ import annotations

import numpy as np


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximizing Youden's J = sensitivity + specificity - 1.

    The manuscript selects the detector threshold on a 3,000-row validation
    partition by this rule and reports the selected value 0.72.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if len(y) != len(s) or len(y) == 0:
        raise ValueError("y_true and scores must be non-empty and aligned")
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Youden's J requires both classes")
    best_j = -np.inf
    best_t = float(np.min(s))
    for threshold in np.unique(s):
        pred = s >= threshold
        sens = int(((pred == 1) & (y == 1)).sum()) / n_pos
        spec = int(((pred == 0) & (y == 0)).sum()) / n_neg
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_t = float(threshold)
    return best_t


def binary_report(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if precision == precision and sensitivity == sensitivity and (precision + sensitivity)
          else float("nan"))
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": _auc(y, np.asarray(scores, dtype=float)),
        "reference_positive_rate": float(y.mean()) if len(y) else float("nan"),
        "threshold": float(threshold),
        "n": int(len(y)),
    }


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann–Whitney form of the ROC AUC.
    correct = 0.0
    for p in pos:
        correct += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(correct / (len(pos) * len(neg)))
