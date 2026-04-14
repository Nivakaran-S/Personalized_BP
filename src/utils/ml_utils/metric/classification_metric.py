"""Multiclass evaluation metrics used by the BP trainer (mirrors experiment_08.ipynb)."""
from __future__ import annotations

import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.entity.artifact_entity import MultiClassMetricArtifact
from src.exception.exception import BPException


def get_multiclass_score(
    y_true, y_pred, y_prob: Optional[np.ndarray] = None
) -> MultiClassMetricArtifact:
    try:
        roc_auc = None
        if y_prob is not None:
            try:
                roc_auc = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
            except (ValueError, IndexError):
                roc_auc = None
        return MultiClassMetricArtifact(
            accuracy=float(accuracy_score(y_true, y_pred)),
            balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
            macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            mcc=float(matthews_corrcoef(y_true, y_pred)),
            macro_precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            macro_recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            roc_auc_ovr_macro=roc_auc,
        )
    except Exception as exc:  # noqa: BLE001
        raise BPException(exc, sys)


def per_class_metrics(
    y_true, y_pred, class_order: List[str]
) -> pd.DataFrame:
    """Precision / recall / specificity / support per class."""
    labels = list(range(len(class_order)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    total = cm.sum()
    for i, cls in enumerate(class_order):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append({
            "class": cls,
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "support": int(tp + fn),
        })
    return pd.DataFrame(rows)


def get_classification_score(y_true, y_pred) -> Dict[str, float]:
    """Legacy binary helper kept for back-compat; returns plain dict."""
    return {
        "f1_score": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision_score": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall_score": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
    }
