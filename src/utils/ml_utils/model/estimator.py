"""Model wrappers used by the training pipeline and the inference API."""
from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from src.exception.exception import BPException


class BPModel:
    """Bundle of (preprocessor, sklearn estimator) for supervised prediction."""

    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, x):
        try:
            x_t = self.preprocessor.transform(x)
            return self.model.predict(x_t)
        except Exception as exc:  # noqa: BLE001
            raise BPException(exc, sys)

    def predict_proba(self, x):
        try:
            x_t = self.preprocessor.transform(x)
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(x_t)
            return None
        except Exception as exc:  # noqa: BLE001
            raise BPException(exc, sys)


class UnsupervisedBPModel:
    """
    Patient-history anomaly detector: IsolationForest + sys/dia median split.
    Intended for long BP histories (>15 readings) where the supervised model's
    3-reading personalization is not applicable.
    """

    def __init__(
        self,
        isolation_forest,
        kmeans,
        cluster_to_class_map: Dict[int, str],
        sys_median: float,
        dia_median: float,
        feature_columns: Optional[list] = None,
    ):
        self.isolation_forest = isolation_forest
        self.kmeans = kmeans
        self.cluster_to_class_map = cluster_to_class_map
        self.sys_median = float(sys_median)
        self.dia_median = float(dia_median)
        self.feature_columns = feature_columns or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "isolation_forest": self.isolation_forest,
            "kmeans": self.kmeans,
            "cluster_to_class_map": self.cluster_to_class_map,
            "sys_median": self.sys_median,
            "dia_median": self.dia_median,
            "feature_columns": self.feature_columns,
        }
