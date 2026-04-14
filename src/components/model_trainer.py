"""
Train 4 supervised classifiers (LR, RF, ET, XGB) with 5-fold stratified CV,
pick the best by macro-F1 (tiebreak balanced accuracy), then fit unsupervised
baselines (KMeans + IsolationForest) plus a rule-based baseline. Saves all
artifacts the notebook produces and writes the final_model/ bundle used by app.py.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

from src.constants import training_pipeline as C
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    MultiClassMetricArtifact,
)
from src.entity.config_entity import ModelTrainerConfig
from src.exception.exception import BPException
from src.logging.logger import logging
from src.utils.main_utils.utils import load_numpy_array_data, load_object, save_object
from src.utils.ml_utils.feature_engineering import second_reading_rules_bp
from src.utils.ml_utils.metric.classification_metric import (
    get_multiclass_score,
    per_class_metrics,
)
from src.utils.ml_utils.model.estimator import BPModel, UnsupervisedBPModel


def _build_supervised_models(num_classes: int = 3):
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=C.RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=C.RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=C.RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_alpha=0.2,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            random_state=C.RANDOM_STATE,
            n_jobs=-1,
        )
    return models


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    counts = Counter(y.tolist())
    total = len(y)
    n_classes = len(counts)
    return np.array([total / (n_classes * counts[int(v)]) for v in y])


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def _load_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
        test = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
        return train[:, :-1], train[:, -1].astype(int), test[:, :-1], test[:, -1].astype(int)

    def _cross_validate(
        self, models: Dict[str, object], X: np.ndarray, y: np.ndarray
    ) -> pd.DataFrame:
        skf = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
        scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"}
        rows = []
        for name, model in models.items():
            try:
                cv_kwargs = dict(cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False)
                if name == "XGBoost":
                    sw = _balanced_sample_weight(y)
                    try:
                        scores = cross_validate(model, X, y, params={"sample_weight": sw}, **cv_kwargs)
                    except TypeError:
                        scores = cross_validate(model, X, y, fit_params={"sample_weight": sw}, **cv_kwargs)
                else:
                    scores = cross_validate(model, X, y, **cv_kwargs)
                rows.append({
                    "model": name,
                    "cv_macro_f1_mean": float(np.mean(scores["test_macro_f1"])),
                    "cv_macro_f1_std": float(np.std(scores["test_macro_f1"])),
                    "cv_balanced_accuracy_mean": float(np.mean(scores["test_balanced_accuracy"])),
                })
            except Exception as exc:  # noqa: BLE001
                logging.warning("CV failed for %s: %s", name, exc)
                rows.append({
                    "model": name,
                    "cv_macro_f1_mean": 0.0,
                    "cv_macro_f1_std": 0.0,
                    "cv_balanced_accuracy_mean": 0.0,
                })
        return pd.DataFrame(rows)

    def _fit_supervised(
        self, models: Dict[str, object], X: np.ndarray, y: np.ndarray
    ) -> Dict[str, object]:
        fitted = {}
        for name, model in models.items():
            if name == "XGBoost":
                sw = _balanced_sample_weight(y)
                model.fit(X, y, sample_weight=sw)
            else:
                model.fit(X, y)
            fitted[name] = model
        return fitted

    def _evaluate_supervised(
        self,
        fitted: Dict[str, object],
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_order: List[str],
    ) -> Tuple[pd.DataFrame, Dict[str, MultiClassMetricArtifact], Dict[str, pd.DataFrame]]:
        rows = []
        metrics_map: Dict[str, MultiClassMetricArtifact] = {}
        per_class_map: Dict[str, pd.DataFrame] = {}
        for name, model in fitted.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            metric = get_multiclass_score(y_test, y_pred, y_prob)
            metrics_map[name] = metric
            per_class_map[name] = per_class_metrics(y_test, y_pred, class_order)
            rows.append({"model": name, **metric.__dict__})
        return pd.DataFrame(rows), metrics_map, per_class_map

    def _fit_unsupervised(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> UnsupervisedBPModel:
        # The ColumnTransformer output is scaled numeric + one-hot cats; K-Means/IsoForest still work.
        kmeans = KMeans(n_clusters=3, random_state=C.RANDOM_STATE, n_init=20)
        kmeans.fit(X_train)

        cluster_labels = kmeans.predict(X_train)
        mapping: Dict[int, int] = {}
        for cluster_id in range(3):
            mask = cluster_labels == cluster_id
            if mask.any():
                majority = int(Counter(y_train[mask].tolist()).most_common(1)[0][0])
                mapping[cluster_id] = majority
            else:
                mapping[cluster_id] = 1  # Normal fallback

        # Non-normal rate drives IsolationForest contamination.
        normal_idx = C.CLASS_ORDER.index("Normal") if "Normal" in C.CLASS_ORDER else 1
        non_normal_rate = float(np.mean(y_train != normal_idx))
        contamination = float(np.clip(non_normal_rate, 0.05, 0.45))
        iso = IsolationForest(
            n_estimators=400,
            contamination=contamination,
            random_state=C.RANDOM_STATE,
            n_jobs=-1,
        )
        iso.fit(X_train)

        # Medians for sys/dia split — read from the transformed train CSV.
        train_csv = self.data_transformation_artifact.train_csv_file_path
        df = pd.read_csv(train_csv)
        sys_median = float(df["sys12mean"].median()) if "sys12mean" in df.columns else 120.0
        dia_median = float(df["dia12mean"].median()) if "dia12mean" in df.columns else 80.0

        class_order = C.CLASS_ORDER
        cluster_to_class = {k: class_order[v] for k, v in mapping.items()}
        return UnsupervisedBPModel(
            isolation_forest=iso,
            kmeans=kmeans,
            cluster_to_class_map=cluster_to_class,
            sys_median=sys_median,
            dia_median=dia_median,
        )

    def _evaluate_unsupervised(
        self,
        unsup: UnsupervisedBPModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_order: List[str],
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        rows = []
        per_class_map: Dict[str, pd.DataFrame] = {}

        kmeans_clusters = unsup.kmeans.predict(X_test)
        kmeans_pred = np.array(
            [class_order.index(unsup.cluster_to_class_map[int(c)]) for c in kmeans_clusters]
        )
        kmeans_metric = get_multiclass_score(y_test, kmeans_pred)
        rows.append({"model": "KMeans3", **kmeans_metric.__dict__})
        per_class_map["KMeans3"] = per_class_metrics(y_test, kmeans_pred, class_order)

        # IsolationForest: anomaly → split by sys/dia mean; we approximate using sys12mean/dia12mean from the test CSV.
        test_csv = self.data_transformation_artifact.test_csv_file_path
        test_df = pd.read_csv(test_csv)
        iso_preds = unsup.isolation_forest.predict(X_test)
        sys_mean_col = test_df["sys12mean"].values if "sys12mean" in test_df.columns else np.full(len(X_test), 120.0)
        dia_mean_col = test_df["dia12mean"].values if "dia12mean" in test_df.columns else np.full(len(X_test), 80.0)

        normal_idx = class_order.index("Normal")
        hypo_idx = class_order.index("Hypotensive")
        hyper_idx = class_order.index("Hypertensive")

        iso_final = np.full(len(X_test), normal_idx)
        for i, p in enumerate(iso_preds):
            if p == -1:
                if sys_mean_col[i] >= unsup.sys_median or dia_mean_col[i] >= unsup.dia_median:
                    iso_final[i] = hyper_idx
                else:
                    iso_final[i] = hypo_idx
        iso_metric = get_multiclass_score(y_test, iso_final)
        rows.append({"model": "IsolationForestSplit", **iso_metric.__dict__})
        per_class_map["IsolationForestSplit"] = per_class_metrics(y_test, iso_final, class_order)

        return pd.DataFrame(rows), per_class_map

    def _evaluate_rule_baseline(
        self, X_test: np.ndarray, y_test: np.ndarray, class_order: List[str]
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        test_csv = self.data_transformation_artifact.test_csv_file_path
        df = pd.read_csv(test_csv)
        preds = []
        for _, row in df.iterrows():
            sbp = float(row["BPXOSY2"]) if "BPXOSY2" in row else float(row["sys12mean"])
            dbp = float(row["BPXODI2"]) if "BPXODI2" in row else float(row["dia12mean"])
            preds.append(class_order.index(second_reading_rules_bp(sbp, dbp)))
        preds = np.array(preds)
        metric = get_multiclass_score(y_test, preds)
        return metric.__dict__, per_class_metrics(y_test, preds, class_order)

    def _select_best(self, cv_df: pd.DataFrame) -> str:
        ranked = cv_df.sort_values(
            by=["cv_macro_f1_mean", "cv_balanced_accuracy_mean"],
            ascending=False,
        )
        return str(ranked.iloc[0]["model"])

    @staticmethod
    def _feature_importance(model, feature_names: List[str], X_train, y_train) -> pd.DataFrame:
        try:
            if hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_)
            elif hasattr(model, "coef_"):
                coef = np.asarray(model.coef_)
                imp = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
            else:
                result = permutation_importance(
                    model, X_train, y_train, n_repeats=5, random_state=C.RANDOM_STATE, n_jobs=-1
                )
                imp = result.importances_mean
            n = min(len(feature_names), len(imp))
            df = pd.DataFrame({
                "feature": feature_names[:n],
                "importance": imp[:n],
            }).sort_values("importance", ascending=False)
            return df
        except Exception as exc:  # noqa: BLE001
            logging.warning("Feature importance failed: %s", exc)
            return pd.DataFrame(columns=["feature", "importance"])

    def _save_outputs(
        self,
        best_name: str,
        best_model,
        preprocessor,
        unsup: UnsupervisedBPModel,
        feature_metadata: Dict,
        cv_df: pd.DataFrame,
        supervised_df: pd.DataFrame,
        unsupervised_df: pd.DataFrame,
        per_class_maps: Dict[str, pd.DataFrame],
        rule_per_class: pd.DataFrame,
        feature_importance_df: pd.DataFrame,
    ) -> Tuple[str, str, str]:
        os.makedirs(self.model_trainer_config.metrics_dir, exist_ok=True)
        supervised_csv = os.path.join(self.model_trainer_config.metrics_dir, "supervised_results_multiclass.csv")
        unsupervised_csv = os.path.join(self.model_trainer_config.metrics_dir, "unsupervised_results_multiclass.csv")
        cv_csv = os.path.join(self.model_trainer_config.metrics_dir, "cv_results_multiclass.csv")
        fi_csv = os.path.join(self.model_trainer_config.metrics_dir, "feature_importance.csv")
        supervised_df.to_csv(supervised_csv, index=False)
        unsupervised_df.to_csv(unsupervised_csv, index=False)
        cv_df.to_csv(cv_csv, index=False)
        feature_importance_df.to_csv(fi_csv, index=False)

        for name, df in per_class_maps.items():
            path = os.path.join(self.model_trainer_config.metrics_dir, f"{name.lower()}_perclass_metrics.csv")
            df.to_csv(path, index=False)
        rule_per_class.to_csv(
            os.path.join(self.model_trainer_config.metrics_dir, "rule_baseline_perclass_metrics.csv"),
            index=False,
        )

        # final_model/ artifacts that app.py loads
        os.makedirs(self.model_trainer_config.saved_model_dir, exist_ok=True)
        save_object(os.path.join(self.model_trainer_config.saved_model_dir, C.MODEL_FILE_NAME), best_model)
        save_object(
            os.path.join(self.model_trainer_config.saved_model_dir, C.UNSUPERVISED_FILE_NAME),
            unsup.to_dict(),
        )
        with open(
            os.path.join(self.model_trainer_config.saved_model_dir, C.FEATURE_METADATA_FILE_NAME),
            "w",
        ) as f:
            json.dump(feature_metadata, f, indent=2)

        # Artifacts/.../trained_model copy
        os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
        save_object(self.model_trainer_config.trained_model_file_path, BPModel(preprocessor, best_model))
        save_object(self.model_trainer_config.unsupervised_model_file_path, unsup.to_dict())

        # Unified bundle (joblib)
        os.makedirs(os.path.dirname(self.model_trainer_config.model_bundle_file_path), exist_ok=True)
        joblib.dump(
            {
                "best_model_name": best_name,
                "model": best_model,
                "preprocessor": preprocessor,
                "unsupervised": unsup.to_dict(),
                "feature_metadata": feature_metadata,
            },
            self.model_trainer_config.model_bundle_file_path,
        )
        return supervised_csv, unsupervised_csv, cv_csv

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            X_train, y_train, X_test, y_test = self._load_arrays()
            class_order = C.CLASS_ORDER
            feature_names = C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES

            models = _build_supervised_models(num_classes=len(class_order))
            logging.info("Running %d-fold CV for %d supervised models.", C.CV_FOLDS, len(models))
            cv_df = self._cross_validate(models, X_train, y_train)
            logging.info("\n%s", cv_df.to_string(index=False))

            fitted = self._fit_supervised(models, X_train, y_train)
            supervised_df, metrics_map, per_class_maps = self._evaluate_supervised(
                fitted, X_test, y_test, class_order
            )

            best_name = self._select_best(cv_df)
            best_model = fitted[best_name]
            logging.info("Best supervised model: %s", best_name)

            unsup = self._fit_unsupervised(X_train, y_train)
            unsupervised_df, unsup_per_class = self._evaluate_unsupervised(
                unsup, X_test, y_test, class_order
            )
            per_class_maps.update(unsup_per_class)

            rule_metric_dict, rule_per_class = self._evaluate_rule_baseline(X_test, y_test, class_order)
            unsupervised_df = pd.concat(
                [unsupervised_df, pd.DataFrame([{"model": "SecondReadingRules", **rule_metric_dict}])],
                ignore_index=True,
            )

            # Train set metrics for the best model
            y_train_pred = best_model.predict(X_train)
            y_train_prob = (
                best_model.predict_proba(X_train) if hasattr(best_model, "predict_proba") else None
            )
            train_metric = get_multiclass_score(y_train, y_train_pred, y_train_prob)
            test_metric = metrics_map[best_name]

            preprocessor = load_object(self.data_transformation_artifact.preprocessor_file_path)
            with open(self.data_transformation_artifact.feature_metadata_file_path) as f:
                feature_metadata = json.load(f)
            feature_metadata["best_model_name"] = best_name

            fi_df = self._feature_importance(best_model, feature_names, X_train, y_train)
            supervised_csv, unsupervised_csv, cv_csv = self._save_outputs(
                best_name, best_model, preprocessor, unsup, feature_metadata,
                cv_df, supervised_df, unsupervised_df, per_class_maps, rule_per_class, fi_df,
            )

            artifact = ModelTrainerArtifact(
                best_model_name=best_name,
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                unsupervised_model_file_path=self.model_trainer_config.unsupervised_model_file_path,
                model_bundle_file_path=self.model_trainer_config.model_bundle_file_path,
                supervised_results_csv=supervised_csv,
                unsupervised_results_csv=unsupervised_csv,
                cv_results_csv=cv_csv,
                train_metrics=train_metric,
                test_metrics=test_metric,
                per_model_test_metrics=metrics_map,
            )
            logging.info("Model trainer artifact: %s", artifact)
            return artifact
        except BPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BPException(exc, sys)
