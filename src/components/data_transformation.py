"""
Feature engineering + personalized alert target + stratified train/test split +
ColumnTransformer preprocessor. Mirrors the model-prep cells of experiment_08.ipynb.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.constants import training_pipeline as C
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import BPException
from src.logging.logger import logging
from src.utils.main_utils.utils import save_numpy_array_data, save_object
from src.utils.ml_utils.feature_engineering import (
    add_normalization_and_proxy_features,
    compute_normalization_stats,
    compute_sys_floor,
    engineer_features,
    make_personalized_alert_type,
)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    @staticmethod
    def _load_dryad_stats(dryad_stats_path: str) -> Dict[str, float]:
        if dryad_stats_path and os.path.exists(dryad_stats_path):
            with open(dryad_stats_path, "r") as f:
                return json.load(f)
        return dict(C.DRYAD_DEFAULTS)

    @staticmethod
    def _build_target(df: pd.DataFrame, sys_floor: float, dia_floor: float) -> pd.Series:
        return df.apply(
            lambda row: make_personalized_alert_type(row, sys_floor=sys_floor, dia_floor=dia_floor),
            axis=1,
        )

    @staticmethod
    def _make_preprocessor() -> ColumnTransformer:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, C.NUMERIC_FEATURES),
                ("cat", categorical_pipe, C.CATEGORICAL_FEATURES),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation.")
            df = pd.read_csv(self.data_validation_artifact.validated_file_path)

            dryad_stats = self._load_dryad_stats(
                os.path.join(
                    os.path.dirname(os.path.dirname(self.data_validation_artifact.validated_file_path)),
                    "..",
                    "data_ingestion",
                    "feature_store",
                    C.DRYAD_STATS_FILE_NAME,
                )
            )
            # Fall back to defaults if path resolution missed; also look in artifacts layout.
            if dryad_stats == dict(C.DRYAD_DEFAULTS):
                candidate = self._find_dryad_stats()
                if candidate:
                    dryad_stats = self._load_dryad_stats(candidate)

            df = engineer_features(df, dryad_stats)

            sys_floor = compute_sys_floor(dryad_stats)
            dia_floor = float(C.DIA_FLOOR)
            logging.info("Target floors: sys=%.3f, dia=%.3f", sys_floor, dia_floor)

            # Build target BEFORE normalization proxies (proxies depend on target-independent features).
            df[C.TARGET_COLUMN] = self._build_target(df, sys_floor=sys_floor, dia_floor=dia_floor)
            df = df.dropna(subset=[C.TARGET_COLUMN])
            logging.info(
                "After target build: %s rows; class mix: %s",
                len(df), df[C.TARGET_COLUMN].value_counts().to_dict(),
            )

            y = df[C.TARGET_COLUMN]
            X_pre = df.drop(columns=[C.TARGET_COLUMN])

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_pre, y,
                test_size=C.TRAIN_TEST_SPLIT_RATIO,
                random_state=C.RANDOM_STATE,
                stratify=y,
            )

            norm_stats = compute_normalization_stats(X_train_raw)
            X_train = add_normalization_and_proxy_features(X_train_raw, norm_stats, dryad_stats)
            X_test = add_normalization_and_proxy_features(X_test_raw, norm_stats, dryad_stats)

            feature_cols = C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES
            X_train = X_train[feature_cols]
            X_test = X_test[feature_cols]

            class_order = C.CLASS_ORDER
            y_train_idx = y_train.map({cls: i for i, cls in enumerate(class_order)}).astype(int).to_numpy()
            y_test_idx = y_test.map({cls: i for i, cls in enumerate(class_order)}).astype(int).to_numpy()

            preprocessor = self._make_preprocessor()
            X_train_trans = preprocessor.fit_transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            if hasattr(X_train_trans, "toarray"):
                X_train_trans = X_train_trans.toarray()
                X_test_trans = X_test_trans.toarray()

            train_arr = np.c_[np.asarray(X_train_trans, dtype=float), y_train_idx]
            test_arr = np.c_[np.asarray(X_test_trans, dtype=float), y_test_idx]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            os.makedirs(os.path.dirname(self.data_transformation_config.train_csv_file_path), exist_ok=True)
            X_train.assign(**{C.TARGET_COLUMN: y_train.values}).to_csv(
                self.data_transformation_config.train_csv_file_path, index=False
            )
            X_test.assign(**{C.TARGET_COLUMN: y_test.values}).to_csv(
                self.data_transformation_config.test_csv_file_path, index=False
            )

            save_object(self.data_transformation_config.preprocessor_file_path, preprocessor)
            save_object(os.path.join(C.SAVED_MODEL_DIR, C.PREPROCESSOR_FILE_NAME), preprocessor)

            feature_metadata = {
                "numeric_features": C.NUMERIC_FEATURES,
                "categorical_features": C.CATEGORICAL_FEATURES,
                "class_order": class_order,
                "norm_stats": norm_stats,
                "dryad_stats": dryad_stats,
                "sys_floor": float(sys_floor),
                "dia_floor": float(dia_floor),
            }
            os.makedirs(os.path.dirname(self.data_transformation_config.feature_metadata_file_path), exist_ok=True)
            with open(self.data_transformation_config.feature_metadata_file_path, "w") as f:
                json.dump(feature_metadata, f, indent=2)
            os.makedirs(C.SAVED_MODEL_DIR, exist_ok=True)
            with open(os.path.join(C.SAVED_MODEL_DIR, C.FEATURE_METADATA_FILE_NAME), "w") as f:
                json.dump(feature_metadata, f, indent=2)

            artifact = DataTransformationArtifact(
                preprocessor_file_path=self.data_transformation_config.preprocessor_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                train_csv_file_path=self.data_transformation_config.train_csv_file_path,
                test_csv_file_path=self.data_transformation_config.test_csv_file_path,
                feature_metadata_file_path=self.data_transformation_config.feature_metadata_file_path,
            )
            logging.info("Data transformation artifact: %s", artifact)
            return artifact
        except BPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BPException(exc, sys)

    @staticmethod
    def _find_dryad_stats() -> str:
        """Locate dryad_stats.json under the current artifacts tree (robust to pathing)."""
        for root, _dirs, files in os.walk(C.ARTIFACT_DIR):
            if C.DRYAD_STATS_FILE_NAME in files:
                return os.path.join(root, C.DRYAD_STATS_FILE_NAME)
        return ""
