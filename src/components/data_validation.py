"""
Validate the merged NHANES dataframe: required columns, pregnancy exclusion,
age gate, BP value ranges. Writes a YAML report summarising filter effects.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.constants import training_pipeline as C
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception.exception import BPException
from src.logging.logger import logging
from src.utils.main_utils.utils import write_yaml_file


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    @staticmethod
    def _validate_required_columns(df: pd.DataFrame) -> Dict[str, bool]:
        required = C.REQUIRED_BP_COLUMNS + C.REQUIRED_DEMO_COLUMNS
        return {col: (col in df.columns) for col in required}

    @staticmethod
    def _clip_ranges(df: pd.DataFrame) -> pd.DataFrame:
        for col in ["BPXOSY1", "BPXOSY2", "BPXOSY3"]:
            if col in df.columns:
                df[col] = df[col].where(
                    df[col].between(*C.SYS_VALID_RANGE), other=np.nan
                )
        for col in ["BPXODI1", "BPXODI2", "BPXODI3"]:
            if col in df.columns:
                df[col] = df[col].where(
                    df[col].between(*C.DIA_VALID_RANGE), other=np.nan
                )
        return df

    @staticmethod
    def _questionnaire_coverage(df: pd.DataFrame) -> Dict[str, float]:
        """Report the % of rows with a non-null answer for each questionnaire column."""
        cols = (
            C.NHANES_BPQ_CODE_COLS
            + C.NHANES_CDQ_CODE_COLS
            + C.NHANES_MCQ_CODE_COLS
            + C.NHANES_DIQ_CODE_COLS
        )
        total = max(len(df), 1)
        return {
            c: round(float(df[c].notna().sum()) / total * 100.0, 2)
            for c in cols
            if c in df.columns
        }

    def _apply_filters(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        counts: Dict[str, int] = {"start": len(df)}

        if "RIDEXPRG" in df.columns:
            df = df[df["RIDEXPRG"].fillna(0) != 1]
            counts["after_pregnancy_exclusion"] = len(df)

        if "RIDAGEYR" in df.columns:
            df = df[df["RIDAGEYR"] >= C.MIN_AGE]
            counts["after_age_gate"] = len(df)

        required = [c for c in C.REQUIRED_BP_COLUMNS + C.REQUIRED_DEMO_COLUMNS if c in df.columns]
        df = df.dropna(subset=required)
        counts["after_required_nonnull"] = len(df)
        return df, counts

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation.")
            df = self.read_data(self.data_ingestion_artifact.feature_store_file_path)

            column_checks = self._validate_required_columns(df)
            missing = [k for k, ok in column_checks.items() if not ok]
            if missing:
                raise BPException(
                    ValueError(f"Missing required columns: {missing}"), sys
                )

            df = self._clip_ranges(df)
            df, counts = self._apply_filters(df)

            os.makedirs(os.path.dirname(self.data_validation_config.validated_file_path), exist_ok=True)
            df.to_csv(self.data_validation_config.validated_file_path, index=False)

            coverage = self._questionnaire_coverage(df)
            logging.info("Questionnaire coverage: %s", coverage)

            report = {
                "required_columns": column_checks,
                "row_counts": counts,
                "final_rows": int(len(df)),
                "questionnaire_coverage_pct": coverage,
            }
            write_yaml_file(self.data_validation_config.validation_report_file_path, report, replace=True)

            artifact = DataValidationArtifact(
                validation_status=True,
                validated_file_path=self.data_validation_config.validated_file_path,
                validation_report_file_path=self.data_validation_config.validation_report_file_path,
            )
            logging.info("Data validation artifact: %s", artifact)
            return artifact
        except BPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BPException(exc, sys)
