"""Orchestrates the four pipeline stages against local NHANES/Dryad data."""
from __future__ import annotations

import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from src.exception.exception import BPException
from src.logging.logger import logging


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            cfg = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data ingestion")
            return DataIngestion(data_ingestion_config=cfg).initiate_data_ingestion()
        except Exception as exc:  # noqa: BLE001
            raise BPException(exc, sys)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            cfg = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data validation")
            return DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=cfg,
            ).initiate_data_validation()
        except Exception as exc:  # noqa: BLE001
            raise BPException(exc, sys)

    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            cfg = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data transformation")
            return DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=cfg,
            ).initiate_data_transformation()
        except Exception as exc:  # noqa: BLE001
            raise BPException(exc, sys)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            cfg = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start model training")
            return ModelTrainer(
                model_trainer_config=cfg,
                data_transformation_artifact=data_transformation_artifact,
            ).initiate_model_trainer()
        except Exception as exc:  # noqa: BLE001
            raise BPException(exc, sys)

    def run_pipeline(self) -> ModelTrainerArtifact:
        try:
            ing = self.start_data_ingestion()
            val = self.start_data_validation(ing)
            trn = self.start_data_transformation(val)
            return self.start_model_trainer(trn)
        except Exception as exc:  # noqa: BLE001
            raise BPException(exc, sys)
