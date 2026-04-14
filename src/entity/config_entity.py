import os
from datetime import datetime

from src.constants import training_pipeline


class TrainingPipelineConfig:
    def __init__(self, timestamp: datetime = None):
        timestamp = (timestamp or datetime.now()).strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.model_dir = training_pipeline.SAVED_MODEL_DIR
        self.output_dir = training_pipeline.OUTPUT_DIR
        self.timestamp: str = timestamp


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME,
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FEATURE_STORE_FILE_NAME,
        )
        self.dryad_stats_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.DRYAD_STATS_FILE_NAME,
        )
        self.nhanes_dir: str = training_pipeline.NHANES_DIR
        self.dryad_dir: str = training_pipeline.DRYAD_DIR
        self.nhanes_bpxo_file: str = training_pipeline.NHANES_BPXO_FILE
        self.nhanes_demo_file: str = training_pipeline.NHANES_DEMO_FILE
        self.nhanes_bmx_file: str = training_pipeline.NHANES_BMX_FILE
        self.nhanes_rxq_file: str = training_pipeline.NHANES_RXQ_FILE
        self.dryad_sleep_file: str = training_pipeline.DRYAD_SLEEP_FILE
        self.dryad_participant_file: str = training_pipeline.DRYAD_PARTICIPANT_FILE


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME,
        )
        self.validated_dir: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        self.validated_file_path: str = os.path.join(
            self.validated_dir, training_pipeline.VALIDATED_FILE_NAME
        )
        self.validation_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME,
        )
        transformed_dir = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        )
        transformed_obj_dir = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
        )
        self.transformed_train_file_path: str = os.path.join(transformed_dir, "train.npy")
        self.transformed_test_file_path: str = os.path.join(transformed_dir, "test.npy")
        self.train_csv_file_path: str = os.path.join(transformed_dir, "train.csv")
        self.test_csv_file_path: str = os.path.join(transformed_dir, "test.csv")
        self.preprocessor_file_path: str = os.path.join(
            transformed_obj_dir, training_pipeline.PREPROCESSOR_FILE_NAME
        )
        self.feature_metadata_file_path: str = os.path.join(
            transformed_obj_dir, training_pipeline.FEATURE_METADATA_FILE_NAME
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME,
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline.MODEL_FILE_NAME,
        )
        self.unsupervised_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline.UNSUPERVISED_FILE_NAME,
        )
        self.metrics_dir: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_METRICS_DIR
        )
        self.model_bundle_file_path: str = os.path.join(
            training_pipeline_config.output_dir,
            training_pipeline.MODEL_BUNDLE_FILE_NAME,
        )
        self.saved_model_dir: str = training_pipeline_config.model_dir
        self.output_dir: str = training_pipeline_config.output_dir
