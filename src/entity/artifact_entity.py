from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str
    dryad_stats_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    validated_file_path: str
    validation_report_file_path: str


@dataclass
class DataTransformationArtifact:
    preprocessor_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    train_csv_file_path: str
    test_csv_file_path: str
    feature_metadata_file_path: str


@dataclass
class MultiClassMetricArtifact:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    mcc: float
    macro_precision: float
    macro_recall: float
    roc_auc_ovr_macro: Optional[float] = None


@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    best_model_name: str
    trained_model_file_path: str
    unsupervised_model_file_path: str
    model_bundle_file_path: str
    supervised_results_csv: str
    unsupervised_results_csv: str
    cv_results_csv: str
    train_metrics: MultiClassMetricArtifact
    test_metrics: MultiClassMetricArtifact
    per_model_test_metrics: Dict[str, MultiClassMetricArtifact] = field(default_factory=dict)
