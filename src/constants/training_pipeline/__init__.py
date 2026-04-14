import os

"""
Common constants for the BP (Blood Pressure) personalized-alert training pipeline.
Mirrors notebooks/experiment_08.ipynb.
"""

PIPELINE_NAME: str = "BPAlert"
ARTIFACT_DIR: str = "Artifacts"

TARGET_COLUMN: str = "alerttype"
CLASS_ORDER = ["Hypotensive", "Normal", "Hypertensive"]

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
FEATURE_STORE_FILE_NAME: str = "nhanes_merged.csv"
VALIDATED_FILE_NAME: str = "validated.csv"
DRYAD_STATS_FILE_NAME: str = "dryad_stats.json"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR = os.path.join("final_model")
MODEL_FILE_NAME: str = "model.pkl"
PREPROCESSOR_FILE_NAME: str = "preprocessor.pkl"
UNSUPERVISED_FILE_NAME: str = "unsupervised.pkl"
FEATURE_METADATA_FILE_NAME: str = "feature_metadata.json"

OUTPUT_DIR: str = "output"
MODEL_BUNDLE_FILE_NAME: str = "best_model_bundle.joblib"

UNSUPERVISED_READINGS_THRESHOLD: int = 15

"""
Data ingestion
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

NHANES_DIR: str = os.path.join("data", "nhanes")
DRYAD_DIR: str = os.path.join("data", "dryad")

NHANES_BPXO_FILE: str = "BPXO_L.xpt"
NHANES_DEMO_FILE: str = "DEMO_L.xpt"
NHANES_BMX_FILE: str = "BMX_L.xpt"
NHANES_RXQ_FILE: str = "RXQ_RX_L.xpt"

DRYAD_SLEEP_FILE: str = "Blood_Pressure_Sleep_Info.xlsx"
DRYAD_PARTICIPANT_FILE: str = "Participant_Information.csv"

"""
Data validation
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_REPORT_DIR: str = "report"
DATA_VALIDATION_REPORT_FILE_NAME: str = "validation_report.yaml"

REQUIRED_BP_COLUMNS = [
    "BPXOSY1", "BPXOSY2", "BPXOSY3",
    "BPXODI1", "BPXODI2", "BPXODI3",
    "BPXOPLS1", "BPXOPLS2", "BPXOPLS3",
]
REQUIRED_DEMO_COLUMNS = ["RIDAGEYR"]
MIN_AGE: int = 18

SYS_VALID_RANGE = (60, 260)
DIA_VALID_RANGE = (30, 160)

"""
Data transformation
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
TRAIN_TEST_SPLIT_RATIO: float = 0.2
RANDOM_STATE: int = 42

SYS_FLOOR_BASE: float = 5.0
DIA_FLOOR: float = 4.0
SYS_FLOOR_DRYAD_COEF: float = 0.35

HYPO_SYS_ABS: float = 90.0
HYPO_DIA_ABS: float = 60.0
HYPO_SYS_REL: float = 100.0
HYPO_DIA_REL: float = 65.0

HYPER_SYS_ABS: float = 140.0
HYPER_DIA_ABS: float = 90.0
HYPER_SYS_REL: float = 130.0
HYPER_DIA_REL: float = 80.0

Z_LOW: float = -1.0
Z_HIGH: float = 1.0

NUMERIC_FEATURES = [
    "RIDAGEYR", "isfemale", "RIDRETH3", "INDFMPIR", "DMDEDUC2",
    "BMXBMI", "BMXWT", "BMXHT", "rxcount", "antihypertensiveflag",
    "BPXOSY1", "BPXOSY2", "BPXODI1", "BPXODI2", "BPXOPLS1", "BPXOPLS2",
    "sys12mean", "dia12mean", "pulse12mean",
    "sys12std", "dia12std", "pulse12std",
    "systrend21", "diatrend21", "pulsetrend21",
    "pp12", "map12", "syscv12", "diacv12",
    "lowincomeflag",
    "morningsurgeproxy", "nondipperrisk", "circadiandysregulationindex",
]
CATEGORICAL_FEATURES = ["obesitycat"]

DRYAD_DEFAULTS = {
    "meanmorningeveningdiff": 3.5,
    "meandippingratio": 0.88,
    "meansysstd": 9.5,
    "meandiastd": 6.5,
}

"""
Model trainer
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_METRICS_DIR: str = "metrics"
CV_FOLDS: int = 5
