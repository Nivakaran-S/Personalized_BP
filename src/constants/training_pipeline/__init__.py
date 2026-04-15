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

# Questionnaire files — clinical history + symptoms
NHANES_BPQ_FILE: str = "BPQ_L.xpt"   # BP / cholesterol history
NHANES_CDQ_FILE: str = "CDQ_L.xpt"   # Cardiovascular symptoms
NHANES_MCQ_FILE: str = "MCQ_L.xpt"   # Medical conditions
NHANES_DIQ_FILE: str = "DIQ_L.xpt"   # Diabetes

# Raw NHANES code columns pulled from each questionnaire (1=Yes, 2=No, 7/9=Refused/DK)
NHANES_BPQ_CODE_COLS = ["BPQ020", "BPQ080", "BPQ050A"]
NHANES_CDQ_CODE_COLS = ["CDQ001", "CDQ008", "CDQ010"]
NHANES_MCQ_CODE_COLS = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]
NHANES_DIQ_CODE_COLS = ["DIQ010"]

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

# Tighter ACC/AHA goals for high-risk patients (diagnosed HTN, on meds, cardiac history, diabetes)
HIGH_RISK_HYPO_SYS_ABS: float = 95.0
HIGH_RISK_HYPO_DIA_ABS: float = 65.0
HIGH_RISK_HYPO_SYS_REL: float = 105.0
HIGH_RISK_HYPO_DIA_REL: float = 70.0
HIGH_RISK_HYPER_SYS_ABS: float = 130.0
HIGH_RISK_HYPER_DIA_ABS: float = 80.0
HIGH_RISK_HYPER_SYS_REL: float = 125.0
HIGH_RISK_HYPER_DIA_REL: float = 75.0

# Symptom-bump cutoffs: if the reading is at/above these AND the patient reports acute
# chest pain, a "Normal" label escalates to "Hypertensive".
SYMPTOM_BUMP_SYS_CUTOFF: float = 125.0
SYMPTOM_BUMP_DIA_CUTOFF: float = 75.0

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
    # Clinical history + symptoms (from BPQ/CDQ/MCQ/DIQ questionnaires)
    "has_diagnosed_htn", "on_antihypertensive", "has_high_cholesterol",
    "chest_pain_flag", "sob_on_exertion_flag", "severe_chest_pain_flag",
    "has_mi", "has_stroke", "has_heart_failure", "has_chd", "has_angina",
    "has_diabetes",
    "cardiac_history_count", "acute_symptom_count", "high_risk_profile",
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
