import os

# ==========================
# Project Configuration
# ==========================

ARTIFACT_DIR = "artifacts"

DATASET_PATH = os.path.join("datasets", "clean_ai4i_ml.csv")

RAW_DATA_PATH    = os.path.join(ARTIFACT_DIR, "raw.csv")
TRAIN_DATA_PATH  = os.path.join(ARTIFACT_DIR, "train.csv")
TEST_DATA_PATH   = os.path.join(ARTIFACT_DIR, "test.csv")

MODEL_PATH       = os.path.join("saved_models", "model.pkl")
PREPROCESSOR_PATH = os.path.join("saved_models", "preprocessor.pkl")

EVALUATION_REPORT_PATH = os.path.join(ARTIFACT_DIR, "evaluation_report.json")
VALIDATION_REPORT_PATH = os.path.join(ARTIFACT_DIR, "validation_report.json")
PLOTS_DIR        = os.path.join(ARTIFACT_DIR, "plots")

# ==========================
# Target & Feature Config
# ==========================

TARGET_COLUMN = "Machine failure"

# Columns to drop (IDs or data-leakage columns)
DROP_COLUMNS = ["Product ID"]

# Numerical feature columns (after dropping IDs)
NUMERICAL_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "TWF", "HDF", "PWF", "OSF", "RNF",
    # engineered features added at runtime:
    "Temp_diff",
    "Power",
    "Wear_rate",
]

# ==========================
# Split / Sampling Config
# ==========================

TEST_SIZE    = 0.20
RANDOM_STATE = 42

# SMOTE settings
SMOTE_SAMPLING_STRATEGY = 0.5   # minority : majority ratio after resampling
SMOTE_K_NEIGHBORS        = 5

# ==========================
# Model Training Config
# ==========================

# Minimum F1-score threshold for model acceptance
EXPECTED_F1_THRESHOLD = 0.60

# Models to train  (name -> short key used for logging)
MODELS_TO_TRAIN = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]