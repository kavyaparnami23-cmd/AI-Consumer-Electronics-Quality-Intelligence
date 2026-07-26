import os
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ─── Data paths ───────────────────────────────────────────────────────────────
CLEAN_AI4I_PATH = os.path.join(BASE_DIR, "datasets", "clean_ai4i_ml.csv")

SAVED_MODELS_TS_DIR = os.path.join(BASE_DIR, "saved_models", "time_series")
ARTIFACTS_TS_DIR    = os.path.join(BASE_DIR, "artifacts",    "time_series")

os.makedirs(SAVED_MODELS_TS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_TS_DIR,    exist_ok=True)

# ─── Autoencoder hyperparameters ──────────────────────────────────────────────
WINDOW_SIZE   = 30          # 30-step rolling context window for autoencoder
STEP_SIZE     = 1
HIDDEN_DIM    = 64
NUM_LAYERS    = 2
LATENT_DIM    = 16
BATCH_SIZE    = 64
EPOCHS        = 30
LEARNING_RATE = 1e-3
PATIENCE      = 7
DEVICE        = torch.device("cpu")

# Reconstruction-error anomaly threshold percentile
# Computed during training on normal (non-failure) windows
THRESHOLD_PERCENTILE = 95   # flag top 5% MSE as anomalous

# ─── Isolation Forest ──────────────────────────────────────────────────────────
IF_N_ESTIMATORS   = 200
IF_CONTAMINATION  = 0.03    # ~3% expected anomaly rate in AI4I dataset

# ─── Feature engineering windows ──────────────────────────────────────────────
ROLLING_WINDOWS = [5, 10, 20]   # rolling mean / std window sizes

# ─── Saved file paths ──────────────────────────────────────────────────────────
AE_MODEL_PATH     = os.path.join(SAVED_MODELS_TS_DIR, "lstm_autoencoder.pth")
IF_MODEL_PATH     = os.path.join(SAVED_MODELS_TS_DIR, "isolation_forest.joblib")
TS_SCALER_PATH    = os.path.join(SAVED_MODELS_TS_DIR, "ts_scaler.joblib")
THRESHOLD_PATH    = os.path.join(SAVED_MODELS_TS_DIR, "ae_threshold.joblib")
TS_REPORT_PATH    = os.path.join(ARTIFACTS_TS_DIR,    "ts_anomaly_report.json")
TS_FEATURES_PATH  = os.path.join(ARTIFACTS_TS_DIR,    "ts_features.csv")
