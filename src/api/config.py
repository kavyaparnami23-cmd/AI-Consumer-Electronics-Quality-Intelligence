"""
config.py
─────────
Central configuration for the FastAPI backend.
All values can be overridden via environment variables.
"""

import os

# ── Server ────────────────────────────────────────────────────────────────────
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))

# ── MLflow ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{DB_PATH}"
)

# ── Saved-model paths (fallback when MLflow registry is unavailable) ──────────
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# Classic ML
CLASSIC_MODEL_PATH      = os.path.join(SAVED_MODELS_DIR, "model.pkl")
CLASSIC_PREPROC_PATH    = os.path.join(SAVED_MODELS_DIR, "preprocessor.pkl")

# Deep Learning
DL_DIR                  = os.path.join(SAVED_MODELS_DIR, "deep_learning")
DL_SCALER_PATH          = os.path.join(DL_DIR, "dl_scaler.joblib")
LSTM_MODEL_PATH         = os.path.join(DL_DIR, "lstm_model.pth")
CNN_MODEL_PATH          = os.path.join(DL_DIR, "cnn_model.pth")

# NLP
NLP_DIR                 = os.path.join(SAVED_MODELS_DIR, "nlp")
TFIDF_MODEL_PATH        = os.path.join(NLP_DIR, "tfidf_model.pkl")
TFIDF_VECTORIZER_PATH   = os.path.join(NLP_DIR, "tfidf_vectorizer.pkl")
TFIDF_LABEL_ENC_PATH    = os.path.join(NLP_DIR, "tfidf_label_encoder.pkl")
DISTILBERT_MODEL_DIR    = os.path.join(NLP_DIR, "distilbert")
DISTILBERT_TOKENIZER_DIR= os.path.join(NLP_DIR, "distilbert_tokenizer")

# Time Series
TS_DIR                  = os.path.join(SAVED_MODELS_DIR, "time_series")
TS_SCALER_PATH          = os.path.join(TS_DIR, "ts_scaler.joblib")
TS_AE_MODEL_PATH        = os.path.join(TS_DIR, "lstm_autoencoder.pth")
TS_AE_THRESHOLD_PATH    = os.path.join(TS_DIR, "ae_threshold.joblib")
TS_SCORE_BOUNDS_PATH    = os.path.join(TS_DIR, "ts_score_bounds.joblib")
TS_ISOFOREST_PATH       = os.path.join(TS_DIR, "isolation_forest.joblib")

# ── Feature dimensions (must match training) ──────────────────────────────────
DL_INPUT_FEATURES   = 8    # number of features used in LSTM/CNN training (see dl_data_prep.py FEATURE_COLS)
DL_WINDOW_SIZE      = 10   # sequence window for DL models (dl_config.WINDOW_SIZE)
TS_WINDOW_SIZE      = 30   # time-window fed to LSTM autoencoder (matches ts_config.WINDOW_SIZE)
NLP_MAX_LENGTH      = 128  # DistilBERT token length
