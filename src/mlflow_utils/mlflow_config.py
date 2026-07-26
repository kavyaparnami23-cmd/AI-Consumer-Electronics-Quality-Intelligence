"""
mlflow_config.py
─────────────────
Central MLflow configuration for the AI Consumer Electronics Quality Intelligence project.
All experiments, model names, and paths are defined here.
"""

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ── MLflow tracking ────────────────────────────────────────────────────────────
# SQLite database backend enables full Model Registry functionality on local OS
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
MLFLOW_TRACKING_URI  = f"sqlite:///{DB_PATH}"
MLFLOW_ARTIFACTS_DIR = os.path.join(BASE_DIR, "mlruns")

# ── Experiment names (one per module) ─────────────────────────────────────────
EXPERIMENT_CLASSIC_ML  = "Classic-ML-Sensor-Failure"
EXPERIMENT_DEEP_LEARNING = "Deep-Learning-Sensor-Failure"
EXPERIMENT_NLP          = "NLP-Review-Sentiment"
EXPERIMENT_TIME_SERIES  = "Time-Series-Anomaly-Detection"

# ── Registered model names (MLflow Model Registry) ────────────────────────────
MODEL_REGISTRY_CLASSIC   = "SensorFailureClassifier"
MODEL_REGISTRY_LSTM      = "SensorLSTM"
MODEL_REGISTRY_CNN       = "Sensor1DCNN"
MODEL_REGISTRY_DISTILBERT = "ReviewSentimentDistilBERT"
MODEL_REGISTRY_AE        = "SensorLSTMAutoencoder"

# ── Tags applied to every run ─────────────────────────────────────────────────
COMMON_TAGS = {
    "project":     "AI-Consumer-Electronics-Quality-Intelligence",
    "dataset":     "AI4I-2020 + Amazon-Reviews",
    "environment": "cpu-only",
    "author":      "mlpipeline",
}
