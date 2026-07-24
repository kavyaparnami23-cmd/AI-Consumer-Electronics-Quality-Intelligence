import os
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Data paths
CLEAN_AI4I_PATH = os.path.join(BASE_DIR, "datasets", "clean_ai4i_ml.csv")
SAVED_MODELS_DL_DIR = os.path.join(BASE_DIR, "saved_models", "deep_learning")
ARTIFACTS_DL_DIR = os.path.join(BASE_DIR, "artifacts", "deep_learning")

os.makedirs(SAVED_MODELS_DL_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DL_DIR, exist_ok=True)

# Hyperparameters
WINDOW_SIZE = 10
STEP_SIZE = 1
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 5
DEVICE = torch.device("cpu")

LSTM_MODEL_PATH = os.path.join(SAVED_MODELS_DL_DIR, "lstm_model.pth")
CNN_MODEL_PATH = os.path.join(SAVED_MODELS_DL_DIR, "cnn_model.pth")
SCALER_PATH = os.path.join(SAVED_MODELS_DL_DIR, "dl_scaler.joblib")
REPORT_PATH = os.path.join(ARTIFACTS_DL_DIR, "dl_evaluation_report.json")
