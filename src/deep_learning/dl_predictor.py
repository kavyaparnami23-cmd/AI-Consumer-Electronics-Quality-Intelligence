import torch
import numpy as np
import joblib

from src.deep_learning.lstm_model import SensorLSTM
from src.deep_learning.cnn_model import Sensor1DCNN
from src.deep_learning.dl_config import LSTM_MODEL_PATH, CNN_MODEL_PATH, SCALER_PATH, WINDOW_SIZE, DEVICE

class DLPredictor:
    def __init__(self, model_type: str = "lstm", input_dim: int = 8):
        self.model_type = model_type.lower()
        self.scaler = joblib.load(SCALER_PATH)

        if self.model_type == "lstm":
            self.model = SensorLSTM(input_dim=input_dim)
            self.model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
        elif self.model_type in ["cnn", "1d-cnn"]:
            self.model = Sensor1DCNN(input_dim=input_dim)
            self.model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, window_data: np.ndarray):
        """
        window_data: shape (seq_len, input_dim) or (batch_size, seq_len, input_dim)
        """
        if window_data.ndim == 2:
            window_data = np.expand_dims(window_data, axis=0)

        # Scale features
        batch_size, seq_len, num_feats = window_data.shape
        reshaped = window_data.reshape(-1, num_feats)
        scaled_reshaped = self.scaler.transform(reshaped)
        scaled_seq = scaled_reshaped.reshape(batch_size, seq_len, num_feats)

        tensor_seq = torch.tensor(scaled_seq, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = self.model(tensor_seq)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

        return {
            "predictions": preds.cpu().numpy().tolist(),
            "probabilities": probs.cpu().numpy().tolist(),
            "model_used": self.model_type.upper()
        }
