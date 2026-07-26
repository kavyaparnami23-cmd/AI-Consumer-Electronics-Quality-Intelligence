"""
ts_predictor.py
───────────────
Real-time anomaly inference class.

Usage:
    predictor = TSPredictor()
    result = predictor.predict(feature_window)   # (WINDOW_SIZE, 126) engineered array
    result = predictor.predict_from_df(df_raw)   # raw sensor DataFrame (auto-engineers)

Returns dict:
    {
        "ae_mse":        float,     # LSTM autoencoder reconstruction error
        "ae_threshold":  float,     # P95 normal-MSE threshold
        "ae_flag":       int,       # 1 if AE MSE > threshold
        "if_flag":       int,       # 1 if Isolation Forest predicts anomaly
        "combined_flag": int,       # 1 if either detector fires
        "anomaly_score": float,     # blended score [0, 1]
        "risk_level":    str,       # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    }
"""

import numpy as np
import torch
import joblib

from src.time_series.anomaly_detector import load_ae_model, load_isolation_forest
from src.time_series.ts_config import TS_SCALER_PATH, WINDOW_SIZE


# Risk thresholds for human-readable risk level
_RISK_LEVELS = [
    (0.75, "CRITICAL"),
    (0.50, "HIGH"),
    (0.25, "MEDIUM"),
    (0.00, "LOW"),
]


def _risk_level(score: float) -> str:
    for threshold, label in _RISK_LEVELS:
        if score >= threshold:
            return label
    return "LOW"


class TSPredictor:
    """
    Loads saved LSTM Autoencoder + Isolation Forest and performs real-time
    anomaly inference.

    Two entry points:
      1. predict(feature_window)  — expects (WINDOW_SIZE, 126) pre-engineered array
      2. predict_from_df(df_raw)  — accepts raw sensor DataFrame, auto-engineers features
    """

    def __init__(self):
        self._scaler    = joblib.load(TS_SCALER_PATH)
        self._input_dim = self._scaler.n_features_in_

        self._ae_model, self._ae_threshold = load_ae_model(self._input_dim)
        self._iso_forest = load_isolation_forest()

        # Score bounds for normalisation
        try:
            bounds_path = TS_SCALER_PATH.replace("ts_scaler", "ts_score_bounds")
            bounds = joblib.load(bounds_path)
            self._if_min = bounds["if_min"]
            self._if_max = bounds["if_max"]
            self._ae_max = bounds["ae_max"]
        except Exception:
            self._if_min = -0.7
            self._if_max =  0.3
            self._ae_max = float(self._ae_threshold * 3)

    def predict(self, feature_window: np.ndarray) -> dict:
        """
        Run anomaly detection on a pre-engineered feature window.

        Args:
            feature_window: np.ndarray shape (WINDOW_SIZE, input_dim=126), RAW (unscaled).
        Returns:
            Anomaly detection result dict.
        """
        if feature_window.shape[0] != WINDOW_SIZE:
            raise ValueError(f"feature_window must have {WINDOW_SIZE} timesteps, got {feature_window.shape[0]}")
        if feature_window.shape[1] != self._input_dim:
            raise ValueError(
                f"feature_window must have {self._input_dim} features, got {feature_window.shape[1]}. "
                f"Use predict_from_df() to pass raw sensor data instead."
            )

        scaled = self._scaler.transform(feature_window)

        # ── Autoencoder ────────────────────────────────────────────────────────
        tensor = torch.tensor(scaled[np.newaxis, :, :], dtype=torch.float32)
        ae_mse  = float(self._ae_model.reconstruction_error(tensor).item())
        ae_flag = int(ae_mse >= self._ae_threshold)

        # ── Isolation Forest (last timestep as point) ─────────────────────────
        last_row     = scaled[-1:, :]
        if_pred      = self._iso_forest.predict(last_row)[0]
        if_score_raw = float(self._iso_forest.score_samples(last_row)[0])
        if_flag      = int(if_pred == -1)

        # Normalise scores to [0, 1]
        if_anom = float(np.clip(
            1.0 - (if_score_raw - self._if_min) / (self._if_max - self._if_min + 1e-8),
            0.0, 1.0
        ))
        ae_anom = float(np.clip(ae_mse / (self._ae_threshold * 3 + 1e-8), 0.0, 1.0))

        anomaly_score = round(0.6 * ae_anom + 0.4 * if_anom, 4)

        return {
            "ae_mse":        round(ae_mse, 6),
            "ae_threshold":  round(self._ae_threshold, 6),
            "ae_flag":       ae_flag,
            "if_flag":       if_flag,
            "combined_flag": int(ae_flag == 1 or if_flag == 1),
            "anomaly_score": anomaly_score,
            "risk_level":    _risk_level(anomaly_score),
        }

    def predict_from_df(self, df_raw) -> dict:
        """
        Convenience wrapper: accepts a raw sensor DataFrame, engineers temporal
        features automatically, and runs detection on the last WINDOW_SIZE rows.

        Args:
            df_raw: pandas DataFrame with >= WINDOW_SIZE rows and BASE_SENSOR_COLS.
        Returns:
            Same dict as predict().
        """
        from src.time_series.ts_features import build_ts_features, get_feature_columns
        df_eng    = build_ts_features(df_raw)
        feat_cols = get_feature_columns(df_eng)
        X         = df_eng[feat_cols].values
        if len(X) < WINDOW_SIZE:
            raise ValueError(
                f"After feature engineering, only {len(X)} rows remain "
                f"(need >= {WINDOW_SIZE}). Provide more data."
            )
        return self.predict(X[-WINDOW_SIZE:])
