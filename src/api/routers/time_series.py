"""
routers/time_series.py
──────────────────────
Endpoints for Time-Series Anomaly Detection (LSTM Autoencoder + Isolation Forest).
Includes both /ts and /timeseries prefixes for full endpoint compatibility.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api import model_loader, config as cfg
from src.api.schemas import AnomalyRequest, AnomalyResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ts", tags=["Time Series"])
timeseries_router = APIRouter(prefix="/timeseries", tags=["Time Series"])


def _normalize_score(score: float, bounds: dict) -> float:
    """Normalise reconstruction error to [0, 1] using training min/max bounds."""
    lo = bounds.get("min", 0.0)
    hi = bounds.get("max", 1.0)
    if hi == lo:
        return 0.0
    return float(np.clip((score - lo) / (hi - lo), 0.0, 1.0))


def _prepare_ts_matrix(window: List[List[float]], scaler: float = None) -> np.ndarray:
    """Convert raw or enriched input window to scaled matrix matching model features."""
    x = np.array(window, dtype=np.float32)
    target_seq_len = len(window)

    # If matrix already has 126 features
    if x.shape[1] == 126:
        if scaler is not None and hasattr(scaler, "transform"):
            return scaler.transform(x)
        return x

    # Build rich features from raw sensor readings
    cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temp_Difference",
        "Power_kW",
        "Torque_Wear_Product",
    ]
    if x.shape[1] < len(cols):
        pad = np.zeros((x.shape[0], len(cols) - x.shape[1]), dtype=np.float32)
        x = np.hstack([x, pad])

    df_raw = pd.DataFrame(x[:, : len(cols)], columns=cols)
    try:
        from src.time_series.ts_features import _ensure_engineered, ROLLING_WINDOWS
        df_raw = _ensure_engineered(df_raw)
        
        all_cols = [c for c in cols if c in df_raw.columns]
        new_cols = {}
        for col in all_cols:
            series = df_raw[col]
            new_cols[f"{col}_roc"] = series.diff().fillna(0)
            new_cols[f"{col}_ewm"] = series.ewm(alpha=0.1, adjust=False).mean()
            for w in ROLLING_WINDOWS:
                rolling = series.rolling(window=w, min_periods=1)
                roll_mean = rolling.mean()
                roll_std = rolling.std().fillna(1e-8).replace(0, 1e-8)
                zscore = (series - roll_mean) / roll_std
                new_cols[f"{col}_rmean_{w}"] = roll_mean
                new_cols[f"{col}_rstd_{w}"] = roll_std
                new_cols[f"{col}_zscore_{w}"] = zscore.fillna(0)
                new_cols[f"{col}_zflag_{w}"] = (zscore.abs() > 3).astype(int)

        df_enriched = pd.concat([df_raw, pd.DataFrame(new_cols, index=df_raw.index)], axis=1)
        zflag_cols = [c for c in df_enriched.columns if c.endswith("_zflag_5")]
        if zflag_cols:
            df_enriched["total_zflag_count"] = df_enriched[zflag_cols].sum(axis=1)

        feat_cols = [
            c
            for c in df_enriched.columns
            if c not in {"Machine failure", "UDI", "Product ID", "Type"}
        ]
        X_mat = df_enriched[feat_cols].values.astype(np.float32)
    except Exception as e:
        logger.warning("Falling back to raw window padding for TS features: %s", e)
        X_mat = x

    # Ensure sequence length matches target_seq_len exactly
    if X_mat.shape[0] < target_seq_len:
        pad_rows = np.zeros((target_seq_len - X_mat.shape[0], X_mat.shape[1]), dtype=np.float32)
        X_mat = np.vstack([pad_rows, X_mat])
    elif X_mat.shape[0] > target_seq_len:
        X_mat = X_mat[-target_seq_len:]

    # Ensure feature dimension matches 126 exactly
    if X_mat.shape[1] < 126:
        pad_cols = np.zeros((X_mat.shape[0], 126 - X_mat.shape[1]), dtype=np.float32)
        X_mat = np.hstack([X_mat, pad_cols])
    elif X_mat.shape[1] > 126:
        X_mat = X_mat[:, :126]

    if scaler is not None and hasattr(scaler, "transform"):
        try:
            X_mat = scaler.transform(X_mat)
        except Exception:
            pass

    return X_mat


@router.post("/anomaly", response_model=AnomalyResponse, summary="Time-series anomaly detection (LSTM Autoencoder)")
@timeseries_router.post("/anomaly", response_model=AnomalyResponse, summary="Time-series anomaly detection (LSTM Autoencoder)")
async def detect_anomaly(req: AnomalyRequest):
    """
    Detects anomalies in a rolling sensor window using the LSTM Autoencoder.

    - **window**: list of `window_size` timesteps, each with sensor values.
    - Accessible at both `/ts/anomaly` and `/timeseries/anomaly`.
    """
    bundle = model_loader.get("ts")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Time-series models are not loaded.")

    window = req.window
    if len(window) < 1:
        raise HTTPException(status_code=422, detail="Window must contain at least 1 timestep.")

    try:
        import torch  # lazy import

        ae = bundle["autoencoder"]
        scaler = bundle["scaler"]
        threshold = float(bundle["threshold"])
        bounds = bundle["score_bounds"]

        X_mat = _prepare_ts_matrix(window, scaler)
        x_t = torch.tensor(X_mat, dtype=torch.float32).unsqueeze(0)

        mse_tensor = ae.reconstruction_error(x_t)
        mse = float(mse_tensor[0])

        norm_score = _normalize_score(mse, bounds)
        is_anomaly = mse > threshold

        return AnomalyResponse(
            is_anomaly=is_anomaly,
            anomaly_score=round(mse, 6),
            threshold=round(threshold, 6),
            normalized_score=round(norm_score, 4),
        )

    except Exception as exc:
        logger.exception("Anomaly detection failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/anomaly/isolation-forest", response_model=AnomalyResponse, summary="Anomaly detection via Isolation Forest")
@timeseries_router.post("/anomaly/isolation-forest", response_model=AnomalyResponse, summary="Anomaly detection via Isolation Forest")
async def detect_anomaly_isoforest(req: AnomalyRequest):
    """
    Detects anomalies using the fitted Isolation Forest model.
    Accessible at `/ts/anomaly/isolation-forest` and `/timeseries/anomaly/isolation-forest`.
    """
    bundle = model_loader.get("ts")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Time-series models are not loaded.")

    if len(req.window) < 1:
        raise HTTPException(status_code=422, detail="Window must contain at least 1 timestep.")

    try:
        iso = bundle["isolation_forest"]
        scaler = bundle["scaler"]
        bounds = bundle["score_bounds"]

        X_mat = _prepare_ts_matrix(req.window, scaler)
        scores = -iso.score_samples(X_mat)
        mean_score = float(scores.mean())
        threshold = float(bundle["threshold"])
        norm_score = _normalize_score(mean_score, bounds)
        is_anomaly = mean_score > threshold

        return AnomalyResponse(
            is_anomaly=is_anomaly,
            anomaly_score=round(mean_score, 6),
            threshold=round(threshold, 6),
            normalized_score=round(norm_score, 4),
        )

    except Exception as exc:
        logger.exception("Isolation Forest anomaly detection failed")
        raise HTTPException(status_code=500, detail=str(exc))
