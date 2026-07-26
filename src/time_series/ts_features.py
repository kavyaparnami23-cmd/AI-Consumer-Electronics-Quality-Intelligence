"""
ts_features.py
──────────────
Temporal feature engineering for sensor time-series data.
Extracts:
  - Rolling mean & rolling std for configurable window sizes
  - Point-level Z-score  (how many σ above the running mean)
  - Rate-of-change (finite-difference derivative) for key sensors
  - Ewm (exponentially-weighted moving average) for trend capture
  - Statistical anomaly indicator: z_score_flag (|z| > 3)

Returns an enriched DataFrame that feeds both the LSTM Autoencoder
and the Isolation Forest models.
"""

import pandas as pd
import numpy as np
from typing import List
from src.time_series.ts_config import ROLLING_WINDOWS

# ─── Sensor columns present in clean_ai4i_ml.csv ──────────────────────────────
BASE_SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ENGINEERED_COLS = [
    "Temp_Difference",
    "Power_kW",
    "Torque_Wear_Product",
]

TARGET_COL = "Machine failure"


def _ensure_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic engineered columns if absent."""
    df = df.copy()
    if "Temp_Difference" not in df.columns:
        df["Temp_Difference"] = (
            df["Process temperature [K]"] - df["Air temperature [K]"]
        )
    if "Power_kW" not in df.columns:
        df["Power_kW"] = (
            df["Rotational speed [rpm]"] * df["Torque [Nm]"] / 9550.0
        )
    if "Torque_Wear_Product" not in df.columns:
        df["Torque_Wear_Product"] = df["Torque [Nm]"] * df["Tool wear [min]"]
    return df


def build_ts_features(df: pd.DataFrame, windows: List[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Build rich temporal features for anomaly detection.
    Uses pd.concat to avoid DataFrame fragmentation performance warnings.

    Args:
        df: Raw / lightly engineered sensor DataFrame (must contain BASE_SENSOR_COLS).
        windows: List of rolling-window sizes to compute stats for.

    Returns:
        DataFrame with original + temporal feature columns.
        Any NaN rows (from rolling start) are dropped.
    """
    df = _ensure_engineered(df)
    all_sensor_cols = [c for c in BASE_SENSOR_COLS + ENGINEERED_COLS if c in df.columns]

    # Collect all new columns in a list then concat once at the end
    new_cols: dict = {}

    for col in all_sensor_cols:
        series = df[col]

        # ── Rate of change (1-step finite difference) ──────────────────────
        new_cols[f"{col}_roc"] = series.diff()

        # ── EWM (exponentially-weighted mean) – α=0.1 ─────────────────────
        new_cols[f"{col}_ewm"] = series.ewm(alpha=0.1, adjust=False).mean()

        for w in windows:
            rolling    = series.rolling(window=w, min_periods=w)
            roll_mean  = rolling.mean()
            roll_std   = rolling.std().replace(0, 1e-8)
            zscore     = (series - roll_mean) / roll_std

            new_cols[f"{col}_rmean_{w}"] = roll_mean
            new_cols[f"{col}_rstd_{w}"]  = roll_std
            new_cols[f"{col}_zscore_{w}"] = zscore
            new_cols[f"{col}_zflag_{w}"]  = (zscore.abs() > 3).astype(int)

    # Build new_cols DataFrame and concat with original in one shot
    new_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # ── Global z-score anomaly count across all 5-step flags ─────────────────
    zflag_cols = [c for c in new_df.columns if c.endswith("_zflag_5")]
    if zflag_cols:
        new_df["total_zflag_count"] = new_df[zflag_cols].sum(axis=1)

    # Drop NaN rows created by rolling windows
    new_df = new_df.dropna().reset_index(drop=True)

    return new_df


def get_feature_columns(df_enriched: pd.DataFrame) -> List[str]:
    """Return all feature column names (exclude target and ID-like columns)."""
    exclude = {TARGET_COL, "UDI", "Product ID", "Type"}
    return [c for c in df_enriched.columns if c not in exclude]
