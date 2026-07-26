"""
routers/classic_ml.py
─────────────────────
Endpoints for Classic ML (Scikit-learn / XGBoost / LogisticRegression) sensor-failure prediction with SHAP explanations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api import model_loader
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    FeatureExplanation,
    PredictRequest,
    PredictResponse,
    SensorFeatures,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/classic", tags=["Classic ML"])

LABELS = {0: "No Failure", 1: "Failure"}

FEATURE_NAMES_14 = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "Temp_diff",
    "Power",
    "Wear_rate",
]


def _build_raw_df(features: SensorFeatures) -> pd.DataFrame:
    """Convert SensorFeatures request schema into raw pandas DataFrame before feature engineering."""
    f = features
    type_val = 1
    if getattr(f, "type_H", 0.0) == 1.0:
        type_val = 2
    elif getattr(f, "type_L", 0.0) == 1.0:
        type_val = 0

    raw_dict = {
        "Type": type_val,
        "Air temperature [K]": f.air_temperature,
        "Process temperature [K]": f.process_temperature,
        "Rotational speed [rpm]": f.rotational_speed,
        "Torque [Nm]": f.torque,
        "Tool wear [min]": f.tool_wear,
        "TWF": 0,
        "HDF": 0,
        "PWF": 0,
        "OSF": 0,
        "RNF": 0,
    }
    df = pd.DataFrame([raw_dict])

    df["Temp_diff"] = f.temp_diff if f.temp_diff is not None else (f.process_temperature - f.air_temperature)
    df["Power"] = f.speed_torque if f.speed_torque is not None else (f.torque * f.rotational_speed)
    df["Wear_rate"] = f.wear_speed if f.wear_speed is not None else (f.tool_wear / (f.rotational_speed + 1e-6))

    return df


def _prepare_batch_input(samples: List[Any]) -> pd.DataFrame:
    """
    Converts list of SensorFeatures, dicts, or flat list feature vectors to a 14-feature DataFrame
    formatted for bundle["preprocessor"].transform().
    """
    rows = []
    for item in samples:
        if isinstance(item, SensorFeatures):
            df_row = _build_raw_df(item)
            rows.append(df_row.iloc[0].to_dict())
        elif isinstance(item, dict):
            sf = SensorFeatures(**item)
            df_row = _build_raw_df(sf)
            rows.append(df_row.iloc[0].to_dict())
        elif isinstance(item, (list, tuple, np.ndarray)):
            arr = list(item)
            # Handle single element nested list case e.g. [[v1, v2...]]
            if len(arr) == 1 and isinstance(arr[0], (list, tuple, np.ndarray)):
                arr = list(arr[0])

            if len(arr) == 14:
                row_dict = {col: float(arr[i]) for i, col in enumerate(FEATURE_NAMES_14)}
                rows.append(row_dict)
            else:
                type_val = float(arr[0]) if len(arr) > 0 else 1.0
                air_temp = float(arr[1]) if len(arr) > 1 else 298.1
                proc_temp = float(arr[2]) if len(arr) > 2 else 308.6
                speed = float(arr[3]) if len(arr) > 3 else 1500.0
                torque = float(arr[4]) if len(arr) > 4 else 40.0
                wear = float(arr[5]) if len(arr) > 5 else 0.0
                twf = float(arr[6]) if len(arr) > 6 else 0.0
                hdf = float(arr[7]) if len(arr) > 7 else 0.0
                pwf = float(arr[8]) if len(arr) > 8 else 0.0
                osf = float(arr[9]) if len(arr) > 9 else 0.0
                rnf = float(arr[10]) if len(arr) > 10 else 0.0

                temp_diff = proc_temp - air_temp
                power = torque * speed
                wear_rate = wear / (speed + 1e-6)

                row_dict = {
                    "Type": type_val,
                    "Air temperature [K]": air_temp,
                    "Process temperature [K]": proc_temp,
                    "Rotational speed [rpm]": speed,
                    "Torque [Nm]": torque,
                    "Tool wear [min]": wear,
                    "TWF": twf,
                    "HDF": hdf,
                    "PWF": pwf,
                    "OSF": osf,
                    "RNF": rnf,
                    "Temp_diff": temp_diff,
                    "Power": power,
                    "Wear_rate": wear_rate,
                }
                rows.append(row_dict)

    df_batch = pd.DataFrame(rows)
    return df_batch[FEATURE_NAMES_14]


def _compute_shap_explanations(
    model: Any, X_scaled: np.ndarray, feature_names: List[str] = FEATURE_NAMES_14
) -> Tuple[Optional[Dict[str, float]], Optional[List[FeatureExplanation]]]:
    """Calculate SHAP values and return feature explanation dictionary and top feature list."""
    try:
        import shap

        explainer = shap.Explainer(model, X_scaled)
        shap_vals = explainer(X_scaled)

        vals = shap_vals.values[0]
        if vals.ndim > 1:
            vals = vals[:, 1] if vals.shape[1] > 1 else vals[:, 0]

        shap_dict = {name: round(float(val), 6) for name, val in zip(feature_names, vals)}

        explanations = [
            FeatureExplanation(
                feature=name,
                shap_value=round(float(val), 6),
                feature_value=round(float(x_val), 4),
            )
            for name, val, x_val in zip(feature_names, vals, X_scaled[0])
        ]
        explanations.sort(key=lambda item: abs(item.shap_value), reverse=True)

        return shap_dict, explanations
    except Exception as exc:
        logger.warning("Could not calculate SHAP values: %s", exc)
        return None, None


@router.post("/predict", response_model=PredictResponse, summary="Sensor failure prediction with SHAP explanations (Classic ML)")
async def predict(req: PredictRequest):
    """
    Predicts whether a sensor will fail given its current readings and returns SHAP explanations.
    """
    bundle = model_loader.get("classic")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Classic ML model is not loaded.")

    try:
        df_raw = _build_raw_df(req.features)
        X_scaled = bundle["preprocessor"].transform(df_raw)

        pred = int(bundle["model"].predict(X_scaled)[0])
        proba = bundle["model"].predict_proba(X_scaled)[0]
        conf = float(proba[pred])

        shap_dict, top_features = _compute_shap_explanations(bundle["model"], X_scaled)

        return PredictResponse(
            prediction=pred,
            confidence=round(conf, 4),
            label=LABELS[pred],
            shap_values=shap_dict,
            top_features=top_features,
        )
    except Exception as exc:
        logger.exception("Classic ML prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/predict/batch", response_model=BatchPredictResponse, summary="Batch sensor failure prediction")
async def predict_batch(req: BatchPredictRequest):
    """
    Predict failure for a batch of samples (supports dicts, SensorFeatures objects, or feature vectors).
    """
    bundle = model_loader.get("classic")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Classic ML model is not loaded.")

    try:
        df_batch = _prepare_batch_input(req.samples)
        X_scaled = bundle["preprocessor"].transform(df_batch)

        preds = bundle["model"].predict(X_scaled).tolist()
        probas = bundle["model"].predict_proba(X_scaled).tolist()
        confs = [float(p[pred]) for p, pred in zip(probas, preds)]
        labels = [LABELS[p] for p in preds]

        return BatchPredictResponse(
            predictions=preds,
            confidences=[round(c, 4) for c in confs],
            labels=labels,
            count=len(preds),
        )
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
