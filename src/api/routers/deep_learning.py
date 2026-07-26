"""
routers/deep_learning.py
─────────────────────────
Endpoints for Deep Learning sensor-failure prediction (LSTM / CNN).
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from src.api import model_loader
from src.api.schemas import DLPredictRequest, DLPredictResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dl", tags=["Deep Learning"])

LABELS = {0: "No Failure", 1: "Failure"}


def _run_inference(model, features: list, bundle) -> tuple[int, float]:
    """Run a PyTorch model forward pass and return (prediction, confidence)."""
    import torch  # lazy import — works whether torch is in venv or system Python

    scaler = bundle["scaler"]
    x = np.array([features], dtype=np.float32)
    x = scaler.transform(x)                            # (1, 12)
    x_t = torch.tensor(x).unsqueeze(1)                 # (1, seq_len=1, 12)

    with torch.no_grad():
        logit = model(x_t)                             # (1,)
        prob  = torch.sigmoid(logit).item()

    pred = 1 if prob >= 0.5 else 0
    conf = prob if pred == 1 else 1.0 - prob
    return pred, round(conf, 4)


@router.post("/predict", response_model=DLPredictResponse, summary="Sensor failure prediction (LSTM or CNN)")
async def dl_predict(req: DLPredictRequest):
    """
    Runs the LSTM or 1D-CNN deep-learning model on a single sensor reading.

    - **features**: list of 12 pre-scaled float values
    - **model**: `"lstm"` (default) or `"cnn"`
    """
    bundle = model_loader.get("dl")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Deep Learning models are not loaded.")

    model_key = req.model.lower()
    if model_key not in ("lstm", "cnn"):
        raise HTTPException(status_code=422, detail="model must be 'lstm' or 'cnn'.")

    try:
        net  = bundle[model_key]
        pred, conf = _run_inference(net, req.features, bundle)
        return DLPredictResponse(
            prediction=pred,
            confidence=conf,
            label=LABELS[pred],
            model_used=model_key.upper(),
        )
    except Exception as exc:
        logger.exception("DL prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
