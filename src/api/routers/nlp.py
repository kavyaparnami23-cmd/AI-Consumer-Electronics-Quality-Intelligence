"""
routers/nlp.py
──────────────
Endpoints for NLP / Sentiment Analysis (TF-IDF and DistilBERT).
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from src.api import model_loader
from src.api.schemas import SentimentRequest, SentimentResponse
from src.api import config as cfg

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/nlp", tags=["NLP / Sentiment"])


def _predict_tfidf(text: str, bundle) -> SentimentResponse:
    vec   = bundle["vectorizer"].transform([text])
    pred  = bundle["model"].predict(vec)[0]
    proba = bundle["model"].predict_proba(vec)[0]
    le    = bundle["label_encoder"]
    label = le.inverse_transform([pred])[0] if hasattr(le, "inverse_transform") else str(pred)
    conf  = float(max(proba))
    return SentimentResponse(sentiment=label, confidence=round(conf, 4), model_used="TF-IDF")


def _predict_distilbert(text: str, bundle) -> SentimentResponse:
    import torch  # lazy import
    tokenizer = bundle["tokenizer"]
    model     = bundle["model"]

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=cfg.NLP_MAX_LENGTH,
    )
    with torch.no_grad():
        logits = model(**encoding).logits          # (1, num_classes)

    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = int(probs.argmax())
    conf     = float(probs[pred_idx])

    # Map id2label from model config
    id2label = model.config.id2label if hasattr(model.config, "id2label") else {0: "negative", 1: "positive"}
    label    = id2label.get(pred_idx, str(pred_idx))

    return SentimentResponse(sentiment=label, confidence=round(conf, 4), model_used="DistilBERT")


@router.post("/sentiment", response_model=SentimentResponse, summary="Sentiment analysis on review text")
async def sentiment(req: SentimentRequest):
    """
    Classifies the sentiment of a product review.

    - **text**: the raw review string
    - **model**: `"tfidf"` (fast, default) or `"distilbert"` (more accurate, slower)
    """
    model_key = req.model.value  # "tfidf" or "distilbert"

    if model_key == "tfidf":
        bundle = model_loader.get("nlp_tfidf")
        if bundle is None:
            raise HTTPException(status_code=503, detail="TF-IDF NLP model is not loaded.")
        try:
            return _predict_tfidf(req.text, bundle)
        except Exception as exc:
            logger.exception("TF-IDF sentiment failed")
            raise HTTPException(status_code=500, detail=str(exc))

    elif model_key == "distilbert":
        bundle = model_loader.get("nlp_distilbert")
        if bundle is None:
            raise HTTPException(status_code=503, detail="DistilBERT model is not loaded.")
        try:
            return _predict_distilbert(req.text, bundle)
        except Exception as exc:
            logger.exception("DistilBERT sentiment failed")
            raise HTTPException(status_code=500, detail=str(exc))

    raise HTTPException(status_code=422, detail="Unknown model selection.")
