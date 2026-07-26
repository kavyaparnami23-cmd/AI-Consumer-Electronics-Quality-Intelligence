"""
model_loader.py
───────────────
Loads all trained models from disk (saved_models/) at startup.
Uses a singleton-style registry so models are only loaded once.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict

import joblib
import numpy as np

from . import config as cfg

logger = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────────────
_registry: Dict[str, Any] = {}
_loaded: Dict[str, bool] = {}


def _try_load(name: str, loader_fn):
    """Run loader_fn, mark as loaded/failed and return result."""
    try:
        result = loader_fn()
        _loaded[name] = True
        logger.info("✅  Loaded model: %s", name)
        return result
    except Exception as exc:
        _loaded[name] = False
        logger.warning("⚠️  Could not load model '%s': %s", name, exc)
        return None


# ── Classic ML ────────────────────────────────────────────────────────────────

def _load_classic():
    model = joblib.load(cfg.CLASSIC_MODEL_PATH)
    preproc = joblib.load(cfg.CLASSIC_PREPROC_PATH)
    return {"model": model, "preprocessor": preproc}


# ── Deep Learning ─────────────────────────────────────────────────────────────

def _load_dl():
    import torch

    scaler = joblib.load(cfg.DL_SCALER_PATH)

    # ── LSTM ──────────────────────────────────────────────────────────────────
    from src.deep_learning.lstm_model import SensorLSTM  # noqa: PLC0415

    lstm = SensorLSTM(input_dim=cfg.DL_INPUT_FEATURES)
    lstm.load_state_dict(
        torch.load(cfg.LSTM_MODEL_PATH, map_location="cpu", weights_only=True)
    )
    lstm.eval()

    # ── CNN ───────────────────────────────────────────────────────────────────
    from src.deep_learning.cnn_model import Sensor1DCNN  # noqa: PLC0415

    cnn = Sensor1DCNN(input_dim=cfg.DL_INPUT_FEATURES)
    cnn.load_state_dict(
        torch.load(cfg.CNN_MODEL_PATH, map_location="cpu", weights_only=True)
    )
    cnn.eval()

    return {"lstm": lstm, "cnn": cnn, "scaler": scaler}


# ── NLP ───────────────────────────────────────────────────────────────────────

def _load_nlp_tfidf():
    model     = joblib.load(cfg.TFIDF_MODEL_PATH)
    vectorizer = joblib.load(cfg.TFIDF_VECTORIZER_PATH)
    le        = joblib.load(cfg.TFIDF_LABEL_ENC_PATH)
    return {"model": model, "vectorizer": vectorizer, "label_encoder": le}


def _load_nlp_distilbert():
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast  # noqa: PLC0415

    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg.DISTILBERT_TOKENIZER_DIR)
    model     = DistilBertForSequenceClassification.from_pretrained(cfg.DISTILBERT_MODEL_DIR)
    model.eval()
    return {"model": model, "tokenizer": tokenizer}


# ── Time Series ───────────────────────────────────────────────────────────────

def _load_ts():
    import torch
    from src.time_series.lstm_autoencoder import LSTMAutoEncoder  # noqa: PLC0415

    scaler     = joblib.load(cfg.TS_SCALER_PATH)
    threshold  = joblib.load(cfg.TS_AE_THRESHOLD_PATH)
    bounds     = joblib.load(cfg.TS_SCORE_BOUNDS_PATH)
    iso_forest = joblib.load(cfg.TS_ISOFOREST_PATH)

    ae = LSTMAutoEncoder(
        input_dim=126,
        hidden_dim=64,
        latent_dim=16,
        num_layers=2,
        seq_len=30,  # WINDOW_SIZE from ts_config.py
    )
    ae.load_state_dict(
        torch.load(cfg.TS_AE_MODEL_PATH, map_location="cpu", weights_only=True)
    )
    ae.eval()

    return {
        "autoencoder": ae,
        "scaler": scaler,
        "threshold": threshold,
        "score_bounds": bounds,
        "isolation_forest": iso_forest,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def load_all_models() -> None:
    """Called once at FastAPI startup."""
    logger.info("Loading Classic ML …")
    _registry["classic"]         = _try_load("classic", _load_classic)

    logger.info("Loading Deep Learning models …")
    _registry["dl"]              = _try_load("dl", _load_dl)

    logger.info("Loading NLP TF-IDF model …")
    _registry["nlp_tfidf"]       = _try_load("nlp_tfidf", _load_nlp_tfidf)

    logger.info("Loading NLP DistilBERT model …")
    _registry["nlp_distilbert"]  = _try_load("nlp_distilbert", _load_nlp_distilbert)

    logger.info("Loading Time-Series models …")
    _registry["ts"]              = _try_load("ts", _load_ts)

    logger.info("Model loading complete. Status: %s", _loaded)


def get(key: str) -> Any:
    """Retrieve a model bundle by key. Returns None if not loaded."""
    return _registry.get(key)


def status() -> Dict[str, bool]:
    """Return a dict mapping model name → whether it was loaded successfully."""
    return dict(_loaded)
