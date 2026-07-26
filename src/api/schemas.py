"""
schemas.py
──────────
Pydantic request / response models for all API endpoints.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ── Shared ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class ModelInfo(BaseModel):
    name: str
    description: str
    loaded: bool


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


# ── Classic ML ────────────────────────────────────────────────────────────────

class SensorFeatures(BaseModel):
    """14 AI4I-2020 raw sensor columns (same order as training)."""
    air_temperature: float          = Field(..., example=298.1,  description="Air temperature [K]")
    process_temperature: float      = Field(..., example=308.6,  description="Process temperature [K]")
    rotational_speed: float         = Field(..., example=1551.0, description="Rotational speed [rpm]")
    torque: float                   = Field(..., example=42.8,   description="Torque [Nm]")
    tool_wear: float                = Field(..., example=0.0,    description="Tool wear [min]")
    # one-hot: Type (H/L/M)
    type_H: float                   = Field(0.0, example=0.0)
    type_L: float                   = Field(0.0, example=0.0)
    type_M: float                   = Field(1.0, example=1.0)
    # engineered
    temp_diff: Optional[float]      = Field(None, description="If None, computed automatically")
    speed_torque: Optional[float]   = Field(None, description="If None, computed automatically")
    wear_torque: Optional[float]    = Field(None, description="If None, computed automatically")
    wear_speed: Optional[float]     = Field(None, description="If None, computed automatically")


class PredictRequest(BaseModel):
    features: SensorFeatures


class FeatureExplanation(BaseModel):
    feature: str                    = Field(..., description="Feature name")
    shap_value: float               = Field(..., description="SHAP contribution score")
    feature_value: float            = Field(..., description="Original input value of the feature")


class PredictResponse(BaseModel):
    prediction: int                 = Field(..., description="0 = No Failure, 1 = Failure")
    confidence: float               = Field(..., description="Probability of the predicted class")
    label: str                      = Field(..., description="Human-readable label")
    shap_values: Optional[Dict[str, float]] = Field(None, description="SHAP contribution values for each feature")
    top_features: Optional[List[FeatureExplanation]] = Field(None, description="Top positive and negative feature explanations")


# ── Deep Learning ─────────────────────────────────────────────────────────────

class DLPredictRequest(BaseModel):
    """Flat list of scaled features for LSTM / CNN (default 8 features)."""
    features: List[float]           = Field(..., min_length=8, max_length=12,
                                           example=[0.1]*8)
    model: str                      = Field("lstm", description="'lstm' or 'cnn'")


class DLPredictResponse(BaseModel):
    prediction: int
    confidence: float
    label: str
    model_used: str


# ── NLP / Sentiment ───────────────────────────────────────────────────────────

class SentimentModelEnum(str, Enum):
    tfidf     = "tfidf"
    distilbert = "distilbert"


class SentimentRequest(BaseModel):
    text: str                       = Field(..., example="Great product, works perfectly!")
    model: SentimentModelEnum       = Field(SentimentModelEnum.tfidf,
                                            description="Which NLP model to use")


class SentimentResponse(BaseModel):
    sentiment: str                  = Field(..., description="positive / negative / neutral")
    confidence: float
    model_used: str


# ── Time Series / Anomaly ─────────────────────────────────────────────────────

class AnomalyRequest(BaseModel):
    """
    A sliding window of sensor readings.
    Each inner list represents one timestep with the same features
    as DLPredictRequest (12 values).
    The window must be exactly `ts_window_size` timesteps (default 20).
    """
    window: List[List[float]]       = Field(
        ...,
        description="Shape [window_size, num_features]. Default window_size=20, num_features=12.",
        example=[[0.0]*12]*20
    )


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float            = Field(..., description="Reconstruction error (lower = more normal)")
    threshold: float
    normalized_score: float         = Field(..., description="Score normalised to [0, 1]")


# ── Batch ─────────────────────────────────────────────────────────────────────

class BatchPredictRequest(BaseModel):
    samples: List[Union[SensorFeatures, List[float]]] = Field(
        ...,
        description="List of SensorFeatures objects or flat feature vectors.",
        example=[
            {
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551.0,
                "torque": 42.8,
                "tool_wear": 0.0,
                "type_H": 0.0,
                "type_L": 0.0,
                "type_M": 1.0,
            }
        ],
    )


class BatchPredictResponse(BaseModel):
    predictions: List[int]
    confidences: List[float]
    labels: List[str]
    count: int


BatchPredictRequest.model_rebuild()
PredictResponse.model_rebuild()
