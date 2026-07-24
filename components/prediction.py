import os
import sys

import numpy as np
import pandas as pd

from logger import logger
from exception import CustomException
from utils import load_object
from config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
    DROP_COLUMNS,
)
from entity.config_entity import PredictionConfig
from components.feature_engineering import add_features


class PredictionPipeline:
    """
    Loads the saved preprocessor and model, applies feature engineering,
    scales input, and returns prediction + probability for a single sample
    or a DataFrame of samples.
    """

    def __init__(self):
        self.prediction_config = PredictionConfig(
            model_path=MODEL_PATH,
            preprocessor_path=PREPROCESSOR_PATH,
            target_column=TARGET_COLUMN,
            drop_columns=DROP_COLUMNS,
        )
        self._model        = None
        self._preprocessor = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _load_artifacts(self):
        if self._model is None:
            self._model        = load_object(self.prediction_config.model_path)
            self._preprocessor = load_object(self.prediction_config.preprocessor_path)

    # ------------------------------------------------------------------
    # Prepare input
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df = add_features(df)
        df = df.drop(columns=self.prediction_config.drop_columns, errors="ignore")
        if self.prediction_config.target_column in df.columns:
            df = df.drop(columns=[self.prediction_config.target_column])
        return self._preprocessor.transform(df)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Raw input with the same columns as the training data
            (excluding target). 'Product ID' can be omitted.

        Returns
        -------
        dict with keys:
            prediction  : list[int]   (0 = No Failure, 1 = Failure)
            probability : list[float] (probability of Failure)
        """
        try:
            self._load_artifacts()
            X = self._prepare(df)
            preds  = self._model.predict(X).tolist()
            probs  = self._model.predict_proba(X)[:, 1].tolist()
            labels = ["No Failure" if p == 0 else "Machine Failure" for p in preds]

            return {
                "prediction":   preds,
                "probability":  [round(p, 4) for p in probs],
                "label":        labels,
            }

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


# ===================================================
# Testing
# ===================================================

if __name__ == "__main__":
    import pandas as pd

    sample = pd.DataFrame([{
        "Product ID":                "M14860",
        "Type":                       1,
        "Air temperature [K]":       298.1,
        "Process temperature [K]":   308.6,
        "Rotational speed [rpm]":    1551,
        "Torque [Nm]":               42.8,
        "Tool wear [min]":            0,
        "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
    }])

    print("=" * 60)
    print("Testing Prediction")
    print("=" * 60)

    predictor = PredictionPipeline()
    result    = predictor.predict(sample)

    print("\nPrediction  :", result["prediction"])
    print("Probability :", result["probability"])
    print("Label       :", result["label"])
    print("=" * 60)
