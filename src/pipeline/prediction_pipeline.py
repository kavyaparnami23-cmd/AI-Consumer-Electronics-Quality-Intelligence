import sys
import pandas as pd

from logger import logger
from exception import CustomException
from components.prediction import PredictionPipeline


class PredictionPipelineRunner:
    """
    Thin wrapper around PredictionPipeline for use by the Flask app
    or CLI callers.
    """

    def __init__(self):
        self._pipeline = PredictionPipeline()

    def predict(self, input_data: dict | pd.DataFrame) -> dict:
        """
        Parameters
        ----------
        input_data : dict or pd.DataFrame
            A single row as a dict or a DataFrame.

        Returns
        -------
        dict : { prediction, probability, label }
        """
        try:
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = input_data

            result = self._pipeline.predict(df)
            logger.info(f"Prediction result : {result}")
            return result

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)
