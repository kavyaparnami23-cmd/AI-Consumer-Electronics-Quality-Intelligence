"""
nlp/nlp_predictor.py -- Inference predictor for single review or batch text sentiment analysis.
"""

import sys
import torch

from logger import logger
from exception import CustomException
from nlp.nlp_config import SENTIMENT_LABELS
from nlp.utils import clean_text
from nlp.tfidf_model import TfidfSentimentModel
from nlp.distilbert_model import DistilBERTSentimentModel


class NLPPredictor:
    """
    Inference class for NLP sentiment prediction.
    Supports both TF-IDF baseline and fine-tuned DistilBERT models.
    """

    def __init__(self, model_type: str = "distilbert"):
        """
        Parameters
        ----------
        model_type : str ("distilbert" or "tfidf")
        """
        self.model_type = model_type.lower()
        self._model = None

    def _load_model(self):
        if self._model is None:
            if self.model_type == "tfidf":
                self._model = TfidfSentimentModel()
                self._model.load()
            else:
                self._model = DistilBERTSentimentModel(pretrained=False)
                self._model.load()

    def predict(self, text_input) -> dict:
        """
        Predict sentiment for raw review text or list of texts.

        Parameters
        ----------
        text_input : str or list[str]

        Returns
        -------
        dict with keys: predictions (list[str]), confidence (list[float]), model_used (str)
        """
        try:
            self._load_model()

            if isinstance(text_input, str):
                texts = [text_input]
            else:
                texts = text_input

            cleaned_texts = [clean_text(t) for t in texts]

            if self.model_type == "tfidf":
                res = self._model.predict(cleaned_texts)
                return {
                    "predictions": res["predictions"],
                    "confidence": res["probabilities"],
                    "model_used": "TF-IDF + LogisticRegression",
                }

            else:

                tokens = self._model.tokenizer(
                    cleaned_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                )

                input_ids = tokens["input_ids"]
                attention_mask = tokens["attention_mask"]

                preds, probs = self._model.predict(input_ids, attention_mask)
                pred_ids = preds.cpu().numpy().tolist()
                prob_vals = probs.cpu().numpy()

                labels = [SENTIMENT_LABELS[i] for i in pred_ids]
                confidences = [round(float(prob_vals[i, pred_ids[i]]), 4) for i in range(len(pred_ids))]

                return {
                    "predictions": labels,
                    "confidence": confidences,
                    "model_used": "DistilBERT",
                }

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


if __name__ == "__main__":
    predictor = NLPPredictor(model_type="tfidf")
    sample_text = "This battery is super reliable and lasts long!"
    out = predictor.predict(sample_text)
    print("TF-IDF Output:", out)
