"""
nlp/tfidf_model.py -- TF-IDF + Logistic Regression baseline for sentiment classification.
"""

import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

from logger import logger
from exception import CustomException
from nlp.nlp_config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
    TFIDF_LABEL_ENCODER_PATH,
    SENTIMENT_LABELS,
    NLP_RANDOM_STATE,
)
from nlp.utils import save_nlp_object, load_nlp_object


class TfidfSentimentModel:
    """
    Classical NLP baseline: TF-IDF vectorization + Logistic Regression.

    Why this baseline matters:
    - Fast to train (seconds vs minutes for BERT)
    - Interpretable (top features per class)
    - Surprisingly competitive on many text tasks
    - Serves as a lower bound for the DistilBERT model
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,      # apply log normalization
        )
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",      # handles sentiment imbalance
            random_state=NLP_RANDOM_STATE,
            solver="lbfgs",
            C=1.0,
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(SENTIMENT_LABELS)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self, train_texts: list, train_labels: list) -> dict:
        """
        Fit the TF-IDF vectorizer and classifier.

        Parameters
        ----------
        train_texts  : list[str]  -- cleaned text
        train_labels : list[str]  -- sentiment strings ("Negative", "Neutral", "Positive")

        Returns
        -------
        dict with training metrics
        """
        try:
            logger.info("TF-IDF model training started")

            # Encode labels
            y_train = self.label_encoder.transform(train_labels)

            # Vectorize
            X_train = self.vectorizer.fit_transform(train_texts)
            logger.info(f"TF-IDF matrix shape : {X_train.shape}")

            # Fit
            self.model.fit(X_train, y_train)

            # Training metrics
            y_pred    = self.model.predict(X_train)
            train_f1  = f1_score(y_train, y_pred, average="weighted")

            logger.info(f"TF-IDF train weighted F1 : {train_f1:.4f}")
            print(f"  TF-IDF train F1 (weighted) : {train_f1:.4f}")

            return {"train_f1": round(train_f1, 4)}

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, texts: list) -> dict:
        """
        Predict sentiment for a list of texts.

        Returns
        -------
        dict with keys: predictions (str list), probabilities (float list)
        """
        X = self.vectorizer.transform(texts)
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)

        labels = self.label_encoder.inverse_transform(y_pred).tolist()
        confidences = [round(float(y_prob[i, y_pred[i]]), 4) for i in range(len(y_pred))]

        return {
            "predictions":  labels,
            "probabilities": confidences,
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self):
        """Persist vectorizer, model, and label encoder to disk."""
        save_nlp_object(TFIDF_VECTORIZER_PATH, self.vectorizer)
        save_nlp_object(TFIDF_MODEL_PATH, self.model)
        save_nlp_object(TFIDF_LABEL_ENCODER_PATH, self.label_encoder)
        logger.info("TF-IDF model artifacts saved")
        print("  TF-IDF model saved")

    def load(self):
        """Load saved artifacts from disk."""
        self.vectorizer    = load_nlp_object(TFIDF_VECTORIZER_PATH)
        self.model         = load_nlp_object(TFIDF_MODEL_PATH)
        self.label_encoder = load_nlp_object(TFIDF_LABEL_ENCODER_PATH)
        logger.info("TF-IDF model artifacts loaded")

    # ------------------------------------------------------------------
    # Top features (interpretability)
    # ------------------------------------------------------------------

    def top_features(self, n: int = 10) -> dict:
        """Return the top-n most important words per sentiment class."""
        feature_names = self.vectorizer.get_feature_names_out()
        result = {}
        for i, label in enumerate(self.label_encoder.classes_):
            coefs = self.model.coef_[i]
            top_idx = coefs.argsort()[-n:][::-1]
            result[label] = [feature_names[j] for j in top_idx]
        return result


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    texts  = ["great product love it", "terrible broke instantly", "it is okay average"]
    labels = ["Positive", "Negative", "Neutral"]

    model = TfidfSentimentModel()
    model.train(texts, labels)
    result = model.predict(["amazing quality"])
    print("Prediction:", result)
