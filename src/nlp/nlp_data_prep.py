"""
nlp/nlp_data_prep.py -- Load, clean, label, and split the Amazon reviews dataset.
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from logger import logger
from exception import CustomException
from nlp.nlp_config import (
    NLP_DATASET_PATH,
    NLP_ARTIFACT_DIR,
    NLP_PROCESSED_TRAIN,
    NLP_PROCESSED_TEST,
    TEXT_COLUMN,
    RATING_COLUMN,
    SENTIMENT_COLUMN,
    SENTIMENT_MAP,
    NLP_TEST_SIZE,
    NLP_RANDOM_STATE,
)
from nlp.utils import clean_text


class NLPDataPrep:
    """
    Handles the full data preparation pipeline for NLP:
      1. Load raw reviews CSV
      2. Clean review text
      3. Map star ratings to sentiment labels
      4. Train / test split
      5. Save processed CSVs
    """

    def __init__(self):
        self.dataset_path = NLP_DATASET_PATH

    def prepare(self) -> dict:
        """
        Returns
        -------
        dict with keys: train_path, test_path, train_df, test_df
        """
        try:
            logger.info("=" * 60)
            logger.info("NLP DATA PREPARATION STARTED")
            logger.info("=" * 60)

            # ---- Load ----
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Loaded reviews dataset : {df.shape}")
            print(f"\nLoaded reviews : {df.shape}")

            # ---- Drop rows with missing text ----
            df = df.dropna(subset=[TEXT_COLUMN, RATING_COLUMN]).reset_index(drop=True)
            logger.info(f"After dropping nulls   : {df.shape}")

            # ---- Clean text ----
            df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)

            # Remove rows where cleaning left empty string
            df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
            logger.info(f"After text cleaning    : {df.shape}")

            # ---- Map rating -> sentiment ----
            df[SENTIMENT_COLUMN] = df[RATING_COLUMN].map(SENTIMENT_MAP)
            logger.info(f"Sentiment distribution :\n{df[SENTIMENT_COLUMN].value_counts().to_string()}")

            print(f"\nSentiment distribution:")
            for label, count in df[SENTIMENT_COLUMN].value_counts().items():
                print(f"  {label:>10} : {count}")

            # ---- Train / Test split ----
            train_df, test_df = train_test_split(
                df,
                test_size=NLP_TEST_SIZE,
                random_state=NLP_RANDOM_STATE,
                stratify=df[SENTIMENT_COLUMN],
            )

            train_df = train_df.reset_index(drop=True)
            test_df  = test_df.reset_index(drop=True)

            logger.info(f"Train size : {len(train_df)} | Test size : {len(test_df)}")
            print(f"\nTrain : {len(train_df)} | Test : {len(test_df)}")

            # ---- Save ----
            os.makedirs(NLP_ARTIFACT_DIR, exist_ok=True)
            train_df.to_csv(NLP_PROCESSED_TRAIN, index=False)
            test_df.to_csv(NLP_PROCESSED_TEST,   index=False)

            logger.info("NLP DATA PREPARATION COMPLETED")

            return {
                "train_path": NLP_PROCESSED_TRAIN,
                "test_path":  NLP_PROCESSED_TEST,
                "train_df":   train_df,
                "test_df":    test_df,
            }

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing NLP Data Prep")
    print("=" * 60)

    prep   = NLPDataPrep()
    result = prep.prepare()

    print(f"\nTrain saved to : {result['train_path']}")
    print(f"Test  saved to : {result['test_path']}")
    print("=" * 60)
