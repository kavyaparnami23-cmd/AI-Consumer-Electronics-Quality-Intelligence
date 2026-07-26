import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from logger import logger
from exception import CustomException
from config import (
    DATASET_PATH,
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TEST_SIZE,
    RANDOM_STATE
)

from entity.config_entity import DataIngestionConfig
from entity.artifact_entity import DataIngestionArtifact


class DataIngestion:

    def __init__(self):

        self.ingestion_config = DataIngestionConfig(
            dataset_path=DATASET_PATH,
            raw_data_path=RAW_DATA_PATH,
            train_data_path=TRAIN_DATA_PATH,
            test_data_path=TEST_DATA_PATH
        )

    def initiate_data_ingestion(self):

        try:

            logger.info("=" * 60)
            logger.info("DATA INGESTION STARTED")
            logger.info("=" * 60)

            print("\nReading Dataset...")

            df = pd.read_csv(self.ingestion_config.dataset_path)

            logger.info(f"Dataset Loaded Successfully")
            logger.info(f"Dataset Shape : {df.shape}")

            print(f"Dataset Shape : {df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logger.info("Raw Dataset Saved Successfully")

            print("\nSplitting Train/Test Dataset...")

            train_set, test_set = train_test_split(
                df,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logger.info(f"Train Shape : {train_set.shape}")
            logger.info(f"Test Shape : {test_set.shape}")

            print(f"\nTrain Shape : {train_set.shape}")
            print(f"Test Shape  : {test_set.shape}")

            logger.info("DATA INGESTION COMPLETED")

            return DataIngestionArtifact(

                train_file_path=self.ingestion_config.train_data_path,

                test_file_path=self.ingestion_config.test_data_path

            )

        except Exception as e:

            logger.error(str(e))

            raise CustomException(e, sys)


# ===================================================
# Testing
# ===================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing Data Ingestion")
    print("=" * 60)

    obj = DataIngestion()

    artifact = obj.initiate_data_ingestion()

    print("\nData Ingestion Completed Successfully\n")

    print("Train File :", artifact.train_file_path)

    print("Test File  :", artifact.test_file_path)

    print("=" * 60)