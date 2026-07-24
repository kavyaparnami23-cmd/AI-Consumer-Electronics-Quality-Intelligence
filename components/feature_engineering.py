import os
import sys

import pandas as pd

from logger import logger
from exception import CustomException
from config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    ARTIFACT_DIR,
)
from entity.config_entity import FeatureEngineeringConfig
from entity.artifact_entity import FeatureEngineeringArtifact


ENGINEERED_TRAIN_PATH = os.path.join(ARTIFACT_DIR, "engineered_train.csv")
ENGINEERED_TEST_PATH  = os.path.join(ARTIFACT_DIR, "engineered_test.csv")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-driven interaction features for the AI4I dataset.

    Features added
    --------------
    Temp_diff  : Process temp − Air temp  (thermal gradient indicator)
    Power      : Torque × Rotational speed (mechanical power proxy, W)
    Wear_rate  : Tool wear / Rotational speed (wear per rotation unit)
    """
    df = df.copy()

    df["Temp_diff"]  = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power"]      = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    df["Wear_rate"]  = df["Tool wear [min]"] / (df["Rotational speed [rpm]"] + 1e-6)

    return df


class FeatureEngineering:

    def __init__(self):
        self.config = FeatureEngineeringConfig(
            train_data_path=TRAIN_DATA_PATH,
            test_data_path=TEST_DATA_PATH,
            engineered_train_path=ENGINEERED_TRAIN_PATH,
            engineered_test_path=ENGINEERED_TEST_PATH,
        )

    def initiate_feature_engineering(self) -> FeatureEngineeringArtifact:
        try:
            logger.info("=" * 60)
            logger.info("FEATURE ENGINEERING STARTED")
            logger.info("=" * 60)

            train_df = pd.read_csv(self.config.train_data_path)
            test_df  = pd.read_csv(self.config.test_data_path)

            logger.info(f"Train shape before FE : {train_df.shape}")
            logger.info(f"Test  shape before FE : {test_df.shape}")

            train_df = add_features(train_df)
            test_df  = add_features(test_df)

            logger.info(f"Train shape after  FE : {train_df.shape}")
            logger.info(f"Test  shape after  FE : {test_df.shape}")
            logger.info(f"New features added    : Temp_diff, Power, Wear_rate")

            os.makedirs(os.path.dirname(self.config.engineered_train_path), exist_ok=True)

            train_df.to_csv(self.config.engineered_train_path, index=False)
            test_df.to_csv(self.config.engineered_test_path,   index=False)

            print(f"\nFeature Engineering Done")
            print(f"  New features : Temp_diff, Power, Wear_rate")
            print(f"  Train shape  : {train_df.shape}")
            print(f"  Test  shape  : {test_df.shape}")

            logger.info("FEATURE ENGINEERING COMPLETED")

            return FeatureEngineeringArtifact(
                engineered_train_path=self.config.engineered_train_path,
                engineered_test_path=self.config.engineered_test_path,
            )

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


# ===================================================
# Testing
# ===================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Feature Engineering")
    print("=" * 60)

    obj      = FeatureEngineering()
    artifact = obj.initiate_feature_engineering()

    print("\nEngineered Train :", artifact.engineered_train_path)
    print("Engineered Test  :", artifact.engineered_test_path)
    print("=" * 60)
