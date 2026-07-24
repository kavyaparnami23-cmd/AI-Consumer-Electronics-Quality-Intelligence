import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from logger import logger
from exception import CustomException
from utils import save_object
from config import (
    ARTIFACT_DIR,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
    DROP_COLUMNS,
    SMOTE_SAMPLING_STRATEGY,
    SMOTE_K_NEIGHBORS,
    RANDOM_STATE,
)
from entity.config_entity import DataTransformationConfig
from entity.artifact_entity import DataTransformationArtifact, FeatureEngineeringArtifact


TRANSFORMED_TRAIN_PATH = os.path.join(ARTIFACT_DIR, "transformed_train.npz")
TRANSFORMED_TEST_PATH  = os.path.join(ARTIFACT_DIR, "transformed_test.npz")


class DataTransformation:

    def __init__(self, fe_artifact: FeatureEngineeringArtifact):
        self.transformation_config = DataTransformationConfig(
            engineered_train_path=fe_artifact.engineered_train_path,
            engineered_test_path=fe_artifact.engineered_test_path,
            preprocessor_path=PREPROCESSOR_PATH,
            target_column=TARGET_COLUMN,
            drop_columns=DROP_COLUMNS,
            smote_sampling_strategy=SMOTE_SAMPLING_STRATEGY,
            smote_k_neighbors=SMOTE_K_NEIGHBORS,
            random_state=RANDOM_STATE,
        )

    # ------------------------------------------------------------------
    # Build preprocessor
    # ------------------------------------------------------------------

    @staticmethod
    def get_preprocessor() -> Pipeline:
        """Return a sklearn Pipeline that applies StandardScaler."""
        return Pipeline(steps=[("scaler", StandardScaler())])

    # ------------------------------------------------------------------
    # Prepare X / y splits
    # ------------------------------------------------------------------

    def _split_features_target(self, df: pd.DataFrame):
        cfg = self.transformation_config
        df  = df.drop(columns=cfg.drop_columns, errors="ignore")
        X   = df.drop(columns=[cfg.target_column])
        y   = df[cfg.target_column]
        return X, y

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("=" * 60)
            logger.info("DATA TRANSFORMATION STARTED")
            logger.info("=" * 60)

            cfg = self.transformation_config

            # Load engineered data
            train_df = pd.read_csv(cfg.engineered_train_path)
            test_df  = pd.read_csv(cfg.engineered_test_path)

            X_train, y_train = self._split_features_target(train_df)
            X_test,  y_test  = self._split_features_target(test_df)

            logger.info(f"X_train shape : {X_train.shape} | class dist : {dict(y_train.value_counts())}")
            logger.info(f"X_test  shape : {X_test.shape}")

            # Fit preprocessor on training data only
            preprocessor = self.get_preprocessor()
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled  = preprocessor.transform(X_test)

            logger.info("StandardScaler fitted on training data")

            # Apply SMOTE to training data only
            smote = SMOTE(
                sampling_strategy=cfg.smote_sampling_strategy,
                k_neighbors=cfg.smote_k_neighbors,
                random_state=cfg.random_state,
            )
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

            logger.info(f"After SMOTE → X_train shape : {X_train_res.shape}")
            logger.info(f"After SMOTE → class dist    : {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

            print(f"\nData Transformation Complete")
            print(f"  Original train class dist : {dict(y_train.value_counts())}")
            print(f"  After SMOTE class dist    : {dict(zip(*np.unique(y_train_res, return_counts=True)))}")
            print(f"  X_train shape : {X_train_res.shape} | X_test shape : {X_test_scaled.shape}")

            # Save preprocessor
            save_object(cfg.preprocessor_path, preprocessor)
            logger.info(f"Preprocessor saved to : {cfg.preprocessor_path}")

            # Save transformed arrays
            os.makedirs(os.path.dirname(TRANSFORMED_TRAIN_PATH), exist_ok=True)

            np.savez_compressed(
                TRANSFORMED_TRAIN_PATH,
                X=X_train_res,
                y=y_train_res,
            )
            np.savez_compressed(
                TRANSFORMED_TEST_PATH,
                X=X_test_scaled,
                y=y_test.values,
            )

            logger.info("DATA TRANSFORMATION COMPLETED")

            return DataTransformationArtifact(
                preprocessor_path=cfg.preprocessor_path,
                transformed_train_path=TRANSFORMED_TRAIN_PATH,
                transformed_test_path=TRANSFORMED_TEST_PATH,
            )

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


# ===================================================
# Testing (requires FE to have run first)
# ===================================================

if __name__ == "__main__":
    from entity.artifact_entity import FeatureEngineeringArtifact
    from config import ARTIFACT_DIR

    fe_art = FeatureEngineeringArtifact(
        engineered_train_path=os.path.join(ARTIFACT_DIR, "engineered_train.csv"),
        engineered_test_path=os.path.join(ARTIFACT_DIR, "engineered_test.csv"),
    )

    print("=" * 60)
    print("Testing Data Transformation")
    print("=" * 60)

    obj      = DataTransformation(fe_art)
    artifact = obj.initiate_data_transformation()

    print("\nPreprocessor Path      :", artifact.preprocessor_path)
    print("Transformed Train Path :", artifact.transformed_train_path)
    print("Transformed Test  Path :", artifact.transformed_test_path)
    print("=" * 60)
