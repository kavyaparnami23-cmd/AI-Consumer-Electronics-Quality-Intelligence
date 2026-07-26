import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb

from logger import logger
from exception import CustomException
from utils import save_object
from config import (
    MODEL_PATH,
    EXPECTED_F1_THRESHOLD,
    RANDOM_STATE,
)
from entity.config_entity import ModelTrainerConfig
from entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact


# ------------------------------------------------------------------
# Model catalogue + hyperparameter search spaces
# ------------------------------------------------------------------

def get_models_and_params(random_state: int) -> dict:
    return {
        "LogisticRegression": {
            "model": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            ),
            "params": {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["lbfgs", "liblinear"],
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 15],
                "min_samples_split": [2, 5, 10],
                "max_features": ["sqrt", "log2"],
            },
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=random_state,
                verbosity=0,
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "scale_pos_weight": [1, 3, 5, 10],
            },
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(
                class_weight="balanced",
                random_state=random_state,
                verbose=-1,
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [-1, 5, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 63, 127],
                "subsample": [0.7, 0.8, 1.0],
            },
        },
    }


class ModelTrainer:

    def __init__(self, transformation_artifact: DataTransformationArtifact):
        self.trainer_config = ModelTrainerConfig(
            preprocessed_train_path=transformation_artifact.transformed_train_path,
            model_path=MODEL_PATH,
            expected_f1_threshold=EXPECTED_F1_THRESHOLD,
            random_state=RANDOM_STATE,
        )
        self.test_path = transformation_artifact.transformed_test_path

    # ------------------------------------------------------------------
    # Training helper
    # ------------------------------------------------------------------

    def _train_and_tune(self, name: str, model_info: dict, X_train, y_train, cv):
        logger.info(f"  Training : {name}")

        search = RandomizedSearchCV(
            estimator=model_info["model"],
            param_distributions=model_info["params"],
            n_iter=20,
            scoring="f1",
            cv=cv,
            random_state=self.trainer_config.random_state,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_estimator = search.best_estimator_
        cv_f1          = search.best_score_

        logger.info(f"    Best params   : {search.best_params_}")
        logger.info(f"    CV F1 score   : {cv_f1:.4f}")

        return best_estimator, cv_f1

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logger.info("=" * 60)
            logger.info("MODEL TRAINING STARTED")
            logger.info("=" * 60)

            cfg = self.trainer_config

            # Load transformed arrays
            train_data = np.load(cfg.preprocessed_train_path, allow_pickle=True)
            test_data  = np.load(self.test_path, allow_pickle=True)

            X_train, y_train = train_data["X"], train_data["y"]
            X_test,  y_test  = test_data["X"],  test_data["y"]

            logger.info(f"X_train : {X_train.shape} | X_test : {X_test.shape}")

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)

            models_params = get_models_and_params(cfg.random_state)

            results = {}
            print("\nTraining models (RandomizedSearchCV with 5-fold CV) ...")

            for name, info in models_params.items():
                print(f"  >> {name} ...", end="", flush=True)
                best_model, cv_f1 = self._train_and_tune(name, info, X_train, y_train, cv)

                # Evaluate on test set
                y_pred = best_model.predict(X_test)
                test_f1 = f1_score(y_test, y_pred)

                results[name] = {
                    "model":   best_model,
                    "cv_f1":   cv_f1,
                    "test_f1": test_f1,
                }
                print(f" CV F1={cv_f1:.4f} | Test F1={test_f1:.4f}")
                logger.info(f"  {name} → CV F1={cv_f1:.4f} | Test F1={test_f1:.4f}")

            # ----------------------------------------------------------
            # Select best model by test F1
            # ----------------------------------------------------------
            best_name  = max(results, key=lambda n: results[n]["test_f1"])
            best_info  = results[best_name]
            best_model = best_info["model"]

            logger.info(f"\nBest Model : {best_name}  (Test F1 = {best_info['test_f1']:.4f})")
            print(f"\n{'='*60}")
            print(f"  Best Model : {best_name}")
            print(f"  Train CV F1: {best_info['cv_f1']:.4f}")
            print(f"  Test F1    : {best_info['test_f1']:.4f}")
            print(f"{'='*60}")

            if best_info["test_f1"] < cfg.expected_f1_threshold:
                msg = (
                    f"Best model F1={best_info['test_f1']:.4f} is below "
                    f"threshold {cfg.expected_f1_threshold}. Pipeline continues but review results."
                )
                logger.warning(msg)
                print(f"\n[WARNING] {msg}")

            # Save best model
            save_object(cfg.model_path, best_model)
            logger.info(f"Model saved to : {cfg.model_path}")

            logger.info("MODEL TRAINING COMPLETED")

            return ModelTrainerArtifact(
                model_path=cfg.model_path,
                best_model_name=best_name,
                train_f1_score=round(best_info["cv_f1"], 4),
                test_f1_score=round(best_info["test_f1"], 4),
            )

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


# ===================================================
# Testing
# ===================================================

if __name__ == "__main__":
    from entity.artifact_entity import DataTransformationArtifact
    from config import ARTIFACT_DIR

    dt_art = DataTransformationArtifact(
        preprocessor_path=os.path.join("saved_models", "preprocessor.pkl"),
        transformed_train_path=os.path.join(ARTIFACT_DIR, "transformed_train.npz"),
        transformed_test_path=os.path.join(ARTIFACT_DIR, "transformed_test.npz"),
    )

    print("=" * 60)
    print("Testing Model Trainer")
    print("=" * 60)

    obj      = ModelTrainer(dt_art)
    artifact = obj.initiate_model_training()

    print("\nBest Model :", artifact.best_model_name)
    print("Train F1   :", artifact.train_f1_score)
    print("Test  F1   :", artifact.test_f1_score)
    print("=" * 60)
