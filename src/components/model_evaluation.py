import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from logger import logger
from exception import CustomException
from utils import load_object
from config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    TEST_DATA_PATH,
    EVALUATION_REPORT_PATH,
    PLOTS_DIR,
    TARGET_COLUMN,
    DROP_COLUMNS,
    EXPECTED_F1_THRESHOLD,
    ARTIFACT_DIR,
)
from entity.config_entity import ModelEvaluationConfig
from entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact
from components.feature_engineering import add_features


class ModelEvaluation:

    def __init__(self, trainer_artifact: ModelTrainerArtifact):
        self.eval_config = ModelEvaluationConfig(
            model_path=trainer_artifact.model_path,
            preprocessor_path=PREPROCESSOR_PATH,
            test_data_path=TEST_DATA_PATH,
            target_column=TARGET_COLUMN,
            evaluation_report_path=EVALUATION_REPORT_PATH,
            plots_dir=PLOTS_DIR,
        )
        self.trainer_artifact = trainer_artifact

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _plot_confusion_matrix(self, cm, save_path: str):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _plot_roc_curve(self, y_true, y_prob, roc_auc: float, save_path: str):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="steelblue", lw=2,
                label=f"ROC Curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logger.info("=" * 60)
            logger.info("MODEL EVALUATION STARTED")
            logger.info("=" * 60)

            cfg = self.eval_config

            # Load model and preprocessor
            model       = load_object(cfg.model_path)
            preprocessor = load_object(cfg.preprocessor_path)

            # Load and prepare test data (raw, before transformation)
            test_df = pd.read_csv(cfg.test_data_path)
            test_df = add_features(test_df)                           # feature engineering
            test_df = test_df.drop(columns=DROP_COLUMNS, errors="ignore")

            X_test = test_df.drop(columns=[cfg.target_column])
            y_test = test_df[cfg.target_column]

            X_test_scaled = preprocessor.transform(X_test)

            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

            # Metrics
            accuracy  = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall    = recall_score(y_test, y_pred, zero_division=0)
            f1        = f1_score(y_test, y_pred, zero_division=0)
            roc_auc   = roc_auc_score(y_test, y_prob)
            cm        = confusion_matrix(y_test, y_pred)
            clf_report = classification_report(y_test, y_pred, target_names=["No Failure", "Failure"])

            model_accepted = f1 >= EXPECTED_F1_THRESHOLD

            logger.info(f"Accuracy  : {accuracy:.4f}")
            logger.info(f"Precision : {precision:.4f}")
            logger.info(f"Recall    : {recall:.4f}")
            logger.info(f"F1 Score  : {f1:.4f}")
            logger.info(f"ROC AUC   : {roc_auc:.4f}")
            logger.info(f"Model Accepted : {model_accepted}")

            print(f"\n{'='*60}")
            print(f"  MODEL EVALUATION RESULTS")
            print(f"{'='*60}")
            print(f"  Best Model  : {self.trainer_artifact.best_model_name}")
            print(f"  Accuracy    : {accuracy:.4f}")
            print(f"  Precision   : {precision:.4f}")
            print(f"  Recall      : {recall:.4f}")
            print(f"  F1 Score    : {f1:.4f}")
            print(f"  ROC AUC     : {roc_auc:.4f}")
            print(f"  Accepted    : {'YES' if model_accepted else 'NO'}") 
            print(f"{'='*60}")
            print(f"\n{clf_report}")

            # Save evaluation report
            os.makedirs(os.path.dirname(cfg.evaluation_report_path), exist_ok=True)
            report = {
                "best_model": self.trainer_artifact.best_model_name,
                "accuracy":   round(accuracy,  4),
                "precision":  round(precision, 4),
                "recall":     round(recall,    4),
                "f1_score":   round(f1,        4),
                "roc_auc":    round(roc_auc,   4),
                "model_accepted": model_accepted,
                "confusion_matrix": cm.tolist(),
                "classification_report": clf_report,
            }
            with open(cfg.evaluation_report_path, "w") as f:
                json.dump(report, f, indent=4)

            logger.info(f"Evaluation report saved to : {cfg.evaluation_report_path}")

            # Save plots
            os.makedirs(cfg.plots_dir, exist_ok=True)
            cm_path  = os.path.join(cfg.plots_dir, "confusion_matrix.png")
            roc_path = os.path.join(cfg.plots_dir, "roc_curve.png")
            self._plot_confusion_matrix(cm, cm_path)
            self._plot_roc_curve(y_test, y_prob, roc_auc, roc_path)
            logger.info(f"Plots saved to : {cfg.plots_dir}")
            print(f"\nPlots saved → {cfg.plots_dir}")

            logger.info("MODEL EVALUATION COMPLETED")

            return ModelEvaluationArtifact(
                evaluation_report_path=cfg.evaluation_report_path,
                accuracy=round(accuracy,  4),
                precision=round(precision, 4),
                recall=round(recall,    4),
                f1_score=round(f1,        4),
                roc_auc=round(roc_auc,   4),
                model_accepted=model_accepted,
            )

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


# ===================================================
# Testing
# ===================================================

if __name__ == "__main__":
    from entity.artifact_entity import ModelTrainerArtifact

    trainer_art = ModelTrainerArtifact(
        model_path=MODEL_PATH,
        best_model_name="Unknown",
        train_f1_score=0.0,
        test_f1_score=0.0,
    )

    print("=" * 60)
    print("Testing Model Evaluation")
    print("=" * 60)

    obj      = ModelEvaluation(trainer_art)
    artifact = obj.initiate_model_evaluation()

    print("\nEvaluation Report :", artifact.evaluation_report_path)
    print("F1 Score          :", artifact.f1_score)
    print("ROC AUC           :", artifact.roc_auc)
    print("=" * 60)
