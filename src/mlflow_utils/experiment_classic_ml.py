"""
experiment_classic_ml.py
─────────────────────────
MLflow experiment for the Classic ML pipeline (XGBoost / LightGBM / RandomForest).
Logs hyperparameters, evaluation metrics, the trained model, and the evaluation report.
"""

import sys
import os
import json
import mlflow

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from mlflow_utils.mlflow_config import (
    EXPERIMENT_CLASSIC_ML, MODEL_REGISTRY_CLASSIC, MLFLOW_TRACKING_URI
)
from mlflow_utils.mlflow_tracker import MLflowTracker


def run_classic_ml_experiment():
    """
    Runs the classic ML training pipeline inside an MLflow run.
    Logs:
      - Model hyperparameters
      - Train / Test F1, Accuracy, Precision, Recall, ROC-AUC
      - Trained model artifact (sklearn-compatible)
      - Evaluation JSON report
    """
    print("\n" + "=" * 60)
    print("  MLflow Experiment: Classic ML Sensor Failure")
    print("=" * 60)

    from components.data_ingestion      import DataIngestion
    from components.data_validation     import DataValidation
    from components.feature_engineering import FeatureEngineering
    from components.data_transformation import DataTransformation
    from components.model_trainer       import ModelTrainer
    from components.model_evaluation    import ModelEvaluation

    ingestion    = DataIngestion()
    ing_art      = ingestion.initiate_data_ingestion()

    validation   = DataValidation()
    val_art      = validation.initiate_data_validation()

    fe           = FeatureEngineering()
    fe_art       = fe.initiate_feature_engineering()

    transformation = DataTransformation(fe_art)
    trans_art      = transformation.initiate_data_transformation()

    trainer      = ModelTrainer(trans_art)
    train_art    = trainer.initiate_model_training()

    evaluator    = ModelEvaluation(train_art)
    eval_art     = evaluator.initiate_model_evaluation()

    # Load evaluation JSON for artifact logging
    eval_report_path = os.path.join(BASE_DIR, "artifacts", "evaluation_report.json")

    # Load trained model for logging
    import joblib
    model_path = os.path.join(BASE_DIR, "saved_models", "model.pkl")
    model = joblib.load(model_path)
    best_name = train_art.best_model_name

    # Start MLflow run
    tracker = MLflowTracker(EXPERIMENT_CLASSIC_ML, tags={"module": "classic_ml"})
    tracker.start(run_name=f"{best_name}-run")

    # Params
    tracker.log_param("model_name",    best_name)
    tracker.log_param("dataset",       "AI4I-2020-Predictive-Maintenance")
    tracker.log_param("smote_applied", True)

    # Metrics
    tracker.log_metrics({
        "train_f1":  float(train_art.train_f1_score),
        "test_f1":   float(train_art.test_f1_score),
        "accuracy":  float(eval_art.accuracy),
        "precision": float(eval_art.precision),
        "recall":    float(eval_art.recall),
        "roc_auc":   float(eval_art.roc_auc),
    })

    # Model artifact
    tracker.log_model_sklearn(model, "sensor_failure_classifier",
                              registered_name=MODEL_REGISTRY_CLASSIC)

    # JSON report
    tracker.log_artifact(eval_report_path, "reports")

    tracker.end()
    print(f"  ✅ Classic ML run logged — Best: {best_name} | Test F1: {train_art.test_f1_score:.4f}")


if __name__ == "__main__":
    run_classic_ml_experiment()
