from dataclasses import dataclass
from typing import Optional


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    validation_report_path: str
    message: str


@dataclass
class FeatureEngineeringArtifact:
    engineered_train_path: str
    engineered_test_path: str


@dataclass
class DataTransformationArtifact:
    preprocessor_path: str
    transformed_train_path: str   # .npz file (X_train, y_train after SMOTE)
    transformed_test_path: str    # .npz file (X_test,  y_test)


@dataclass
class ModelTrainerArtifact:
    model_path: str
    best_model_name: str
    train_f1_score: float
    test_f1_score: float


@dataclass
class ModelEvaluationArtifact:
    evaluation_report_path: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    model_accepted: bool
