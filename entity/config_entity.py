from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    dataset_path: str
    raw_data_path: str
    train_data_path: str
    test_data_path: str


@dataclass
class DataValidationConfig:
    train_data_path: str
    test_data_path: str
    validation_report_path: str
    required_columns: list
    target_column: str


@dataclass
class FeatureEngineeringConfig:
    train_data_path: str
    test_data_path: str
    engineered_train_path: str
    engineered_test_path: str


@dataclass
class DataTransformationConfig:
    engineered_train_path: str
    engineered_test_path: str
    preprocessor_path: str
    target_column: str
    drop_columns: list
    smote_sampling_strategy: float
    smote_k_neighbors: int
    random_state: int


@dataclass
class ModelTrainerConfig:
    preprocessed_train_path: str   # path to transformed train array (.npz)
    model_path: str
    expected_f1_threshold: float
    random_state: int


@dataclass
class ModelEvaluationConfig:
    model_path: str
    preprocessor_path: str
    test_data_path: str
    target_column: str
    evaluation_report_path: str
    plots_dir: str


@dataclass
class PredictionConfig:
    model_path: str
    preprocessor_path: str
    target_column: str
    drop_columns: list
