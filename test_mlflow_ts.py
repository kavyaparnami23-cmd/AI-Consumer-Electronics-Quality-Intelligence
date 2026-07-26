"""
Quick smoke-test: log the already-saved TS results into MLflow
without re-running training. Uses saved models + report JSON.
"""
import sys, os, json
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import mlflow, joblib, torch
from src.mlflow_utils.mlflow_config import (
    MLFLOW_TRACKING_URI, EXPERIMENT_TIME_SERIES, MODEL_REGISTRY_AE
)
from src.mlflow_utils.mlflow_tracker import MLflowTracker
from src.time_series.ts_config import (
    TS_REPORT_PATH, AE_MODEL_PATH, IF_MODEL_PATH,
    HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, WINDOW_SIZE,
    EPOCHS, LEARNING_RATE, BATCH_SIZE, PATIENCE,
    THRESHOLD_PERCENTILE, IF_N_ESTIMATORS, IF_CONTAMINATION,
    TS_FEATURES_PATH, THRESHOLD_PATH, TS_SCALER_PATH,
)
from src.time_series.anomaly_detector import load_ae_model

# Load report
with open(TS_REPORT_PATH) as f:
    report = json.load(f)

input_dim = joblib.load(TS_SCALER_PATH).n_features_in_
ae_model, ae_threshold = load_ae_model(input_dim)

tracker = MLflowTracker(EXPERIMENT_TIME_SERIES, tags={"module": "time_series", "run_type": "log_existing"})
tracker.start(run_name="LSTM-AE-IsolationForest-v1")

tracker.log_params({
    "ae_type": "LSTM-Autoencoder", "window_size": WINDOW_SIZE,
    "hidden_dim": HIDDEN_DIM, "latent_dim": LATENT_DIM,
    "num_layers": NUM_LAYERS, "batch_size": BATCH_SIZE,
    "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
    "patience": PATIENCE, "ae_threshold_pct": THRESHOLD_PERCENTILE,
    "if_n_estimators": IF_N_ESTIMATORS, "if_contamination": IF_CONTAMINATION,
    "input_dim": input_dim, "train_on_normals_only": True,
    "anomaly_score_blend": "60pct_AE+40pct_IF",
})

tracker.log_metrics({
    "ae_threshold":       ae_threshold,
    "precision":          report.get("precision",          0),
    "recall":             report.get("recall",             0),
    "f1_score":           report.get("f1_score",           0),
    "roc_auc":            report.get("roc_auc",            0),
    "average_precision":  report.get("average_precision",  0),
    "detection_rate_pct": report.get("detection_rate_pct",0),
    "true_positives":     float(report.get("true_positives",  0)),
    "false_positives":    float(report.get("false_positives", 0)),
    "n_true_failures":    float(report.get("n_true_failures", 0)),
})

tracker.log_model_pytorch(ae_model, "lstm_autoencoder", registered_name=MODEL_REGISTRY_AE)
tracker.log_artifact(TS_REPORT_PATH,   "reports")
tracker.log_artifact(IF_MODEL_PATH,    "models")

tracker.end()
print(f"\n✅ TS experiment logged successfully!")
print(f"   ROC-AUC: {report.get('roc_auc')} | Detection Rate: {report.get('detection_rate_pct')}%")
print(f"\n   Run:  python start_mlflow_ui.py")
print(f"   Then: http://localhost:5000")
