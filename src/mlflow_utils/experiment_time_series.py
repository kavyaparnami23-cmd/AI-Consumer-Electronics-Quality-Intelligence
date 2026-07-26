"""
experiment_time_series.py
──────────────────────────
MLflow experiment for the Time Series Anomaly Detection pipeline.

Logs:
  - Autoencoder architecture (hidden_dim, latent_dim, window_size, epochs)
  - Isolation Forest config (n_estimators, contamination)
  - Evaluation metrics (ROC-AUC, AP, F1, detection_rate)
  - AE threshold value
  - PyTorch AE model + IF model as artifacts
  - TS anomaly report JSON + feature CSV
"""

import sys
import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from mlflow_utils.mlflow_config import (
    EXPERIMENT_TIME_SERIES, MODEL_REGISTRY_AE, MLFLOW_TRACKING_URI
)
from mlflow_utils.mlflow_tracker import MLflowTracker


def run_ts_experiment():
    """
    Trains the full TS anomaly detection pipeline inside an MLflow run.
    Emits per-epoch AE training loss as step metrics for chart view.
    """
    print("\n" + "=" * 60)
    print("  MLflow Experiment: Time Series Anomaly Detection")
    print("=" * 60)

    import numpy as np
    import pandas as pd
    import joblib
    import torch

    from time_series.ts_config import (
        CLEAN_AI4I_PATH, TS_SCALER_PATH, AE_MODEL_PATH, IF_MODEL_PATH,
        THRESHOLD_PATH, TS_REPORT_PATH, TS_FEATURES_PATH,
        WINDOW_SIZE, STEP_SIZE, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS,
        BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE,
        THRESHOLD_PERCENTILE, IF_N_ESTIMATORS, IF_CONTAMINATION,
    )
    from time_series.ts_features   import build_ts_features, get_feature_columns, TARGET_COL
    from time_series.lstm_autoencoder import LSTMAutoEncoder
    from sklearn.preprocessing import StandardScaler

    # ── Build features ────────────────────────────────────────────────────────
    df_raw    = pd.read_csv(CLEAN_AI4I_PATH)
    df        = build_ts_features(df_raw)
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].values
    y = df[TARGET_COL].values if TARGET_COL in df.columns else None

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, TS_SCALER_PATH)

    input_dim   = X_scaled.shape[1]
    normal_mask = (y == 0) if y is not None else np.ones(len(X_scaled), dtype=bool)
    X_normal    = X_scaled[normal_mask]

    # ── MLflow run ────────────────────────────────────────────────────────────
    tracker = MLflowTracker(EXPERIMENT_TIME_SERIES, tags={"module": "time_series"})
    tracker.start(run_name="LSTM-AE-IsolationForest")

    tracker.log_params({
        "ae_type":              "LSTM-Autoencoder",
        "input_dim":            input_dim,
        "window_size":          WINDOW_SIZE,
        "hidden_dim":           HIDDEN_DIM,
        "latent_dim":           LATENT_DIM,
        "num_layers":           NUM_LAYERS,
        "batch_size":           BATCH_SIZE,
        "epochs":               EPOCHS,
        "learning_rate":        LEARNING_RATE,
        "patience":             PATIENCE,
        "ae_threshold_pct":     THRESHOLD_PERCENTILE,
        "if_n_estimators":      IF_N_ESTIMATORS,
        "if_contamination":     IF_CONTAMINATION,
        "normal_train_samples": int(normal_mask.sum()),
        "total_samples":        len(X_scaled),
        "n_features":           input_dim,
        "train_on_normals_only": True,
        "anomaly_score_blend":  "60%AE+40%IF",
    })

    # ── Train AE with per-epoch MLflow logging ────────────────────────────────
    from torch.utils.data import DataLoader, TensorDataset

    model = LSTMAutoEncoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        seq_len=WINDOW_SIZE,
    )

    seqs = []
    for i in range(0, len(X_normal) - WINDOW_SIZE + 1, STEP_SIZE):
        seqs.append(X_normal[i : i + WINDOW_SIZE])
    seqs   = np.array(seqs, dtype=np.float32)
    tensor = torch.tensor(seqs)
    loader = DataLoader(TensorDataset(tensor), batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion   = torch.nn.MSELoss()
    best_loss   = float("inf")
    patience_ct = 0
    best_state  = None

    print(f"  Training AE on {len(seqs):,} normal sequences ...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for (bx,) in loader:
            optimizer.zero_grad()
            recon = model(bx)
            loss  = criterion(recon, bx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(bx)
        ep_loss /= len(seqs)
        scheduler.step(ep_loss)

        # Log epoch to MLflow
        tracker.log_metric("ae_train_loss", ep_loss, step=epoch)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{EPOCHS} | Loss: {ep_loss:.6f}")

        if ep_loss < best_loss:
            best_loss   = ep_loss
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, AE_MODEL_PATH)

    # Compute threshold
    model.eval()
    all_mse = []
    with torch.no_grad():
        for i in range(0, len(tensor), 256):
            mse = model.reconstruction_error(tensor[i : i + 256])
            all_mse.extend(mse.numpy().tolist())
    threshold = float(np.percentile(all_mse, THRESHOLD_PERCENTILE))
    joblib.dump(threshold, THRESHOLD_PATH)
    print(f"  AE Threshold (P{THRESHOLD_PERCENTILE}): {threshold:.6f}")
    tracker.log_metric("ae_threshold", threshold)

    # ── Train Isolation Forest ────────────────────────────────────────────────
    from time_series.anomaly_detector import train_isolation_forest, score_anomalies
    iso_forest = train_isolation_forest(X_scaled)

    # ── Score & evaluate ──────────────────────────────────────────────────────
    scores_df = score_anomalies(X_scaled, model, threshold, iso_forest)
    if y is not None:
        from time_series.ts_evaluator import evaluate_anomaly_detection
        metrics = evaluate_anomaly_detection(
            y,
            scores_df["combined_flag"].values,
            scores_df["anomaly_score"].values,
        )
        tracker.log_metrics({
            "precision":          metrics["precision"],
            "recall":             metrics["recall"],
            "f1_score":           metrics["f1_score"],
            "roc_auc":            metrics["roc_auc"],
            "average_precision":  metrics["average_precision"],
            "detection_rate_pct": metrics["detection_rate_pct"],
            "true_positives":     float(metrics["true_positives"]),
            "false_positives":    float(metrics["false_positives"]),
        })

    # ── Log model & artifacts ─────────────────────────────────────────────────
    tracker.log_model_pytorch(model, "lstm_autoencoder",
                              registered_name=MODEL_REGISTRY_AE)
    tracker.log_artifact(TS_REPORT_PATH,    "reports")
    tracker.log_artifact(TS_FEATURES_PATH,  "features")
    tracker.log_artifact(IF_MODEL_PATH,     "models")

    tracker.end()
    if y is not None:
        print(f"  ✅ TS run logged | ROC-AUC: {metrics['roc_auc']:.4f} | "
              f"Detection Rate: {metrics['detection_rate_pct']}%")
    else:
        print("  ✅ TS run logged (no ground-truth labels).")


if __name__ == "__main__":
    run_ts_experiment()
