"""
ts_training_pipeline.py
────────────────────────
Orchestrates the full Time Series Anomaly Detection pipeline:

  Step 1: Load & engineer temporal features from AI4I sensor data.
  Step 2: Scale features and save scaler.
  Step 3: Train LSTM Autoencoder on *normal-only* windows.
  Step 4: Train Isolation Forest on all enriched features.
  Step 5: Score the full dataset with both models.
  Step 6: Evaluate combined predictions vs Machine failure labels.
  Step 7: Save artifacts & feature CSV.

Run from project root:
    python -c "from src.pipeline.ts_training_pipeline import TSTrainingPipeline; TSTrainingPipeline().run()"
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

from src.time_series.ts_config import (
    CLEAN_AI4I_PATH,
    SAVED_MODELS_TS_DIR,
    ARTIFACTS_TS_DIR,
    WINDOW_SIZE,
    TS_SCALER_PATH,
    TS_FEATURES_PATH,
)
from src.time_series.ts_features   import build_ts_features, get_feature_columns, TARGET_COL
from src.time_series.anomaly_detector import train_autoencoder, train_isolation_forest, score_anomalies
from src.time_series.ts_evaluator  import evaluate_anomaly_detection
from sklearn.preprocessing import StandardScaler


_SEP = "=" * 60


class TSTrainingPipeline:
    """End-to-end Time Series Anomaly Detection training pipeline."""

    def run(self):
        print(f"\n{_SEP}")
        print("  TIME SERIES ANOMALY DETECTION PIPELINE")
        print(f"{_SEP}\n")

        # ─────────────────────────────────────────────────────────────────────
        # Step 1 — Load raw data & build temporal features
        # ─────────────────────────────────────────────────────────────────────
        print("[1/6] Loading & engineering temporal features ...")
        df_raw = pd.read_csv(CLEAN_AI4I_PATH)
        df     = build_ts_features(df_raw)

        # Save enriched feature CSV for inspection / downstream use
        df.to_csv(TS_FEATURES_PATH, index=False)
        print(f"  Rows after feature engineering: {len(df):,}")
        print(f"  Feature CSV saved → {TS_FEATURES_PATH}")

        feat_cols = get_feature_columns(df)
        X = df[feat_cols].values
        y = df[TARGET_COL].values if TARGET_COL in df.columns else None

        print(f"  Feature dimensions: {X.shape[1]}")

        # ─────────────────────────────────────────────────────────────────────
        # Step 2 — Scale features & persist scaler
        # ─────────────────────────────────────────────────────────────────────
        print("\n[2/6] Scaling features ...")
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, TS_SCALER_PATH)
        print(f"  Scaler saved → {TS_SCALER_PATH}")

        # ─────────────────────────────────────────────────────────────────────
        # Step 3 — Train LSTM Autoencoder on *normal* windows
        # ─────────────────────────────────────────────────────────────────────
        print("\n[3/6] Training LSTM Autoencoder ...")
        if y is not None:
            normal_mask = (y == 0)
            X_normal    = X_scaled[normal_mask]
        else:
            X_normal = X_scaled  # No labels → treat all as normal (unsupervised)

        input_dim   = X_scaled.shape[1]
        ae_model, ae_threshold = train_autoencoder(X_normal, input_dim)

        # ─────────────────────────────────────────────────────────────────────
        # Step 4 — Train Isolation Forest on all features
        # ─────────────────────────────────────────────────────────────────────
        print("\n[4/6] Training Isolation Forest ...")
        iso_forest = train_isolation_forest(X_scaled)

        # ── Save normalisation bounds for TSPredictor ──────────────────────
        # Compute IF score bounds on training data
        if_scores_all = iso_forest.score_samples(X_scaled)
        ae_mse_sample = []
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from src.time_series.ts_config import WINDOW_SIZE, STEP_SIZE
        seqs_tmp = []
        for i in range(0, len(X_scaled) - WINDOW_SIZE + 1, STEP_SIZE):
            seqs_tmp.append(X_scaled[i : i + WINDOW_SIZE])
        if seqs_tmp:
            t = torch.tensor(np.array(seqs_tmp, dtype=np.float32))
            ae_model.eval()
            with torch.no_grad():
                for i in range(0, len(t), 256):
                    mse_b = ae_model.reconstruction_error(t[i:i+256])
                    ae_mse_sample.extend(mse_b.cpu().numpy().tolist())

        bounds = {
            "if_min": float(if_scores_all.min()),
            "if_max": float(if_scores_all.max()),
            "ae_max": float(max(ae_mse_sample)) if ae_mse_sample else 1.0,
        }
        joblib.dump(bounds, TS_SCALER_PATH.replace("ts_scaler", "ts_score_bounds"))

        # ─────────────────────────────────────────────────────────────────────
        # Step 5 — Score full dataset
        # ─────────────────────────────────────────────────────────────────────
        print("\n[5/6] Scoring full dataset with dual anomaly signals ...")
        scores_df = score_anomalies(X_scaled, ae_model, ae_threshold, iso_forest)

        # Attach to main DataFrame
        for col in scores_df.columns:
            df[col] = scores_df[col].values

        # ─────────────────────────────────────────────────────────────────────
        # Step 6 — Evaluate
        # ─────────────────────────────────────────────────────────────────────
        print("\n[6/6] Evaluating anomaly detection performance ...")
        if y is not None:
            y_pred         = df["combined_flag"].values
            anomaly_scores = df["anomaly_score"].values
            metrics        = evaluate_anomaly_detection(y, y_pred, anomaly_scores)
        else:
            print("  No ground-truth labels found — skipping supervised evaluation.")
            metrics = {}

        print(f"\n{_SEP}")
        f1  = metrics.get("f1_score",  "N/A")
        auc = metrics.get("roc_auc",   "N/A")
        dr  = metrics.get("detection_rate_pct", "N/A")
        print(f"  TS Pipeline Complete | F1: {f1} | ROC-AUC: {auc} | Detection Rate: {dr}%")
        print(f"{_SEP}\n")

        return metrics
