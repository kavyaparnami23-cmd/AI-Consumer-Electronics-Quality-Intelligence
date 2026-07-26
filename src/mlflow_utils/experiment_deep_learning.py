"""
experiment_deep_learning.py
────────────────────────────
MLflow experiment for the Deep Learning pipeline (BiLSTM + Attention & 1D-CNN).

Logs:
  - Architecture hyperparameters (hidden_dim, num_layers, window_size, etc.)
  - Per-epoch train_loss / val_loss / val_f1  (visible as charts in MLflow UI)
  - Final Accuracy, Precision, Recall, F1 for both LSTM and 1D-CNN
  - PyTorch model weights (LSTM registered in Model Registry)
  - Evaluation JSON report
"""

import sys
import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import mlflow
import torch

from mlflow_utils.mlflow_config import (
    EXPERIMENT_DEEP_LEARNING, MODEL_REGISTRY_LSTM, MODEL_REGISTRY_CNN,
    MLFLOW_TRACKING_URI,
)
from mlflow_utils.mlflow_tracker import MLflowTracker
from deep_learning.dl_config import (
    CLEAN_AI4I_PATH, LSTM_MODEL_PATH, CNN_MODEL_PATH,
    WINDOW_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE, REPORT_PATH,
)
from deep_learning.dl_data_prep  import prepare_dl_data, FEATURE_COLS
from deep_learning.lstm_model    import SensorLSTM
from deep_learning.cnn_model     import Sensor1DCNN
from deep_learning.dl_trainer    import DLTrainer, FocalLoss
from deep_learning.dl_evaluator  import DLEvaluator


def run_dl_experiment():
    """Train both DL models with MLflow tracking (epoch-level logging)."""
    print("\n" + "=" * 60)
    print("  MLflow Experiment: Deep Learning Sensor Failure")
    print("=" * 60)

    # ── Data prep ─────────────────────────────────────────────────────────────
    train_loader, val_loader, input_dim = prepare_dl_data(CLEAN_AI4I_PATH)
    n_features = input_dim

    # ─────────────────────────────────────────────────────────────────────────
    # Run 1: BiLSTM + Attention
    # ─────────────────────────────────────────────────────────────────────────
    tracker_lstm = MLflowTracker(EXPERIMENT_DEEP_LEARNING, tags={"module": "deep_learning", "model": "BiLSTM"})
    tracker_lstm.start(run_name="BiLSTM-Attention")

    try:
        tracker_lstm.log_params({
            "model_type":    "BiLSTM-Attention",
            "input_dim":     n_features,
            "hidden_dim":    128,
            "num_layers":    2,
            "window_size":   WINDOW_SIZE,
            "batch_size":    BATCH_SIZE,
            "epochs":        EPOCHS,
            "learning_rate": LEARNING_RATE,
            "patience":      PATIENCE,
            "loss_fn":       "FocalLoss(alpha=0.75,gamma=2.0)",
            "optimizer":     "AdamW",
            "scheduler":     "ReduceLROnPlateau",
            "bidirectional": True,
            "attention":     "TemporalAttention",
            "dropout":       0.3,
            "oversampling":  "GaussianNoiseJitter(ratio=3.0)",
        })

        lstm_model   = SensorLSTM(input_dim=n_features)
        lstm_trainer = DLTrainer(lstm_model, LSTM_MODEL_PATH)

        best_f1 = _train_with_mlflow(lstm_trainer, train_loader, val_loader, tracker_lstm)

        evaluator = DLEvaluator()
        evaluator.evaluate_model(lstm_model, val_loader, "LSTM")
        evaluator.save_report()
        report = evaluator.results

        lstm_metrics = report.get("LSTM", {})
        tracker_lstm.log_metrics({
            "final_accuracy":  lstm_metrics.get("accuracy",  0),
            "final_precision": lstm_metrics.get("precision", 0),
            "final_recall":    lstm_metrics.get("recall",    0),
            "final_f1":        lstm_metrics.get("f1_score",  0),
            "best_val_f1":     best_f1,
        })

        tracker_lstm.log_model_pytorch(lstm_model, "lstm_model",
                                       registered_name=MODEL_REGISTRY_LSTM)
        tracker_lstm.log_artifact(REPORT_PATH, "reports")
        print(f"  ✅ BiLSTM run logged | Best Val F1: {best_f1:.4f}")
    finally:
        tracker_lstm.end()

    # ─────────────────────────────────────────────────────────────────────────
    # Run 2: 1D-CNN
    # ─────────────────────────────────────────────────────────────────────────
    tracker_cnn = MLflowTracker(EXPERIMENT_DEEP_LEARNING, tags={"module": "deep_learning", "model": "1D-CNN"})
    tracker_cnn.start(run_name="1D-CNN")

    try:
        tracker_cnn.log_params({
            "model_type":    "1D-CNN",
            "input_dim":     n_features,
            "window_size":   WINDOW_SIZE,
            "batch_size":    BATCH_SIZE,
            "epochs":        EPOCHS,
            "learning_rate": LEARNING_RATE,
            "patience":      PATIENCE,
            "loss_fn":       "FocalLoss(alpha=0.75,gamma=2.0)",
            "optimizer":     "AdamW",
        })

        cnn_model   = Sensor1DCNN(input_dim=n_features)
        cnn_trainer = DLTrainer(cnn_model, CNN_MODEL_PATH)
        cnn_best_f1 = _train_with_mlflow(cnn_trainer, train_loader, val_loader, tracker_cnn)

        evaluator.evaluate_model(cnn_model, val_loader, "1D-CNN")
        evaluator.save_report()
        report = evaluator.results
        cnn_metrics = report.get("1D-CNN", {})

        tracker_cnn.log_metrics({
            "final_accuracy":  cnn_metrics.get("accuracy",  0),
            "final_precision": cnn_metrics.get("precision", 0),
            "final_recall":    cnn_metrics.get("recall",    0),
            "final_f1":        cnn_metrics.get("f1_score",  0),
            "best_val_f1":     cnn_best_f1,
        })

        tracker_cnn.log_model_pytorch(cnn_model, "cnn_model",
                                      registered_name=MODEL_REGISTRY_CNN)
        tracker_cnn.log_artifact(REPORT_PATH, "reports")
        print(f"  ✅ 1D-CNN run logged | Best Val F1: {cnn_best_f1:.4f}")
    finally:
        tracker_cnn.end()


def _train_with_mlflow(trainer: DLTrainer, train_loader, val_loader, tracker: MLflowTracker) -> float:
    """
    Wraps DLTrainer.train() to capture epoch-level metrics for MLflow.
    Returns best validation F1.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score as sk_f1
    from deep_learning.dl_trainer import FocalLoss
    from deep_learning.dl_config  import EPOCHS, PATIENCE, LEARNING_RATE, DEVICE

    model      = trainer.model
    save_path  = trainer.model_save_path
    criterion  = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_f1     = 0.0
    patience_ct = 0
    best_state  = None

    model.to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        epoch_loss /= len(train_loader.dataset)

        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(DEVICE)
                y_val = y_val.to(DEVICE)
                logits   = model(X_val)
                val_loss += criterion(logits, y_val).item() * len(X_val)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
                all_preds.extend(preds)
                all_labels.extend(y_val.cpu().numpy().astype(int))
        val_loss /= len(val_loader.dataset)
        val_f1 = sk_f1(all_labels, all_preds, average="binary", zero_division=0)

        scheduler.step(val_f1)

        tracker.log_epoch(epoch, epoch_loss, val_loss, val_f1)

        print(f"  Epoch {epoch:02d}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1     = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, save_path)

    print(f"  Best Validation F1: {best_f1:.4f}")
    return best_f1


if __name__ == "__main__":
    run_dl_experiment()
