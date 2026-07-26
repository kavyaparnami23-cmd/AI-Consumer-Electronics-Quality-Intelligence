"""
anomaly_detector.py
───────────────────
Dual-signal anomaly detection combining:
  1. LSTM Autoencoder reconstruction error (unsupervised, sequence-aware)
  2. Isolation Forest on enriched temporal features (point-based, robust)

Final anomaly score = weighted combination of normalised scores from both.
Final anomaly flag  = (ae_flag OR if_flag) for high recall on rare failures.

Training flow:
  - Fit both models on normal (non-failure) windows only to learn
    the distribution of healthy sensor operation.
  - Calibrate AE threshold via percentile on normal training MSE.
"""

import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.time_series.lstm_autoencoder import LSTMAutoEncoder
from src.time_series.ts_config import (
    WINDOW_SIZE, STEP_SIZE, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE,
    THRESHOLD_PERCENTILE, IF_N_ESTIMATORS, IF_CONTAMINATION,
    DEVICE, AE_MODEL_PATH, IF_MODEL_PATH, TS_SCALER_PATH, THRESHOLD_PATH,
)


# ─── Sliding-window helper ────────────────────────────────────────────────────

def _make_sequences(X: np.ndarray, window: int = WINDOW_SIZE, step: int = STEP_SIZE):
    """Convert a 2-D feature array into overlapping 3-D sequence windows."""
    seqs = []
    for i in range(0, len(X) - window + 1, step):
        seqs.append(X[i : i + window])
    return np.array(seqs, dtype=np.float32)


# ─── AE training ──────────────────────────────────────────────────────────────

def train_autoencoder(X_normal: np.ndarray, input_dim: int) -> tuple:
    """
    Train the LSTM Autoencoder on *normal* (non-failure) sequences only.

    Returns:
        model  – trained LSTMAutoEncoder
        threshold – float MSE above which a sequence is flagged anomalous
    """
    model = LSTMAutoEncoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        seq_len=WINDOW_SIZE,
    ).to(DEVICE)

    seqs = _make_sequences(X_normal)
    if len(seqs) == 0:
        raise ValueError("No normal training sequences available (check data).")

    tensor = torch.tensor(seqs, dtype=torch.float32).to(DEVICE)
    dataset = TensorDataset(tensor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = torch.nn.MSELoss()

    best_loss   = float("inf")
    patience_ct = 0

    print(f"  Training LSTM Autoencoder on {len(seqs):,} normal sequences "
          f"(window={WINDOW_SIZE}, input_dim={input_dim}) ...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (batch_x,) in loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss  = criterion(recon, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)

        epoch_loss /= len(seqs)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss   = epoch_loss
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{EPOCHS} | Loss: {epoch_loss:.6f}")

        if patience_ct >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best loss={best_loss:.6f})")
            break

    model.load_state_dict(best_state)

    # ── Compute reconstruction errors on normal data → set threshold ──────────
    model.eval()
    all_mse = []
    with torch.no_grad():
        for (batch_x,) in DataLoader(TensorDataset(tensor), batch_size=256):
            mse = model.reconstruction_error(batch_x)
            all_mse.extend(mse.cpu().numpy().tolist())

    threshold = float(np.percentile(all_mse, THRESHOLD_PERCENTILE))
    print(f"  AE Threshold (P{THRESHOLD_PERCENTILE}): {threshold:.6f}")

    # Persist
    torch.save(model.state_dict(), AE_MODEL_PATH)
    joblib.dump(threshold, THRESHOLD_PATH)

    return model, threshold


# ─── Isolation Forest training ────────────────────────────────────────────────

def train_isolation_forest(X_features: np.ndarray) -> IsolationForest:
    """Train Isolation Forest on the full enriched feature set (no labels)."""
    print(f"  Training Isolation Forest on {len(X_features):,} rows ...")
    clf = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_features)
    joblib.dump(clf, IF_MODEL_PATH)
    return clf


# ─── Dual-signal scoring on full dataset ─────────────────────────────────────

def score_anomalies(
    X_scaled: np.ndarray,
    model:     LSTMAutoEncoder,
    threshold: float,
    clf:       IsolationForest,
) -> pd.DataFrame:
    """
    Compute per-row anomaly scores from both models and combine.

    Returns:
        DataFrame with columns:
            ae_mse, ae_flag, if_score, if_flag, combined_flag, anomaly_score
    """
    n = len(X_scaled)
    ae_mse  = np.full(n, np.nan)

    # AE scores only for rows that have a full window ending at them
    seqs_all = _make_sequences(X_scaled)
    tensor   = torch.tensor(seqs_all, dtype=torch.float32).to(DEVICE)

    model.eval()
    all_mse = []
    with torch.no_grad():
        for i in range(0, len(tensor), 256):
            batch = tensor[i : i + 256]
            mse   = model.reconstruction_error(batch)
            all_mse.extend(mse.cpu().numpy().tolist())

    # Align AE MSE scores to their ending row index (window_size - 1 … n-1)
    offset = WINDOW_SIZE - 1
    for j, mse_val in enumerate(all_mse):
        ae_mse[offset + j * STEP_SIZE] = mse_val

    # Forward-fill NaN for first (window_size-1) rows
    ae_series = pd.Series(ae_mse).bfill().ffill().fillna(float(np.nanmax(ae_mse)))

    ae_flag  = (ae_series >= threshold).astype(int).values

    # Isolation Forest
    if_preds  = clf.predict(X_scaled)        # +1 = normal, -1 = anomaly
    if_scores = clf.score_samples(X_scaled)  # lower = more anomalous
    if_flag   = (if_preds == -1).astype(int)

    # Normalise IF scores to [0,1] anomaly probability
    # score_samples range is arbitrary; invert and scale
    if_anom_score = 1.0 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)

    # Normalised AE score
    ae_anom_score = ae_series.values / (ae_series.max() + 1e-8)

    # Combined anomaly score: 60 % AE (sequence-aware) + 40 % IF
    combined_score = 0.6 * ae_anom_score + 0.4 * if_anom_score

    # Flag if either detector fires (union → high recall)
    combined_flag  = ((ae_flag == 1) | (if_flag == 1)).astype(int)

    return pd.DataFrame({
        "ae_mse":         ae_series.values,
        "ae_flag":        ae_flag,
        "if_score":       if_scores,
        "if_flag":        if_flag,
        "anomaly_score":  combined_score,
        "combined_flag":  combined_flag,
    })


# ─── Load saved models ────────────────────────────────────────────────────────

def load_ae_model(input_dim: int) -> tuple:
    model = LSTMAutoEncoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        seq_len=WINDOW_SIZE,
    ).to(DEVICE)
    model.load_state_dict(torch.load(AE_MODEL_PATH, map_location=DEVICE))
    model.eval()
    threshold = joblib.load(THRESHOLD_PATH)
    return model, threshold


def load_isolation_forest() -> IsolationForest:
    return joblib.load(IF_MODEL_PATH)
