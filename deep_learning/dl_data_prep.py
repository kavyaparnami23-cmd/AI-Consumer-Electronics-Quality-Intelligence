import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

from src.deep_learning.dl_config import WINDOW_SIZE, STEP_SIZE, BATCH_SIZE, SCALER_PATH

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp_Difference",
    "Power_kW",
    "Torque_Wear_Product"
]

TARGET_COL = "Machine failure"


class SequenceDataset(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y_seq, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_dl_data(df_path: str, window_size=WINDOW_SIZE, train_ratio=0.8, is_train=True):
    df = pd.read_csv(df_path)
    
    # Feature engineering if not already present
    if "Temp_Difference" not in df.columns:
        df["Temp_Difference"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    if "Power_kW" not in df.columns:
        df["Power_kW"] = df["Rotational speed [rpm]"] * df["Torque [Nm]"] / 9550.0
    if "Torque_Wear_Product" not in df.columns:
        df["Torque_Wear_Product"] = df["Torque [Nm]"] * df["Tool wear [min]"]

    available_features = [col for col in FEATURE_COLS if col in df.columns]
    X_raw = df[available_features].values
    y_raw = df[TARGET_COL].values

    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X_raw)

    # Generate sliding sequences
    X_seq, y_seq = [], []
    for i in range(0, len(X_scaled) - window_size + 1, STEP_SIZE):
        X_seq.append(X_scaled[i : i + window_size])
        # Target is whether a machine failure occurs in the final step of the window
        y_seq.append(y_raw[i + window_size - 1])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Train / Val / Test split
    split_idx = int(len(X_seq) * train_ratio)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, len(available_features)
