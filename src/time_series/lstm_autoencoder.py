"""
lstm_autoencoder.py
───────────────────
Sequence-to-Sequence LSTM Autoencoder for unsupervised anomaly detection.

Architecture:
  Encoder: Multi-layer LSTM → compresses sequence into a fixed latent vector.
  Decoder: Single LSTM that reconstructs the full input sequence from the
           latent vector by repeating it across timesteps.

Anomaly criterion:
  Reconstruction MSE per sequence.  Normal sequences have low MSE;
  anomalous sensor patterns have high MSE because they deviate from
  the distribution learned on failure-free windows.
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """Compresses (batch, seq_len, input_dim) → (batch, latent_dim)."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (B, T, F)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, B, hidden_dim) — take top layer
        last_hidden = h_n[-1]           # (B, hidden_dim)
        latent = self.fc(last_hidden)   # (B, latent_dim)
        return latent


class LSTMDecoder(nn.Module):
    """Reconstructs (batch, latent_dim) → (batch, seq_len, input_dim)."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent):
        # latent: (B, latent_dim)
        init_input = self.fc(latent).unsqueeze(1)           # (B, 1, hidden_dim)
        repeated   = init_input.repeat(1, self.seq_len, 1)  # (B, T, hidden_dim)
        out, _     = self.lstm(repeated)                    # (B, T, hidden_dim)
        recon      = self.output_proj(out)                  # (B, T, F)
        return recon


class LSTMAutoEncoder(nn.Module):
    """Full Encoder-Decoder autoencoder for sensor sequence reconstruction."""

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_layers: int = 2,
        seq_len:    int = 30,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout)

    def forward(self, x):
        latent = self.encoder(x)   # (B, latent_dim)
        recon  = self.decoder(latent)  # (B, T, F)
        return recon

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-sample MSE reconstruction error.

        Args:
            x: (B, T, F) input tensor
        Returns:
            (B,) tensor of MSE values
        """
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            # MSE averaged over timesteps and features
            mse = ((x - recon) ** 2).mean(dim=(1, 2))
        return mse
