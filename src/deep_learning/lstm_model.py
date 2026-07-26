import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        attn_weights = F.softmax(self.attn(lstm_out), dim=1) # (batch_size, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1) # (batch_size, hidden_dim)
        return context, attn_weights

class SensorLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super(SensorLSTM, self).__init__()
        # Bidirectional LSTM for full sequence context
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.bilstm(x) # (batch_size, seq_len, hidden_dim * 2)
        
        # Temporal Attention pooling over timesteps
        context, _ = self.attention(lstm_out) # (batch_size, hidden_dim * 2)
        
        context = self.norm(context)
        out = self.dropout(self.relu(self.fc1(context)))
        logits = self.fc2(out).squeeze(-1)
        return logits
