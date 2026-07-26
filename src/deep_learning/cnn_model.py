import torch
import torch.nn as nn

class Sensor1DCNN(nn.Module):
    def __init__(self, input_dim: int, num_filters: int = 64, kernel_size: int = 3):
        super(Sensor1DCNN, self).__init__()
        # Conv1d expects shape: (batch_size, input_dim, seq_len)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_filters * 2, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x input shape: (batch_size, seq_len, input_dim) -> transpose to (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.pool(out).squeeze(-1)
        out = self.relu3(self.fc1(out))
        logits = self.fc2(out).squeeze(-1)
        return logits
