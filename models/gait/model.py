import torch
import torch.nn as nn


class FOGModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)   # (batch, seq, features)

        _, (h, _) = self.lstm(x)

        return self.fc(h[-1])
