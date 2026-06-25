from functools import lru_cache
import json
import joblib

import torch
import torch.nn as nn

from services.model_storage_service import (
    model_storage_service
)


# ── Model architecture ────────────────────────────────────────────────

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)


class FOGModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(
            ResidualBlock(9, 32),
            nn.MaxPool1d(2),
            nn.Dropout(0.20),
            ResidualBlock(32, 64),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            ResidualBlock(64, 128),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.30,
            batch_first=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.40),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 1)
        outputs, _ = self.lstm(x)
        weights = torch.softmax(self.attention(outputs), dim=1)
        context = torch.sum(weights * outputs, dim=1)
        return self.classifier(context)


# ── Temperature scaler ────────────────────────────────────────────────

class TemperatureScaler(nn.Module):

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature


# ── Component loader ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_components():

    # ── PyTorch model ─────────────────────────────────────────────────
    model_path = model_storage_service.download_model(
        "gait/daphnet/gait_model.pt"
    )
    model = FOGModel()
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()

    # ── Temperature scaler ────────────────────────────────────────────
    ts_path = model_storage_service.download_model(
        "gait/daphnet/temperature_scaler.pt"
    )
    temperature_scaler = TemperatureScaler()
    temperature_scaler.load_state_dict(
        torch.load(ts_path, map_location="cpu")
    )
    temperature_scaler.eval()

    # ── sklearn scaler ────────────────────────────────────────────────
    scaler_path = model_storage_service.download_model(
        "gait/daphnet/scaler.pkl"
    )
    scaler = joblib.load(scaler_path)

    # ── SHAP background ───────────────────────────────────────────────
    # Load real training samples saved during training (background.pt).
    # Falls back to zeros if the artifact is absent, but real-data
    # backgrounds produce more meaningful SHAP attributions.
    # Kept on CPU — GradientExplainer must run on CPU for SHAP compat.
    try:
        background_path = model_storage_service.download_model(
            "gait/daphnet/background.pt"
        )
        background = torch.load(background_path, map_location="cpu")
    except Exception:
        background = torch.zeros(1, 9, 256)

    # ── Metadata ──────────────────────────────────────────────────────
    metadata_path = model_storage_service.download_model(
        "gait/daphnet/metadata.json"
    )
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return {
        "model": model,
        "temperature_scaler": temperature_scaler,
        "scaler": scaler,
        "background": background,
        "metadata": metadata,
    }