"""
models/gait/model.py

PyTorch is imported lazily inside load_components() so that importing this
module at application startup does not immediately allocate PyTorch RAM.
"""
from functools import lru_cache
import json
import joblib

from services.model_storage_service import model_storage_service


def _get_torch():
    import torch
    return torch

def _get_nn():
    import torch.nn as nn
    return nn


# ── Model architecture ────────────────────────────────────────────────

class ResidualBlock:
    """Placeholder — built inside build_fog_model() to defer torch import."""
    pass


def build_fog_model():
    """Build FOGModel architecture. torch imported here, not at module top."""
    torch = _get_torch()
    nn = _get_nn()

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(out_channels)
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm1d(out_channels),
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
                input_size=128, hidden_size=128, num_layers=2,
                bidirectional=True, dropout=0.30, batch_first=True,
            )
            self.attention = nn.Sequential(
                nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1),
            )
            self.classifier = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.30), nn.Linear(128, 2),
            )

        def forward(self, x):
            feat = self.features(x)
            feat = feat.permute(0, 2, 1)
            lstm_out, _ = self.lstm(feat)
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
            return self.classifier(context)

    return FOGModel()


def build_temperature_scaler():
    """Build TemperatureScaler. torch imported here."""
    torch = _get_torch()
    nn = _get_nn()

    class TemperatureScaler(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1))
        def forward(self, logits):
            return logits / self.temperature

    return TemperatureScaler()


# ── Component loader ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_components():
    """
    Download and return all gait model components.
    torch is imported here (inside the function), not at module top.
    """
    torch = _get_torch()

    # ── PyTorch model ─────────────────────────────────────────────────
    model_path = model_storage_service.download_model("gait/gait_model.pt")
    model = build_fog_model()
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    # ── Temperature scaler ────────────────────────────────────────────
    ts_path = model_storage_service.download_model("gait/temperature_scaler.pt")
    temperature_scaler = build_temperature_scaler()
    temperature_scaler.load_state_dict(
        torch.load(ts_path, map_location="cpu", weights_only=True)
    )
    temperature_scaler.eval()

    # ── sklearn scaler ────────────────────────────────────────────────
    scaler_path = model_storage_service.download_model("gait/scaler.pkl")
    scaler = joblib.load(scaler_path)

    # ── Metadata ──────────────────────────────────────────────────────
    metadata_path = model_storage_service.download_model("gait/metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # ── Decision threshold & validation AUC ──────────────────────────
    try:
        dt_path = model_storage_service.download_model("gait/decision_threshold.pkl")
        metadata["decision_threshold"] = float(joblib.load(dt_path))
    except Exception:
        pass

    try:
        auc_path = model_storage_service.download_model("gait/validation_auc.pkl")
        metadata["validation_auc"] = float(joblib.load(auc_path))
    except Exception:
        pass

    # ── SHAP background ───────────────────────────────────────────────
    background = torch.zeros(1, 9, int(metadata.get("window_size", 256)))

    return {
        "model": model,
        "temperature_scaler": temperature_scaler,
        "scaler": scaler,
        "background": background,
        "metadata": metadata,
    }
