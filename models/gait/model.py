from functools import lru_cache

import torch
import torch.nn as nn

from services.model_storage_service import model_storage_service


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
        x = x.permute(0, 2, 1)

        _, (h, _) = self.lstm(x)

        return self.fc(h[-1])


@lru_cache(maxsize=1)
def load_components():
    """
    Downloads the trained gait model from Hugging Face and loads it.
    """

    local_path = model_storage_service.download_model(
        "gait/gait_model.pt"
    )

    model = FOGModel()

    model.load_state_dict(
        torch.load(local_path, map_location=torch.device("cpu"))
    )

    model.eval()

    return {
        "model": model
    }