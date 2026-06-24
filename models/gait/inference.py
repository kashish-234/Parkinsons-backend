import numpy as np
import torch

from models.base.contracts import ModelOutput
from .model import FOGModel


class GaitFOGModel:

    MODEL_ID = "gait_cnn_lstm_v1"

    def __init__(self):
        self.model = FOGModel()

        # Load trained model weights
        self.model.load_state_dict(
            torch.load("gait_model.pt", map_location=torch.device("cpu"))
        )

        self.model.eval()

    def predict(self, sample: np.ndarray) -> ModelOutput:
        """
        Parameters
        ----------
        sample : np.ndarray
            Preprocessed gait sample of shape (9, 128)

        Returns
        -------
        ModelOutput
        """

        if sample.ndim == 2:
            sample = np.expand_dims(sample, axis=0)

        x = torch.tensor(sample, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(x)
            probs = torch.softmax(output, dim=1)

            probability = float(probs[0, 1])

        predicted_label = int(probability >= 0.4)

        eps = 1e-6
        probability = min(max(probability, eps), 1 - eps)

        raw_logit = float(
            np.log(probability / (1 - probability))
        )

        return ModelOutput(
            model_id=self.MODEL_ID,
            modality="gait",
            dataset="daphnet_fog",
            probability=probability,
            shap_features=[],
            raw_logit=raw_logit,
            mc_samples=[],
            metadata={
                "predicted_label": predicted_label,
                "decision_threshold": 0.4,
            },
        )