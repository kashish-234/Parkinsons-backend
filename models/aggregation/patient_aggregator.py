import numpy as np

def aggregate_predictions(predictions: list[float]) -> float:

    if not predictions:
        return 0.0

    return float(
        np.median(predictions)
    )