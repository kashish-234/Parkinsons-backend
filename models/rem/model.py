"""
model.py
--------
Defines the REMEnsemble model architecture used for REM sleep behaviour
disorder / Parkinson's disease / healthy-control classification.

Classes
-------
_SubModel     – thin wrapper pairing a sklearn/XGBoost estimator with its scaler.
REMEnsemble   – voting/averaging ensemble consumed by inference.py.

Utilities
---------
load_ensemble(path)        – load a serialised REMEnsemble from disk.
save_ensemble(model, path) – persist a REMEnsemble to disk.

Label conventions (matches the notebook)
-----------------------------------------
PREFIX_TO_LABEL : dict[str, int]  "PD"->0, "RB"->1, "HC"->2  (used during training)
LABEL_NAMES     : dict[int, str]  0->"PD", 1->"RB", 2->"HC"  (used during inference)
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List


# ---------------------------------------------------------------------------
# Label mapping  –– aligned with the notebook
# ---------------------------------------------------------------------------

# Participant-code prefix → integer label  (as in the notebook's .map() call)
PREFIX_TO_LABEL: Dict[str, int] = {
    "PD": 0,  # Parkinson's Disease
    "RB": 1,  # REM Behaviour Disorder
    "HC": 2,  # Healthy Control
}

# Integer label → human-readable string  (inverse; used by predict_labels)
LABEL_NAMES: Dict[int, str] = {v: k for k, v in PREFIX_TO_LABEL.items()}


# ---------------------------------------------------------------------------
# _SubModel
# ---------------------------------------------------------------------------

class _SubModel:
    """
    Thin wrapper pairing a fitted estimator with its StandardScaler.

    Matches the notebook's _SubModel exactly so that joblib-serialised
    objects saved during training load cleanly here.

    Attributes
    ----------
    model   : fitted sklearn / XGBoost estimator
    scaler  : fitted StandardScaler applied before prediction
    """

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X) -> np.ndarray:
        """Scale X, then return class predictions."""
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X) -> np.ndarray:
        """Scale X, then return class probability matrix (n_samples × 3)."""
        return self.model.predict_proba(self.scaler.transform(X))


# ---------------------------------------------------------------------------
# REMEnsemble
# ---------------------------------------------------------------------------

class REMEnsemble:
    """
    Fused ensemble of heterogeneous classifiers.

    Matches the notebook's REMEnsemble exactly so that joblib-serialised
    objects load without compatibility issues.

    Parameters
    ----------
    models        : dict mapping model name → _SubModel instance
    feature_names : ordered list of feature column names expected at inference
    fusion_method : "voting"  – hard majority vote across sub-models (default)
                    "average" – soft average of predicted probabilities

    Required public attributes (consumed by inference.py)
    ------------------------------------------------------
    .feature_names  list[str]
    .models         dict[str, _SubModel]
    .fusion_method  str
    """

    def __init__(
        self,
        models: Dict[str, "_SubModel"],
        feature_names: List[str],
        fusion_method: str = "voting",
    ):
        if fusion_method not in ("voting", "average"):
            raise ValueError(
                f"fusion_method must be 'voting' or 'average', got '{fusion_method}'"
            )
        self.models = models
        self.feature_names = feature_names
        self.fusion_method = fusion_method

    # ------------------------------------------------------------------
    # Core prediction methods  (match notebook behaviour exactly)
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Average predicted probability matrices across all sub-models.

        Returns
        -------
        np.ndarray of shape (n_samples, 3)
        """
        self._validate_columns(X)
        proba_list = [
            sub.predict_proba(X[self.feature_names]) for sub in self.models.values()
        ]
        return np.mean(proba_list, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels using the configured fusion method.

        "voting"  → hard majority vote (notebook default)
        "average" → argmax of averaged probabilities

        Returns
        -------
        np.ndarray of shape (n_samples,) with integer labels {0, 1, 2}
        """
        self._validate_columns(X)
        X_feat = X[self.feature_names]

        if self.fusion_method == "average":
            return np.argmax(self.predict_proba(X), axis=1)

        # Hard voting  –– identical to the notebook's np.bincount approach
        preds = np.column_stack(
            [sub.predict(X_feat) for sub in self.models.values()]
        )
        return np.apply_along_axis(
            lambda row: np.bincount(row.astype(int), minlength=3).argmax(),
            axis=1,
            arr=preds,
        )

    def predict_labels(self, X: pd.DataFrame) -> List[str]:
        """Return human-readable label strings ("PD" / "RB" / "HC")."""
        return [LABEL_NAMES[i] for i in self.predict(X)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_columns(self, X: pd.DataFrame) -> None:
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing features: {missing}")

    def __repr__(self) -> str:
        return (
            f"REMEnsemble(models={list(self.models.keys())}, "
            f"n_features={len(self.feature_names)}, "
            f"fusion='{self.fusion_method}')"
        )


# ---------------------------------------------------------------------------
# Serialisation utilities
# ---------------------------------------------------------------------------

def save_ensemble(model: REMEnsemble, path: str) -> None:
    """Persist a REMEnsemble to *path* using joblib."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"[model] Saved ensemble → {path}")


def load_ensemble(path: str) -> REMEnsemble:
    """Load a REMEnsemble from *path*. Raises FileNotFoundError if absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
    model = joblib.load(path)
    if not isinstance(model, REMEnsemble):
        raise TypeError(f"Expected REMEnsemble, got {type(model)}")
    print(f"[model] Loaded ensemble ← {path}")
    return model
