"""
models/rem/inference.py
-----------------------
REM sleep behaviour disorder inference pipeline.

FIX C3: Changed bare 'from canonical import' / 'from model import' to
fully-qualified package paths so this module works when imported as
models.rem.inference (not just when run as a script).
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# FIX C3: fully-qualified imports
from models.rem.canonical import LABEL_NAMES, preprocess_inference
from models.rem.model import REMEnsemble, load_ensemble

warnings.filterwarnings("ignore")

_ENSEMBLE_FILE = "rem_ensemble.pkl"
_METADATA_FILE = "modality_result.json"
_SHAP_FILE     = "shap_explainer.pkl"
_ENCODERS_FILE = "encoders.pkl"
_MEDIANS_FILE  = "median_values.pkl"


class REMInferencePipeline:

    def __init__(
        self,
        ensemble: REMEnsemble,
        metadata: dict,
        encoders: Optional[dict] = None,
        median_values: Optional[pd.Series] = None,
        shap_explainer=None,
    ):
        self.ensemble = ensemble
        self.metadata = metadata
        self.encoders = encoders or {}
        self.median_values = (
            median_values if median_values is not None else pd.Series(dtype=float)
        )
        self.shap_explainer = shap_explainer

    # ------------------------------------------------------------------
    # FIX C4: new classmethod that accepts pre-downloaded local paths
    # Called by get_rem_pipeline() in inference_pipeline.py
    # ------------------------------------------------------------------

    @classmethod
    def from_local(
        cls,
        ensemble_path: str,
        metadata_path: str,
        shap_path: Optional[str] = None,
        encoders_path: Optional[str] = None,
        medians_path: Optional[str] = None,
    ) -> "REMInferencePipeline":
        """
        Build pipeline from pre-downloaded local file paths.
        Used by the FastAPI service (artifacts come from HuggingFace).
        """
        ensemble = load_ensemble(ensemble_path)

        with open(metadata_path) as f:
            metadata = json.load(f)

        encoders: dict = {}
        if encoders_path and Path(encoders_path).exists():
            encoders = joblib.load(encoders_path)

        median_values = pd.Series(dtype=float)
        if medians_path and Path(medians_path).exists():
            median_values = joblib.load(medians_path)

        shap_explainer = None
        if shap_path and Path(shap_path).exists():
            shap_explainer = joblib.load(shap_path)

        return cls(ensemble, metadata, encoders, median_values, shap_explainer)

    # ------------------------------------------------------------------
    # Original from_artifacts kept for standalone / CLI use
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(cls, artifact_dir: str = "output") -> "REMInferencePipeline":
        artifact_dir = Path(artifact_dir)

        ensemble = load_ensemble(str(artifact_dir / _ENSEMBLE_FILE))

        metadata_path = artifact_dir / _METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        with open(metadata_path) as f:
            metadata = json.load(f)

        encoders_path = artifact_dir / _ENCODERS_FILE
        encoders = joblib.load(encoders_path) if encoders_path.exists() else {}

        medians_path = artifact_dir / _MEDIANS_FILE
        median_values = (
            joblib.load(medians_path) if medians_path.exists() else pd.Series(dtype=float)
        )

        shap_path = artifact_dir / _SHAP_FILE
        shap_explainer = joblib.load(shap_path) if shap_path.exists() else None

        return cls(ensemble, metadata, encoders, median_values, shap_explainer)

    def register_preprocessing(self, encoders: dict, median_values: pd.Series) -> None:
        self.encoders = encoders
        self.median_values = median_values

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.encoders or self.median_values.empty:
            # Graceful degradation: return df aligned to feature_names without preprocessing
            # (works when encoders weren't saved to HF — REM ensemble handles NaN internally)
            import warnings
            warnings.warn(
                "REM encoders/medians not available — passing raw features. "
                "Save encoders.pkl and median_values.pkl during training for best results.",
                UserWarning,
            )
            cols = [c for c in self.ensemble.feature_names if c in df.columns]
            return df[cols].copy()

        return preprocess_inference(
            df,
            encoders=self.encoders,
            median_values=self.median_values,
            feature_names=self.ensemble.feature_names,
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare(df)
        return self.ensemble.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare(df)
        return self.ensemble.predict_proba(X)

    def predict_df(self, df: pd.DataFrame, include_shap: bool = False) -> pd.DataFrame:
        X = self._prepare(df)
        proba = self.ensemble.predict_proba(X)
        preds = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)

        results = pd.DataFrame(
            {
                "predicted_label": [LABEL_NAMES[p] for p in preds],
                "predicted_class": preds,
                "confidence": confidence,
                "prob_PD": proba[:, 0],
                "prob_RB": proba[:, 1],
                "prob_HC": proba[:, 2],
            },
            index=df.index,
        )

        if include_shap:
            results = self._append_shap(results, X)

        return results

    def _append_shap(self, results: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        if self.shap_explainer is None:
            warnings.warn("SHAP explainer not loaded; skipping.", UserWarning)
            return results

        xgb_sub = self.ensemble.models.get("XGBoost")
        if xgb_sub is None:
            return results

        X_scaled = xgb_sub.scaler.transform(X)
        shap_values = self.shap_explainer.shap_values(X_scaled)

        if isinstance(shap_values, list):
            preds = results["predicted_class"].values
            shap_arr = np.array([shap_values[p][i] for i, p in enumerate(preds)])
        else:
            shap_arr = shap_values

        shap_df = pd.DataFrame(
            shap_arr,
            columns=[f"shap_{c}" for c in self.ensemble.feature_names],
            index=results.index,
        )
        return pd.concat([results, shap_df], axis=1)


# CLI entry point — only runs when script is called directly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="REM inference CLI")
    parser.add_argument("--input",     "-i", required=True)
    parser.add_argument("--output",    "-o", default="predictions.csv")
    parser.add_argument("--artifacts", "-a", default="output")
    parser.add_argument("--shap",      action="store_true")
    args = parser.parse_args()

    pipeline = REMInferencePipeline.from_artifacts(args.artifacts)
    df = pd.read_csv(args.input)
    results = pipeline.predict_df(df, include_shap=args.shap)
    results.to_csv(args.output, index=False)
    print(f"Predictions saved → {args.output}")