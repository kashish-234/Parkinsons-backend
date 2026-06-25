"""
inference.py
------------
End-to-end inference pipeline for the REM sleep behaviour disorder ensemble.

Typical usage
-------------
    python inference.py --input data/new_patients.csv --output results.csv

Or programmatically:

    from inference import REMInferencePipeline

    pipe = REMInferencePipeline.from_artifacts("output/")
    results = pipe.predict_df(df_raw)

Artifacts expected in the artifact directory
--------------------------------------------
    rem_ensemble.pkl        – serialised REMEnsemble  (saved by training notebook)
    modality_result.json    – metadata dict            (saved by training notebook)
    encoders.pkl            – fitted LabelEncoders     (must be saved by training)
    median_values.pkl       – column fill-medians      (must be saved by training)
    shap_explainer.pkl      – TreeSHAP explainer       (optional; saved by training)

Note: encoders.pkl and median_values.pkl are NOT saved by the original notebook.
Add these two lines at the end of training to enable full self-contained inference:

    joblib.dump(encoders,      "output/encoders.pkl")
    joblib.dump(median_values, "output/median_values.pkl")

where `encoders` and `median_values` come from canonical.preprocess_inference's
counterpart training call (see canonical.py for the full preprocess() signature).
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# canonical.py: preprocessing logic and label constants shared across scripts
from canonical import LABEL_NAMES, preprocess_inference

# model.py: REMEnsemble class definition + load helper
from model import REMEnsemble, load_ensemble

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Artifact filenames  –– must match names used in the training notebook
# ---------------------------------------------------------------------------

_ENSEMBLE_FILE = "rem_ensemble.pkl"
_METADATA_FILE = "modality_result.json"
_SHAP_FILE     = "shap_explainer.pkl"
_ENCODERS_FILE = "encoders.pkl"
_MEDIANS_FILE  = "median_values.pkl"


# ---------------------------------------------------------------------------
# REMInferencePipeline
# ---------------------------------------------------------------------------

class REMInferencePipeline:
    """
    Wraps a trained REMEnsemble with preprocessing and optional SHAP explanation.

    Attributes
    ----------
    ensemble       : REMEnsemble
    metadata       : dict  – training metadata from modality_result.json
    encoders       : dict[str, LabelEncoder]  – categorical encoders from training
    median_values  : pd.Series  – column fill-medians from training
    shap_explainer : TreeExplainer or None  – loaded if shap_explainer.pkl exists
    """

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
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(cls, artifact_dir: str = "output") -> "REMInferencePipeline":
        """
        Load all artifacts from *artifact_dir* and return a ready pipeline.

        Raises FileNotFoundError if the ensemble or metadata file is missing.
        """
        artifact_dir = Path(artifact_dir)

        # Required ── ensemble
        ensemble = load_ensemble(str(artifact_dir / _ENSEMBLE_FILE))

        # Required ── metadata
        metadata_path = artifact_dir / _METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"[inference] Loaded metadata        ← {metadata_path}")

        # Optional ── preprocessing objects (needed for raw-CSV inference)
        encoders_path = artifact_dir / _ENCODERS_FILE
        encoders = joblib.load(encoders_path) if encoders_path.exists() else {}
        if encoders:
            print(f"[inference] Loaded encoders        ← {encoders_path}")
        else:
            print(
                f"[inference] WARNING: {_ENCODERS_FILE} not found. "
                "Call register_preprocessing() before running inference on raw CSVs."
            )

        medians_path = artifact_dir / _MEDIANS_FILE
        median_values = (
            joblib.load(medians_path) if medians_path.exists() else pd.Series(dtype=float)
        )
        if not median_values.empty:
            print(f"[inference] Loaded median_values   ← {medians_path}")
        else:
            print(
                f"[inference] WARNING: {_MEDIANS_FILE} not found. "
                "Call register_preprocessing() before running inference on raw CSVs."
            )

        # Optional ── SHAP explainer
        shap_path = artifact_dir / _SHAP_FILE
        shap_explainer = joblib.load(shap_path) if shap_path.exists() else None
        if shap_explainer is not None:
            print(f"[inference] Loaded SHAP explainer  ← {shap_path}")

        return cls(ensemble, metadata, encoders, median_values, shap_explainer)

    def register_preprocessing(
        self, encoders: dict, median_values: pd.Series
    ) -> None:
        """
        Attach preprocessing objects after construction.

        Useful when inference is called immediately after training in the
        same process and the objects are already in memory.
        """
        self.encoders = encoders
        self.median_values = median_values

    # ------------------------------------------------------------------
    # Internal preprocessing
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply canonical preprocessing and return an aligned feature DataFrame.

        Uses preprocess_inference() from canonical.py — the same steps as
        training but with saved encoders/medians (no re-fitting).
        """
        if not self.encoders or self.median_values.empty:
            raise RuntimeError(
                "Preprocessing objects (encoders / median_values) are not loaded.\n"
                "Either save them during training (see module docstring) or call "
                "register_preprocessing(encoders, median_values) before inference."
            )
        return preprocess_inference(
            df,
            encoders=self.encoders,
            median_values=self.median_values,
            feature_names=self.ensemble.feature_names,
        )

    # ------------------------------------------------------------------
    # Public prediction methods
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return integer class predictions for *df*.

        Parameters
        ----------
        df : Raw input DataFrame (columns as in the training CSV).

        Returns
        -------
        np.ndarray of shape (n_samples,) with values in {0, 1, 2}.
        """
        X = self._prepare(df)
        return self.ensemble.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return averaged class probabilities for *df*.

        Returns
        -------
        np.ndarray of shape (n_samples, 3)  –– [P(PD), P(RB), P(HC)]
        """
        X = self._prepare(df)
        return self.ensemble.predict_proba(X)

    def predict_df(self, df: pd.DataFrame, include_shap: bool = False) -> pd.DataFrame:
        """
        Run full inference and return a results DataFrame.

        Parameters
        ----------
        df           : Raw input DataFrame.
        include_shap : Append per-feature SHAP values (XGBoost sub-model)
                       when a shap_explainer is loaded.

        Returns
        -------
        pd.DataFrame with columns:
            predicted_label  – str  ("PD", "RB", "HC")
            predicted_class  – int  (0, 1, 2)
            confidence       – float  (max predicted probability)
            prob_PD          – float
            prob_RB          – float
            prob_HC          – float
            [shap_<feature>] – float per feature, only if include_shap=True
        """
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

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def _append_shap(self, results: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """Compute SHAP values via the XGBoost sub-model and append as columns."""
        if self.shap_explainer is None:
            warnings.warn(
                "SHAP explainer not loaded; skipping SHAP columns.", UserWarning
            )
            return results

        xgb_sub = self.ensemble.models.get("XGBoost")
        if xgb_sub is None:
            warnings.warn(
                "No 'XGBoost' key in ensemble.models; skipping SHAP.", UserWarning
            )
            return results

        # Scale with the XGBoost sub-model's own scaler (matches training)
        X_scaled = xgb_sub.scaler.transform(X)
        shap_values = self.shap_explainer.shap_values(X_scaled)

        # shap_values is a list of arrays (one per class) for multi-class XGBoost
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

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a concise model summary to stdout."""
        print("=" * 60)
        print("REM Ensemble – Inference Pipeline Summary")
        print("=" * 60)
        print(f"  Model ID        : {self.metadata.get('model_id', 'N/A')}")
        print(f"  Model version   : {self.metadata.get('model_version', 'N/A')}")
        print(f"  Fusion method   : {self.metadata.get('fusion_method', 'N/A')}")
        print(f"  Train accuracy  : {self.metadata.get('ensemble_accuracy', 'N/A')}")
        print(f"  Train F1        : {self.metadata.get('ensemble_f1', 'N/A')}")
        print(f"  Sub-models      : {list(self.ensemble.models.keys())}")
        print(f"  Features        : {len(self.ensemble.feature_names)}")
        print(f"  SHAP available  : {self.shap_explainer is not None}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the REM sleep behaviour disorder ensemble."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to raw input CSV (same column format as training data).",
    )
    parser.add_argument(
        "--output", "-o", default="predictions.csv",
        help="Path to write predictions CSV (default: predictions.csv).",
    )
    parser.add_argument(
        "--artifacts", "-a", default="output",
        help="Directory containing model artifacts (default: output/).",
    )
    parser.add_argument(
        "--shap", action="store_true",
        help="Append per-feature SHAP values (XGBoost) to the output.",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print model summary before running inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    pipeline = REMInferencePipeline.from_artifacts(args.artifacts)

    if args.summary:
        pipeline.summary()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    print(f"[inference] Loaded {len(df)} row(s) from {args.input}")

    results = pipeline.predict_df(df, include_shap=args.shap)
    print(f"\n[inference] Results preview:\n{results[['predicted_label', 'confidence']].head()}")

    results.to_csv(args.output, index=False)
    print(f"\n[inference] Predictions saved → {args.output}")


if __name__ == "__main__":
    main()
