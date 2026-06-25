"""
canonical.py
------------
Single source of truth for preprocessing constants and logic.

Imported by inference.py to apply the exact same encoding, coercion,
and NaN-fill steps that were applied during training — without re-fitting
any transformers (which would cause data leakage).

Constants
---------
CATEGORICAL_COLS  : columns that require LabelEncoding
DROP_COLS         : columns removed before the feature matrix is built
PREFIX_TO_LABEL   : participant-code prefix → integer class label
LABEL_NAMES       : integer class label → human-readable string

Functions
---------
preprocess_inference(df, encoders, median_values, feature_names)
    Apply saved preprocessing to raw inference rows.
    Does NOT fit anything — pure transform using objects saved from training.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, List


# ---------------------------------------------------------------------------
# Column constants  –– must match the notebook exactly
# ---------------------------------------------------------------------------

CATEGORICAL_COLS: List[str] = [
    "Gender",
    "Positive  history  of  Parkinson  disease  in  family",
    "Antidepressant  therapy",
    "Antiparkinsonian  medication",
    "Antipsychotic  medication",
    "Benzodiazepine  medication",
]

DROP_COLS: List[str] = [
    "Participant  code",
    "label",
]

# ---------------------------------------------------------------------------
# Label constants  –– kept here so inference.py has a single import source
# ---------------------------------------------------------------------------

# Participant-code prefix → integer label  (matches notebook .map() call)
PREFIX_TO_LABEL: Dict[str, int] = {
    "PD": 0,  # Parkinson's Disease
    "RB": 1,  # REM Behaviour Disorder
    "HC": 2,  # Healthy Control
}

# Integer label → human-readable string  (used in inference output)
LABEL_NAMES: Dict[int, str] = {v: k for k, v in PREFIX_TO_LABEL.items()}


# ---------------------------------------------------------------------------
# Inference-time preprocessing
# ---------------------------------------------------------------------------

def preprocess_inference(
    df: pd.DataFrame,
    encoders: dict,
    median_values: pd.Series,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Apply saved preprocessing to a raw inference DataFrame.

    Reproduces the notebook's preprocessing pipeline without fitting
    any new transformers, so training and serving are always consistent.

    Steps
    -----
    1. Strip whitespace from column names.
    2. Label-encode CATEGORICAL_COLS using *encoders* saved from training.
    3. Coerce remaining object columns to numeric.
    4. Fill NaNs with *median_values* saved from training.
    5. Select and order columns to match *feature_names*.

    Parameters
    ----------
    df            : Raw input rows (one or more); may omit 'Participant  code'
                    and 'label' — they are ignored if present.
    encoders      : {col: LabelEncoder} dict saved during training.
    median_values : Column medians (pd.Series) saved during training.
    feature_names : Ordered list of feature columns the model expects.

    Returns
    -------
    pd.DataFrame aligned to *feature_names*, ready to pass to the ensemble.
    """
    df = df.copy()

    # 1. Normalise column names (strip leading/trailing whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Drop target / id columns if they leaked into the inference frame
    existing_drop = [c for c in DROP_COLS if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)

    # 2. Label-encode categoricals using fitted encoders from training
    for col, le in encoders.items():
        if col not in df.columns:
            continue
        df[col] = le.transform(df[col].astype(str).str.strip())

    # 3. Coerce any remaining object columns to numeric
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Fill NaNs with training medians
    df = df.fillna(median_values)

    # 5. Align to expected feature order
    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing model features: {missing}")

    return df[feature_names]
