import os
import re
import glob
import joblib

import numpy as np
import pandas as pd

import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

WINDOW_SIZE = 256
STRIDE = 128

BATCH_SIZE = 32

RANDOM_STATE = 42

ARTIFACT_DIR = os.path.join(
    os.getcwd(),
    "artifacts"
)

os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True
)

def validate_dataset(files):

    if len(files) == 0:
        raise FileNotFoundError(
            "No gait files found."
        )

    print("=" * 60)
    print(f"Patients Found : {len(files)}")
    print("=" * 60)

def create_windows(df,patient_id):
    windows = []
    labels = []
    groups = []

    X_raw = df.iloc[:, 1:-1].values
    y_raw = (df.iloc[:, -1].values > 0).astype(int)

    for i in range(0,len(X_raw) - WINDOW_SIZE,STRIDE):

        window_x = X_raw[
            i:i + WINDOW_SIZE
        ]

        fog_ratio = np.mean(y_raw[i:i + WINDOW_SIZE])

        window_y = int(fog_ratio >= 0.50)

        windows.append(window_x)
        labels.append(window_y)
        groups.append(patient_id)

    return windows, labels, groups

def print_dataset_statistics(X,y,groups):

    print()
    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print("Total Windows :", len(X))
    print("Patients      :", len(np.unique(groups)))
    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for cls, cnt in zip(unique, counts):
        print(
            f"Class {cls}: {cnt}"
        )

    print("=" * 60)

def compute_weights(
    y_train
):

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


def load_and_preprocess(data_path):

    files = sorted(
        glob.glob(os.path.join(data_path, "*.txt"))
    )

    validate_dataset(files)

    all_windows = []
    all_labels = []
    all_groups = []

    for file_path in files:

        patient_id = re.findall(
            r"(S\d+)",
            os.path.basename(file_path)
        )[0]

        df = pd.read_csv(
            file_path,
            sep=" ",
            header=None,
            engine="c"
        )

        # remove invalid labels
        df = df[df.iloc[:, -1] != -1]

        windows, labels, patient_groups = create_windows(df,patient_id)

        all_windows.extend(windows)
        all_labels.extend(labels)
        all_groups.extend(patient_groups)

    X = np.array(all_windows)       # (N, WINDOW, 9)
    y = np.array(all_labels)        # (N,)
    groups = np.array(all_groups)   # (N,)
    print_dataset_statistics(X,y,groups)
    print()

    print("=" * 60)
    print("FOG Ratio")
    print("=" * 60)

    print(f"FOG Windows    : {np.sum(y == 1)}")
    print(f"Normal Windows : {np.sum(y == 0)}")
    print(f"FOG Ratio      : {np.mean(y):.4f}")

    print("=" * 60)

    # ── Group-aware train / temp split (70 / 30) ──────────────────────
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_STATE
    )

    train_idx, temp_idx = next(
        gss.split(X, y, groups)
    )

    X_train = X[train_idx]
    y_train = y[train_idx]
    groups_train = groups[train_idx]

    X_temp = X[temp_idx]
    y_temp = y[temp_idx]
    groups_temp = groups[temp_idx]

    # ── Group-aware val / test split (50 / 50 of temp) ────────────────
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    val_idx, test_idx = next(
        gss.split(X_temp, y_temp, groups_temp)
    )

    X_val = X_temp[val_idx]
    y_val = y_temp[val_idx]

    X_test = X_temp[test_idx]
    y_test = y_temp[test_idx]

    groups_val = groups_temp[val_idx]
    groups_test = groups_temp[test_idx]

    assert len(set(groups_train) & set(groups_val)) == 0, \
        "Patient leakage between train and validation."

    assert len(set(groups_train) & set(groups_test)) == 0, \
        "Patient leakage between train and test."

    assert len(set(groups_val) & set(groups_test)) == 0, \
        "Patient leakage between validation and test."

    print("✓ No patient leakage detected.")

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # ── Scaler: fit on train, apply to all splits ──────────────────────
    # Fit on flattened (N*WINDOW, 9) so per-sensor stats are correct
    scaler = StandardScaler()

    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)

    def transform_windows(X):
        original_shape = X.shape
        X = X.reshape(-1, X.shape[-1])
        X = scaler.transform(X)
        return X.reshape(original_shape)

    X_train = transform_windows(X_train)
    X_val   = transform_windows(X_val)
    X_test  = transform_windows(X_test)
    class_weights = compute_weights(y_train)
    joblib.dump(class_weights,os.path.join(ARTIFACT_DIR,"class_weights.pkl"))

    joblib.dump(scaler,os.path.join(
        ARTIFACT_DIR,
        "scaler.pkl"
    ))

    joblib.dump({
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "random_state": RANDOM_STATE
    },
    os.path.join(
        ARTIFACT_DIR,
        "preprocessing_config.pkl"))

    # ── Transpose to (N, 9, WINDOW) for CNN Conv1d ────────────────────
    X_train_cnn = np.transpose(X_train, (0, 2, 1))
    X_val_cnn   = np.transpose(X_val,   (0, 2, 1))
    X_test_cnn  = np.transpose(X_test,  (0, 2, 1))

    # ── LightGBM flat arrays: (N, WINDOW*9) ───────────────────────────
    X_train_lgbm = X_train.reshape(X_train.shape[0], -1)
    X_val_lgbm   = X_val.reshape(X_val.shape[0],     -1)
    X_test_lgbm  = X_test.reshape(X_test.shape[0],   -1)

    # ── PyTorch DataLoaders (CNN branch) ──────────────────────────────
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_cnn, dtype=torch.float32),
            torch.tensor(y_train,     dtype=torch.long)
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_cnn, dtype=torch.float32),
            torch.tensor(y_val,     dtype=torch.long)
        ),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test_cnn, dtype=torch.float32),
            torch.tensor(y_test,     dtype=torch.long)
        ),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
    train_loader,
    val_loader,
    test_loader,
    X_train_lgbm,
    X_val_lgbm,
    X_test_lgbm,
    y_train,
    y_val,
    y_test,
    scaler,
    class_weights,
    groups_train,
    groups_val,
    groups_test)
