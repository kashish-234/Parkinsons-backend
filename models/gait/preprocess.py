import os
import re
import glob
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import TensorDataset, DataLoader

WINDOW = 128
STRIDE = 64


def load_and_preprocess(data_path):

    files = sorted(
        glob.glob(os.path.join(data_path, "*.txt"))
    )

    print(f"Found {len(files)} files")

    if len(files) == 0:
        raise ValueError("No files loaded. Check your dataset path.")

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

        X_raw = df.iloc[:, 1:-1].values        # 9 sensor columns
        y_raw = (df.iloc[:, -1].values > 0).astype(int)

        for i in range(0, len(X_raw) - WINDOW, STRIDE):

            window_x = X_raw[i:i + WINDOW]

            window_y = int(
                np.max(y_raw[i:i + WINDOW])
            )

            all_windows.append(window_x)
            all_labels.append(window_y)
            all_groups.append(patient_id)

    X = np.array(all_windows)       # (N, WINDOW, 9)
    y = np.array(all_labels)        # (N,)
    groups = np.array(all_groups)   # (N,)

    print(X.shape)
    print(y.shape)

    # ── Group-aware train / temp split (70 / 30) ──────────────────────
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42
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
        random_state=42
    )

    val_idx, test_idx = next(
        gss.split(X_temp, y_temp, groups_temp)
    )

    X_val = X_temp[val_idx]
    y_val = y_temp[val_idx]

    X_test = X_temp[test_idx]
    y_test = y_temp[test_idx]

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

    joblib.dump(scaler, "scaler.pkl")

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
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_cnn, dtype=torch.float32),
            torch.tensor(y_val,     dtype=torch.long)
        ),
        batch_size=32,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test_cnn, dtype=torch.float32),
            torch.tensor(y_test,     dtype=torch.long)
        ),
        batch_size=32,
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
    )
