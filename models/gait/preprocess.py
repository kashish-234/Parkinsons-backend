import pandas as pd
import numpy as np
import glob
import os
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def create_windows(X, y, window=128, stride=64):
    Xs, ys = [], []

    for i in range(0, len(X) - window, stride):
        Xs.append(X[i:i+window])
        ys.append(max(y[i:i+window]))  # if any FOG → 1

    return np.array(Xs), np.array(ys)


def load_and_preprocess(data_path):
    # get all .txt files
    files = glob.glob(os.path.join(data_path, "*.txt"))

    print("Total files found:", len(files))

    data = []

    for f in files:
        print("Loading:", f)
        df = pd.read_csv(f, sep=" ", header=None)

        # remove invalid labels (-1)
        df = df[df.iloc[:, -1] != -1]

        data.append(df)

    if len(data) == 0:
        raise ValueError("No files loaded. Check your dataset path.")

    # combine all files
    df = pd.concat(data, ignore_index=True)

    # split features and labels
    X = df.iloc[:, 1:-1].values   # 9 sensor columns
    y = df.iloc[:, -1].values     # labels

    # convert to binary
    y = (y > 0).astype(int)

    print("Classes:", np.unique(y))

    # normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # create time windows
    X, y = create_windows(X, y)

    # reshape for CNN
    X = np.transpose(X, (0, 2, 1))

    print("Shape after windowing:", X.shape, y.shape)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    # create PyTorch loaders
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=32, shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.long)),
        batch_size=32, shuffle=False
    )

    return train_loader, test_loader
