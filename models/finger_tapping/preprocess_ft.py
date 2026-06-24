import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


DROP_COLUMNS = [
    "Unnamed: 0",
    "filename",
    "Comment1",
    "Comment3",
    "Comment4",
    "Rating",
    "Rating1",
    "Rating2",
    "Rating3",
    "Rating4",
    "CheckDifficult1",
    "CheckDifficult3",
    "CheckDifficult4",
]


def load_and_preprocess(data_path):

    df = pd.read_csv(data_path)

    print(df.shape)

    # ── Drop irrelevant columns ────────────────────────────────────────
    df = df.drop(columns=DROP_COLUMNS)

    print(df.shape)

    # ── Encode categoricals ────────────────────────────────────────────
    label_hand = LabelEncoder()
    label_gender = LabelEncoder()

    df["hand"] = label_hand.fit_transform(df["hand"])
    df["gender"] = label_gender.fit_transform(df["gender"])

    df["diagnosed"] = (
        df["diagnosed"]
        .map({"no": 0, "yes": 1})
        .astype(int)
    )

    # ── Features and target ────────────────────────────────────────────
    X = df.drop(columns=["diagnosed"])
    y = df["diagnosed"]

    print(X.shape)
    print(y.shape)
    print(y.value_counts())

    # ── Stratified train / temp split (70 / 30) ───────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    # ── Stratified val / test split (50 / 50 of temp) ─────────────────
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    # ── Scaler: fit on train only, apply to all splits ─────────────────
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, "scaler.pkl")

    # ── Save feature names ─────────────────────────────────────────────
    joblib.dump(
        list(X.columns),
        "selected_feature_names.pkl"
    )

    print("Feature list saved.")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        list(X.columns),
        scaler,
    )
