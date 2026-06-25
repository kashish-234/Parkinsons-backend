from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SpectSample:
    subject: str
    label: int
    dicom_files: Sequence[Path]


class SpectDataset(Dataset):
    def __init__(self, subjects, labels, root_dir, slices_per_subject=5):
        self.subjects = list(subjects)
        self.labels = labels
        self.root_dir = Path(root_dir)
        self.slices_per_subject = slices_per_subject

        self.samples = self.build_samples()

    @staticmethod
    def _normalize_subject(subject: object) -> str:
        try:
            return str(int(float(subject)))
        except Exception:
            return str(subject).strip()

    def build_samples(self):
        samples = []

        for subject in self.subjects:
            subject_id = self._normalize_subject(subject)
            folder = self.root_dir / subject_id
            if not folder.exists():
                continue

            dicom_files = []
            for root, _, files in os.walk(folder):
                for filename in files:
                    if filename.lower().endswith(".dcm"):
                        dicom_files.append(Path(root) / filename)

            if len(dicom_files) == 0:
                continue

            dicom_files.sort()

            # select central slices
            mid = len(dicom_files) // 2
            half = self.slices_per_subject // 2
            start = max(0, mid - half)
            end = min(len(dicom_files), mid + half + 1)
            selected = dicom_files[start:end]

            for path in selected:
                samples.append((path, int(self.labels[subject]), subject_id))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, subject = self.samples[idx]

        dcm = pydicom.dcmread(str(path))
        img = dcm.pixel_array.astype(np.float32)

        # normalize
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)

        # resize
        img = cv2.resize(img, (224, 224))

        # 3 channel
        img = np.stack([img] * 3, axis=0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long), subject


def load_subject_table(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["Subject"] = df["Subject"].astype(str)
    df["label"] = df["Group"].map({"PD": 1, "Prodromal": 0, "Control": 0})
    df = df.dropna(subset=["Subject", "label"]).reset_index(drop=True)
    return df


def build_subject_label_map(subject_df: pd.DataFrame) -> Dict[str, int]:
    return dict(zip(subject_df["Subject"].astype(str), subject_df["label"].astype(int)))


def make_subject_folds(
    subject_df: pd.DataFrame,
    n_splits: int = 3,
    random_state: int = 42,
) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def iter_subject_folds(
    subject_df: pd.DataFrame,
    n_splits: int = 3,
    random_state: int = 42,
):
    x = subject_df["Subject"].astype(str)
    y = subject_df["label"].astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
        train_subjects = x.iloc[train_idx].tolist()
        val_subjects = x.iloc[val_idx].tolist()
        yield fold, train_subjects, val_subjects
