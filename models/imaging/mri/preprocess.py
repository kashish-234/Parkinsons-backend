from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MRISample:
    subject: str
    image_data_id: str
    label_name: Optional[str]
    dicom_files: Sequence[Path]


class MRIDicomDataset(Dataset):
    """
    PyTorch Dataset for loading MRI DICOM data using Subject IDs from MRI_4_10_2026.csv.

    Expected CSV columns:
    - Subject (required)
    - Image Data ID (recommended)
    - Group (optional label source, e.g. PD / Control / Prodromal)

    Returns a dictionary per sample:
    - image: torch.FloatTensor
      - shape [1, H, W] when load_mode="middle_slice"
      - shape [1, D, H, W] when load_mode="volume"
    - target: torch.LongTensor scalar (or -1 if label is unavailable)
    - subject: str
    - image_data_id: str
    """

    def __init__(
        self,
        csv_path: str | Path,
        mri_root: str | Path = "Data/PPMI/mri/PPMI",
        load_mode: str = "middle_slice",
        target_size: Optional[tuple[int, int] | tuple[int, int, int]] = (224, 224),
        subject_column: str = "Subject",
        image_id_column: str = "Image Data ID",
        label_column: str = "Group",
        label_map: Optional[Dict[str, int]] = None,
        unique_subjects: bool = False,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        if load_mode not in {"middle_slice", "volume"}:
            raise ValueError("load_mode must be either 'middle_slice' or 'volume'.")

        self.csv_path = Path(csv_path)
        self.mri_root = Path(mri_root)
        self.load_mode = load_mode
        self.target_size = target_size
        self.subject_column = subject_column
        self.image_id_column = image_id_column
        self.label_column = label_column
        self.transform = transform

        if label_map is None:
            label_map = {"Control": 0, "Prodromal": 1, "PD": 2}
        self.label_map = label_map

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if not self.mri_root.exists():
            raise FileNotFoundError(f"MRI root folder not found: {self.mri_root}")

        self.samples = self._build_samples(unique_subjects=unique_subjects)
        if not self.samples:
            raise RuntimeError("No MRI samples were found. Check CSV columns and MRI folder paths.")

    def _build_samples(self, unique_subjects: bool) -> List[MRISample]:
        samples: List[MRISample] = []
        seen_subjects: set[str] = set()

        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header row.")
            if self.subject_column not in reader.fieldnames:
                raise ValueError(
                    f"CSV column '{self.subject_column}' not found. Available: {reader.fieldnames}"
                )

            for row in reader:
                raw_subject = (row.get(self.subject_column) or "").strip()
                if not raw_subject:
                    continue

                subject = self._normalize_subject(raw_subject)
                if unique_subjects and subject in seen_subjects:
                    continue

                image_data_id = (row.get(self.image_id_column) or "").strip()
                label_name = (row.get(self.label_column) or "").strip() or None

                dicom_files = self._resolve_dicom_files(subject=subject, image_data_id=image_data_id)
                if not dicom_files:
                    continue

                samples.append(
                    MRISample(
                        subject=subject,
                        image_data_id=image_data_id,
                        label_name=label_name,
                        dicom_files=dicom_files,
                    )
                )
                seen_subjects.add(subject)

        return samples

    @staticmethod
    def _normalize_subject(subject: str) -> str:
        # Handles numeric fields read as strings like "75426" or "75426.0".
        try:
            return str(int(float(subject)))
        except ValueError:
            return subject.strip()

    def _resolve_dicom_files(self, subject: str, image_data_id: str) -> List[Path]:
        subject_dir = self.mri_root / subject
        if not subject_dir.exists():
            return []

        # First try to match the specific image id from the CSV.
        if image_data_id:
            for candidate in subject_dir.rglob(image_data_id):
                if candidate.is_dir():
                    files = self._list_dicom_files(candidate)
                    if files:
                        return files

        # Fallback: use the first directory under this subject that has DICOM files.
        for candidate in [subject_dir] + [p for p in subject_dir.rglob("*") if p.is_dir()]:
            files = self._list_dicom_files(candidate)
            if files:
                return files

        return []

    @staticmethod
    def _list_dicom_files(folder: Path) -> List[Path]:
        files = [p for p in folder.glob("*") if p.is_file()]

        dicom_like: List[Path] = []
        for f in files:
            if f.suffix.lower() == ".dcm":
                dicom_like.append(f)
                continue

            # Some DICOM files can be extensionless.
            try:
                pydicom.dcmread(str(f), stop_before_pixels=True)
                dicom_like.append(f)
            except Exception:
                continue

        if not dicom_like:
            return []

        return sorted(dicom_like, key=MRIDicomDataset._dicom_sort_key)

    @staticmethod
    def _dicom_sort_key(path: Path) -> tuple[int, str]:
        try:
            dcm = pydicom.dcmread(str(path), stop_before_pixels=True)
            instance_number = int(getattr(dcm, "InstanceNumber", 0))
            return (instance_number, path.name)
        except Exception:
            return (0, path.name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        volume = self._load_volume(sample.dicom_files)
        image = self._preprocess(volume)

        if self.transform is not None:
            image = self.transform(image)

        if sample.label_name is None:
            target_value = -1
        else:
            target_value = self.label_map.get(sample.label_name, -1)

        return {
            "image": image,
            "target": torch.tensor(target_value, dtype=torch.long),
            "subject": sample.subject,
            "image_data_id": sample.image_data_id,
        }

    def _load_volume(self, dicom_files: Sequence[Path]) -> np.ndarray:
        slices: List[np.ndarray] = []
        for path in dicom_files:
            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array.astype(np.float32)
            slices.append(arr)

        if not slices:
            raise RuntimeError("No pixel data loaded for MRI sample.")

        return np.stack(slices, axis=0)  # [D, H, W]

    @staticmethod
    def _normalize(volume: np.ndarray) -> np.ndarray:
        v_min = float(volume.min())
        v_max = float(volume.max())
        if v_max <= v_min:
            return np.zeros_like(volume, dtype=np.float32)
        return ((volume - v_min) / (v_max - v_min)).astype(np.float32)

    @staticmethod
    def _to_tensor(volume: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(volume.astype(np.float32))

    def _resize_2d(self, image: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        # image: [1, H, W] -> [1, H_out, W_out]
        resized = F.interpolate(
            image.unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    def _resize_3d(self, volume: torch.Tensor, target_size: tuple[int, int, int]) -> torch.Tensor:
        # volume: [1, D, H, W] -> [1, D_out, H_out, W_out]
        resized = F.interpolate(
            volume.unsqueeze(0),
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    def _preprocess(self, volume: np.ndarray) -> torch.Tensor:
        normalized = self._normalize(volume)

        if self.load_mode == "middle_slice":
            center_idx = normalized.shape[0] // 2
            image_2d = normalized[center_idx]  # [H, W]
            image = self._to_tensor(image_2d).unsqueeze(0)  # [1, H, W]

            if self.target_size is not None:
                if len(self.target_size) != 2:
                    raise ValueError(
                        "For load_mode='middle_slice', target_size must be (H, W)."
                    )
                image = self._resize_2d(image, self.target_size)
            return image

        # load_mode == "volume"
        image = self._to_tensor(normalized).unsqueeze(0)  # [1, D, H, W]

        if self.target_size is not None:
            if len(self.target_size) == 2:
                depth = image.shape[1]
                size_3d = (depth, self.target_size[0], self.target_size[1])
            elif len(self.target_size) == 3:
                size_3d = self.target_size
            else:
                raise ValueError(
                    "For load_mode='volume', target_size must be (H, W) or (D, H, W)."
                )
            image = self._resize_3d(image, size_3d)

        return image
