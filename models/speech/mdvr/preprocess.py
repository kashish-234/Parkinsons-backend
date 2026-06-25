
"""
Better acoustic feature extraction for MDVR / Parkinson's speech.

Keeps MDVP-style feature names for compatibility with your CANONICAL_MAP,
and adds a richer set of spectral, MFCC, energy, and voicing features.

Optional dependencies:
- parselmouth (for formants / harmonicity if available)
"""

from __future__ import annotations

from pathlib import Path
import math
import re
import warnings
from typing import Dict, Iterable, Tuple

import librosa
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import parselmouth  # type: ignore
    HAS_PARSEL_MOUTH = True
except Exception:
    HAS_PARSEL_MOUTH = False

def _clean_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).flatten()
    x = x[np.isfinite(x)]
    return x

def _safe_stats(x: np.ndarray, prefix: str) -> Dict[str, float]:
    """Return robust summary statistics for a 1D array."""
    x = _clean_array(x)
    if x.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_q75": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_skew": 0.0,
            f"{prefix}_kurtosis": 0.0,
        }

    q25, q75 = np.percentile(x, [25, 75])
    mu = float(np.mean(x))
    sd = float(np.std(x))
    mn = float(np.min(x))
    mx = float(np.max(x))
    med = float(np.median(x))
    rng = float(mx - mn)

    # safe skew/kurtosis without scipy dependency
    if sd > 1e-12:
        z = (x - mu) / sd
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4) - 3.0)
    else:
        skew = 0.0
        kurt = 0.0

    return {
        f"{prefix}_mean": mu,
        f"{prefix}_std": sd,
        f"{prefix}_min": mn,
        f"{prefix}_max": mx,
        f"{prefix}_median": med,
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_range": rng,
        f"{prefix}_skew": skew,
        f"{prefix}_kurtosis": kurt,
    }


def _safe_mean(x: np.ndarray) -> float:
    x = _clean_array(x)
    return float(np.mean(x)) if x.size else 0.0


def _safe_std(x: np.ndarray) -> float:
    x = _clean_array(x)
    return float(np.std(x)) if x.size else 0.0


def _safe_median(x: np.ndarray) -> float:
    x = _clean_array(x)
    return float(np.median(x)) if x.size else 0.0


def _safe_max(x: np.ndarray) -> float:
    x = _clean_array(x)
    return float(np.max(x)) if x.size else 0.0


def _safe_min(x: np.ndarray) -> float:
    x = _clean_array(x)
    return float(np.min(x)) if x.size else 0.0


def _safe_range(x: np.ndarray) -> float:
    x = _clean_array(x)
    return float(np.max(x) - np.min(x)) if x.size else 0.0


def _slope(y: np.ndarray) -> float:
    y = _clean_array(y)
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=float)
    try:
        m = np.polyfit(x, y, 1)[0]
        return float(m)
    except Exception:
        return 0.0


def _voiced_ratio(f0: np.ndarray) -> float:
    f0 = _clean_array(f0)
    if f0.size == 0:
        return 0.0
    return float(np.mean(f0 > 0))


def _frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(y) < frame_length:
        pad = frame_length - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    return librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)


# -----------------------------
# Pitch / jitter / shimmer
# -----------------------------

def _pitch_contour(y: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 500.0) -> np.ndarray:
    """
    Return pitch contour in Hz.
    Try pyin first; fall back to yin.
    """
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=int(0.050 * sr),
            hop_length=int(0.015 * sr),
        )
        if f0 is None:
            raise RuntimeError("pyin returned None")
        f0 = np.asarray(f0, dtype=float)
        f0 = np.where(np.isfinite(f0), f0, 0.0)
        return f0
    except Exception:
        try:
            f0 = librosa.yin(
                y,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                frame_length=int(0.050 * sr),
                hop_length=int(0.015 * sr),
            )
            f0 = np.asarray(f0, dtype=float)
            f0 = np.where(np.isfinite(f0), f0, 0.0)
            return f0
        except Exception:
            return np.array([], dtype=float)


def extract_pitch_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    f0 = _pitch_contour(y, sr)
    f0v = f0[f0 > 0]

    if f0v.size == 0:
        return {
            "MDVP_Fo_Hz_": 0.0,
            "MDVP_Fhi_Hz_": 0.0,
            "MDVP_Flo_Hz_": 0.0,
            "pitch_std": 0.0,
            "pitch_median": 0.0,
            "pitch_range": 0.0,
            "pitch_q25": 0.0,
            "pitch_q75": 0.0,
            "pitch_skew": 0.0,
            "pitch_kurtosis": 0.0,
            "pitch_slope": 0.0,
            "voiced_ratio": 0.0,
            "pitch_frames": 0.0,
        }

    stats = _safe_stats(f0v, "pitch")
    voiced_ratio = _voiced_ratio(f0)
    slope = _slope(f0v)

    return {
        "MDVP_Fo_Hz_": float(np.mean(f0v)),
        "MDVP_Fhi_Hz_": float(np.max(f0v)),
        "MDVP_Flo_Hz_": float(np.min(f0v)),
        "pitch_std": stats["pitch_std"],
        "pitch_median": stats["pitch_median"],
        "pitch_range": stats["pitch_range"],
        "pitch_q25": stats["pitch_q25"],
        "pitch_q75": stats["pitch_q75"],
        "pitch_skew": stats["pitch_skew"],
        "pitch_kurtosis": stats["pitch_kurtosis"],
        "pitch_slope": slope,
        "voiced_ratio": voiced_ratio,
        "pitch_frames": float(f0v.size),
    }


def extract_jitter_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Approximate MDVP jitter features from the pitch contour.
    If you later add Praat/parselmouth extraction, you can swap this implementation.
    """
    f0 = _pitch_contour(y, sr)
    f0 = f0[f0 > 0]

    if f0.size < 3:
        return {
            "MDVP_Jitter_": 0.0,
            "MDVP_Jitter_Abs_": 0.0,
            "MDVP_RAP": 0.0,
            "MDVP_PPQ": 0.0,
            "Jitter_DDP": 0.0,
        }

    periods = 1.0 / np.clip(f0, 1e-8, None)
    periods = np.asarray(periods, dtype=float)

    # local jitter
    local_diffs = np.abs(np.diff(periods))
    local_mean = np.mean(periods)
    jitter_local = (np.mean(local_diffs) / local_mean) * 100.0 if local_mean > 0 else 0.0

    # RAP: mean abs(period_i - mean(period_{i-1}, period_i, period_{i+1})) / mean(period)
    if periods.size >= 3:
        rap_vals = []
        for i in range(1, len(periods) - 1):
            local_avg = np.mean(periods[i - 1 : i + 2])
            rap_vals.append(abs(periods[i] - local_avg))
        jitter_rap = (np.mean(rap_vals) / local_mean) if rap_vals and local_mean > 0 else 0.0
    else:
        jitter_rap = 0.0

    # PPQ: 5-point analog
    if periods.size >= 5:
        ppq_vals = []
        for i in range(2, len(periods) - 2):
            local_avg = np.mean(periods[i - 2 : i + 3])
            ppq_vals.append(abs(periods[i] - local_avg))
        jitter_ppq = (np.mean(ppq_vals) / local_mean) if ppq_vals and local_mean > 0 else 0.0
    else:
        jitter_ppq = 0.0

    # DDP is 3 * RAP in standard MDVP-style definitions
    jitter_ddp = 3.0 * jitter_rap

    return {
        "MDVP_Jitter_": float(jitter_local),
        "MDVP_Jitter_Abs_": float(np.mean(local_diffs)),
        "MDVP_RAP": float(jitter_rap),
        "MDVP_PPQ": float(jitter_ppq),
        "Jitter_DDP": float(jitter_ddp),
    }


def extract_shimmer_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Approximate shimmer from frame-wise RMS energy.
    """
    frame_length = int(0.050 * sr)
    hop_length = int(0.015 * sr)
    frames = _frame_signal(y, frame_length, hop_length)
    amps = np.array([np.sqrt(np.mean(frame ** 2)) for frame in frames.T], dtype=float)
    amps = amps[np.isfinite(amps)]
    amps = amps[amps > 0]

    if amps.size < 3:
        return {
            "MDVP_Shimmer": 0.0,
            "MDVP_Shimmer_dB_": 0.0,
            "Shimmer_APQ3": 0.0,
            "Shimmer_APQ5": 0.0,
            "MDVP_APQ": 0.0,
            "Shimmer_DDA": 0.0,
        }

    local_diffs = np.abs(np.diff(amps))
    mean_amp = np.mean(amps)

    shimmer_local = (np.mean(local_diffs) / mean_amp) * 100.0 if mean_amp > 0 else 0.0
    shimmer_db = 20.0 * np.log10((np.mean(amps[1:]) + 1e-10) / (np.mean(amps[:-1]) + 1e-10))

    if amps.size >= 3:
        apq3_vals = []
        for i in range(1, len(amps) - 1):
            apq3_vals.append(abs(amps[i] - np.mean(amps[i - 1 : i + 2])))
        shimmer_apq3 = np.mean(apq3_vals) / mean_amp if mean_amp > 0 else 0.0
    else:
        shimmer_apq3 = 0.0

    if amps.size >= 5:
        apq5_vals = []
        for i in range(2, len(amps) - 2):
            apq5_vals.append(abs(amps[i] - np.mean(amps[i - 2 : i + 3])))
        shimmer_apq5 = np.mean(apq5_vals) / mean_amp if mean_amp > 0 else 0.0
    else:
        shimmer_apq5 = 0.0

    shimmer_apq = (np.std(local_diffs) / mean_amp) if mean_amp > 0 else 0.0
    shimmer_dda = float(np.mean(np.abs(np.diff(np.diff(amps))))) if amps.size >= 4 else 0.0

    return {
        "MDVP_Shimmer": float(shimmer_local),
        "MDVP_Shimmer_dB_": float(shimmer_db),
        "Shimmer_APQ3": float(shimmer_apq3),
        "Shimmer_APQ5": float(shimmer_apq5),
        "MDVP_APQ": float(shimmer_apq),
        "Shimmer_DDA": float(shimmer_dda),
    }


# -----------------------------
# Harmonics / noise / complexity
# -----------------------------

def extract_hnr_nhr_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Estimate HNR / NHR.
    Prefer parselmouth harmonicity when available; otherwise use a harmonic/residual proxy.
    """
    if HAS_PARSEL_MOUTH:
        try:
            snd = parselmouth.Sound(y, sampling_frequency=sr)
            harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=75)
            values = harm.values.flatten()
            values = values[np.isfinite(values)]
            values = values[values > -200]
            if values.size > 0:
                hnr = float(np.nanmean(values))
                nhr = float(1.0 / (10 ** (hnr / 10.0) + 1e-10))
                return {"NHR": max(nhr, 0.0), "HNR": max(hnr, 0.0)}
        except Exception:
            pass

    harmonic = librosa.effects.harmonic(y)
    residual = y - harmonic
    harmonic_energy = float(np.sum(harmonic ** 2))
    noise_energy = float(np.sum(residual ** 2))

    hnr = 10.0 * np.log10((harmonic_energy + 1e-10) / (noise_energy + 1e-10))
    nhr = 1.0 / (10.0 ** (hnr / 10.0) + 1e-10)

    return {
        "NHR": float(max(nhr, 0.0)),
        "HNR": float(max(hnr, 0.0)),
    }


def extract_complexity_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Simplified RPDE / DFA / spread / D2 / PPE-like measures.
    These are proxies, but they still often help as extra signals.
    """
    frame_length = int(0.050 * sr)
    hop_length = int(0.015 * sr)
    frames = _frame_signal(y, frame_length, hop_length)

    rms = np.array([np.sqrt(np.mean(frame ** 2)) for frame in frames.T], dtype=float)
    rms = rms[np.isfinite(rms)]
    rms = rms[rms > 0]

    if rms.size < 4:
        return {
            "RPDE": 0.0,
            "DFA": 0.0,
            "spread1": 0.0,
            "spread2": 0.0,
            "D2": 0.0,
            "PPE": 0.0,
        }

    log_rms = np.log(rms + 1e-10)
    diff1 = np.diff(log_rms)
    diff2 = np.diff(diff1) if diff1.size >= 2 else np.array([0.0])

    rpde = float(np.std(diff1))
    dfa = float(_slope(log_rms))
    spread1 = float(np.std(log_rms))
    spread2 = float(np.mean(np.abs(diff1)))
    d2 = float(np.std(log_rms) / (np.mean(np.abs(log_rms)) + 1e-10))
    ppe = float(-np.sum(np.abs(log_rms) * np.log(np.abs(log_rms) + 1e-10)))

    return {
        "RPDE": max(rpde, 0.0),
        "DFA": float(dfa),
        "spread1": float(spread1),
        "spread2": float(spread2),
        "D2": float(d2),
        "PPE": float(ppe),
    }


# -----------------------------
# Spectral / MFCC / energy
# -----------------------------

def extract_spectral_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    feat = {}

    def add(prefix: str, arr: np.ndarray):
        feat.update(_safe_stats(arr, prefix))

    # time-domain
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    rms = librosa.feature.rms(y=y)[0]

    # frequency-domain
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    poly = librosa.feature.poly_features(y=y, sr=sr, order=2)

    add("zcr", zcr)
    add("rms", rms)
    add("spectral_centroid", centroid)
    add("spectral_bandwidth", bandwidth)
    add("spectral_rolloff", rolloff)
    add("spectral_flatness", flatness)
    add("polycoef_1", poly[0])
    add("polycoef_2", poly[1] if poly.shape[0] > 1 else np.array([0.0]))

    # spectral contrast: 7 bands by default
    for i in range(contrast.shape[0]):
        add(f"spectral_contrast_{i+1}", contrast[i])

    # extra stable summaries
    feat["spectral_centroid_slope"] = _slope(centroid)
    feat["rms_slope"] = _slope(rms)
    feat["zcr_slope"] = _slope(zcr)

    return feat


def extract_mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = 13) -> Dict[str, float]:
    feat = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    for i in range(n_mfcc):
        feat.update(_safe_stats(mfcc[i], f"mfcc_{i+1}"))
        feat.update(_safe_stats(delta1[i], f"delta_mfcc_{i+1}"))
        feat.update(_safe_stats(delta2[i], f"delta2_mfcc_{i+1}"))

    return feat


def extract_formant_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Optional formant extraction from parselmouth.
    """
    if not HAS_PARSEL_MOUTH:
        return {
            "formant_f1_mean": 0.0,
            "formant_f2_mean": 0.0,
            "formant_f3_mean": 0.0,
            "formant_f1_std": 0.0,
            "formant_f2_std": 0.0,
            "formant_f3_std": 0.0,
        }

    try:
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        formant = snd.to_formant_burg(time_step=0.01)
        times = np.linspace(0.0, snd.duration, num=max(int(snd.duration / 0.01), 10))
        f1, f2, f3 = [], [], []

        for t in times:
            f1.append(formant.get_value_at_time(1, t))
            f2.append(formant.get_value_at_time(2, t))
            f3.append(formant.get_value_at_time(3, t))

        f1 = _clean_array(np.array(f1, dtype=float))
        f2 = _clean_array(np.array(f2, dtype=float))
        f3 = _clean_array(np.array(f3, dtype=float))

        return {
            "formant_f1_mean": _safe_mean(f1),
            "formant_f2_mean": _safe_mean(f2),
            "formant_f3_mean": _safe_mean(f3),
            "formant_f1_std": _safe_std(f1),
            "formant_f2_std": _safe_std(f2),
            "formant_f3_std": _safe_std(f3),
        }
    except Exception:
        return {
            "formant_f1_mean": 0.0,
            "formant_f2_mean": 0.0,
            "formant_f3_mean": 0.0,
            "formant_f1_std": 0.0,
            "formant_f2_std": 0.0,
            "formant_f3_std": 0.0,
        }


def extract_pause_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Pause / silence statistics. Helpful if recordings contain long silences or hesitations.
    """
    intervals = librosa.effects.split(y, top_db=30)
    total = len(y)

    if total <= 0 or len(intervals) == 0:
        return {
            "silence_ratio": 1.0,
            "speech_ratio": 0.0,
            "num_segments": 0.0,
            "mean_segment_len": 0.0,
            "std_segment_len": 0.0,
        }

    seg_lengths = np.array([end - start for start, end in intervals], dtype=float)
    speech_samples = float(np.sum(seg_lengths))
    silence_samples = float(max(total - speech_samples, 0.0))

    return {
        "silence_ratio": float(silence_samples / total),
        "speech_ratio": float(speech_samples / total),
        "num_segments": float(len(intervals)),
        "mean_segment_len": float(np.mean(seg_lengths)),
        "std_segment_len": float(np.std(seg_lengths)),
    }


# -----------------------------
# Main feature extraction
# -----------------------------

def extract_features_from_wav(wav_path: str) -> Dict[str, float]:
    """
    Extract a rich feature set from a .wav file.

    Returned keys are designed to work well with your existing canonical mapping.
    """
    wav_path = str(wav_path)
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    if y.size == 0:
        return {}

    # trim leading/trailing silence, then normalize
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size == 0:
        return {}

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    features: Dict[str, float] = {}
    features["duration_sec"] = float(len(y) / sr)
    features["sample_rate"] = float(sr)
    features["num_samples"] = float(len(y))

    # core legacy features
    features.update(extract_pitch_features(y, sr))
    features.update(extract_jitter_features(y, sr))
    features.update(extract_shimmer_features(y, sr))
    features.update(extract_hnr_nhr_features(y, sr))
    features.update(extract_complexity_features(y, sr))

    # richer acoustic features
    features.update(extract_spectral_features(y, sr))
    features.update(extract_mfcc_features(y, sr, n_mfcc=13))
    features.update(extract_formant_features(y, sr))
    features.update(extract_pause_features(y, sr))

    # simple intensity summary
    feat_rms = librosa.feature.rms(y=y)[0]
    features["intensity_mean"] = _safe_mean(feat_rms)
    features["intensity_std"] = _safe_std(feat_rms)
    features["intensity_max"] = _safe_max(feat_rms)

    return features


# -----------------------------
# Dataset processing
# -----------------------------

def infer_subject_id(wav_file: Path) -> str:
    """
    Try to derive a stable subject/group ID from filename.
    Adjust this if your MDVR naming convention is known.

    Default behavior:
    - If there are underscores/hyphens, use the first token.
    - Otherwise use the stem as-is.
    """
    stem = wav_file.stem

    # try common tokenized patterns first
    tokenized = re.split(r"[_-]+", stem)
    if len(tokenized) > 1 and tokenized[0]:
        return tokenized[0]

    # if the name starts with letters+digits, use that prefix
    m = re.match(r"^([A-Za-z]+\d+)", stem)
    if m:
        return m.group(1)

    return stem


def process_mdvr_dataset(dataset_root: str) -> Dict[str, Dict]:
    """
    Process all .wav files in the MDVR dataset structure.

    Expected structure (example):
        root/
          ReadText/
            HC/
            PD/
          SpontaneousDialogue/
            HC/
            PD/

    Returns:
        dict[record_id] = {
            "label": 0/1,
            "subject_id": group id for splitting,
            "features": {...},
            "file": "path/to/audio.wav",
        }
    """
    root = Path(dataset_root)
    data: Dict[str, Dict] = {}

    for task in ["ReadText", "SpontaneousDialogue"]:
        task_dir = root / task
        if not task_dir.exists():
            continue

        for class_type, label in [("HC", 0), ("PD", 1)]:
            class_dir = task_dir / class_type
            if not class_dir.exists():
                continue

            for wav_file in sorted(class_dir.glob("*.wav")):
                try:
                    rec_id = f"{task}_{class_type}_{wav_file.stem}"
                    subject_id = infer_subject_id(wav_file)
                    features = extract_features_from_wav(str(wav_file))

                    data[rec_id] = {
                        "label": int(label),
                        "subject_id": subject_id,
                        "features": features,
                        "file": str(wav_file),
                        "task": task,
                        "class_type": class_type,
                    }
                except Exception as e:
                    print(f"Error processing {wav_file}: {e}")

    return data


def features_to_dataframe(data: Dict[str, Dict]):
    """
    Convert the processed dataset dict into a feature dataframe.
    """
    import pandas as pd

    rows = []
    for rec_id, item in data.items():
        row = {
            "record_id": rec_id,
            "label": item["label"],
            "subject_id": item.get("subject_id", rec_id),
            "file": item.get("file", ""),
            "task": item.get("task", ""),
            "class_type": item.get("class_type", ""),
        }
        row.update(item.get("features", {}))
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
