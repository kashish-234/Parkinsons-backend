import re
 
_DIRECT_MAP = {
    "centroid_mean":    "spectral_centroid_mean",
    "centroid_std":     "spectral_centroid_std",
    "centroid_median":  "spectral_centroid_median",
    "centroid_min":     "spectral_centroid_min",
    "centroid_max":     "spectral_centroid_max",
    "bandwidth_mean":   "spectral_bandwidth_mean",
    "bandwidth_std":    "spectral_bandwidth_std",
    "bandwidth_median": "spectral_bandwidth_median",
    "bandwidth_min":    "spectral_bandwidth_min",
    "bandwidth_max":    "spectral_bandwidth_max",
    "rolloff_mean":     "spectral_rolloff_mean",
    "rolloff_std":      "spectral_rolloff_std",
    "rolloff_median":   "spectral_rolloff_median",
    "rolloff_min":      "spectral_rolloff_min",
    "rolloff_max":      "spectral_rolloff_max",
    "rms_mean":         "rms_energy_mean",
    "rms_std":          "rms_energy_std",
    "rms_median":       "rms_energy_median",
    "rms_min":          "rms_energy_min",
    "rms_max":          "rms_energy_max",
 
    # Shimmer — already similar to canonical but make explicit
    "shimmer_local":    "shimmer",
    "shimmer_apq3":     "shimmer_apq3",
    "shimmer_apq5":     "shimmer_apq5",
 
    # These are already canonical — listed for explicitness
    "hnr":              "hnr",
    "zcr_mean":         "zcr_mean",
    "pitch_mean":       "pitch_mean",
    "pitch_std":        "pitch_std",
    "pitch_median":     "pitch_median",
    "pitch_min":        "pitch_min",
    "pitch_max":        "pitch_max",
}
 
 
def canonicalize(raw_name: str) -> str:
    """
    Map a raw Italian-pipeline feature name to the canonical vocabulary.
    mfcc_N_stat names (mfcc_1_mean, mfcc_7_std, etc.) are already canonical
    and fall through the cleaned-lowercase fallback unchanged.
    """
    if raw_name in _DIRECT_MAP:
        return _DIRECT_MAP[raw_name]
    return re.sub(r"[^A-Za-z0-9]+", "_", raw_name).strip("_").lower()