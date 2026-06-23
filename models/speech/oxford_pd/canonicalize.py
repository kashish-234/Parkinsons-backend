"""
Canonical feature-name mapping for the Oxford (UCI) speech sub-model.

Goal: every speech sub-model (oxford LGBM, HuBERT, Whisper, CNN spectrogram)
must emit SHAPFeature.name values from the SAME vocabulary, so
IntraModalFuser._merge_shap can match features across models by name
instead of treating "MDVP_Jitter_" and "jitter_local" as unrelated.
"""

# original (sanitized) column name -> canonical name used project-wide
CANONICAL_MAP = {
    "MDVP_Fo_Hz_":        "pitch_mean",
    "MDVP_Fhi_Hz_":       "pitch_max",
    "MDVP_Flo_Hz_":       "pitch_min",
    "MDVP_Jitter_":       "jitter",       # MDVP:Jitter(%)
    "MDVP_Jitter_Abs_":   "jitter_abs",
    "MDVP_RAP":           "jitter_rap",
    "MDVP_PPQ":           "jitter_ppq",
    "Jitter_DDP":         "jitter_ddp",
    "MDVP_Shimmer":       "shimmer",
    "MDVP_Shimmer_dB_":   "shimmer_db",
    "Shimmer_APQ3":       "shimmer_apq3",
    "Shimmer_APQ5":       "shimmer_apq5",
    "MDVP_APQ":           "shimmer_apq",
    "Shimmer_DDA":        "shimmer_dda",
    "NHR":                "nhr",
    "HNR":                "hnr",
    "RPDE":               "rpde",
    "DFA":                "dfa",
    "spread1":            "spread1",
    "spread2":            "spread2",
    "D2":                 "d2",
    "PPE":                "ppe",
}


def canonicalize(name: str) -> str:
    """
    Map a sanitized/raw feature name to its canonical cross-model name.
    Falls back to the input unchanged (lowercased) if no mapping exists,
    so unmapped features don't silently disappear — they just won't merge
    with other sub-models until you add them to CANONICAL_MAP.
    """
    if name in CANONICAL_MAP:
        return CANONICAL_MAP[name]
    return name.strip("_").lower()