import re
# Raw preprocess.py output names -> shared canonical vocabulary.
# Only the MDVP-style names need explicit mapping; mfcc_*, delta_mfcc_*,

_DIRECT_MAP = {
    # --- Pitch ---
    "MDVP_Fo_Hz_":        "pitch_mean",
    "MDVP_Fhi_Hz_":       "pitch_max",
    "MDVP_Flo_Hz_":       "pitch_min",

    # --- Jitter ---
    "MDVP_Jitter_":       "jitter",
    "MDVP_Jitter_Abs_":   "jitter_abs",
    "MDVP_RAP":           "jitter_rap",
    "MDVP_PPQ":           "jitter_ppq",
    "Jitter_DDP":         "jitter_ddp",

    # --- Shimmer ---
    "MDVP_Shimmer":       "shimmer",
    "MDVP_Shimmer_dB_":   "shimmer_db",
    "Shimmer_APQ3":       "shimmer_apq3",
    "Shimmer_APQ5":       "shimmer_apq5",
    "MDVP_APQ":           "shimmer_apq",
    "Shimmer_DDA":        "shimmer_dda",

    # --- Harmonicity ---
    "NHR":                "nhr",
    "HNR":                "hnr",

    # --- Complexity (already lowercase in preprocess.py output,
    #     but listed here for explicitness) ---
    "RPDE":               "rpde",
    "DFA":                "dfa",
    "spread1":            "spread1",
    "spread2":            "spread2",
    "D2":                 "d2",
    "PPE":                "ppe",
}


def canonicalize(raw_name: str) -> str:
    if raw_name in _DIRECT_MAP:
        return _DIRECT_MAP[raw_name]

    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", raw_name).strip("_").lower()
    return cleaned