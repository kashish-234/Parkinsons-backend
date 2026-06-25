"""
Canonical feature-name mapping for the Sakar / UCI PD Speech Features
sub-model (pd_speech_features.csv, 752 raw feature columns).

Same goal as every other sub-model's canonicalize.py: every speech
sub-model must emit SHAPFeature.name values from the same vocabulary
so IntraModalFuser._merge_shap can match attributions across models by
name (e.g. Oxford's "MDVP_Jitter_" and this dataset's "locPctJitter"
both become "jitter").

Six raw families present in this CSV, mapped here:
  1. Baseline:    PPE, DFA, RPDE, jitter/shimmer variants, harmonicity, pulses
  2. Intensity / Formant / Bandwidth: meanIntensity, f1-f4, b1-b4, GQ_*
  3. Vocal fold:  GNE_*, VFER_*, IMF_*
  4. MFCC:        mean/std of MFCC coefficients + delta + delta-delta (ordinal-named)
  5. Wavelet:     Ea/Ed, det/app entropy & TKEO (decomposition levels 1-10)
  6. TQWT:        energy/entropy/mean/std/min/max/skew/kurtosis (levels 1-36)
"""
import re

# ---------------------------------------------------------------------------
# 1. Baseline features -- direct overlap with Oxford's canonical vocabulary
# ---------------------------------------------------------------------------
_DIRECT_MAP = {
    "PPE": "ppe",
    "DFA": "dfa",
    "RPDE": "rpde",
    "numPulses": "num_pulses",
    "numPeriodsPulses": "num_periods_pulses",
    "meanPeriodPulses": "period_mean",
    "stdDevPeriodPulses": "period_std",

    "locPctJitter": "jitter",
    "locAbsJitter": "jitter_abs",
    "rapJitter": "jitter_rap",
    "ppq5Jitter": "jitter_ppq",
    "ddpJitter": "jitter_ddp",

    "locShimmer": "shimmer",
    "locDbShimmer": "shimmer_db",
    "apq3Shimmer": "shimmer_apq3",
    "apq5Shimmer": "shimmer_apq5",
    "apq11Shimmer": "shimmer_apq11",
    "ddaShimmer": "shimmer_dda",

    "meanAutoCorrHarmonicity": "autocorr_harmonicity",
    "meanNoiseToHarmHarmonicity": "nhr",
    "meanHarmToNoiseHarmonicity": "hnr",

    # --- Intensity ---
    "minIntensity": "intensity_min",
    "maxIntensity": "intensity_max",
    "meanIntensity": "intensity_mean",

    # --- Formants & bandwidths ---
    "f1": "formant_1", "f2": "formant_2", "f3": "formant_3", "f4": "formant_4",
    "b1": "bandwidth_1", "b2": "bandwidth_2", "b3": "bandwidth_3", "b4": "bandwidth_4",

    # --- Glottal quotient ---
    "GQ_prc5_95": "gq_prc5_95",
    "GQ_std_cycle_open": "gq_std_cycle_open",
    "GQ_std_cycle_closed": "gq_std_cycle_closed",

    # --- Glottal-to-noise excitation ---
    "GNE_mean": "gne_mean",
    "GNE_std": "gne_std",
    "GNE_SNR_TKEO": "gne_snr_tkeo",
    "GNE_SNR_SEO": "gne_snr_seo",
    "GNE_NSR_TKEO": "gne_nsr_tkeo",
    "GNE_NSR_SEO": "gne_nsr_seo",

    # --- Vocal fold excitation ratio ---
    "VFER_mean": "vfer_mean",
    "VFER_std": "vfer_std",
    "VFER_entropy": "vfer_entropy",
    "VFER_SNR_TKEO": "vfer_snr_tkeo",
    "VFER_SNR_SEO": "vfer_snr_seo",
    "VFER_NSR_TKEO": "vfer_nsr_tkeo",
    "VFER_NSR_SEO": "vfer_nsr_seo",

    # --- Intrinsic mode function (empirical mode decomposition) ---
    "IMF_SNR_SEO": "imf_snr_seo",
    "IMF_SNR_TKEO": "imf_snr_tkeo",
    "IMF_SNR_entropy": "imf_snr_entropy",
    "IMF_NSR_SEO": "imf_nsr_seo",
    "IMF_NSR_TKEO": "imf_nsr_tkeo",
    "IMF_NSR_entropy": "imf_nsr_entropy",

    "mean_Log_energy": "log_energy_mean",
    "std_Log_energy": "log_energy_std",
    "mean_delta_log_energy": "log_energy_delta_mean",
    "std_delta_log_energy": "log_energy_delta_std",
    "mean_delta_delta_log_energy": "log_energy_delta_delta_mean",
    "std_delta_delta_log_energy": "log_energy_delta_delta_std",

    "Ea": "wavelet_ea",
    "Ea2": "wavelet_ea2",
}

# Ordinal words used by this dataset's MFCC/delta naming -> integer string
_ORDINAL_TO_INT = {
    "0th": "0", "1st": "1", "2nd": "2", "3rd": "3", "4th": "4",
    "5th": "5", "6th": "6", "7th": "7", "8th": "8", "9th": "9",
    "10th": "10", "11th": "11", "12th": "12",
}
_ORD_PATTERN = "|".join(_ORDINAL_TO_INT.keys())

# ---------------------------------------------------------------------------
# 2. Regex families -- MFCC / delta / delta-delta / wavelet / TQWT
#    Order matters: more specific patterns must come before general ones.
# ---------------------------------------------------------------------------
_FAMILY_PATTERNS = [
    # MFCC coefficients: mean_MFCC_0th_coef / std_MFCC_12th_coef
    (re.compile(rf"^mean_MFCC_({_ORD_PATTERN})_coef$"),
     lambda m: f"mfcc_{_ORDINAL_TO_INT[m.group(1)]}_mean"),
    (re.compile(rf"^std_MFCC_({_ORD_PATTERN})_coef$"),
     lambda m: f"mfcc_{_ORDINAL_TO_INT[m.group(1)]}_std"),

    # Delta-delta (2nd derivative) -- check BEFORE plain delta since both
    # contain "delta" as a substring of the column name.
    # Two naming styles appear in this file:
    #   mean_delta_delta_0th  (0th coefficient only)
    #   mean_1st_delta_delta  (1st-12th coefficients)
    (re.compile(rf"^mean_delta_delta_({_ORD_PATTERN})$"),
     lambda m: f"mfcc_{_ORDINAL_TO_INT[m.group(1)]}_delta_delta_mean"),
    (re.compile(rf"^std_delta_delta_({_ORD_PATTERN})$"),
     lambda m: f"mfcc_{_ORDINAL_TO_INT[m.group(1)]}_delta_delta_std"),
    (re.compile(rf"^mean_({_ORD_PATTERN})_delta_delta$"),
     lambda m: f"mfcc_{_ORDINAL_TO_INT[m.group(1)]}_delta_delta_mean"),
    (re.compile(rf"^std_({_ORD_PATTERN})_delta_delta$"),
     lambda m: f"mfcc_{_ORDINAL_TO_INT[m.group(1)]}_delta_delta_std"),

    # Plain delta (1st derivative): mean_0th_delta / std_12th_delta
    (re.compile(rf"^mean_({_ORD_PATTERN})_delta$"),
     lambda m: f"mfcc_{_ORDINAL_TO_INT[m.group(1)]}_delta_mean"),
    (re.compile(rf"^std_({_ORD_PATTERN})_delta$"),
     lambda m: f"mfcc_{_ORDINAL_TO_INT[m.group(1)]}_delta_std"),

    # --- Wavelet family (decomposition levels 1-10) ---
    (re.compile(r"^Ed_(\d+)_coef$"),                    lambda m: f"wavelet_ed_{m.group(1)}"),
    (re.compile(r"^Ed2_(\d+)_coef$"),                   lambda m: f"wavelet_ed2_{m.group(1)}"),
    (re.compile(r"^det_entropy_shannon_(\d+)_coef$"),   lambda m: f"wavelet_det_entropy_shannon_{m.group(1)}"),
    (re.compile(r"^det_entropy_log_(\d+)_coef$"),       lambda m: f"wavelet_det_entropy_log_{m.group(1)}"),
    (re.compile(r"^det_TKEO_mean_(\d+)_coef$"),         lambda m: f"wavelet_det_tkeo_mean_{m.group(1)}"),
    (re.compile(r"^det_TKEO_std_(\d+)_coef$"),          lambda m: f"wavelet_det_tkeo_std_{m.group(1)}"),
    (re.compile(r"^app_entropy_shannon_(\d+)_coef$"),   lambda m: f"wavelet_app_entropy_shannon_{m.group(1)}"),
    (re.compile(r"^app_entropy_log_(\d+)_coef$"),       lambda m: f"wavelet_app_entropy_log_{m.group(1)}"),
    (re.compile(r"^app_det_TKEO_mean_(\d+)_coef$"),     lambda m: f"wavelet_app_tkeo_mean_{m.group(1)}"),
    (re.compile(r"^app_TKEO_std_(\d+)_coef$"),          lambda m: f"wavelet_app_tkeo_std_{m.group(1)}"),
    # Long-term (LT) variants of the same families
    (re.compile(r"^det_LT_entropy_shannon_(\d+)_coef$"), lambda m: f"wavelet_det_lt_entropy_shannon_{m.group(1)}"),
    (re.compile(r"^det_LT_entropy_log_(\d+)_coef$"),     lambda m: f"wavelet_det_lt_entropy_log_{m.group(1)}"),
    (re.compile(r"^det_LT_TKEO_mean_(\d+)_coef$"),       lambda m: f"wavelet_det_lt_tkeo_mean_{m.group(1)}"),
    (re.compile(r"^det_LT_TKEO_std_(\d+)_coef$"),        lambda m: f"wavelet_det_lt_tkeo_std_{m.group(1)}"),
    (re.compile(r"^app_LT_entropy_shannon_(\d+)_coef$"), lambda m: f"wavelet_app_lt_entropy_shannon_{m.group(1)}"),
    (re.compile(r"^app_LT_entropy_log_(\d+)_coef$"),     lambda m: f"wavelet_app_lt_entropy_log_{m.group(1)}"),
    (re.compile(r"^app_LT_TKEO_mean_(\d+)_coef$"),       lambda m: f"wavelet_app_lt_tkeo_mean_{m.group(1)}"),
    (re.compile(r"^app_LT_TKEO_std_(\d+)_coef$"),        lambda m: f"wavelet_app_lt_tkeo_std_{m.group(1)}"),

    # --- TQWT family (decomposition levels 1-36) ---
    (re.compile(r"^tqwt_energy_dec_(\d+)$"),           lambda m: f"tqwt_energy_{m.group(1)}"),
    (re.compile(r"^tqwt_entropy_shannon_dec_(\d+)$"),  lambda m: f"tqwt_entropy_shannon_{m.group(1)}"),
    (re.compile(r"^tqwt_entropy_log_dec_(\d+)$"),      lambda m: f"tqwt_entropy_log_{m.group(1)}"),
    (re.compile(r"^tqwt_TKEO_mean_dec_(\d+)$"),        lambda m: f"tqwt_tkeo_mean_{m.group(1)}"),
    (re.compile(r"^tqwt_TKEO_std_dec_(\d+)$"),         lambda m: f"tqwt_tkeo_std_{m.group(1)}"),
    (re.compile(r"^tqwt_medianValue_dec_(\d+)$"),      lambda m: f"tqwt_median_{m.group(1)}"),
    (re.compile(r"^tqwt_meanValue_dec_(\d+)$"),        lambda m: f"tqwt_mean_{m.group(1)}"),
    (re.compile(r"^tqwt_stdValue_dec_(\d+)$"),         lambda m: f"tqwt_std_{m.group(1)}"),
    (re.compile(r"^tqwt_minValue_dec_(\d+)$"),         lambda m: f"tqwt_min_{m.group(1)}"),
    (re.compile(r"^tqwt_maxValue_dec_(\d+)$"),         lambda m: f"tqwt_max_{m.group(1)}"),
    (re.compile(r"^tqwt_skewnessValue_dec_(\d+)$"),    lambda m: f"tqwt_skewness_{m.group(1)}"),
    (re.compile(r"^tqwt_kurtosisValue_dec_(\d+)$"),    lambda m: f"tqwt_kurtosis_{m.group(1)}"),
]


def canonicalize(raw_name: str) -> str:
    """
    Map a raw Sakar/UCI-dataset feature name to its canonical cross-model
    name. Checked in order: exact match in _DIRECT_MAP, then each regex
    family pattern, then a cleaned-lowercase fallback so unmapped features
    still surface in SHAP output (just without cross-model merging).
    """
    if raw_name in _DIRECT_MAP:
        return _DIRECT_MAP[raw_name]

    for pattern, fn in _FAMILY_PATTERNS:
        m = pattern.match(raw_name)
        if m:
            return fn(m)

    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", raw_name).strip("_").lower()
    return cleaned