import logging

logger = logging.getLogger(__name__)

FEATURE_MAP = {

    # Wrist Movement
    "wrist_mvmnt_x_median": "Median Wrist Movement X",
    "wrist_mvmnt_x_min": "Minimum Wrist Movement X",

    "wrist_mvmnt_y_median": "Median Wrist Movement Y",
    "wrist_mvmnt_y_min": "Minimum Wrist Movement Y",
    "wrist_mvmnt_y_max": "Maximum Wrist Movement Y",

    "wrist_mvmnt_dist_min": "Minimum Wrist Movement Distance",

    # Periodicity
    "aperiodicity_denoised": "Aperiodicity",
    "aperiodicity_trimmed": "Aperiodicity (Trimmed)",

    "periodEntropy_denoised": "Period Entropy",

    "periodVarianceNorm_denoised": "Normalized Period Variance",

    "periodVarianceNorm_trimmed": "Normalized Period Variance (Trimmed)",

    "period_median_denoised": "Median Period",

    "period_quartile_range_denoised": "Period IQR",

    "period_quartile_range_trimmed": "Period IQR (Trimmed)",

    "period_min_denoised": "Minimum Period",

    # Freeze
    "numFreeze_denoised": "Freeze Count",

    "numFreeze_trimmed": "Freeze Count (Trimmed)",

    "maxFreezeDuration_denoised": "Maximum Freeze Duration",

    "maxFreezeDuration_trimmed": "Maximum Freeze Duration (Trimmed)",

    "numInterruptions_denoised": "Movement Interruptions",

    # Frequency
    "frequency_quartile_range_denoised": "Frequency IQR",

    "frequency_min_denoised": "Minimum Frequency",

    "frequency_stdev_denoised": "Frequency Standard Deviation",

    "frequency_lr_fitness_r2_denoised": "Frequency Trend R²",

    "frequency_lr_slope_denoised": "Frequency Trend Slope",

    "frequency_lr_fitness_r2_trimmed": "Frequency Trend R² (Trimmed)",

    "frequency_lr_slope_trimmed": "Frequency Trend Slope (Trimmed)",

    "frequency_fit_min_degree_denoised": "Minimum Frequency Polynomial Degree",

    "frequency_fit_min_degree_trimmed": "Minimum Frequency Polynomial Degree (Trimmed)",

    # Amplitude
    "amplitude_median_denoised": "Median Amplitude",

    "amplitude_quartile_range_denoised": "Amplitude IQR",

    "amplitude_max_denoised": "Maximum Amplitude",

    "amplitude_stdev_denoised": "Amplitude Standard Deviation",

    "amplitude_entropy_denoised": "Amplitude Entropy",

    "amplitude_stdev_trimmed": "Amplitude Standard Deviation (Trimmed)",

    "amplitude_decrement_fitness_r2_denoised": "Amplitude Decrement R²",

    "amplitude_decrement_slope_denoised": "Amplitude Decrement Slope",

    "amplitude_decrement_end_to_mean_denoised": "Amplitude End-to-Mean Ratio",

    "amplitude_decrement_fit_min_degree_denoised": "Amplitude Polynomial Degree",

    "amplitude_decrement_last_to_first_half_denoised": "Amplitude Last-to-First Half Ratio",

    "amplitude_decrement_fitness_r2_trimmed": "Amplitude Decrement R² (Trimmed)",

    "amplitude_decrement_slope_trimmed": "Amplitude Decrement Slope (Trimmed)",

    "amplitude_decrement_end_to_mean_trimmed": "Amplitude End-to-Mean Ratio (Trimmed)",

    "amplitude_decrement_fit_min_degree_trimmed": "Amplitude Polynomial Degree (Trimmed)",

    # Speed
    "speed_median_denoised": "Median Speed",

    "speed_quartile_range_denoised": "Speed IQR",

    "speed_min_denoised": "Minimum Speed",

    "speed_max_denoised": "Maximum Speed",

    "speed_median_trimmed": "Median Speed (Trimmed)",

    "speed_min_trimmed": "Minimum Speed (Trimmed)",

    # Acceleration
    "acceleration_min_denoised": "Minimum Acceleration",

    "acceleration_min_trimmed": "Minimum Acceleration (Trimmed)",

    # Peaks
    "num_peaks_trimmed": "Peak Count",

    # Patient
    "hand": "Dominant Hand",

    "gender": "Gender",

    "age": "Age"
}


def canonicalize(feature_name: str) -> str:
    """
    Convert raw feature names into human-readable names.
    """

    if feature_name in FEATURE_MAP:
        return FEATURE_MAP[feature_name]

    logger.warning(
        "Feature '%s' not found in FEATURE_MAP; falling back to title-case.",
        feature_name,
    )

    return (
        feature_name
        .replace("_", " ")
        .title()
    )