"""
Canonical feature-name mapping for the REM Sleep Behavior Disorder dataset
(130 participants: PD=30, RB=50, HC=50; 63 feature columns after stripping).

Same goal as every other sub-model's canonicalize.py: map raw feature
names to a shared vocabulary so IntraModalFuser._merge_shap can match
attributions across models by name.

Feature families present in dataset.csv:
  1. Demographics / clinical:   Age, Gender, family history, medications,
                                 Levodopa, Clonazepam, Hoehn&Yahr, UPDRS III
  2. UPDRS III motor items:     Speech, Facial Expression, Tremor (items 18-31)
  3. Speech acoustics (set 1):  Entropy, Rate, Acceleration, Pause, Voiced,
                                 Gaping, Unvoiced, Fricatives, Loudness,
                                 Respiration, Latency (columns 41-52)
  4. Speech acoustics (set 2):  Same features, second recording (columns 53-64)
"""
import re

# ---------------------------------------------------------------------------
# Direct exact-match mappings (stripped column names from dataset.csv)
# ---------------------------------------------------------------------------
_DIRECT_MAP = {
    # ── Demographics / clinical ──────────────────────────────────────────
    "Age  (years)":                                             "age",
    "Gender":                                                   "gender",
    "Positive  history  of  Parkinson  disease  in  family":   "family_pd_history",
    "Age  of  disease  onset  (years)":                        "disease_onset_age",
    "Duration  of  disease  from  first  symptoms  (years)":   "disease_duration",
    "Antidepressant  therapy":                                  "antidepressant",
    "Antiparkinsonian  medication":                             "antiparkinsonian",
    "Antipsychotic  medication":                                "antipsychotic",
    "Benzodiazepine  medication":                               "benzodiazepine",
    "Levodopa  equivalent  (mg/day)":                          "levodopa_equiv",
    "Clonazepam  (mg/day)":                                    "clonazepam",
    "Overview  of  motor  examination:  Hoehn  &  Yahr  scale  (-)":  "hoehn_yahr",
    "Overview  of  motor  examination:  UPDRS  III  total  (-)":       "updrs_iii_total",

    # ── UPDRS III motor sub-items ────────────────────────────────────────
    "18.  Speech":                                  "updrs_speech",
    "19.  Facial  Expression":                      "updrs_facial_expression",
    "20.  Tremor  at  Rest  -  head":               "tremor_rest_head",
    "20.  Tremor  at  Rest  -  RUE":                "tremor_rest_rue",
    "20.  Tremor  at  Rest  -  LUE":                "tremor_rest_lue",
    "20.  Tremor  at  Rest  -  RLE":                "tremor_rest_rle",
    "20.  Tremor  at  Rest  -  LLE":                "tremor_rest_lle",
    "21.  Action  or  Postural  Tremor  -  RUE":    "tremor_postural_rue",
    "21.  Action  or  Postural  Tremor  -  LUE":    "tremor_postural_lue",
    "22.  Rigidity  -  neck":                       "rigidity_neck",
    "22.  Rigidity  -  RUE":                        "rigidity_rue",
    "22.  Rigidity  -  LUE":                        "rigidity_lue",
    "22.  Rigidity  -  RLE":                        "rigidity_rle",
    "22.  Rigidity  -  LLE":                        "rigidity_lle",
    "23.Finger  Taps  -  RUE":                      "finger_taps_rue",
    "23.Finger  Taps  -  LUE":                      "finger_taps_lue",
    "24.  Hand  Movements  -  RUE":                 "hand_movements_rue",
    "24.  Hand  Movements  -  LUE":                 "hand_movements_lue",
    "25.  Rapid  Alternating  Movements  -  RUE":   "rapid_alt_movements_rue",
    "25.  Rapid  Alternating  Movements  -  LUE":   "rapid_alt_movements_lue",
    "26.  Leg  Agility  -  RLE":                    "leg_agility_rle",
    "26.  Leg  Agility  -  LLE":                    "leg_agility_lle",
    "27.  Arising  from  Chair":                    "arising_from_chair",
    "28.  Posture":                                 "posture",
    "29.  Gait":                                    "gait",
    "30.  Postural  Stability":                     "postural_stability",
    "31.  Body  Bradykinesia  and  Hypokinesia":    "bradykinesia",

    # ── Speech acoustics — recording 1 ──────────────────────────────────
    "Entropy  of  speech  timing  (-)":             "speech_entropy_1",
    "Rate  of  speech  timing  (-/min)":            "speech_rate_1",
    "Acceleration  of  speech  timing  (-/min2)":   "speech_acceleration_1",
    "Duration  of  pause  intervals  (ms)":         "pause_duration_1",
    "Duration  of  voiced  intervals  (ms)":        "voiced_duration_1",
    "Gaping  in-between  voiced  intervals  (-/min)": "voiced_gaping_1",
    "Duration  of  unvoiced  stops  (ms)":          "unvoiced_stops_1",
    "Decay  of  unvoiced  fricatives  (‰/min)":     "fricatives_decay_1",
    "Relative  loudness  of  respiration  (dB)":    "respiration_loudness_1",
    "Pause  intervals  per  respiration  (-)":      "pause_per_respiration_1",
    "Rate  of  speech  respiration  (-/min)":       "respiration_rate_1",
    "Latency  of  respiratory  exchange  (ms)":     "respiration_latency_1",

    # ── Speech acoustics — recording 2 (.1 suffix added by pandas) ──────
    "Entropy  of  speech  timing  (-) .1":           "speech_entropy_2",
    "Rate  of  speech  timing  (-/min) .1":          "speech_rate_2",
    "Acceleration  of  speech  timing  (-/min2) .1": "speech_acceleration_2",
    "Duration  of  pause  intervals  (ms) .1":       "pause_duration_2",
    "Duration  of  voiced  intervals  (ms) .1":      "voiced_duration_2",
    "Gaping  in-between  voiced  Intervals  (-/min)": "voiced_gaping_2",
    "Duration  of  unvoiced  stops  (ms) .1":        "unvoiced_stops_2",
    "Decay  of  unvoiced  fricatives  (‰/min) .1":   "fricatives_decay_2",
    "Relative  loudness  of  respiration  (dB) .1":  "respiration_loudness_2",
    "Pause  intervals  per  respiration  (-) .1":    "pause_per_respiration_2",
    "Rate  of  speech  respiration  (-/min) .1":     "respiration_rate_2",
    "Latency  of  respiratory  exchange  (ms) .1":   "respiration_latency_2",
}

# No regex families needed — all columns are direct-mapped above.
_FAMILY_PATTERNS = []


def canonicalize(raw_name: str) -> str:
    """
    Map a raw REM dataset feature name to its canonical cross-model name.
    Checked in order: exact match in _DIRECT_MAP, then cleaned-lowercase
    fallback so unmapped features still surface in SHAP output.
    """
    if raw_name in _DIRECT_MAP:
        return _DIRECT_MAP[raw_name]

    for pattern, fn in _FAMILY_PATTERNS:
        m = pattern.match(raw_name)
        if m:
            return fn(m)

    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", raw_name).strip("_").lower()
    return cleaned
