import re

def canonicalize(name: str) -> str:
    """
    Normalize a raw feature name to a clean, lowercase, underscore-separated
    identifier safe for use as a SHAP feature name.
    """
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())