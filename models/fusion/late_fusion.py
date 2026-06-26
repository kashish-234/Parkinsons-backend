"""
Late Fusion Model for Parkinson's Disease Prediction
 
Combines predictions from 6 modalities (speech, tapping, gait, rem, handwriting, neuroimaging)
into a single fused prediction with uncertainty quantification.
 
Each input is a ModalityResult from intra-modal fusion (or single model for non-speech).
"""
 
import logging
import numpy as np
from typing import List, Dict, Tuple
from models.base.contracts import ModalityResult, FusedResult, SHAPFeature
 
logger = logging.getLogger(__name__)
 
class LateFusionModel:
    """
    Combines predictions from multiple modalities into final risk assessment.
    
    Strategy:
    - Uses modality-specific weights (based on validation performance)
    - Handles missing modalities gracefully
    - Propagates uncertainty across modalities
    - Merges SHAP features from all modalities
    """
    
    # Modality weights based on validation AUC/performance
    # These should be updated as you train and validate each modality
    DEFAULT_WEIGHTS = {
        "speech": 0.25,           # Speech is most informative
        "tapping": 0.20,          # Motor activity (finger tapping)
        "gait": 0.20,             # Motor activity (gait)
        "rem": 0.15,              # REM sleep behavior
        "handwriting": 0.15,      # Fine motor control
        "neuroimaging": 0.05,     # Neuroimaging (if available)
    }
    
    # Confidence thresholds for risk categorization
    RISK_THRESHOLDS = {
        "low": 0.40,              # P(PD) < 0.40 → Low risk
        "borderline": 0.65,       # 0.40 ≤ P(PD) < 0.65 → Borderline
        # P(PD) ≥ 0.65 → High risk
    }
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize Late Fusion Model.
        
        Args:
            weights: Dict mapping modality names to fusion weights.
                    If None, uses DEFAULT_WEIGHTS.
                    Weights will be normalized internally.
        """
        if weights is None:
            weights = self.DEFAULT_WEIGHTS.copy()
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to positive value")
        
        self.weights = {k: v / total for k, v in weights.items()}
        self.modality_order = ["speech", "tapping", "gait", "rem", "handwriting", "neuroimaging"]
        
        logger.info(f"LateFusionModel initialized with weights: {self.weights}")
    
    def fuse(self, modality_results: List[ModalityResult], 
             patient_id: str, job_id: str) -> FusedResult:
        """
        Fuse multiple modality predictions into final result.
        
        Args:
            modality_results: List of ModalityResult from each modality
            patient_id: Patient identifier
            job_id: Job/prediction session identifier
        
        Returns:
            FusedResult with final probability, risk label, and confidence intervals
        """
        
        # Filter for available modalities
        available_results = [r for r in modality_results if r.available]
        
        if not available_results:
            raise ValueError("No available modality data for fusion")
        
        # Get probabilities and weights for available modalities
        available_modalities = {r.modality: r for r in available_results}
        
        # Weighted probability fusion
        fused_prob, modality_weights = self._weighted_average(available_results)
        
        # Propagate uncertainty from modality CIs
        ci_low, ci_high = self._propagate_uncertainty(available_results, fused_prob)
        
        # Merge SHAP features from all modalities
        merged_shap = self._merge_shap_features(available_results)
        
        # Determine risk label
        risk_label = self._classify_risk(fused_prob)
        
        # Create fused result
        fused = FusedResult(
            job_id=job_id,
            patient_id=patient_id,
            probability=fused_prob,
            ci_low=ci_low,
            ci_high=ci_high,
            risk_label=risk_label,
            modality_results=modality_results,
            modality_weights=modality_weights,
            fusion_model_version="v1.0"
        )
        
        logger.info(
            f"Fused prediction: patient={patient_id}, "
            f"prob={fused_prob:.4f}, risk={risk_label}, "
            f"modalities={list(available_modalities.keys())}"
        )
        
        return fused
    
    def _weighted_average(self, results: List[ModalityResult]) -> Tuple[float, Dict[str, float]]:
        """
        Compute weighted average of modality probabilities.
        
        Uses stored weights, normalized by available modalities.
        
        Returns:
            (fused_probability, modality_weights_used)
        """
        probabilities = {}
        result_weights = {}
        
        for result in results:
            modality = result.modality
            # Use stored weight, default to 1.0 if modality not in dict
            weight = self.weights.get(modality, 1.0)
            probabilities[modality] = result.probability
            result_weights[modality] = weight
        
        # Normalize weights for available modalities
        total_weight = sum(result_weights.values())
        normalized_weights = {k: v / total_weight for k, v in result_weights.items()}
        
        # Weighted average
        fused_prob = sum(
            normalized_weights[m] * probabilities[m] 
            for m in probabilities.keys()
        )
        
        return float(fused_prob), normalized_weights
    
    def _propagate_uncertainty(self, results: List[ModalityResult], 
                               fused_prob: float) -> Tuple[float, float]:
        """
        Propagate uncertainty from individual modality CIs to fused CI.
        
        Strategy: 
        - Pool CI bounds across modalities
        - Widen by disagreement between modalities
        
        Returns:
            (ci_low, ci_high)
        """
        if not results:
            # Fallback CI
            return max(0.0, fused_prob - 0.25), min(1.0, fused_prob + 0.25)
        
        # Collect CI bounds and probabilities
        ci_lows = [r.ci_low for r in results]
        ci_highs = [r.ci_high for r in results]
        probs = [r.probability for r in results]
        
        # Pool bounds
        pooled_low = min(ci_lows)
        pooled_high = max(ci_highs)
        
        # Widen by inter-modality disagreement
        if len(results) > 1:
            prob_std = float(np.std(probs))
            pooled_low = max(0.0, min(pooled_low, fused_prob - prob_std))
            pooled_high = min(1.0, max(pooled_high, fused_prob + prob_std))
        
        return float(pooled_low), float(pooled_high)
    
    def _merge_shap_features(self, results: List[ModalityResult], 
                            top_n: int = 15) -> List[SHAPFeature]:
        """
        Merge top SHAP features from all modalities.
        
        Creates modality-prefixed feature names to avoid collisions:
        "speech::pitch_mean", "tapping::tap_duration_mean", etc.
        
        Args:
            results: List of ModalityResults
            top_n: Number of top features to return
        
        Returns:
            List of top SHAPFeature sorted by absolute value
        """
        feature_map: Dict[str, float] = {}
        
        for result in results:
            modality = result.modality
            for shap_feat in result.shap_features:
                # Prefix feature name with modality
                prefixed_name = f"{modality}::{shap_feat.name}"
                # Accumulate SHAP values (in case same feature appears in multiple modalities)
                feature_map[prefixed_name] = feature_map.get(prefixed_name, 0.0) + shap_feat.value
        
        # Rank by absolute value
        ranked = sorted(
            feature_map.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Create SHAPFeature objects for top N
        shap_features = [
            SHAPFeature(
                name=name.split("::")[-1] + f" ({name.split('::')[0]})",  # Add modality in display
                value=float(value),
                rank=rank + 1
            )
            for rank, (name, value) in enumerate(ranked[:top_n])
        ]
        
        return shap_features
    
    def _classify_risk(self, probability: float) -> str:
        """
        Classify risk level based on probability.

        Returns:
            Risk label: "Positive" (P(PD) >= 0.5) or "Negative"
        """
        return "Positive" if probability >= 0.5 else "Negative"
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update fusion weights (e.g., after new validation results).
        
        Args:
            new_weights: Dict of modality → weight
        """
        total = sum(new_weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to positive value")
        
        self.weights = {k: v / total for k, v in new_weights.items()}
        logger.info(f"LateFusionModel weights updated: {self.weights}")
 
 
# Singleton instance for use throughout backend
late_fusion_model = LateFusionModel()