"""
REM Model Inference Module
Provides inference functions for trained REM ensemble model
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Union, Tuple, Optional, Dict, List
import json
import warnings

warnings.filterwarnings('ignore')


class REMInference:
    """
    Inference engine for REM Sleep Behavior Disorder classification
    Loads trained ensemble model and provides prediction methods
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 default_model: str = 'output/rem_ensemble.pkl'):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained ensemble model (pkl file)
            default_model: Default model path if model_path not provided
        """
        self.model_path = model_path or default_model
        self.ensemble = None
        self.feature_names = None
        self.label_map = {0: 'No REM Disorder', 1: 'REM Disorder', 2: 'Uncertain'}
        
        # Load model if path exists
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print(f"Warning: Model not found at {self.model_path}")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load trained ensemble model
        
        Args:
            model_path: Path to model file
        """
        path = model_path or self.model_path
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        
        try:
            self.ensemble = joblib.load(path)
            self.feature_names = self.ensemble.feature_names
            print(f"✓ Model loaded from {path}")
            print(f"  Models in ensemble: {list(self.ensemble.models.keys())}")
            print(f"  Fusion method: {self.ensemble.fusion_method}")
            print(f"  Feature count: {len(self.feature_names) if self.feature_names else 'Unknown'}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], 
                return_proba: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on input data
        
        Args:
            X: Input features (DataFrame or array)
            return_proba: Whether to return probabilities
        
        Returns:
            Predictions or (predictions, probabilities) tuple
        """
        if self.ensemble is None:
            raise RuntimeError("No model loaded. Call load_model() first")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Validate features
        self._validate_features(X)
        
        # Get predictions
        predictions = self.ensemble.predict(X)
        
        if return_proba:
            probabilities = self.ensemble.predict_proba(X)
            return predictions, probabilities
        else:
            return predictions
    
    def predict_with_confidence(self, X: Union[pd.DataFrame, np.ndarray],
                                threshold: float = 0.7) -> pd.DataFrame:
        """
        Make predictions with confidence scores
        
        Args:
            X: Input features
            threshold: Confidence threshold for decisions
        
        Returns:
            DataFrame with predictions and confidence
        """
        predictions, probabilities = self.predict(X, return_proba=True)
        
        # Calculate confidence as max probability
        confidence = np.max(probabilities, axis=1)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': predictions,
            'prediction_label': [self.label_map.get(p, 'Unknown') for p in predictions],
            'confidence': confidence,
            'high_confidence': confidence >= threshold
        })
        
        # Add class probabilities
        for class_idx, label in self.label_map.items():
            if class_idx < probabilities.shape[1]:
                results[f'prob_{label}'] = probabilities[:, class_idx]
        
        return results
    
    def predict_batch(self, X_list: List[Union[pd.DataFrame, np.ndarray]],
                      return_proba: bool = True) -> List:
        """
        Make predictions on multiple batches
        
        Args:
            X_list: List of input features
            return_proba: Whether to return probabilities
        
        Returns:
            List of predictions (or tuples if return_proba=True)
        """
        results = []
        for X in X_list:
            pred = self.predict(X, return_proba=return_proba)
            results.append(pred)
        
        return results
    
    def predict_single(self, X_single: Union[dict, list, np.ndarray, pd.Series],
                       return_proba: bool = True) -> Union[int, Tuple[int, np.ndarray]]:
        """
        Make prediction on a single sample
        
        Args:
            X_single: Single sample (dict, list, array, or Series)
            return_proba: Whether to return probabilities
        
        Returns:
            Prediction class or (class, probabilities) tuple
        """
        # Convert to DataFrame
        if isinstance(X_single, dict):
            X_df = pd.DataFrame([X_single])
        elif isinstance(X_single, (list, np.ndarray)):
            X_df = pd.DataFrame([X_single], columns=self.feature_names)
        elif isinstance(X_single, pd.Series):
            X_df = X_single.to_frame().T
        else:
            raise ValueError("Unsupported input type")
        
        predictions, probabilities = self.predict(X_df, return_proba=True)
        
        if return_proba:
            return int(predictions[0]), probabilities[0]
        else:
            return int(predictions[0])
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get aggregated feature importance from ensemble
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            Dictionary of feature importance scores
        """
        if self.ensemble is None:
            raise RuntimeError("No model loaded")
        
        importance = self.ensemble.get_aggregated_feature_importance()
        
        # Return top N features
        return dict(list(importance.items())[:top_n])
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model metadata
        """
        if self.ensemble is None:
            return {'status': 'No model loaded'}
        
        info = {
            'status': 'Model loaded',
            'fusion_method': self.ensemble.fusion_method,
            'models': list(self.ensemble.models.keys()),
            'n_models': len(self.ensemble.models),
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'features': self.feature_names[:10] if self.feature_names else None  # First 10
        }
        
        return info
    
    def _validate_features(self, X: pd.DataFrame) -> None:
        """
        Validate input features match expected features
        
        Args:
            X: Input DataFrame
        """
        if self.feature_names is None:
            print("Warning: Feature names not available for validation")
            return
        
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        extra_features = set(X.columns) - set(self.feature_names)
        if extra_features:
            print(f"Warning: Extra features will be ignored: {extra_features}")
        
        # Select only required features in correct order
        X = X[self.feature_names]


# Convenience functions for simple usage
def load_rem_model(model_path: str) -> REMInference:
    """
    Load REM model for inference
    
    Args:
        model_path: Path to trained model pkl file
    
    Returns:
        REMInference instance
    """
    return REMInference(model_path=model_path)


def predict_rem(X: Union[pd.DataFrame, np.ndarray], 
                model_path: str = 'output/rem_ensemble.pkl') -> np.ndarray:
    """
    Make REM predictions using default model
    
    Args:
        X: Input features
        model_path: Path to model
    
    Returns:
        Predictions array
    """
    engine = REMInference(model_path=model_path)
    return engine.predict(X, return_proba=False)


def predict_rem_proba(X: Union[pd.DataFrame, np.ndarray], 
                      model_path: str = 'output/rem_ensemble.pkl') -> np.ndarray:
    """
    Make REM predictions with probabilities
    
    Args:
        X: Input features
        model_path: Path to model
    
    Returns:
        Predictions and probabilities
    """
    engine = REMInference(model_path=model_path)
    predictions, probabilities = engine.predict(X, return_proba=True)
    return predictions, probabilities


# Example usage
if __name__ == "__main__":
    # Example 1: Basic inference
    print("="*60)
    print("REM Model Inference Example")
    print("="*60)
    
    # Initialize inference engine
    engine = REMInference(model_path='output/rem_ensemble.pkl')
    
    # Get model info
    print("\nModel Information:")
    info = engine.get_model_info()
    for key, value in info.items():
        if key != 'features':
            print(f"  {key}: {value}")
    
    # Get feature importance
    print("\nTop 10 Features by Importance:")
    importance = engine.get_feature_importance(top_n=10)
    for idx, (feat, imp) in enumerate(importance.items(), 1):
        print(f"  {idx}. {feat}: {imp:.4f}")
    
    # Example 2: Single prediction
    print("\n" + "="*60)
    print("Single Sample Prediction Example")
    print("="*60)
    
    # Create sample data (replace with actual features)
    # This is a placeholder - use real features from your dataset
    sample_features = {
        'feature_1': 1.0,
        'feature_2': 0.5,
        # ... add all required features
    }
    
    try:
        pred, proba = engine.predict_single(sample_features, return_proba=True)
        print(f"\nPrediction: {engine.label_map.get(pred, 'Unknown')}")
        print(f"Confidence: {np.max(proba):.4f}")
        print(f"Probabilities: {proba}")
    except Exception as e:
        print(f"Note: {e}")
