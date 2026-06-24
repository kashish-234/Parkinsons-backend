"""
REM Data Canonicalization Module
Standardizes and validates input data for inference
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple, Optional, List
import json
import warnings

warnings.filterwarnings('ignore')


class REMCanonicalizer:
    """
    Canonicalizes and validates REM input data
    Handles data type conversions, missing values, scaling, and normalization
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None,
                 scaler_stats: Optional[Dict] = None):
        """
        Initialize canonicalizer
        
        Args:
            feature_names: Expected feature names
            scaler_stats: Min/max or mean/std for scaling
        """
        self.feature_names = feature_names
        self.scaler_stats = scaler_stats or {}
        self.validation_rules = {}
    
    def canonicalize(self, X: Union[pd.DataFrame, dict, list, np.ndarray]) -> pd.DataFrame:
        """
        Canonicalize input data to standard format
        
        Args:
            X: Input data (various formats)
        
        Returns:
            Standardized DataFrame
        """
        # Convert to DataFrame
        if isinstance(X, dict):
            X_df = pd.DataFrame([X])
        elif isinstance(X, list):
            X_df = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
        elif isinstance(X, pd.Series):
            X_df = X.to_frame().T
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")
        
        # Reset index
        X_df = X_df.reset_index(drop=True)
        
        return X_df
    
    def handle_missing_values(self, X: pd.DataFrame, 
                             method: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in data
        
        Args:
            X: Input DataFrame
            method: 'mean', 'median', 'forward_fill', 'drop'
        
        Returns:
            DataFrame with missing values handled
        """
        X_clean = X.copy()
        
        missing_count = X_clean.isnull().sum()
        if missing_count.sum() == 0:
            print("✓ No missing values found")
            return X_clean
        
        print(f"Handling {missing_count.sum()} missing values...")
        
        if method == 'mean':
            X_clean = X_clean.fillna(X_clean.mean())
        elif method == 'median':
            X_clean = X_clean.fillna(X_clean.median())
        elif method == 'forward_fill':
            X_clean = X_clean.fillna(method='ffill').fillna(method='bfill')
        elif method == 'drop':
            X_clean = X_clean.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"✓ Missing values handled using {method} method")
        
        return X_clean
    
    def validate_ranges(self, X: pd.DataFrame, 
                       validation_rules: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate data is within expected ranges
        
        Args:
            X: Input DataFrame
            validation_rules: Dict of feature -> (min, max) tuples
        
        Returns:
            Tuple of (validated_data, violations_summary)
        """
        rules = validation_rules or self.validation_rules
        violations = {}
        X_clean = X.copy()
        
        for feature, (min_val, max_val) in rules.items():
            if feature in X_clean.columns:
                mask = (X_clean[feature] < min_val) | (X_clean[feature] > max_val)
                if mask.any():
                    violations[feature] = mask.sum()
                    # Clip to valid range
                    X_clean[feature] = X_clean[feature].clip(min_val, max_val)
        
        if violations:
            print(f"Warning: {len(violations)} feature(s) had out-of-range values:")
            for feat, count in violations.items():
                print(f"  - {feat}: {count} samples clipped")
        else:
            print("✓ All features within valid ranges")
        
        return X_clean, violations
    
    def normalize(self, X: pd.DataFrame, 
                 method: str = 'standard') -> pd.DataFrame:
        """
        Normalize features
        
        Args:
            X: Input DataFrame
            method: 'standard' (z-score) or 'minmax' (0-1 scaling)
        
        Returns:
            Normalized DataFrame
        """
        X_norm = X.copy()
        
        if method == 'standard':
            mean = X_norm.mean()
            std = X_norm.std()
            X_norm = (X_norm - mean) / (std + 1e-8)  # Avoid division by zero
            print("✓ Features normalized using standard scaling (z-score)")
        
        elif method == 'minmax':
            min_vals = X_norm.min()
            max_vals = X_norm.max()
            X_norm = (X_norm - min_vals) / (max_vals - min_vals + 1e-8)
            print("✓ Features normalized using min-max scaling (0-1)")
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return X_norm
    
    def select_features(self, X: pd.DataFrame, 
                       features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Select specific features from DataFrame
        
        Args:
            X: Input DataFrame
            features: List of feature names to select
        
        Returns:
            DataFrame with selected features only
        """
        target_features = features or self.feature_names
        
        if target_features is None:
            return X
        
        missing = set(target_features) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features in data: {missing}")
        
        return X[target_features]
    
    def deduplicate(self, X: pd.DataFrame, 
                   subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            X: Input DataFrame
            subset: Columns to consider for deduplication
        
        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(X)
        X_dedup = X.drop_duplicates(subset=subset)
        removed = initial_count - len(X_dedup)
        
        if removed > 0:
            print(f"✓ Removed {removed} duplicate rows")
        else:
            print("✓ No duplicates found")
        
        return X_dedup
    
    def detect_outliers(self, X: pd.DataFrame, 
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Dict[str, List[int]]:
        """
        Detect outliers in data
        
        Args:
            X: Input DataFrame
            method: 'iqr' (interquartile range) or 'zscore'
            threshold: Threshold for outlier detection
        
        Returns:
            Dictionary mapping feature names to list of outlier row indices
        """
        outliers = {}
        
        if method == 'iqr':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            for col in X.columns:
                outlier_mask = (X[col] < lower_bound[col]) | (X[col] > upper_bound[col])
                if outlier_mask.any():
                    outliers[col] = X.index[outlier_mask].tolist()
        
        elif method == 'zscore':
            for col in X.columns:
                z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
                outlier_mask = z_scores > threshold
                if outlier_mask.any():
                    outliers[col] = X.index[outlier_mask].tolist()
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        if outliers:
            print(f"Detected outliers in {len(outliers)} feature(s):")
            for feat, indices in outliers.items():
                print(f"  - {feat}: {len(indices)} outlier(s) at rows {indices[:5]}")
        else:
            print("✓ No outliers detected")
        
        return outliers
    
    def preprocess_pipeline(self, X: Union[pd.DataFrame, dict, list, np.ndarray],
                           handle_missing: bool = True,
                           remove_duplicates: bool = True,
                           normalize: bool = False,
                           select_features: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            X: Input data
            handle_missing: Whether to handle missing values
            remove_duplicates: Whether to remove duplicates
            normalize: Whether to normalize features
            select_features: Whether to select specific features
        
        Returns:
            Preprocessed DataFrame
        """
        print("\n" + "="*60)
        print("REM DATA PREPROCESSING PIPELINE")
        print("="*60 + "\n")
        
        # Step 1: Canonicalize
        print("Step 1: Canonicalizing input data...")
        X_canonical = self.canonicalize(X)
        print(f"✓ Data shape: {X_canonical.shape}")
        
        # Step 2: Remove duplicates
        if remove_duplicates:
            print("\nStep 2: Removing duplicates...")
            X_canonical = self.deduplicate(X_canonical)
        
        # Step 3: Handle missing values
        if handle_missing:
            print("\nStep 3: Handling missing values...")
            X_canonical = self.handle_missing_values(X_canonical)
        
        # Step 4: Select features
        if select_features and self.feature_names:
            print("\nStep 4: Selecting features...")
            X_canonical = self.select_features(X_canonical)
        
        # Step 5: Normalize
        if normalize:
            print("\nStep 5: Normalizing features...")
            X_canonical = self.normalize(X_canonical)
        
        # Step 6: Detect outliers (informational only)
        print("\nStep 6: Detecting outliers...")
        self.detect_outliers(X_canonical)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60 + "\n")
        
        return X_canonical
    
    def save_config(self, filepath: str) -> None:
        """Save canonicalizer configuration"""
        config = {
            'feature_names': self.feature_names,
            'scaler_stats': self.scaler_stats,
            'validation_rules': {k: [v[0], v[1]] for k, v in self.validation_rules.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Configuration saved to {filepath}")
    
    def load_config(self, filepath: str) -> None:
        """Load canonicalizer configuration"""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.feature_names = config.get('feature_names')
        self.scaler_stats = config.get('scaler_stats', {})
        
        rules = config.get('validation_rules', {})
        self.validation_rules = {k: tuple(v) for k, v in rules.items()}
        
        print(f"✓ Configuration loaded from {filepath}")


# Convenience functions
def canonicalize_rem_data(X: Union[pd.DataFrame, dict, list, np.ndarray],
                          handle_missing: bool = True,
                          remove_duplicates: bool = True,
                          normalize: bool = False) -> pd.DataFrame:
    """
    Quick canonicalization of REM data
    
    Args:
        X: Input data
        handle_missing: Whether to handle missing values
        remove_duplicates: Whether to remove duplicates
        normalize: Whether to normalize features
    
    Returns:
        Preprocessed DataFrame
    """
    canonicalizer = REMCanonicalizer()
    return canonicalizer.preprocess_pipeline(
        X,
        handle_missing=handle_missing,
        remove_duplicates=remove_duplicates,
        normalize=normalize,
        select_features=False
    )


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("REM Canonicalization Example")
    print("="*60)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature_1': [1.0, 2.0, np.nan, 4.0, 1.0],
        'feature_2': [0.5, 0.6, 0.7, 0.8, 0.5],
        'feature_3': [100, 200, 300, 400, 100]
    })
    
    # Initialize canonicalizer
    canonicalizer = REMCanonicalizer(
        feature_names=['feature_1', 'feature_2', 'feature_3']
    )
    
    # Run preprocessing pipeline
    processed_data = canonicalizer.preprocess_pipeline(
        sample_data,
        handle_missing=True,
        remove_duplicates=True,
        normalize=True
    )
    
    print("Processed data:")
    print(processed_data)
