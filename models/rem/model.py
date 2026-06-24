"""
Consolidated REM Model Module
Combines XGBoost, Random Forest, and ensemble fusion logic
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from typing import Tuple, Dict, Optional, List, Any
from dataclasses import dataclass
import json
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc
        }


class BaseModel:
    """Base class for model wrappers"""
    
    def __init__(self, model_name: str, random_state: int = 42):
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        self.metrics = None
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Evaluate model on test set"""
        if not self.is_trained:
            raise ValueError(f"{self.model_name} model not trained yet")
        
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Handle multi-class AUC
        if len(np.unique(y)) > 2:
            auc = roc_auc_score(y, probabilities, multi_class='ovr', average='weighted')
        else:
            auc = roc_auc_score(y, probabilities[:, 1])
        
        metrics = ModelMetrics(
            accuracy=accuracy_score(y, predictions),
            precision=precision_score(y, predictions, average='weighted', zero_division=0),
            recall=recall_score(y, predictions, average='weighted', zero_division=0),
            f1_score=f1_score(y, predictions, average='weighted', zero_division=0),
            auc_roc=auc
        )
        
        self.metrics = metrics
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        raise NotImplementedError
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        raise NotImplementedError
    
    def save(self, filepath: str) -> None:
        """Save model to disk"""
        joblib.dump(self, filepath)
        print(f"✓ {self.model_name} saved to {filepath}")
    
    @staticmethod
    def load(filepath: str):
        """Load model from disk"""
        model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
        return model


class XGBModel(BaseModel):
    """XGBoost model for REM classification"""
    
    def __init__(self, model_name: str = "XGBoost", random_state: int = 42, **params):
        super().__init__(model_name, random_state)
        
        # Default parameters
        default_params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 1.0,
            'reg_alpha': 0.5,
            'random_state': random_state,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss'
        }
        
        default_params.update(params)
        self.params = default_params
        self.model = xgb.XGBClassifier(**default_params)
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_split: float = 0.2) -> Dict:
        """Train XGBoost model"""
        print(f"Training {self.model_name}...")
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train with early stopping
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        self.is_trained = True
        
        val_score = self.model.score(X_val_scaled, y_val)
        print(f"✓ {self.model_name} training completed")
        print(f"  Validation Accuracy: {val_score:.4f}")
        
        return {'validation_accuracy': val_score}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost"""
        if not self.is_trained:
            return {}
        
        importance_dict = self.model.get_booster().get_score(importance_type='weight')
        
        # Normalize importance scores
        if importance_dict:
            max_importance = max(importance_dict.values())
            return {k: v / max_importance for k, v in importance_dict.items()}
        
        return {}


class RFModel(BaseModel):
    """Random Forest model for REM classification"""
    
    def __init__(self, model_name: str = "RandomForest", random_state: int = 42, **params):
        super().__init__(model_name, random_state)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'random_state': random_state,
            'n_jobs': -1
        }
        
        default_params.update(params)
        self.params = default_params
        self.model = RandomForestClassifier(**default_params)
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_split: float = 0.2) -> Dict:
        """Train Random Forest model"""
        print(f"Training {self.model_name}...")
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        oob_score = self.model.oob_score_ if self.model.oob_score else None
        val_score = self.model.score(X_val_scaled, y_val)
        
        print(f"✓ {self.model_name} training completed")
        if oob_score:
            print(f"  OOB Score: {oob_score:.4f}")
        print(f"  Validation Accuracy: {val_score:.4f}")
        
        return {
            'oob_score': oob_score,
            'validation_accuracy': val_score
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest"""
        if not self.is_trained or not self.feature_names:
            return {}
        
        importances = self.model.feature_importances_
        feature_importance = {
            name: float(imp) for name, imp in zip(self.feature_names, importances)
        }
        
        # Normalize to 0-1 range
        max_imp = max(feature_importance.values()) if feature_importance else 1
        return {k: v / max_imp for k, v in feature_importance.items()}


class REMEnsemble:
    """Fused ensemble combining multiple REM models"""
    
    def __init__(self, fusion_method: str = "voting", 
                 fusion_weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble
        
        Args:
            fusion_method: 'voting', 'weighted_voting', 'averaging', 'weighted_averaging'
            fusion_weights: Weights for weighted fusion methods
        """
        self.fusion_method = fusion_method
        self.fusion_weights = fusion_weights or {}
        self.models: Dict[str, BaseModel] = {}
        self.feature_names = None
        self.is_trained = False
    
    def add_model(self, model_name: str, model: BaseModel) -> None:
        """Add a model to the ensemble"""
        if not model.is_trained:
            raise ValueError(f"Model {model_name} must be trained before adding to ensemble")
        
        self.models[model_name] = model
        
        if self.feature_names is None:
            self.feature_names = model.feature_names
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions"""
        all_predictions = []
        all_probabilities = []
        
        for model_name, model in self.models.items():
            preds = model.predict(X)
            probs = model.predict_proba(X)
            all_predictions.append(preds)
            all_probabilities.append(probs)
        
        if self.fusion_method == "weighted_voting":
            fused_preds = self._weighted_voting(all_predictions)
        elif self.fusion_method == "averaging":
            fused_preds = self._averaging(all_probabilities)
        elif self.fusion_method == "weighted_averaging":
            fused_preds = self._weighted_averaging(all_probabilities)
        else:  # voting
            fused_preds = self._voting(all_predictions)
        
        return fused_preds
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble probabilities"""
        all_probabilities = []
        
        for model_name, model in self.models.items():
            probs = model.predict_proba(X)
            all_probabilities.append(probs)
        
        # Average probabilities
        fused_probs = np.mean(all_probabilities, axis=0)
        return fused_probs
    
    def _voting(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """Majority voting fusion"""
        stacked = np.column_stack(predictions_list)
        fused = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), 
            axis=1, 
            arr=stacked
        )
        return fused
    
    def _weighted_voting(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """Weighted voting fusion"""
        if not self.fusion_weights:
            return self._voting(predictions_list)
        
        n_classes = int(np.max(np.concatenate(predictions_list))) + 1
        n_samples = predictions_list[0].shape[0]
        weighted_votes = np.zeros((n_samples, n_classes))
        
        for idx, (model_name, preds) in enumerate(zip(self.models.keys(), predictions_list)):
            weight = self.fusion_weights.get(model_name, 1.0)
            for i, pred in enumerate(preds):
                weighted_votes[i, int(pred)] += weight
        
        return np.argmax(weighted_votes, axis=1)
    
    def _averaging(self, probabilities_list: List[np.ndarray]) -> np.ndarray:
        """Probability averaging fusion"""
        averaged_probs = np.mean(probabilities_list, axis=0)
        return np.argmax(averaged_probs, axis=1)
    
    def _weighted_averaging(self, probabilities_list: List[np.ndarray]) -> np.ndarray:
        """Weighted probability averaging fusion"""
        if not self.fusion_weights:
            return self._averaging(probabilities_list)
        
        weights = [self.fusion_weights.get(name, 1.0) for name in self.models.keys()]
        weights = np.array(weights) / np.sum(weights)  # Normalize
        
        weighted_probs = np.zeros_like(probabilities_list[0])
        for probs, weight in zip(probabilities_list, weights):
            weighted_probs += weight * probs
        
        return np.argmax(weighted_probs, axis=1)
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> ModelMetrics:
        """Evaluate ensemble performance"""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Handle multi-class AUC
        if len(np.unique(y)) > 2:
            auc = roc_auc_score(y, probabilities, multi_class='ovr', average='weighted')
        else:
            auc = roc_auc_score(y, probabilities[:, 1])
        
        metrics = ModelMetrics(
            accuracy=accuracy_score(y, predictions),
            precision=precision_score(y, predictions, average='weighted', zero_division=0),
            recall=recall_score(y, predictions, average='weighted', zero_division=0),
            f1_score=f1_score(y, predictions, average='weighted', zero_division=0),
            auc_roc=auc
        )
        
        return metrics
    
    def get_aggregated_feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance across all models"""
        all_importances = {}
        n_models = len(self.models)
        
        for model_name, model in self.models.items():
            importance = model.get_feature_importance()
            for feat, imp_value in importance.items():
                if feat not in all_importances:
                    all_importances[feat] = 0
                all_importances[feat] += imp_value / n_models
        
        # Sort by importance
        return dict(sorted(all_importances.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, filepath: str) -> None:
        """Save ensemble to disk"""
        joblib.dump(self, filepath)
        print(f"✓ Ensemble saved to {filepath}")
    
    @staticmethod
    def load(filepath: str):
        """Load ensemble from disk"""
        ensemble = joblib.load(filepath)
        print(f"✓ Ensemble loaded from {filepath}")
        return ensemble
