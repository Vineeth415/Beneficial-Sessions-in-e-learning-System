"""
Machine learning model module for e-learning session analysis.
Handles model training, evaluation, and prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class SessionClassifier:
    """Machine learning classifier for identifying beneficial e-learning sessions."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'random_state': 42,
                    'learning_rate': 0.1
                }
            },
            'logistic': {
                'class': LogisticRegression,
                'params': {
                    'random_state': 42,
                    'max_iter': 1000,
                    'C': 1.0
                }
            }
        }
        
    def _initialize_model(self):
        """Initialize the model based on the specified type."""
        if self.model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        config = self.model_configs[self.model_type]
        self.model = config['class'](**config['params'])
        logger.info(f"Initialized {self.model_type} model")
        
    def prepare_features(self, features: np.ndarray, feature_names: Optional[list] = None) -> np.ndarray:
        """
        Prepare features for training/prediction.
        
        Args:
            features: Feature array
            feature_names: Names of the features
            
        Returns:
            Prepared feature array
        """
        if feature_names is not None:
            self.feature_names = feature_names
            
        # Handle NaN values
        if np.isnan(features).any():
            logger.warning("NaN values found in features. Replacing with median.")
            features = np.nan_to_num(features, nan=np.nanmedian(features))
            
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled
        
    def train(self, features: np.ndarray, labels: np.ndarray, 
              test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            features: Training features
            labels: Training labels
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing training results
        """
        if self.model is None:
            self._initialize_model()
            
        # Prepare features
        features_scaled = self.prepare_features(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        
        # Train model
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, features_scaled, labels, cv=5, scoring='accuracy')
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names or range(len(self.model.feature_importances_)), 
                                        self.model.feature_importances_))
        
        self.is_trained = True
        
        results = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': feature_importance
        }
        
        logger.info(f"Training completed. Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
        return results
        
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            features: Feature array for prediction
            
        Returns:
            Tuple of (predictions, prediction_probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Prepare features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        
        return predictions, probabilities
        
    def hyperparameter_tuning(self, features: np.ndarray, labels: np.ndarray, 
                            param_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            features: Training features
            labels: Training labels
            param_grid: Parameter grid for tuning
            
        Returns:
            Dictionary containing tuning results
        """
        if self.model is None:
            self._initialize_model()
            
        # Default parameter grids
        if param_grid is None:
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            elif self.model_type == 'logistic':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2']
                }
        
        # Prepare features
        features_scaled = self.prepare_features(features)
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(features_scaled, labels)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Hyperparameter tuning completed. Best score: {grid_search.best_score_:.4f}")
        return results
        
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }
        
        if self.is_trained and hasattr(self.model, 'feature_importances_'):
            info['feature_importance'] = dict(zip(
                self.feature_names or range(len(self.model.feature_importances_)),
                self.model.feature_importances_
            ))
            
        return info