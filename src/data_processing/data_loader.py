"""
Data loader module for e-learning session analysis.
Handles loading and preprocessing of session data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionDataLoader:
    """Loads and preprocesses e-learning session data."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the data file or directory
        """
        self.data_path = Path(data_path)
        self.data = None
        self.features = None
        self.labels = None
        
    def load_data(self, file_format: str = 'csv') -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            file_format: Format of the data file ('csv', 'json', 'excel')
            
        Returns:
            Loaded data as pandas DataFrame
        """
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
                
            if file_format.lower() == 'csv':
                self.data = pd.read_csv(self.data_path)
            elif file_format.lower() == 'json':
                self.data = pd.read_json(self.data_path)
            elif file_format.lower() == 'excel':
                self.data = pd.read_excel(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
            logger.info(f"Successfully loaded data with shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self) -> bool:
        """
        Validate the loaded data for required columns and data types.
        
        Returns:
            True if data is valid, False otherwise
        """
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            return False
            
        required_columns = [
            'session_id', 'user_id', 'duration', 'interactions',
            'completion_rate', 'score', 'timestamp'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Check for data types
        if not pd.api.types.is_numeric_dtype(self.data['duration']):
            logger.error("Duration column must be numeric")
            return False
            
        if not pd.api.types.is_numeric_dtype(self.data['score']):
            logger.error("Score column must be numeric")
            return False
            
        # Check for missing values
        missing_values = self.data[required_columns].isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Missing values found: {missing_values.to_dict()}")
            
        logger.info("Data validation completed successfully")
        return True
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the data for machine learning.
        
        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Create a copy to avoid modifying original data
        processed_data = self.data.copy()
        
        # Handle missing values
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        processed_data[numeric_columns] = processed_data[numeric_columns].fillna(
            processed_data[numeric_columns].median()
        )
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in processed_data.columns:
            processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'])
            
        # Create additional features
        processed_data['hour_of_day'] = processed_data['timestamp'].dt.hour
        processed_data['day_of_week'] = processed_data['timestamp'].dt.dayofweek
        
        # Calculate session efficiency only if both score and duration exist
        if 'score' in processed_data.columns and 'duration' in processed_data.columns:
            processed_data['efficiency'] = processed_data['score'] / processed_data['duration']
            
            # Remove outliers using multiple approaches
            outliers_removed = 0
            
            # 1. Remove individual column outliers first (values > 3 standard deviations from mean)
            for col in ['score', 'duration']:
                if col in processed_data.columns:
                    col_mean = processed_data[col].mean()
                    col_std = processed_data[col].std()
                    col_outlier_mask = abs(processed_data[col] - col_mean) > 3 * col_std
                    outliers_removed += col_outlier_mask.sum()
                    processed_data = processed_data[~col_outlier_mask]
            
            # 2. Remove efficiency outliers (sessions with efficiency > 2 standard deviations from mean)
            # Recalculate efficiency after column outlier removal
            processed_data['efficiency'] = processed_data['score'] / processed_data['duration']
            efficiency_mean = processed_data['efficiency'].mean()
            efficiency_std = processed_data['efficiency'].std()
            efficiency_outlier_mask = abs(processed_data['efficiency'] - efficiency_mean) > 2 * efficiency_std
            outliers_removed += efficiency_outlier_mask.sum()
            
            # Apply efficiency outlier removal
            processed_data = processed_data[~efficiency_outlier_mask]
            
            logger.info(f"Removed {outliers_removed} outliers during preprocessing")
        
        logger.info(f"Preprocessed data shape: {processed_data.shape}")
        return processed_data
    
    def extract_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from the data.
        
        Args:
            data: Preprocessed DataFrame
            
        Returns:
            Tuple of (features, labels)
        """
        feature_columns = [
            'duration', 'interactions', 'completion_rate', 'score',
            'hour_of_day', 'day_of_week', 'efficiency'
        ]
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_columns if col in data.columns]
        if len(available_features) < 4:  # Minimum required features
            raise ValueError(f"Insufficient features available: {available_features}")
            
        self.features = data[available_features].values
        
        # Create binary labels: 1 for beneficial sessions (score > median), 0 otherwise
        score_median = data['score'].median()
        self.labels = (data['score'] > score_median).astype(int).values
        
        logger.info(f"Extracted {self.features.shape[1]} features and {len(self.labels)} labels")
        return self.features, self.labels
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded data.
        
        Returns:
            Dictionary containing data summary statistics
        """
        if self.data is None:
            return {"error": "No data loaded"}
            
        # Convert timestamp to datetime for proper date operations
        timestamp_col = pd.to_datetime(self.data['timestamp'])
        
        summary = {
            "total_sessions": len(self.data),
            "unique_users": self.data['user_id'].nunique(),
            "date_range": {
                "start": timestamp_col.min().strftime('%Y-%m-%d'),
                "end": timestamp_col.max().strftime('%Y-%m-%d')
            },
            "score_stats": {
                "mean": self.data['score'].mean(),
                "median": self.data['score'].median(),
                "std": self.data['score'].std()
            },
            "duration_stats": {
                "mean": self.data['duration'].mean(),
                "median": self.data['duration'].median(),
                "std": self.data['duration'].std()
            }
        }
        
        return summary