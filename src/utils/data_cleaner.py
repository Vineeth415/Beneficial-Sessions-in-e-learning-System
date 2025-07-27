"""
Data cleaner module for e-learning session analysis.
Handles duplicate removal, data validation, and cleaning operations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class SessionDataCleaner:
    """Cleans and validates e-learning session data."""
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.cleaning_report = {}
        
    def remove_duplicates(self, data: pd.DataFrame, 
                         subset: Optional[List[str]] = None,
                         keep: str = 'first') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove duplicate records from the dataset.
        
        Args:
            data: DataFrame to clean
            subset: Columns to consider for duplicate detection
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            Tuple of (cleaned_data, cleaning_stats)
        """
        try:
            initial_count = len(data)
            
            # Remove duplicates
            cleaned_data = data.drop_duplicates(subset=subset, keep=keep)
            
            removed_count = initial_count - len(cleaned_data)
            
            stats = {
                'initial_count': initial_count,
                'final_count': len(cleaned_data),
                'removed_count': removed_count,
                'removal_percentage': (removed_count / initial_count) * 100 if initial_count > 0 else 0
            }
            
            self.cleaning_report['duplicate_removal'] = stats
            
            logger.info(f"Removed {removed_count} duplicate records ({stats['removal_percentage']:.2f}%)")
            
            return cleaned_data, stats
            
        except Exception as e:
            logger.error(f"Error removing duplicates: {str(e)}")
            raise
            
    def validate_data_types(self, data: pd.DataFrame, 
                           expected_types: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate and convert data types.
        
        Args:
            data: DataFrame to validate
            expected_types: Dictionary mapping column names to expected types
            
        Returns:
            Tuple of (validated_data, validation_stats)
        """
        try:
            validation_stats = {
                'converted_columns': [],
                'failed_conversions': [],
                'type_errors': []
            }
            
            validated_data = data.copy()
            
            for column, expected_type in expected_types.items():
                if column not in validated_data.columns:
                    validation_stats['type_errors'].append(f"Column '{column}' not found")
                    continue
                    
                try:
                    if expected_type == 'datetime':
                        validated_data[column] = pd.to_datetime(validated_data[column], errors='coerce')
                    elif expected_type == 'numeric':
                        validated_data[column] = pd.to_numeric(validated_data[column], errors='coerce')
                    elif expected_type == 'int':
                        validated_data[column] = pd.to_numeric(validated_data[column], errors='coerce').astype('Int64')
                    elif expected_type == 'float':
                        validated_data[column] = pd.to_numeric(validated_data[column], errors='coerce')
                    elif expected_type == 'string':
                        validated_data[column] = validated_data[column].astype(str)
                        
                    validation_stats['converted_columns'].append(column)
                    
                except Exception as e:
                    validation_stats['failed_conversions'].append(f"Column '{column}': {str(e)}")
                    
            self.cleaning_report['type_validation'] = validation_stats
            
            logger.info(f"Type validation completed. Converted {len(validation_stats['converted_columns'])} columns")
            
            return validated_data, validation_stats
            
        except Exception as e:
            logger.error(f"Error validating data types: {str(e)}")
            raise
            
    def handle_missing_values(self, data: pd.DataFrame, 
                            strategy: str = 'median',
                            threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle missing values in the dataset.
        
        Args:
            data: DataFrame to clean
            strategy: Strategy for handling missing values ('median', 'mean', 'drop', 'forward_fill')
            threshold: Maximum fraction of missing values allowed before dropping column
            
        Returns:
            Tuple of (cleaned_data, missing_value_stats)
        """
        try:
            missing_stats = {
                'initial_missing': data.isnull().sum().to_dict(),
                'columns_dropped': [],
                'values_filled': {},
                'final_missing': {}
            }
            
            cleaned_data = data.copy()
            
            # Check for columns with too many missing values
            missing_fractions = data.isnull().sum() / len(data)
            columns_to_drop = missing_fractions[missing_fractions > threshold].index.tolist()
            
            if columns_to_drop:
                cleaned_data = cleaned_data.drop(columns=columns_to_drop)
                missing_stats['columns_dropped'] = columns_to_drop
                logger.warning(f"Dropped columns with >{threshold*100}% missing values: {columns_to_drop}")
            
            # Handle remaining missing values
            for column in cleaned_data.columns:
                if cleaned_data[column].isnull().any():
                    if strategy == 'drop':
                        cleaned_data = cleaned_data.dropna(subset=[column])
                    elif strategy == 'median':
                        if pd.api.types.is_numeric_dtype(cleaned_data[column]):
                            fill_value = cleaned_data[column].median()
                            cleaned_data[column] = cleaned_data[column].fillna(fill_value)
                            missing_stats['values_filled'][column] = {'strategy': 'median', 'value': fill_value}
                    elif strategy == 'mean':
                        if pd.api.types.is_numeric_dtype(cleaned_data[column]):
                            fill_value = cleaned_data[column].mean()
                            cleaned_data[column] = cleaned_data[column].fillna(fill_value)
                            missing_stats['values_filled'][column] = {'strategy': 'mean', 'value': fill_value}
                    elif strategy == 'forward_fill':
                        cleaned_data[column] = cleaned_data[column].fillna(method='ffill')
                        missing_stats['values_filled'][column] = {'strategy': 'forward_fill'}
            
            missing_stats['final_missing'] = cleaned_data.isnull().sum().to_dict()
            
            self.cleaning_report['missing_value_handling'] = missing_stats
            
            logger.info(f"Missing value handling completed. Filled values in {len(missing_stats['values_filled'])} columns")
            
            return cleaned_data, missing_stats
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
            
    def remove_outliers(self, data: pd.DataFrame, 
                       columns: List[str],
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove outliers from specified columns.
        
        Args:
            data: DataFrame to clean
            columns: Columns to check for outliers
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Tuple of (cleaned_data, outlier_stats)
        """
        try:
            outlier_stats = {
                'outliers_removed': {},
                'total_removed': 0
            }
            
            cleaned_data = data.copy()
            initial_count = len(cleaned_data)
            
            for column in columns:
                if column not in cleaned_data.columns:
                    continue
                    
                if method == 'iqr':
                    Q1 = cleaned_data[column].quantile(0.25)
                    Q3 = cleaned_data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outliers_mask = (cleaned_data[column] < lower_bound) | (cleaned_data[column] > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs((cleaned_data[column] - cleaned_data[column].mean()) / cleaned_data[column].std())
                    outliers_mask = z_scores > threshold
                    
                else:
                    raise ValueError(f"Unsupported outlier detection method: {method}")
                
                outliers_count = outliers_mask.sum()
                if outliers_count > 0:
                    cleaned_data = cleaned_data[~outliers_mask]
                    outlier_stats['outliers_removed'][column] = outliers_count
                    outlier_stats['total_removed'] += outliers_count
                    
            final_count = len(cleaned_data)
            outlier_stats['removal_percentage'] = (outlier_stats['total_removed'] / initial_count) * 100 if initial_count > 0 else 0
            
            self.cleaning_report['outlier_removal'] = outlier_stats
            
            logger.info(f"Removed {outlier_stats['total_removed']} outliers ({outlier_stats['removal_percentage']:.2f}%)")
            
            return cleaned_data, outlier_stats
            
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            raise
            
    def validate_session_data(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate session data for required fields and logical consistency.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        try:
            validation_results = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'checks_passed': []
            }
            
            # Check required columns
            required_columns = ['session_id', 'user_id', 'duration', 'score']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                validation_results['errors'].append(f"Missing required columns: {missing_columns}")
                validation_results['is_valid'] = False
            
            # Check data types
            if 'duration' in data.columns and not pd.api.types.is_numeric_dtype(data['duration']):
                validation_results['errors'].append("Duration column must be numeric")
                validation_results['is_valid'] = False
                
            if 'score' in data.columns and not pd.api.types.is_numeric_dtype(data['score']):
                validation_results['errors'].append("Score column must be numeric")
                validation_results['is_valid'] = False
            
            # Check logical constraints
            if 'duration' in data.columns:
                if (data['duration'] <= 0).any():
                    validation_results['warnings'].append("Found sessions with duration <= 0")
                    
                if (data['duration'] > 480).any():  # More than 8 hours
                    validation_results['warnings'].append("Found sessions with duration > 8 hours")
            
            if 'score' in data.columns:
                if (data['score'] < 0).any() or (data['score'] > 100).any():
                    validation_results['warnings'].append("Found scores outside valid range [0, 100]")
            
            if 'completion_rate' in data.columns:
                if (data['completion_rate'] < 0).any() or (data['completion_rate'] > 1).any():
                    validation_results['warnings'].append("Found completion rates outside valid range [0, 1]")
            
            # Check for duplicate session IDs
            if 'session_id' in data.columns:
                duplicate_sessions = data['session_id'].duplicated().sum()
                if duplicate_sessions > 0:
                    validation_results['warnings'].append(f"Found {duplicate_sessions} duplicate session IDs")
            
            # Check data completeness
            total_sessions = len(data)
            if total_sessions == 0:
                validation_results['errors'].append("No session data found")
                validation_results['is_valid'] = False
            elif total_sessions < 10:
                validation_results['warnings'].append(f"Very few sessions ({total_sessions}) for reliable analysis")
            
            # Record passed checks
            if validation_results['is_valid']:
                validation_results['checks_passed'].append("All required columns present")
                validation_results['checks_passed'].append("Data types are valid")
                validation_results['checks_passed'].append(f"Dataset contains {total_sessions} sessions")
            
            self.cleaning_report['data_validation'] = validation_results
            
            if validation_results['is_valid']:
                logger.info("Data validation passed")
            else:
                logger.warning("Data validation failed")
                
            return validation_results['is_valid'], validation_results
            
        except Exception as e:
            logger.error(f"Error validating session data: {str(e)}")
            raise
            
    def clean_session_data(self, data: pd.DataFrame, 
                          remove_duplicates: bool = True,
                          handle_missing: bool = True,
                          remove_outliers: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive data cleaning pipeline.
        
        Args:
            data: DataFrame to clean
            remove_duplicates: Whether to remove duplicates
            handle_missing: Whether to handle missing values
            remove_outliers: Whether to remove outliers
            
        Returns:
            Tuple of (cleaned_data, cleaning_report)
        """
        try:
            cleaned_data = data.copy()
            cleaning_steps = []
            
            # Step 1: Validate data
            is_valid, validation_results = self.validate_session_data(cleaned_data)
            if not is_valid:
                logger.error("Data validation failed. Cannot proceed with cleaning.")
                return cleaned_data, {'validation_failed': validation_results}
            
            cleaning_steps.append('validation')
            
            # Step 2: Remove duplicates
            if remove_duplicates:
                cleaned_data, duplicate_stats = self.remove_duplicates(cleaned_data)
                cleaning_steps.append('duplicate_removal')
            
            # Step 3: Handle missing values
            if handle_missing:
                cleaned_data, missing_stats = self.handle_missing_values(cleaned_data)
                cleaning_steps.append('missing_value_handling')
            
            # Step 4: Remove outliers
            if remove_outliers:
                numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_columns:
                    cleaned_data, outlier_stats = self.remove_outliers(cleaned_data, numeric_columns)
                    cleaning_steps.append('outlier_removal')
            
            # Create comprehensive cleaning report
            comprehensive_report = {
                'cleaning_steps': cleaning_steps,
                'initial_count': len(data),
                'final_count': len(cleaned_data),
                'removal_percentage': ((len(data) - len(cleaned_data)) / len(data)) * 100 if len(data) > 0 else 0,
                'detailed_reports': self.cleaning_report
            }
            
            logger.info(f"Data cleaning completed. Removed {len(data) - len(cleaned_data)} records")
            
            return cleaned_data, comprehensive_report
            
        except Exception as e:
            logger.error(f"Error in data cleaning pipeline: {str(e)}")
            raise
            
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all cleaning operations performed.
        
        Returns:
            Dictionary containing cleaning summary
        """
        return self.cleaning_report