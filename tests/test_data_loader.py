"""
Unit tests for the data loader module.
Tests data loading, validation, and preprocessing functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_processing.data_loader import SessionDataLoader


class TestSessionDataLoader:
    """Test cases for SessionDataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'session_id': [f'session_{i}' for i in range(1, 11)],
            'user_id': [f'user_{i % 3 + 1}' for i in range(10)],
            'duration': [30, 45, 60, 20, 90, 35, 50, 25, 70, 40],
            'interactions': [10, 15, 20, 8, 25, 12, 18, 9, 22, 14],
            'completion_rate': [0.8, 0.9, 0.7, 0.6, 0.95, 0.85, 0.75, 0.5, 0.88, 0.82],
            'score': [75, 85, 70, 60, 90, 80, 72, 55, 87, 78],
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='h')
        })
        
        # Create temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_initialization(self):
        """Test SessionDataLoader initialization."""
        loader = SessionDataLoader("test_path.csv")
        assert loader.data_path == Path("test_path.csv")
        assert loader.data is None
        assert loader.features is None
        assert loader.labels is None
    
    def test_load_data_csv(self):
        """Test loading CSV data."""
        loader = SessionDataLoader(self.temp_file.name)
        data = loader.load_data(file_format='csv')
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 10
        assert list(data.columns) == list(self.sample_data.columns)
        assert loader.data is not None
    
    def test_load_data_nonexistent_file(self):
        """Test loading data from non-existent file."""
        loader = SessionDataLoader("nonexistent_file.csv")
        
        with pytest.raises(FileNotFoundError):
            loader.load_data()
    
    def test_load_data_unsupported_format(self):
        """Test loading data with unsupported format."""
        loader = SessionDataLoader(self.temp_file.name)
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load_data(file_format='txt')
    
    def test_validate_data_success(self):
        """Test successful data validation."""
        loader = SessionDataLoader(self.temp_file.name)
        loader.load_data()
        
        result = loader.validate_data()
        assert result is True
    
    def test_validate_data_no_data_loaded(self):
        """Test validation when no data is loaded."""
        loader = SessionDataLoader(self.temp_file.name)
        
        result = loader.validate_data()
        assert result is False
    
    def test_validate_data_missing_columns(self):
        """Test validation with missing required columns."""
        # Create data with missing columns
        incomplete_data = self.sample_data.drop(columns=['session_id', 'score'])
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        incomplete_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        loader = SessionDataLoader(temp_file.name)
        loader.load_data()
        
        result = loader.validate_data()
        assert result is False
        
        # Clean up
        os.unlink(temp_file.name)
    
    def test_validate_data_wrong_data_types(self):
        """Test validation with wrong data types."""
        # Create data with wrong data types
        wrong_type_data = self.sample_data.copy()
        wrong_type_data['duration'] = wrong_type_data['duration'].astype(str)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        wrong_type_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        loader = SessionDataLoader(temp_file.name)
        loader.load_data()
        
        # Directly modify the loaded data to have wrong types
        # This bypasses the CSV save/load issue where pandas auto-converts
        loader.data['duration'] = loader.data['duration'].astype(str)
        
        result = loader.validate_data()
        assert result is False
        
        # Clean up
        os.unlink(temp_file.name)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        loader = SessionDataLoader(self.temp_file.name)
        loader.load_data()
        
        processed_data = loader.preprocess_data()
        
        assert isinstance(processed_data, pd.DataFrame)
        assert 'hour_of_day' in processed_data.columns
        assert 'day_of_week' in processed_data.columns
        assert 'efficiency' in processed_data.columns
        assert len(processed_data) <= len(self.sample_data)  # May remove outliers
    
    def test_preprocess_data_no_data_loaded(self):
        """Test preprocessing when no data is loaded."""
        loader = SessionDataLoader(self.temp_file.name)
        
        with pytest.raises(ValueError, match="No data loaded"):
            loader.preprocess_data()
    
    def test_extract_features(self):
        """Test feature extraction."""
        loader = SessionDataLoader(self.temp_file.name)
        loader.load_data()
        processed_data = loader.preprocess_data()
        
        features, labels = loader.extract_features(processed_data)
        
        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert len(features) == len(labels)
        assert features.shape[1] >= 4  # At least 4 features
        assert set(labels).issubset({0, 1})  # Binary labels
    
    def test_extract_features_insufficient_features(self):
        """Test feature extraction with insufficient features."""
        # Create data with very few features
        minimal_data = self.sample_data[['session_id', 'user_id', 'duration']].copy()
        minimal_data['timestamp'] = pd.date_range('2024-01-01', periods=len(minimal_data), freq='h')
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        minimal_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        loader = SessionDataLoader(temp_file.name)
        loader.load_data()
        processed_data = loader.preprocess_data()
        
        with pytest.raises(ValueError, match="Insufficient features"):
            loader.extract_features(processed_data)
        
        # Clean up
        os.unlink(temp_file.name)
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        loader = SessionDataLoader(self.temp_file.name)
        loader.load_data()
        
        summary = loader.get_data_summary()
        
        assert isinstance(summary, dict)
        assert 'total_sessions' in summary
        assert 'unique_users' in summary
        assert 'date_range' in summary
        assert summary['total_sessions'] == 10
        assert summary['unique_users'] == 3
    
    def test_get_data_summary_no_data(self):
        """Test data summary when no data is loaded."""
        loader = SessionDataLoader(self.temp_file.name)
        
        summary = loader.get_data_summary()
        assert summary == {"error": "No data loaded"}
    
    def test_handle_missing_values(self):
        """Test handling of missing values in preprocessing."""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'score'] = np.nan
        data_with_missing.loc[1, 'duration'] = np.nan
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        data_with_missing.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        loader = SessionDataLoader(temp_file.name)
        loader.load_data()
        
        # Should not raise an error
        processed_data = loader.preprocess_data()
        assert len(processed_data) > 0
        
        # Clean up
        os.unlink(temp_file.name)
    
    def test_outlier_removal(self):
        """Test outlier removal in preprocessing."""
        # Create data with extreme outliers
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[0, 'score'] = 1000  # Extreme outlier
        data_with_outliers.loc[1, 'duration'] = 1000  # Extreme outlier
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        data_with_outliers.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        loader = SessionDataLoader(temp_file.name)
        loader.load_data()
        
        processed_data = loader.preprocess_data()
        
        # Should remove outliers
        assert len(processed_data) < len(data_with_outliers)
        
        # Clean up
        os.unlink(temp_file.name)
    
    def test_efficiency_calculation_edge_cases(self):
        """Test efficiency calculation with edge cases."""
        # Test with zero duration
        edge_case_data = self.sample_data.copy()
        edge_case_data.loc[0, 'duration'] = 0
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        edge_case_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        loader = SessionDataLoader(temp_file.name)
        loader.load_data()
        
        # Should handle zero duration gracefully
        processed_data = loader.preprocess_data()
        assert 'efficiency' in processed_data.columns
        
        # Clean up
        os.unlink(temp_file.name)


if __name__ == "__main__":
    pytest.main([__file__])