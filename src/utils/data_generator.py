"""
Data generator module for creating sample e-learning session data.
Used for testing and demonstration purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Optional, Dict, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)


class SessionDataGenerator:
    """Generates synthetic e-learning session data for testing."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_session_data(self, num_sessions: int = 1000, 
                            start_date: str = "2024-01-01",
                            end_date: str = "2024-06-30") -> pd.DataFrame:
        """
        Generate synthetic e-learning session data.
        
        Args:
            num_sessions: Number of sessions to generate
            start_date: Start date for the data
            end_date: End date for the data
            
        Returns:
            DataFrame containing synthetic session data
        """
        try:
            # Generate session IDs
            session_ids = [f"session_{i:06d}" for i in range(1, num_sessions + 1)]
            
            # Generate user IDs (assume 100 unique users)
            num_users = min(100, num_sessions // 10)
            user_ids = [f"user_{i:03d}" for i in range(1, num_users + 1)]
            
            # Generate timestamps
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            date_range = (end_dt - start_dt).days
            
            timestamps = []
            for _ in range(num_sessions):
                random_days = random.randint(0, date_range)
                random_hours = random.randint(0, 23)
                random_minutes = random.randint(0, 59)
                timestamp = start_dt + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
                timestamps.append(timestamp)
            
            # Generate session durations (5-180 minutes, skewed towards shorter sessions)
            durations = np.random.gamma(shape=2, scale=20, size=num_sessions)
            durations = np.clip(durations, 5, 180)
            
            # Generate interaction counts (correlated with duration)
            interactions = np.random.poisson(durations / 3) + np.random.randint(1, 10, num_sessions)
            interactions = np.clip(interactions, 1, 100)
            
            # Generate completion rates (0.1 to 1.0, correlated with duration)
            base_completion = np.random.beta(2, 2, num_sessions)
            duration_factor = np.clip(durations / 60, 0.1, 1.0)
            completion_rates = np.clip(base_completion * duration_factor, 0.1, 1.0)
            
            # Generate scores (0-100, correlated with completion rate and interactions)
            base_scores = np.random.normal(70, 15, num_sessions)
            interaction_bonus = interactions * 0.5
            completion_bonus = completion_rates * 20
            scores = np.clip(base_scores + interaction_bonus + completion_bonus, 0, 100)
            
            # Create DataFrame
            data = pd.DataFrame({
                'session_id': session_ids,
                'user_id': np.random.choice(user_ids, num_sessions),
                'duration': durations.round(2),
                'interactions': interactions,
                'completion_rate': completion_rates.round(3),
                'score': scores.round(2),
                'timestamp': timestamps
            })
            
            # Sort by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Generated {num_sessions} synthetic session records")
            return data
            
        except Exception as e:
            logger.error(f"Error generating session data: {str(e)}")
            raise
            
    def generate_beneficial_sessions(self, data: pd.DataFrame, 
                                   beneficial_ratio: float = 0.3) -> pd.DataFrame:
        """
        Generate additional beneficial sessions with higher scores and engagement.
        
        Args:
            data: Original session data
            beneficial_ratio: Ratio of beneficial sessions to generate
            
        Returns:
            DataFrame with additional beneficial sessions
        """
        try:
            num_beneficial = int(len(data) * beneficial_ratio)
            
            # Create beneficial sessions with higher scores and engagement
            beneficial_data = data.sample(n=num_beneficial, random_state=self.seed).copy()
            
            # Enhance scores for beneficial sessions
            beneficial_data['score'] = beneficial_data['score'] * 1.3 + np.random.normal(5, 3, num_beneficial)
            beneficial_data['score'] = np.clip(beneficial_data['score'], 0, 100)
            
            # Increase interactions
            beneficial_data['interactions'] = beneficial_data['interactions'] * 1.5 + np.random.randint(5, 15, num_beneficial)
            
            # Increase completion rates
            beneficial_data['completion_rate'] = beneficial_data['completion_rate'] * 1.2 + np.random.normal(0.1, 0.05, num_beneficial)
            beneficial_data['completion_rate'] = np.clip(beneficial_data['completion_rate'], 0.1, 1.0)
            
            # Modify session IDs to avoid duplicates
            beneficial_data['session_id'] = [f"beneficial_session_{i:06d}" for i in range(1, num_beneficial + 1)]
            
            # Combine original and beneficial data
            combined_data = pd.concat([data, beneficial_data], ignore_index=True)
            combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Added {num_beneficial} beneficial sessions")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error generating beneficial sessions: {str(e)}")
            raise
            
    def add_noise_to_data(self, data: pd.DataFrame, noise_level: float = 0.05) -> pd.DataFrame:
        """
        Add realistic noise to the data.
        
        Args:
            data: Original session data
            noise_level: Level of noise to add (0-1)
            
        Returns:
            DataFrame with added noise
        """
        try:
            noisy_data = data.copy()
            
            # Add noise to numerical columns
            numerical_cols = ['duration', 'interactions', 'completion_rate', 'score']
            
            for col in numerical_cols:
                if col in noisy_data.columns:
                    noise = np.random.normal(0, noise_level * noisy_data[col].std(), len(noisy_data))
                    noisy_data[col] = noisy_data[col] + noise
                    
                    # Ensure values stay within reasonable bounds
                    if col == 'duration':
                        noisy_data[col] = np.clip(noisy_data[col], 1, 300)
                    elif col == 'interactions':
                        noisy_data[col] = np.clip(noisy_data[col], 1, 200)
                    elif col == 'completion_rate':
                        noisy_data[col] = np.clip(noisy_data[col], 0, 1)
                    elif col == 'score':
                        noisy_data[col] = np.clip(noisy_data[col], 0, 100)
            
            logger.info(f"Added noise with level {noise_level}")
            return noisy_data
            
        except Exception as e:
            logger.error(f"Error adding noise to data: {str(e)}")
            raise
            
    def create_duplicate_sessions(self, data: pd.DataFrame, 
                                duplicate_ratio: float = 0.1) -> pd.DataFrame:
        """
        Create duplicate sessions for testing duplicate removal.
        
        Args:
            data: Original session data
            duplicate_ratio: Ratio of duplicates to create
            
        Returns:
            DataFrame with duplicates
        """
        try:
            num_duplicates = int(len(data) * duplicate_ratio)
            
            # Select random sessions to duplicate
            duplicate_indices = np.random.choice(len(data), num_duplicates, replace=False)
            duplicates = data.iloc[duplicate_indices].copy()
            
            # Modify session IDs to make them appear as duplicates
            duplicates['session_id'] = [f"duplicate_{i:06d}" for i in range(1, num_duplicates + 1)]
            
            # Combine original and duplicate data
            combined_data = pd.concat([data, duplicates], ignore_index=True)
            
            logger.info(f"Added {num_duplicates} duplicate sessions")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error creating duplicate sessions: {str(e)}")
            raise
            
    def save_sample_data(self, filepath: str, num_sessions: int = 1000) -> None:
        """
        Generate and save sample data to file.
        
        Args:
            filepath: Path where to save the data
            num_sessions: Number of sessions to generate
        """
        try:
            # Generate base data
            data = self.generate_session_data(num_sessions)
            
            # Add beneficial sessions
            data = self.generate_beneficial_sessions(data)
            
            # Add noise
            data = self.add_noise_to_data(data)
            
            # Add duplicates
            data = self.create_duplicate_sessions(data)
            
            # Save to file
            if filepath.endswith('.csv'):
                data.to_csv(filepath, index=False)
            elif filepath.endswith('.json'):
                data.to_json(filepath, orient='records', date_format='iso')
            else:
                raise ValueError("Unsupported file format. Use .csv or .json")
                
            logger.info(f"Sample data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving sample data: {str(e)}")
            raise
            
    def get_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the generated data.
        
        Args:
            data: Session data DataFrame
            
        Returns:
            Dictionary containing data statistics
        """
        try:
            stats = {
                'total_sessions': len(data),
                'unique_users': data['user_id'].nunique(),
                'date_range': {
                    'start': data['timestamp'].min().strftime('%Y-%m-%d'),
                    'end': data['timestamp'].max().strftime('%Y-%m-%d')
                },
                'duration_stats': {
                    'mean': data['duration'].mean(),
                    'median': data['duration'].median(),
                    'std': data['duration'].std(),
                    'min': data['duration'].min(),
                    'max': data['duration'].max()
                },
                'score_stats': {
                    'mean': data['score'].mean(),
                    'median': data['score'].median(),
                    'std': data['score'].std(),
                    'min': data['score'].min(),
                    'max': data['score'].max()
                },
                'interaction_stats': {
                    'mean': data['interactions'].mean(),
                    'median': data['interactions'].median(),
                    'std': data['interactions'].std(),
                    'min': data['interactions'].min(),
                    'max': data['interactions'].max()
                },
                'completion_rate_stats': {
                    'mean': data['completion_rate'].mean(),
                    'median': data['completion_rate'].median(),
                    'std': data['completion_rate'].std(),
                    'min': data['completion_rate'].min(),
                    'max': data['completion_rate'].max()
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating data statistics: {str(e)}")
            raise