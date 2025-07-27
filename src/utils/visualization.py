"""
Visualization module for e-learning session analysis.
Handles plotting and data visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SessionVisualizer:
    """Visualization class for e-learning session data."""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.output_dir = Path("outputs/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_session_distribution(self, data: pd.DataFrame, save_plot: bool = True) -> None:
        """
        Plot distribution of session characteristics.
        
        Args:
            data: DataFrame containing session data
            save_plot: Whether to save the plot to file
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            fig.suptitle('Session Distribution Analysis', fontsize=16, fontweight='bold')
            
            # Duration distribution
            axes[0, 0].hist(data['duration'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Session Duration Distribution')
            axes[0, 0].set_xlabel('Duration (minutes)')
            axes[0, 0].set_ylabel('Frequency')
            
            # Score distribution
            axes[0, 1].hist(data['score'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Session Score Distribution')
            axes[0, 1].set_xlabel('Score')
            axes[0, 1].set_ylabel('Frequency')
            
            # Interactions distribution
            axes[1, 0].hist(data['interactions'], bins=30, alpha=0.7, color='salmon', edgecolor='black')
            axes[1, 0].set_title('Session Interactions Distribution')
            axes[1, 0].set_xlabel('Number of Interactions')
            axes[1, 0].set_ylabel('Frequency')
            
            # Completion rate distribution
            axes[1, 1].hist(data['completion_rate'], bins=30, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 1].set_title('Completion Rate Distribution')
            axes[1, 1].set_xlabel('Completion Rate')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(self.output_dir / 'session_distribution.png', dpi=300, bbox_inches='tight')
                logger.info("Session distribution plot saved")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating session distribution plot: {str(e)}")
            raise
            
    def plot_correlation_matrix(self, data: pd.DataFrame, save_plot: bool = True) -> None:
        """
        Plot correlation matrix of numerical features.
        
        Args:
            data: DataFrame containing session data
            save_plot: Whether to save the plot to file
        """
        try:
            # Select numerical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            correlation_matrix = data[numerical_cols].corr()
            
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5)
            plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
                logger.info("Correlation matrix plot saved")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix plot: {str(e)}")
            raise
            
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                              save_plot: bool = True) -> None:
        """
        Plot feature importance from trained model.
        
        Args:
            feature_importance: Dictionary of feature names and their importance scores
            save_plot: Whether to save the plot to file
        """
        try:
            if not feature_importance:
                logger.warning("No feature importance data provided")
                return
                
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            features, importance = zip(*sorted_features)
            
            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(features)), importance, color='steelblue', alpha=0.8)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance Analysis', fontsize=16, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
                logger.info("Feature importance plot saved")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            raise
            
    def plot_model_performance(self, results: Dict[str, Any], save_plot: bool = True) -> None:
        """
        Plot model performance metrics.
        
        Args:
            results: Dictionary containing model performance results
            save_plot: Whether to save the plot to file
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
            
            # Accuracy and ROC AUC
            metrics = ['Accuracy', 'ROC AUC']
            values = [results.get('accuracy', 0), results.get('roc_auc', 0)]
            
            axes[0, 0].bar(metrics, values, color=['lightblue', 'lightgreen'], alpha=0.8)
            axes[0, 0].set_title('Model Performance Metrics')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_ylim(0, 1)
            
            # Add value labels
            for i, v in enumerate(values):
                axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # Cross-validation scores
            if 'cv_mean' in results and 'cv_std' in results:
                cv_scores = [results['cv_mean'] - results['cv_std'], 
                           results['cv_mean'], 
                           results['cv_mean'] + results['cv_std']]
                cv_labels = ['Mean - Std', 'Mean', 'Mean + Std']
                
                axes[0, 1].bar(cv_labels, cv_scores, color=['red', 'blue', 'red'], alpha=0.7)
                axes[0, 1].set_title('Cross-Validation Performance')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Confusion matrix
            if 'confusion_matrix' in results:
                cm = np.array(results['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
                axes[1, 0].set_title('Confusion Matrix')
                axes[1, 0].set_xlabel('Predicted')
                axes[1, 0].set_ylabel('Actual')
            
            # Classification report (text)
            if 'classification_report' in results:
                axes[1, 1].text(0.1, 0.5, results['classification_report'], 
                               transform=axes[1, 1].transAxes, fontsize=10,
                               verticalalignment='center', fontfamily='monospace')
                axes[1, 1].set_title('Classification Report')
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(self.output_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
                logger.info("Model performance plot saved")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating model performance plot: {str(e)}")
            raise
            
    def plot_time_series_analysis(self, data: pd.DataFrame, save_plot: bool = True) -> None:
        """
        Plot time series analysis of session data.
        
        Args:
            data: DataFrame containing session data with timestamp column
            save_plot: Whether to save the plot to file
        """
        try:
            if 'timestamp' not in data.columns:
                logger.warning("Timestamp column not found for time series analysis")
                return
                
            # Convert timestamp to datetime if needed
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Resample data by day
            daily_data = data.set_index('timestamp').resample('D').agg({
                'score': 'mean',
                'duration': 'mean',
                'interactions': 'mean',
                'completion_rate': 'mean'
            }).dropna()
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Time Series Analysis of Session Metrics', fontsize=16, fontweight='bold')
            
            # Score over time
            axes[0, 0].plot(daily_data.index, daily_data['score'], marker='o', linewidth=2)
            axes[0, 0].set_title('Average Score Over Time')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Average Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Duration over time
            axes[0, 1].plot(daily_data.index, daily_data['duration'], marker='s', linewidth=2, color='orange')
            axes[0, 1].set_title('Average Duration Over Time')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Average Duration (minutes)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Interactions over time
            axes[1, 0].plot(daily_data.index, daily_data['interactions'], marker='^', linewidth=2, color='green')
            axes[1, 0].set_title('Average Interactions Over Time')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Average Interactions')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Completion rate over time
            axes[1, 1].plot(daily_data.index, daily_data['completion_rate'], marker='d', linewidth=2, color='red')
            axes[1, 1].set_title('Average Completion Rate Over Time')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Average Completion Rate')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(self.output_dir / 'time_series_analysis.png', dpi=300, bbox_inches='tight')
                logger.info("Time series analysis plot saved")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating time series analysis plot: {str(e)}")
            raise
            
    def create_summary_dashboard(self, data: pd.DataFrame, results: Optional[Dict[str, Any]] = None,
                               save_plot: bool = True) -> None:
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            data: DataFrame containing session data
            results: Optional dictionary containing model results
            save_plot: Whether to save the plot to file
        """
        try:
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle('E-Learning Session Analysis Dashboard', fontsize=20, fontweight='bold')
            
            # Create grid layout
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
            # Session distribution (top left)
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.hist(data['score'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            ax1.set_title('Session Score Distribution', fontweight='bold')
            ax1.set_xlabel('Score')
            ax1.set_ylabel('Frequency')
            
            # Duration vs Score scatter (top right)
            ax2 = fig.add_subplot(gs[0, 2:])
            ax2.scatter(data['duration'], data['score'], alpha=0.6, color='orange')
            ax2.set_title('Duration vs Score Relationship', fontweight='bold')
            ax2.set_xlabel('Duration (minutes)')
            ax2.set_ylabel('Score')
            
            # Model performance (if available)
            if results:
                ax3 = fig.add_subplot(gs[1, :2])
                metrics = ['Accuracy', 'ROC AUC']
                values = [results.get('accuracy', 0), results.get('roc_auc', 0)]
                bars = ax3.bar(metrics, values, color=['lightgreen', 'lightcoral'], alpha=0.8)
                ax3.set_title('Model Performance', fontweight='bold')
                ax3.set_ylabel('Score')
                ax3.set_ylim(0, 1)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
                
                # Feature importance (if available)
                if 'feature_importance' in results and results['feature_importance']:
                    ax4 = fig.add_subplot(gs[1, 2:])
                    feature_importance = results['feature_importance']
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    features, importance = zip(*sorted_features)
                    
                    ax4.barh(range(len(features)), importance, color='steelblue', alpha=0.8)
                    ax4.set_yticks(range(len(features)))
                    ax4.set_yticklabels(features)
                    ax4.set_xlabel('Importance')
                    ax4.set_title('Top 5 Feature Importance', fontweight='bold')
                    ax4.invert_yaxis()
            
            # Time series (bottom)
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                daily_data = data.set_index('timestamp').resample('D')['score'].mean().dropna()
                
                ax5 = fig.add_subplot(gs[2:, :])
                ax5.plot(daily_data.index, daily_data.values, marker='o', linewidth=2, color='purple')
                ax5.set_title('Average Score Over Time', fontweight='bold')
                ax5.set_xlabel('Date')
                ax5.set_ylabel('Average Score')
                ax5.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(self.output_dir / 'dashboard.png', dpi=300, bbox_inches='tight')
                logger.info("Dashboard plot saved")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise