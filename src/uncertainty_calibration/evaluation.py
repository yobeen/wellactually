# src/uncertainty_calibration/evaluation.py
#!/usr/bin/env python3
"""
Evaluation metrics for uncertainty calibration quality.
Implements comprehensive calibration assessment framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import (
    log_loss, brier_score_loss, roc_auc_score, accuracy_score,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve
import logging

logger = logging.getLogger(__name__)

class CalibrationEvaluator:
    """
    Comprehensive evaluation of uncertainty calibration quality.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate_calibration(self, y_true: np.ndarray, y_prob: np.ndarray,
                           n_bins: int = 10) -> Dict[str, float]:
        """
        Comprehensive calibration evaluation.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary of calibration metrics
        """
        
        metrics = {}
        
        # Expected Calibration Error (ECE)
        metrics['ECE'] = self.expected_calibration_error(y_true, y_prob, n_bins)
        
        # Maximum Calibration Error (MCE)
        metrics['MCE'] = self.maximum_calibration_error(y_true, y_prob, n_bins)
        
        # Brier Score (reliability + resolution - uncertainty)
        metrics['Brier_Score'] = brier_score_loss(y_true, y_prob)
        
        # Decompose Brier Score
        reliability, resolution, uncertainty = self.brier_decomposition(y_true, y_prob, n_bins)
        metrics['Reliability'] = reliability  # Lower is better (calibration)
        metrics['Resolution'] = resolution    # Higher is better (sharpness)
        metrics['Uncertainty'] = uncertainty  # Inherent uncertainty in data
        
        # Log Loss (proper scoring rule)
        metrics['Log_Loss'] = log_loss(y_true, y_prob)
        
        # ROC AUC (discrimination ability)
        metrics['ROC_AUC'] = roc_auc_score(y_true, y_prob)
        
        # Accuracy at 0.5 threshold
        y_pred = (y_prob > 0.5).astype(int)
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        
        # Sharpness (how far predictions are from 0.5)
        metrics['Sharpness'] = self.sharpness(y_prob)
        
        # Overconfidence/Underconfidence
        overconf, underconf = self.confidence_bias(y_true, y_prob, n_bins)
        metrics['Overconfidence'] = overconf
        metrics['Underconfidence'] = underconf
        
        return metrics
    
    def expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray,
                                 n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE measures the difference between prediction confidence and accuracy.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def maximum_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray,
                                n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error (MCE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error
    
    def brier_decomposition(self, y_true: np.ndarray, y_prob: np.ndarray,
                          n_bins: int = 10) -> Tuple[float, float, float]:
        """
        Decompose Brier Score into Reliability, Resolution, and Uncertainty.
        
        Brier Score = Reliability - Resolution + Uncertainty
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        overall_accuracy = y_true.mean()
        n_samples = len(y_true)
        
        reliability = 0
        resolution = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            n_in_bin = in_bin.sum()
            
            if n_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                # Reliability: calibration error weighted by bin size
                reliability += (n_in_bin / n_samples) * (avg_confidence_in_bin - accuracy_in_bin) ** 2
                
                # Resolution: how much bin accuracy differs from overall accuracy
                resolution += (n_in_bin / n_samples) * (accuracy_in_bin - overall_accuracy) ** 2
        
        # Uncertainty: inherent uncertainty in the data
        uncertainty = overall_accuracy * (1 - overall_accuracy)
        
        return reliability, resolution, uncertainty
    
    def sharpness(self, y_prob: np.ndarray) -> float:
        """
        Calculate sharpness - how far predictions are from uninformative (0.5).
        Higher sharpness = more confident predictions.
        """
        return np.mean(np.abs(y_prob - 0.5))
    
    def confidence_bias(self, y_true: np.ndarray, y_prob: np.ndarray,
                       n_bins: int = 10) -> Tuple[float, float]:
        """
        Calculate overconfidence and underconfidence.
        
        Returns:
            (overconfidence, underconfidence) where:
            - overconfidence: average amount predictions exceed accuracy
            - underconfidence: average amount predictions fall below accuracy
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        overconf_sum = 0
        underconf_sum = 0
        total_weight = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                diff = avg_confidence_in_bin - accuracy_in_bin
                
                if diff > 0:  # Overconfident
                    overconf_sum += diff * prop_in_bin
                else:  # Underconfident
                    underconf_sum += (-diff) * prop_in_bin
                
                total_weight += prop_in_bin
        
        return overconf_sum, underconf_sum
    
    def evaluate_by_model(self, df: pd.DataFrame, prob_column: str = 'predicted_prob',
                         true_column: str = 'is_correct',
                         model_column: str = 'model_name') -> pd.DataFrame:
        """
        Evaluate calibration separately for each model.
        
        Args:
            df: Dataframe with predictions and true labels
            prob_column: Column with predicted probabilities
            true_column: Column with true labels
            model_column: Column with model identifiers
            
        Returns:
            Dataframe with metrics per model
        """
        
        model_metrics = []
        
        for model_name in df[model_column].unique():
            model_df = df[df[model_column] == model_name]
            
            if len(model_df) == 0:
                continue
            
            y_true = model_df[true_column].values
            y_prob = model_df[prob_column].values
            
            metrics = self.evaluate_calibration(y_true, y_prob)
            metrics['model_name'] = model_name
            metrics['n_samples'] = len(model_df)
            
            model_metrics.append(metrics)
        
        return pd.DataFrame(model_metrics)
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                             n_bins: int = 10, title: str = "Calibration Curve",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration curve (reliability diagram).
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot calibration curve
        ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                label="Model", linewidth=2, markersize=8)
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.8)
        
        # Add histogram of prediction confidences
        ax2 = ax.twinx()
        ax2.hist(y_prob, bins=n_bins, alpha=0.3, color='gray', label='Prediction Distribution')
        ax2.set_ylabel('Count', color='gray')
        
        # Formatting
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add ECE to plot
        ece = self.expected_calibration_error(y_true, y_prob, n_bins)
        ax.text(0.05, 0.95, f'ECE: {ece:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, model_metrics: pd.DataFrame,
                            metric: str = 'ECE',
                            title: Optional[str] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of calibration metrics across models.
        
        Args:
            model_metrics: DataFrame with metrics per model
            metric: Metric to plot
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        
        if title is None:
            title = f"Model Comparison: {metric}"
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort models by metric value
        sorted_df = model_metrics.sort_values(metric)
        
        # Create bar plot
        bars = ax.bar(range(len(sorted_df)), sorted_df[metric])
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if metric in ['ECE', 'MCE', 'Brier_Score', 'Log_Loss']:  # Lower is better
                color_intensity = sorted_df[metric].iloc[i] / sorted_df[metric].max()
                bar.set_color(plt.cm.Reds(color_intensity))
            else:  # Higher is better
                color_intensity = sorted_df[metric].iloc[i] / sorted_df[metric].max()
                bar.set_color(plt.cm.Greens(color_intensity))
        
        # Formatting
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticks(range(len(sorted_df)))
        ax.set_xticklabels(sorted_df['model_name'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_evaluation_report(self, y_true: np.ndarray, y_prob: np.ndarray,
                               model_name: str = "Model") -> Dict[str, Any]:
        """
        Create comprehensive evaluation report.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation results and plots
        """
        
        # Calculate metrics
        metrics = self.evaluate_calibration(y_true, y_prob)
        
        # Create plots
        calibration_fig = self.plot_calibration_curve(
            y_true, y_prob, title=f"Calibration Curve - {model_name}"
        )
        
        # Create summary
        report = {
            'model_name': model_name,
            'metrics': metrics,
            'plots': {
                'calibration_curve': calibration_fig
            },
            'summary': {
                'total_samples': len(y_true),
                'positive_rate': y_true.mean(),
                'mean_confidence': y_prob.mean(),
                'confidence_std': y_prob.std()
            }
        }
        
        return report

def evaluate_calibration_quality(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> Dict[str, float]:
    """
    Convenience function to evaluate calibration quality.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration metrics
        
    Returns:
        Dictionary of calibration metrics
    """
    evaluator = CalibrationEvaluator()
    return evaluator.evaluate_calibration(y_true, y_prob, n_bins)