# src/uncertainty_calibration/evaluation.py
"""
Evaluation metrics for uncertainty calibration quality.
Implements Expected Calibration Error, Brier score, and other calibration metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from typing import Dict, List, Any, Optional, Tuple

class CalibrationEvaluator:
    """Evaluates calibration quality of uncertainty estimates."""
    
    def __init__(self):
        pass
    
    def expected_calibration_error(self, 
                                 y_true: np.ndarray, 
                                 y_prob: np.ndarray, 
                                 n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            ECE score (lower is better)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = y_true[in_bin].mean()
                # Average confidence in this bin
                avg_confidence_in_bin = y_prob[in_bin].mean()
                # ECE contribution
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def maximum_calibration_error(self, 
                                y_true: np.ndarray, 
                                y_prob: np.ndarray, 
                                n_bins: int = 10) -> float:
        """
        Calculate Maximum Calibration Error (MCE).
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            MCE score (lower is better)
        """
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
    
    def reliability_diagram_data(self, 
                               y_true: np.ndarray, 
                               y_prob: np.ndarray, 
                               n_bins: int = 10) -> Dict[str, np.ndarray]:
        """
        Generate data for reliability diagram.
        
        Returns:
            Dict with bin data for plotting
        """
        fraction_correct, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        # Calculate bin counts
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_counts = np.zeros(n_bins)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            bin_counts[i] = in_bin.sum()
        
        return {
            'fraction_correct': fraction_correct,
            'mean_predicted': mean_predicted,
            'bin_counts': bin_counts,
            'bin_boundaries': bin_boundaries
        }
    
    def sharpness(self, y_prob: np.ndarray) -> float:
        """
        Calculate sharpness (how discriminative the predictions are).
        
        Higher sharpness means predictions are more concentrated 
        away from 0.5 (more confident).
        """
        return np.var(y_prob)
    
    def comprehensive_evaluation(self, 
                               y_true: np.ndarray, 
                               y_prob: np.ndarray,
                               n_bins: int = 10) -> Dict[str, float]:
        """
        Calculate comprehensive calibration metrics.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary of calibration metrics
        """
        metrics = {}
        
        # Calibration metrics
        metrics['ECE'] = self.expected_calibration_error(y_true, y_prob, n_bins)
        metrics['MCE'] = self.maximum_calibration_error(y_true, y_prob, n_bins)
        
        # Performance metrics
        metrics['Brier_Score'] = brier_score_loss(y_true, y_prob)
        metrics['Log_Loss'] = log_loss(y_true, y_prob)
        metrics['AUC'] = roc_auc_score(y_true, y_prob)
        
        # Prediction quality
        metrics['Sharpness'] = self.sharpness(y_prob)
        metrics['Accuracy'] = (y_true == (y_prob > 0.5)).mean()
        
        # Confidence statistics
        metrics['Mean_Confidence'] = np.mean(y_prob)
        metrics['Confidence_Std'] = np.std(y_prob)
        
        # Overconfidence/underconfidence
        metrics['Mean_Predicted_Prob'] = np.mean(y_prob)
        metrics['Base_Rate'] = np.mean(y_true)
        metrics['Overconfidence'] = metrics['Mean_Predicted_Prob'] - metrics['Base_Rate']
        
        return metrics
    
    def evaluate_by_groups(self, 
                         df: pd.DataFrame,
                         group_col: str,
                         prob_col: str = 'calibrated_confidence',
                         true_col: str = 'is_correct') -> Dict[str, Dict]:
        """
        Evaluate calibration by groups (e.g., by model, temperature).
        
        Args:
            df: DataFrame with predictions and groups
            group_col: Column to group by
            prob_col: Column with predicted probabilities
            true_col: Column with true labels
            
        Returns:
            Dict mapping group values to metrics
        """
        group_metrics = {}
        
        for group_value in df[group_col].unique():
            group_data = df[df[group_col] == group_value]
            
            if len(group_data) < 10:  # Need minimum samples
                continue
            
            y_true = group_data[true_col].values
            y_prob = group_data[prob_col].values
            
            group_metrics[str(group_value)] = self.comprehensive_evaluation(y_true, y_prob)
        
        return group_metrics
    
    def plot_reliability_diagram(self, 
                               y_true: np.ndarray, 
                               y_prob: np.ndarray,
                               title: str = "Reliability Diagram",
                               n_bins: int = 10,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot reliability diagram for calibration visualization.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            title: Plot title
            n_bins: Number of bins
            save_path: Optional path to save plot
            
        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reliability diagram
        diagram_data = self.reliability_diagram_data(y_true, y_prob, n_bins)
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.plot(diagram_data['mean_predicted'], diagram_data['fraction_correct'], 
                'o-', label='Model calibration')
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title(f'{title} - Calibration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram of predictions
        ax2.hist(y_prob, bins=n_bins, alpha=0.7, density=True)
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Density')
        ax2.set_title(f'{title} - Prediction Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_calibration_by_group(self, 
                                df: pd.DataFrame,
                                group_col: str,
                                prob_col: str = 'calibrated_confidence',
                                true_col: str = 'is_correct',
                                title: str = "Calibration by Group",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration metrics by group.
        
        Args:
            df: DataFrame with predictions and groups
            group_col: Column to group by
            prob_col: Column with predicted probabilities
            true_col: Column with true labels
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            matplotlib Figure
        """
        group_metrics = self.evaluate_by_groups(df, group_col, prob_col, true_col)
        
        # Extract metrics for plotting
        groups = list(group_metrics.keys())
        ece_scores = [group_metrics[g]['ECE'] for g in groups]
        brier_scores = [group_metrics[g]['Brier_Score'] for g in groups]
        auc_scores = [group_metrics[g]['AUC'] for g in groups]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # ECE by group
        ax1.bar(groups, ece_scores)
        ax1.set_ylabel('Expected Calibration Error')
        ax1.set_title('ECE by Group')
        ax1.tick_params(axis='x', rotation=45)
        
        # Brier Score by group
        ax2.bar(groups, brier_scores)
        ax2.set_ylabel('Brier Score')
        ax2.set_title('Brier Score by Group')
        ax2.tick_params(axis='x', rotation=45)
        
        # AUC by group
        ax3.bar(groups, auc_scores)
        ax3.set_ylabel('AUC')
        ax3.set_title('AUC by Group')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def compare_calibration_methods(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple calibration methods.
    
    Args:
        results_dict: Dict mapping method names to evaluation results
        
    Returns:
        DataFrame comparing methods
    """
    comparison_data = []
    
    for method_name, metrics in results_dict.items():
        row = {'Method': method_name}
        row.update(metrics)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    return df


def evaluate_pipeline_calibration(pipeline, test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate calibration quality of a trained pipeline.
    
    Args:
        pipeline: Trained UncertaintyCalibrationPipeline
        test_df: Test dataframe with ground truth
        
    Returns:
        Comprehensive evaluation results
    """
    evaluator = CalibrationEvaluator()
    
    # Apply calibration
    calibrated_df = pipeline.batch_calibrate(test_df)
    
    # Overall evaluation
    y_true = calibrated_df['is_correct'].values
    y_prob = calibrated_df['calibrated_confidence'].values
    
    overall_metrics = evaluator.comprehensive_evaluation(y_true, y_prob)
    
    # By-model evaluation
    model_metrics = evaluator.evaluate_by_groups(
        calibrated_df, 'model_id', 'calibrated_confidence', 'is_correct'
    )
    
    # By-temperature evaluation (if available)
    temp_metrics = {}
    if 'temperature' in calibrated_df.columns:
        temp_metrics = evaluator.evaluate_by_groups(
            calibrated_df, 'temperature', 'calibrated_confidence', 'is_correct'
        )
    
    return {
        'overall_metrics': overall_metrics,
        'by_model_metrics': model_metrics,
        'by_temperature_metrics': temp_metrics,
        'calibrated_data': calibrated_df
    }