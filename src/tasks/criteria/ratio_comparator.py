# src/uncertainty_calibration/criteria_assessment/ratio_comparator.py
"""
Ratio comparator for criteria assessment.
Compares criteria-based target scores with human preferences from training data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ComparisonResult:
    """Container for individual comparison result."""
    repo_a: str
    repo_b: str
    target_score_a: float
    target_score_b: float
    predicted_ratio: float
    human_choice: float
    human_multiplier: float
    human_preference: str  # "A", "B", "Equal"
    predicted_preference: str  # "A", "B", "Equal"
    directional_agreement: bool
    ratio_error: float
    comparison_metadata: Dict[str, Any]

@dataclass
class ComparisonSummary:
    """Container for overall comparison analysis."""
    total_comparisons: int
    directional_accuracy: float
    mean_ratio_error: float
    median_ratio_error: float
    correlation_coefficient: float
    agreement_by_preference: Dict[str, Dict[str, Any]]
    comparison_details: List[ComparisonResult]
    analysis_warnings: List[str]

class RatioComparator:
    """
    Compares criteria-based target scores with human preferences.
    """
    
    def __init__(self):
        """Initialize the ratio comparator."""
        self.equal_threshold = 1.2  # Human multiplier <= 1.2 considered "Equal"
        self.prediction_equal_threshold = 1.1  # Predicted ratio within 10% considered "Equal"
        
    def compare_with_training_data(self, target_scores: Dict[str, float], 
                                 train_csv_path: str = "data/raw/train.csv") -> ComparisonSummary:
        """
        Compare target scores with human preferences from training data.
        
        Args:
            target_scores: Dict mapping repo URL -> target score
            train_csv_path: Path to training CSV file
            
        Returns:
            ComparisonSummary with analysis results
        """
        warnings = []
        
        try:
            # Load training data
            train_df = pd.read_csv(train_csv_path)
            logger.info(f"Loaded {len(train_df)} training comparisons")
            
            # Process each comparison
            comparison_results = []
            
            for _, row in train_df.iterrows():
                try:
                    result = self._process_single_comparison(row, target_scores)
                    if result:
                        comparison_results.append(result)
                except Exception as e:
                    warnings.append(f"Error processing comparison {row.get('repo_a', 'unknown')} vs {row.get('repo_b', 'unknown')}: {e}")
            
            logger.info(f"Successfully processed {len(comparison_results)} comparisons")
            
            # Generate summary analysis
            summary = self._generate_comparison_summary(comparison_results, warnings)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error comparing with training data: {e}")
            warnings.append(f"Failed to load training data: {e}")
            return self._create_empty_summary(warnings)
    
    def _process_single_comparison(self, row: pd.Series, target_scores: Dict[str, float]) -> ComparisonResult:
        """
        Process a single comparison from training data.
        
        Args:
            row: Row from training DataFrame
            target_scores: Dict mapping repo URL -> target score
            
        Returns:
            ComparisonResult or None if repos not found
        """
        repo_a = row.get('repo_a', '')
        repo_b = row.get('repo_b', '')
        human_choice = row.get('choice', 1.0)
        human_multiplier = row.get('multiplier', 1.0)
        
        # Check if both repos have target scores
        if repo_a not in target_scores or repo_b not in target_scores:
            return None
        
        target_score_a = target_scores[repo_a]
        target_score_b = target_scores[repo_b]
        
        # Calculate predicted ratio
        if target_score_b > 0:
            predicted_ratio = target_score_a / target_score_b
        else:
            predicted_ratio = float('inf') if target_score_a > 0 else 1.0
        
        # Determine human preference
        human_preference = self._get_human_preference(human_choice, human_multiplier)
        
        # Determine predicted preference
        predicted_preference = self._get_predicted_preference(predicted_ratio)
        
        # Check directional agreement
        directional_agreement = self._check_directional_agreement(human_preference, predicted_preference)
        
        # Calculate ratio error
        ratio_error = self._calculate_ratio_error(predicted_ratio, human_multiplier, human_choice)
        
        # Gather metadata
        metadata = {
            'parent': row.get('parent', ''),
            'juror': row.get('juror', ''),
            'timestamp': row.get('timestamp', ''),
            'reasoning': row.get('reasoning', '')
        }
        
        return ComparisonResult(
            repo_a=repo_a,
            repo_b=repo_b,
            target_score_a=target_score_a,
            target_score_b=target_score_b,
            predicted_ratio=predicted_ratio,
            human_choice=human_choice,
            human_multiplier=human_multiplier,
            human_preference=human_preference,
            predicted_preference=predicted_preference,
            directional_agreement=directional_agreement,
            ratio_error=ratio_error,
            comparison_metadata=metadata
        )
    
    def _get_human_preference(self, choice: float, multiplier: float) -> str:
        """Determine human preference from choice and multiplier."""
        if multiplier <= self.equal_threshold:
            return "Equal"
        elif choice == 1.0:
            return "A"
        elif choice == 2.0:
            return "B"
        else:
            return "Equal"
    
    def _get_predicted_preference(self, ratio: float) -> str:
        """Determine predicted preference from ratio."""
        if 1.0 / self.prediction_equal_threshold <= ratio <= self.prediction_equal_threshold:
            return "Equal"
        elif ratio > self.prediction_equal_threshold:
            return "A"
        else:
            return "B"
    
    def _check_directional_agreement(self, human_pref: str, predicted_pref: str) -> bool:
        """Check if human and predicted preferences agree directionally."""
        # Exact match
        if human_pref == predicted_pref:
            return True
        
        # Both prefer same repo (A or B) vs Equal
        if human_pref in ["A", "B"] and predicted_pref in ["A", "B"]:
            return human_pref == predicted_pref
        
        return False
    
    def _calculate_ratio_error(self, predicted_ratio: float, human_multiplier: float, human_choice: float) -> float:
        """Calculate error between predicted ratio and human judgment."""
        
        # Convert human judgment to comparable ratio
        if human_choice == 1.0:  # Human prefers A
            human_ratio = human_multiplier
        elif human_choice == 2.0:  # Human prefers B
            human_ratio = 1.0 / human_multiplier
        else:  # Equal
            human_ratio = 1.0
        
        # Handle infinite ratios
        if np.isinf(predicted_ratio) or np.isinf(human_ratio):
            if np.isinf(predicted_ratio) and np.isinf(human_ratio):
                return 0.0  # Both infinite
            else:
                return float('inf')  # One infinite, one finite
        
        # Calculate absolute log ratio error (symmetric)
        try:
            log_error = abs(np.log(predicted_ratio) - np.log(human_ratio))
            return log_error
        except (ValueError, ZeroDivisionError):
            return float('inf')
    
    def _generate_comparison_summary(self, results: List[ComparisonResult], warnings: List[str]) -> ComparisonSummary:
        """Generate summary statistics from comparison results."""
        
        if not results:
            return self._create_empty_summary(warnings + ["No valid comparisons found"])
        
        # Overall statistics
        total_comparisons = len(results)
        directional_agreements = sum(1 for r in results if r.directional_agreement)
        directional_accuracy = directional_agreements / total_comparisons
        
        # Ratio error statistics (excluding infinite errors)
        finite_errors = [r.ratio_error for r in results if not np.isinf(r.ratio_error)]
        
        if finite_errors:
            mean_ratio_error = np.mean(finite_errors)
            median_ratio_error = np.median(finite_errors)
        else:
            mean_ratio_error = float('inf')
            median_ratio_error = float('inf')
            warnings.append("All ratio errors are infinite")
        
        # Correlation between predicted and human ratios
        try:
            predicted_ratios = []
            human_ratios = []
            
            for r in results:
                if not np.isinf(r.predicted_ratio):
                    predicted_ratios.append(r.predicted_ratio)
                    
                    # Convert human judgment to ratio
                    if r.human_choice == 1.0:
                        human_ratio = r.human_multiplier
                    elif r.human_choice == 2.0:
                        human_ratio = 1.0 / r.human_multiplier
                    else:
                        human_ratio = 1.0
                    
                    human_ratios.append(human_ratio)
            
            if len(predicted_ratios) >= 2:
                correlation = np.corrcoef(predicted_ratios, human_ratios)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
                warnings.append("Insufficient data for correlation calculation")
                
        except Exception as e:
            correlation = 0.0
            warnings.append(f"Error calculating correlation: {e}")
        
        # Agreement by preference type
        agreement_by_preference = self._analyze_agreement_by_preference(results)
        
        return ComparisonSummary(
            total_comparisons=total_comparisons,
            directional_accuracy=directional_accuracy,
            mean_ratio_error=mean_ratio_error,
            median_ratio_error=median_ratio_error,
            correlation_coefficient=correlation,
            agreement_by_preference=agreement_by_preference,
            comparison_details=results,
            analysis_warnings=warnings
        )
    
    def _analyze_agreement_by_preference(self, results: List[ComparisonResult]) -> Dict[str, Dict[str, Any]]:
        """Analyze agreement rates by preference type."""
        
        preference_groups = {"A": [], "B": [], "Equal": []}
        
        for result in results:
            if result.human_preference in preference_groups:
                preference_groups[result.human_preference].append(result)
        
        analysis = {}
        
        for pref_type, group_results in preference_groups.items():
            if group_results:
                agreements = sum(1 for r in group_results if r.directional_agreement)
                accuracy = agreements / len(group_results)
                
                finite_errors = [r.ratio_error for r in group_results if not np.isinf(r.ratio_error)]
                avg_error = np.mean(finite_errors) if finite_errors else float('inf')
                
                analysis[pref_type] = {
                    "count": len(group_results),
                    "agreements": agreements,
                    "accuracy": accuracy,
                    "average_ratio_error": avg_error
                }
            else:
                analysis[pref_type] = {
                    "count": 0,
                    "agreements": 0,
                    "accuracy": 0.0,
                    "average_ratio_error": float('inf')
                }
        
        return analysis
    
    def _create_empty_summary(self, warnings: List[str]) -> ComparisonSummary:
        """Create an empty summary for error cases."""
        return ComparisonSummary(
            total_comparisons=0,
            directional_accuracy=0.0,
            mean_ratio_error=float('inf'),
            median_ratio_error=float('inf'),
            correlation_coefficient=0.0,
            agreement_by_preference={"A": {"count": 0, "accuracy": 0.0}, 
                                   "B": {"count": 0, "accuracy": 0.0}, 
                                   "Equal": {"count": 0, "accuracy": 0.0}},
            comparison_details=[],
            analysis_warnings=warnings
        )
    
    def save_comparison_results(self, summary: ComparisonSummary, output_path: str):
        """Save comparison results to CSV file."""
        try:
            # Convert results to DataFrame
            results_data = []
            
            for result in summary.comparison_details:
                results_data.append({
                    'repo_a': result.repo_a,
                    'repo_b': result.repo_b,
                    'target_score_a': result.target_score_a,
                    'target_score_b': result.target_score_b,
                    'predicted_ratio': result.predicted_ratio,
                    'human_choice': result.human_choice,
                    'human_multiplier': result.human_multiplier,
                    'human_preference': result.human_preference,
                    'predicted_preference': result.predicted_preference,
                    'directional_agreement': result.directional_agreement,
                    'ratio_error': result.ratio_error,
                    'parent': result.comparison_metadata.get('parent', ''),
                    'juror': result.comparison_metadata.get('juror', '')
                })
            
            df = pd.DataFrame(results_data)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved {len(results_data)} comparison results to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving comparison results: {e}")
            raise