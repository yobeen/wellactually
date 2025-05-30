# src/uncertainty_calibration/criteria_assessment/target_score_calculator.py
"""
Target score calculator for criteria assessment.
Calculates weighted target scores from criteria assessments.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TargetScoreResult:
    """Container for target score calculation results."""
    target_score: float
    weighted_contributions: Dict[str, float]  # criterion -> weighted_score
    total_possible_score: float
    normalized_score: float  # 0-1 scale
    calculation_method: str
    calculation_warnings: List[str]

class TargetScoreCalculator:
    """
    Calculates weighted target scores from criteria assessments.
    """
    
    def __init__(self):
        """Initialize the target score calculator."""
        self.min_score = 1.0  # Minimum score per criterion
        self.max_score = 10.0  # Maximum score per criterion
        
    def calculate_target_score(self, criteria_scores: Dict[str, Dict[str, Any]]) -> TargetScoreResult:
        """
        Calculate weighted target score from criteria assessments.
        
        Args:
            criteria_scores: Dict mapping criterion -> {score, weight, reasoning}
            
        Returns:
            TargetScoreResult with calculated scores and metadata
        """
        warnings = []
        
        if not criteria_scores:
            warnings.append("No criteria scores provided")
            return self._create_default_result(warnings)
        
        # Extract scores and weights
        scores = {}
        weights = {}
        
        for criterion, assessment in criteria_scores.items():
            # Validate and extract score
            score = assessment.get("score", 5.0)
            if not isinstance(score, (int, float)):
                score = 5.0
                warnings.append(f"Invalid score for {criterion}, using default (5.0)")
            elif not (self.min_score <= score <= self.max_score):
                score = max(self.min_score, min(self.max_score, score))
                warnings.append(f"Score for {criterion} out of range, clamped to {score}")
            
            scores[criterion] = float(score)
            
            # Validate and extract weight
            weight = assessment.get("weight", 0.0)
            if not isinstance(weight, (int, float)) or weight < 0:
                weight = 0.0
                warnings.append(f"Invalid weight for {criterion}, using 0.0")
            
            weights[criterion] = float(weight)
        
        # Check weight sum
        total_weight = sum(weights.values())
        if total_weight == 0:
            warnings.append("All weights are zero, cannot calculate meaningful score")
            return self._create_default_result(warnings)
        
        # Calculate weighted contributions
        weighted_contributions = {}
        target_score = 0.0
        
        for criterion in scores.keys():
            weighted_score = scores[criterion] * weights[criterion]
            weighted_contributions[criterion] = weighted_score
            target_score += weighted_score
        
        # Calculate total possible score (if all criteria scored max)
        total_possible_score = self.max_score * total_weight
        
        # Calculate normalized score (0-1 scale)
        if total_possible_score > 0:
            normalized_score = target_score / total_possible_score
        else:
            normalized_score = 0.0
            warnings.append("Cannot normalize score: total possible score is zero")
        
        logger.debug(f"Calculated target score: {target_score:.3f} (normalized: {normalized_score:.3f})")
        
        return TargetScoreResult(
            target_score=target_score,
            weighted_contributions=weighted_contributions,
            total_possible_score=total_possible_score,
            normalized_score=normalized_score,
            calculation_method="weighted_sum",
            calculation_warnings=warnings
        )
    
    def calculate_alternative_scores(self, criteria_scores: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate alternative scoring methods for comparison.
        
        Args:
            criteria_scores: Dict mapping criterion -> {score, weight, reasoning}
            
        Returns:
            Dictionary with alternative score calculations
        """
        if not criteria_scores:
            return {}
        
        # Extract scores
        scores = [assessment.get("score", 5.0) for assessment in criteria_scores.values()]
        weights = [assessment.get("weight", 0.0) for assessment in criteria_scores.values()]
        
        # Simple average (ignoring weights)
        simple_average = sum(scores) / len(scores) if scores else 0.0
        
        # Weighted average (normalized by weight sum)
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_average = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            weighted_average = simple_average
        
        # Median score
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        if n % 2 == 0:
            median_score = (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2
        else:
            median_score = sorted_scores[n//2]
        
        # Min and max scores
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        
        return {
            "simple_average": simple_average,
            "weighted_average": weighted_average,
            "median": median_score,
            "minimum": min_score,
            "maximum": max_score,
            "range": max_score - min_score
        }
    
    def get_score_distribution(self, criteria_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the distribution of scores across criteria.
        
        Args:
            criteria_scores: Dict mapping criterion -> {score, weight, reasoning}
            
        Returns:
            Dictionary with score distribution statistics
        """
        if not criteria_scores:
            return {}
        
        scores = []
        weights = []
        weighted_scores = []
        
        for assessment in criteria_scores.values():
            score = assessment.get("score", 5.0)
            weight = assessment.get("weight", 0.0)
            
            scores.append(score)
            weights.append(weight)
            weighted_scores.append(score * weight)
        
        # Basic statistics
        n = len(scores)
        mean_score = sum(scores) / n if n > 0 else 0.0
        
        # Calculate variance and std dev
        if n > 1:
            variance = sum((s - mean_score) ** 2 for s in scores) / (n - 1)
            std_dev = variance ** 0.5
        else:
            variance = 0.0
            std_dev = 0.0
        
        # Score distribution by ranges
        score_ranges = {
            "low (1-3)": sum(1 for s in scores if 1 <= s <= 3),
            "medium (4-7)": sum(1 for s in scores if 4 <= s <= 7),
            "high (8-10)": sum(1 for s in scores if 8 <= s <= 10)
        }
        
        # Weight distribution
        total_weight = sum(weights)
        weight_distribution = {}
        if total_weight > 0:
            weight_distribution = {
                criterion: {"weight": weight, "percentage": weight / total_weight * 100}
                for criterion, weight in zip(criteria_scores.keys(), weights)
            }
        
        return {
            "score_statistics": {
                "count": n,
                "mean": mean_score,
                "median": sorted(scores)[n//2] if n > 0 else 0.0,
                "std_dev": std_dev,
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0
            },
            "score_ranges": score_ranges,
            "weight_statistics": {
                "total_weight": total_weight,
                "mean_weight": sum(weights) / n if n > 0 else 0.0,
                "weight_distribution": weight_distribution
            },
            "weighted_score_sum": sum(weighted_scores)
        }
    
    def compare_scores(self, target_score_a: float, target_score_b: float) -> Dict[str, Any]:
        """
        Compare two target scores and calculate ratio.
        
        Args:
            target_score_a: First target score
            target_score_b: Second target score
            
        Returns:
            Dictionary with comparison results
        """
        # Calculate ratio (how many times A is better than B)
        if target_score_b > 0:
            ratio_a_to_b = target_score_a / target_score_b
        else:
            ratio_a_to_b = float('inf') if target_score_a > 0 else 1.0
        
        # Calculate difference
        score_difference = target_score_a - target_score_b
        
        # Determine preference
        if abs(score_difference) < 0.1:  # Very close scores
            preference = "Equal"
            confidence = "Low"
        elif score_difference > 0:
            preference = "A"
            confidence = "High" if score_difference > 1.0 else "Medium"
        else:
            preference = "B"
            confidence = "High" if abs(score_difference) > 1.0 else "Medium"
        
        return {
            "score_a": target_score_a,
            "score_b": target_score_b,
            "ratio_a_to_b": ratio_a_to_b,
            "ratio_b_to_a": 1.0 / ratio_a_to_b if ratio_a_to_b != 0 else float('inf'),
            "score_difference": score_difference,
            "absolute_difference": abs(score_difference),
            "preference": preference,
            "confidence": confidence
        }
    
    def _create_default_result(self, warnings: List[str]) -> TargetScoreResult:
        """Create a default result for error cases."""
        return TargetScoreResult(
            target_score=5.0,  # Middle value
            weighted_contributions={},
            total_possible_score=10.0,
            normalized_score=0.5,
            calculation_method="default_fallback",
            calculation_warnings=warnings
        )