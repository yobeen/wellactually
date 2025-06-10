# src/api/score_based_comparator.py
"""
Score-based repository comparison using criteria assessment data.
Performs ratio calculations and generates comparison reasoning.
"""

import math
import random
import logging
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass

from src.api.criteria.criteria_assessment_loader import CriteriaAssessment

logger = logging.getLogger(__name__)

@dataclass
class ComparisonResult:
    """Container for comparison results."""
    choice: int  # 1 for A, 2 for B
    multiplier: float
    choice_uncertainty: float
    multiplier_uncertainty: float
    reasoning_uncertainty: float
    explanation: str
    score_a: float
    score_b: float
    ratio: float

class ScoreBasedComparator:
    """
    Performs score-based comparisons between repository assessments.
    """
    
    def __init__(self):
        """Initialize the comparator."""
        self.min_multiplier = 1.0
        self.max_multiplier = 999.0
        
    def compare_repositories(self, assessment_a: CriteriaAssessment, 
                           assessment_b: CriteriaAssessment) -> ComparisonResult:
        """
        Compare two repository assessments using target scores.
        
        Args:
            assessment_a: First repository assessment
            assessment_b: Second repository assessment
            
        Returns:
            ComparisonResult with choice, multiplier, uncertainties, and reasoning
        """
        logger.debug(f"Comparing {assessment_a.repository_name} vs {assessment_b.repository_name}")
        
        # Extract target scores
        score_a = assessment_a.target_score
        score_b = assessment_b.target_score
        
        logger.debug(f"Scores: {assessment_a.repository_name}={score_a:.3f}, "
                    f"{assessment_b.repository_name}={score_b:.3f}")
        
        # Calculate comparison metrics
        choice, multiplier, ratio = self._calculate_comparison_metrics(score_a, score_b)
        
        # Calculate uncertainties with special case check
        choice_uncertainty, multiplier_uncertainty, reasoning_uncertainty = self._calculate_uncertainties(
            assessment_a, assessment_b, abs(score_a - score_b)
        )
        
        # Fixed explanation for assessment-based comparisons
        explanation = "Assessment-based comparison"
        
        result = ComparisonResult(
            choice=choice,
            multiplier=multiplier,
            choice_uncertainty=choice_uncertainty,
            multiplier_uncertainty=multiplier_uncertainty,
            reasoning_uncertainty=reasoning_uncertainty,
            explanation=explanation,
            score_a=score_a,
            score_b=score_b,
            ratio=ratio
        )
        
        logger.info(f"Comparison result: choice={choice}, multiplier={multiplier:.2f}, "
                   f"uncertainties=({choice_uncertainty:.3f}, {multiplier_uncertainty:.3f}, {reasoning_uncertainty:.3f})")
        
        return result
    
    def _calculate_comparison_metrics(self, score_a: float, score_b: float) -> Tuple[int, float, float]:
        """
        Calculate choice, multiplier, and ratio from target scores.
        
        Args:
            score_a: Target score for repository A
            score_b: Target score for repository B
            
        Returns:
            (choice, multiplier, ratio) tuple
        """
        # Handle edge cases
        if score_a <= 0 or score_b <= 0:
            logger.warning(f"Invalid scores: A={score_a}, B={score_b}")
            return random.choice([1, 2]), 1.0, 1.0
        
        # Calculate ratio and determine choice
        if score_a > score_b:
            choice = 1
            ratio = score_a / score_b
        elif score_b > score_a:
            choice = 2
            ratio = score_b / score_a
        else:
            # Equal scores
            choice = random.choice([1, 2])
            ratio = 1.0
        
        # Calculate multiplier using exponential transformation
        try:
            multiplier = math.exp(2 * ratio)
            
            # Clamp to reasonable range
            multiplier = max(self.min_multiplier, min(self.max_multiplier, multiplier))
            
        except (OverflowError, ValueError):
            logger.warning(f"Multiplier calculation overflow for ratio {ratio}")
            multiplier = self.max_multiplier
        
        return choice, multiplier, ratio
    
    def _calculate_uncertainties(self, assessment_a: CriteriaAssessment, 
                               assessment_b: CriteriaAssessment, 
                               score_difference: float) -> Tuple[float, float, float]:
        """
        Calculate uncertainty measures with special case detection.
        
        Args:
            assessment_a: First repository assessment
            assessment_b: Second repository assessment
            score_difference: Absolute difference between scores
            
        Returns:
            (choice_uncertainty, multiplier_uncertainty, reasoning_uncertainty) tuple
        """
        # Check for parsing failure indicator (all uncertainties = 0.5)
        def has_parsing_failure(assessment: CriteriaAssessment) -> bool:
            if not assessment.criteria_scores:
                return True
            
            uncertainties = [
                criterion.get("raw_uncertainty", 0.5)
                for criterion in assessment.criteria_scores.values()
            ]
            
            # Check if all uncertainties are approximately 0.5 (within floating point tolerance)
            return len(uncertainties) > 0 and all(abs(u - 0.5) < 1e-10 for u in uncertainties)
        
        # Detect parsing failures
        parsing_failure_a = has_parsing_failure(assessment_a)
        parsing_failure_b = has_parsing_failure(assessment_b)
        
        if parsing_failure_a or parsing_failure_b:
            logger.warning(f"Parsing failure detected - using fallback uncertainties")
            logger.debug(f"Failure flags: A={parsing_failure_a}, B={parsing_failure_b}")
            
            # Use specified fallback values
            return 0.1, 0.4, 0.18
        
        # Normal uncertainty calculation based on score differences
        # Lower uncertainty when scores differ more significantly
        base_uncertainty = 1.0 / (1.0 + score_difference * 2.0)
        
        choice_uncertainty = max(0.05, min(0.95, base_uncertainty))
        multiplier_uncertainty = max(0.1, min(0.9, base_uncertainty * 1.2))
        reasoning_uncertainty = max(0.05, min(0.8, base_uncertainty * 1.1))
        
        return choice_uncertainty, multiplier_uncertainty, reasoning_uncertainty
    
    def _generate_reasoning(self, assessment_a: CriteriaAssessment, 
                          assessment_b: CriteriaAssessment, 
                          choice: int, ratio: float, 
                          score_a: float, score_b: float) -> str:
        """
        Generate comparison reasoning based on assessment data.
        
        Args:
            assessment_a: First repository assessment
            assessment_b: Second repository assessment
            choice: Chosen repository (1 or 2)
            ratio: Score ratio
            score_a: Target score for repository A
            score_b: Target score for repository B
            
        Returns:
            Formatted reasoning string
        """
        try:
            # Determine which repository was chosen
            chosen_repo = assessment_a if choice == 1 else assessment_b
            other_repo = assessment_b if choice == 1 else assessment_a
            chosen_score = score_a if choice == 1 else score_b
            other_score = score_b if choice == 1 else score_a
            
            # Start with overall comparison
            reasoning_parts = []
            
            # Overall score comparison
            score_diff = abs(chosen_score - other_score)
            if score_diff < 0.1:
                reasoning_parts.append(
                    f"{chosen_repo.repository_name} and {other_repo.repository_name} have very similar "
                    f"target scores ({chosen_score:.2f} vs {other_score:.2f}), making this a close comparison."
                )
            else:
                reasoning_parts.append(
                    f"{chosen_repo.repository_name} (score: {chosen_score:.2f}) outperforms "
                    f"{other_repo.repository_name} (score: {other_score:.2f}) "
                    f"with a {ratio:.2f}x advantage in overall assessment."
                )
            
            # Add simple score-based reasoning
            if score_diff >= 1.0:
                reasoning_parts.append(
                    f"The chosen repository demonstrates superior performance across "
                    f"multiple evaluation criteria, resulting in a significant scoring advantage."
                )
            
            # Add overall reasoning if available
            if chosen_repo.overall_reasoning:
                reasoning_parts.append(f"Assessment summary: {chosen_repo.overall_reasoning}")
            
            return " ".join(reasoning_parts)
            
        except Exception as e:
            logger.warning(f"Error generating reasoning: {e}")
            
            # Fallback reasoning
            chosen_name = assessment_a.repository_name if choice == 1 else assessment_b.repository_name
            return (f"{chosen_name} was selected based on criteria assessment scores "
                   f"({score_a:.2f} vs {score_b:.2f}).")