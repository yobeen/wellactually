# src/uncertainty_calibration/criteria_assessment/weight_validator.py
"""
Weight validator for criteria assessment.
Validates and normalizes weights to ensure they sum to 1.0.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WeightValidationResult:
    """Container for weight validation results."""
    original_weights: Dict[str, float]
    normalized_weights: Dict[str, float]
    original_sum: float
    normalized_sum: float
    deviation_from_template: float
    validation_warnings: List[str]
    significant_deviation: bool
    normalization_applied: bool

class WeightValidator:
    """
    Validates and normalizes criterion weights for criteria assessment.
    """
    
    def __init__(self):
        """Initialize the weight validator."""
        # Template weights for comparison
        self.template_weights = {
            "core_protocol": 0.25,
            "market_adoption": 0.20,
            "developer_ecosystem": 0.15,
            "general_purpose_tools": 0.10,
            "security_infrastructure": 0.10,
            "defi_infrastructure": 0.05,
            "data_analytics": 0.05,
            "innovation_research": 0.03,
            "ecosystem_coordination": 0.03,
            "community_trust": 0.02,
            "user_applications": 0.02
        }
        
        # Validation thresholds
        self.max_deviation_warning = 0.1  # Warn if sum deviates by more than 10%
        self.max_deviation_error = 0.5    # Error if sum deviates by more than 50%
        self.normalization_threshold = 0.05  # Normalize if deviation > 5%
    
    def validate_and_normalize_weights(self, weights: Dict[str, float]) -> WeightValidationResult:
        """
        Validate criterion weights and normalize if necessary.
        
        Args:
            weights: Dictionary of criterion weights
            
        Returns:
            WeightValidationResult with validation info and normalized weights
        """
        warnings = []
        original_weights = weights.copy()
        original_sum = sum(weights.values())
        
        logger.debug(f"Validating weights with sum: {original_sum:.3f}")
        
        # Check for missing criteria
        missing_criteria = set(self.template_weights.keys()) - set(weights.keys())
        if missing_criteria:
            for criterion in missing_criteria:
                weights[criterion] = self.template_weights[criterion]
                warnings.append(f"Added missing criterion '{criterion}' with template weight")
        
        # Check for extra criteria
        extra_criteria = set(weights.keys()) - set(self.template_weights.keys())
        if extra_criteria:
            for criterion in extra_criteria:
                del weights[criterion]
                warnings.append(f"Removed unknown criterion '{criterion}'")
        
        # Recalculate sum after additions/removals
        current_sum = sum(weights.values())
        
        # Check for zero or negative weights
        for criterion, weight in weights.items():
            if weight <= 0:
                weights[criterion] = self.template_weights[criterion]
                warnings.append(f"Replaced non-positive weight for '{criterion}' with template value")
        
        # Recalculate sum after fixing non-positive weights
        current_sum = sum(weights.values())
        
        # Determine if normalization is needed
        sum_deviation = abs(current_sum - 1.0)
        needs_normalization = sum_deviation > self.normalization_threshold
        significant_deviation = sum_deviation > self.max_deviation_warning
        
        # Apply normalization if needed
        normalized_weights = weights.copy()
        normalization_applied = False
        
        if needs_normalization and current_sum > 0:
            normalized_weights = {k: v / current_sum for k, v in weights.items()}
            normalization_applied = True
            logger.debug(f"Normalized weights from sum {current_sum:.3f} to 1.0")
            
            if significant_deviation:
                warnings.append(f"Significant weight deviation: sum was {current_sum:.3f}, normalized to 1.0")
            else:
                warnings.append(f"Minor weight normalization: sum was {current_sum:.3f}")
        
        elif current_sum == 0:
            # Fallback to template weights
            normalized_weights = self.template_weights.copy()
            normalization_applied = True
            warnings.append("All weights were zero, using template weights")
        
        # Calculate deviation from template
        template_deviation = self._calculate_template_deviation(normalized_weights)
        
        # Add warnings for significant template deviation
        if template_deviation > 0.2:  # 20% average deviation
            warnings.append(f"Weights significantly differ from template (avg deviation: {template_deviation:.1%})")
        
        # Final validation
        final_sum = sum(normalized_weights.values())
        
        return WeightValidationResult(
            original_weights=original_weights,
            normalized_weights=normalized_weights,
            original_sum=original_sum,
            normalized_sum=final_sum,
            deviation_from_template=template_deviation,
            validation_warnings=warnings,
            significant_deviation=significant_deviation,
            normalization_applied=normalization_applied
        )
    
    def _calculate_template_deviation(self, weights: Dict[str, float]) -> float:
        """
        Calculate average deviation from template weights.
        
        Args:
            weights: Dictionary of criterion weights
            
        Returns:
            Average absolute deviation from template weights
        """
        total_deviation = 0.0
        count = 0
        
        for criterion, weight in weights.items():
            if criterion in self.template_weights:
                template_weight = self.template_weights[criterion]
                deviation = abs(weight - template_weight)
                total_deviation += deviation
                count += 1
        
        return total_deviation / count if count > 0 else 0.0
    
    def get_weight_comparison_summary(self, validation_result: WeightValidationResult) -> Dict[str, Any]:
        """
        Generate a summary comparing weights to template.
        
        Args:
            validation_result: Weight validation result
            
        Returns:
            Dictionary with comparison summary
        """
        comparison = {}
        
        for criterion in self.template_weights.keys():
            template_weight = self.template_weights[criterion]
            original_weight = validation_result.original_weights.get(criterion, 0.0)
            normalized_weight = validation_result.normalized_weights.get(criterion, 0.0)
            
            comparison[criterion] = {
                "template": template_weight,
                "original": original_weight,
                "normalized": normalized_weight,
                "deviation_from_template": abs(normalized_weight - template_weight),
                "relative_change": ((normalized_weight / template_weight) - 1.0) if template_weight > 0 else 0.0
            }
        
        return {
            "criterion_comparisons": comparison,
            "summary": {
                "original_sum": validation_result.original_sum,
                "normalized_sum": validation_result.normalized_sum,
                "avg_template_deviation": validation_result.deviation_from_template,
                "normalization_applied": validation_result.normalization_applied,
                "warning_count": len(validation_result.validation_warnings)
            }
        }
    
    def validate_single_weight(self, criterion: str, weight: float) -> Dict[str, Any]:
        """
        Validate a single criterion weight.
        
        Args:
            criterion: Criterion name
            weight: Weight value
            
        Returns:
            Validation result for single weight
        """
        warnings = []
        
        # Check if criterion is known
        if criterion not in self.template_weights:
            warnings.append(f"Unknown criterion: {criterion}")
            is_valid = False
        else:
            is_valid = True
        
        # Check weight range
        if weight < 0:
            warnings.append(f"Negative weight for {criterion}: {weight}")
            is_valid = False
        elif weight > 1.0:
            warnings.append(f"Weight exceeds 1.0 for {criterion}: {weight}")
        
        # Check deviation from template
        if criterion in self.template_weights:
            template_weight = self.template_weights[criterion]
            deviation = abs(weight - template_weight)
            relative_deviation = deviation / template_weight if template_weight > 0 else float('inf')
            
            if relative_deviation > 1.0:  # 100% deviation
                warnings.append(f"Large deviation from template for {criterion}: {relative_deviation:.1%}")
        else:
            template_weight = None
            deviation = None
            relative_deviation = None
        
        return {
            "criterion": criterion,
            "weight": weight,
            "template_weight": template_weight,
            "deviation": deviation,
            "relative_deviation": relative_deviation,
            "is_valid": is_valid,
            "warnings": warnings
        }