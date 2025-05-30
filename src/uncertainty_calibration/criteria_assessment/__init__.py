# src/uncertainty_calibration/criteria_assessment/__init__.py
"""
Criteria Assessment Package for Repository Evaluation.

This package provides a comprehensive framework for assessing repositories
against multiple criteria and comparing results with human preferences.
"""

from .repo_extractor import RepositoryExtractor
from .criteria_prompt_generator import CriteriaPromptGenerator
from .fuzzy_response_parser import FuzzyCriteriaResponseParser, ParsedCriteriaResponse
from .weight_validator import WeightValidator, WeightValidationResult
from .target_score_calculator import TargetScoreCalculator, TargetScoreResult
from .ratio_comparator import RatioComparator, ComparisonResult, ComparisonSummary
from ....scripts.criteria_assessment_main import CriteriaAssessmentPipeline

__all__ = [
    # Core components
    'RepositoryExtractor',
    'CriteriaPromptGenerator', 
    'FuzzyCriteriaResponseParser',
    'WeightValidator',
    'TargetScoreCalculator',
    'RatioComparator',
    
    # Main pipeline
    'CriteriaAssessmentPipeline',
    
    # Data classes
    'ParsedCriteriaResponse',
    'WeightValidationResult',
    'TargetScoreResult',
    'ComparisonResult',
    'ComparisonSummary'
]

__version__ = "1.0.0"
__author__ = "Criteria Assessment Team"
__description__ = "Multi-criteria repository assessment and comparison framework"