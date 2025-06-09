# src/tasks/l1/__init__.py
"""
Level 1 Analysis Package

This package provides comprehensive Level 1 validation analysis including single model analysis,
multi-model comparisons, voting analysis, and visualization components.
"""

from .l1_analysis import *
from .l1_core_analysis import (
    run_analysis,
    run_single_model_analysis,
    run_multi_model_analysis,
    analyze_accuracy,
    analyze_precision_rejection
)
from .l1_multi_model import generate_cross_model_analysis
from .l1_utils import (
    group_data_by_model,
    sanitize_model_name,
    generate_timestamp,
    create_results_directory
)
from .l1_visualization import (
    create_single_model_plots,
    create_model_accuracy_comparison
)
from .l1_voting_analysis import run_voting_analysis
from .level1_prompts import Level1PromptGenerator

__all__ = [
    # Main analysis functions
    'run_analysis',
    'run_single_model_analysis', 
    'run_multi_model_analysis',
    'run_voting_analysis',
    
    # Analysis components
    'analyze_accuracy',
    'analyze_precision_rejection',
    'generate_cross_model_analysis',
    
    # Utilities
    'group_data_by_model',
    'sanitize_model_name',
    'generate_timestamp',
    'create_results_directory',
    
    # Visualization
    'create_single_model_plots',
    'create_model_accuracy_comparison',
    
    # Prompts
    'Level1PromptGenerator'
]

__version__ = "1.0.0"
__author__ = "L1 Analysis Team"
__description__ = "Level 1 validation analysis and uncertainty calibration framework"