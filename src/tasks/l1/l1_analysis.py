# src/uncertainty_calibration/l1_analysis.py
"""
Backward compatibility layer for l1_analysis.
Imports from the new refactored modules to maintain existing API.
"""

# Import all functions from the new modules to maintain backward compatibility
from .l1_core_analysis import (
    run_analysis,
    run_single_model_analysis,
    run_multi_model_analysis,
    analyze_accuracy,
    analyze_precision_rejection,
    run_analysis_legacy  # For extra backward compatibility
)

from .l1_utils import (
    group_data_by_model,
    sanitize_model_name,
    generate_timestamp,
    create_results_directory,
    save_single_model_metadata,
    save_multi_model_metadata,
    convert_calibration_points_to_dataframe,
    validate_calibration_data
)

from .l1_visualization import (
    create_single_model_plots,
    create_model_accuracy_comparison,
    create_uncertainty_distribution_comparison,
    create_precision_rejection_comparison
)

from .l1_multi_model import (
    generate_cross_model_analysis,
    create_cross_model_summary,
    perform_statistical_tests,
    create_model_rankings,
    analyze_uncertainty_correlations,
    calculate_uncertainty_calibration_score,
    calculate_model_agreement_matrix
)

from .l1_voting_analysis import (
    run_voting_analysis,
    analyze_voting_with_per_model_rejection,
    apply_per_model_rejection,
    voting_with_dual_mode,
    majority_vote_with_uncertainty_tiebreak,
    analyze_voting_model_contributions
)

# Re-export the main function for backward compatibility
__all__ = [
    # Core analysis functions
    'run_analysis',
    'run_single_model_analysis', 
    'run_multi_model_analysis',
    'analyze_accuracy',
    'analyze_precision_rejection',
    
    # Utility functions
    'group_data_by_model',
    'sanitize_model_name',
    'generate_timestamp',
    'create_results_directory',
    'convert_calibration_points_to_dataframe',
    
    # Visualization functions
    'create_single_model_plots',
    'create_model_accuracy_comparison',
    
    # Multi-model analysis
    'generate_cross_model_analysis',
    'create_cross_model_summary',
    'perform_statistical_tests',
    
    # Voting analysis (NEW)
    'run_voting_analysis',
    'analyze_voting_with_per_model_rejection',
    'voting_with_dual_mode',
    'majority_vote_with_uncertainty_tiebreak',
    
    # Backward compatibility
    'run_analysis_legacy'
]

# For extra backward compatibility, create an alias
def run_analysis_original(*args, **kwargs):
    """Alias for the original run_analysis function."""
    return run_analysis(*args, **kwargs)

# Print deprecation warning for direct imports from this module
import warnings

def _deprecated_import_warning():
    warnings.warn(
        "Importing from l1_analysis.py is deprecated. "
        "Please import from the specific modules: "
        "l1_core_analysis, l1_visualization, l1_multi_model, l1_voting_analysis, or l1_utils",
        DeprecationWarning,
        stacklevel=3
    )