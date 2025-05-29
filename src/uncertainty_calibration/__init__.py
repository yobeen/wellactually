# src/uncertainty_calibration/__init__.py
"""
Uncertainty Calibration Package

LightGBM-based uncertainty calibration framework for aggregating
heterogeneous uncertainty measures from multiple LLMs with caching support.
"""

from .cache_manager import (
    CacheManager
)

from .model_metadata import (
    MODEL_PARAMS,
    MODEL_PROVIDERS, 
    MODEL_ARCHITECTURES,
    get_model_params,
    get_model_provider,
    get_model_architecture,
    get_model_metadata,
    validate_model_id,
    TEMPERATURE_SWEEP
)

from .response_parser import (
    ModelResponse,
    ResponseParser,
    parse_response,
    calculate_uncertainty
)

from .data_collection import (
    UncertaintyDataCollector,
    CalibrationDataPoint
)

from .feature_engineering import (
    CalibrationFeatureEngineer,
    prepare_calibration_features
)

from .lightgbm_trainer import (
    LightGBMCalibrationTrainer,
    train_calibration_model
)

from .evaluation import (
    CalibrationEvaluator,
    evaluate_calibration_quality
)

from .calibration_pipeline import (
    UncertaintyCalibrationPipeline,
    CalibrationPipelineConfig,
    create_pipeline_from_config
)

from .multi_model_engine import (
    MultiModelEngine
)

__version__ = "0.1.0"
__author__ = "Cross-Model Uncertainty Aggregation Team"

# Main pipeline entry point
def run_calibration_pipeline(config_path: str):
    """
    Run the complete uncertainty calibration pipeline.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Pipeline results dictionary
    """
    pipeline = create_pipeline_from_config(config_path)
    return pipeline.run_complete_pipeline()

# Convenience function for direct usage
def calibrate_uncertainties(raw_uncertainties, model_names, temperatures, model_path: str):
    """
    Apply trained calibration model to new uncertainties.
    
    Args:
        raw_uncertainties: List of raw uncertainty scores
        model_names: List of model identifiers
        temperatures: List of temperature values
        model_path: Path to trained calibration model
        
    Returns:
        Calibrated confidence scores
    """
    from .lightgbm_trainer import LightGBMCalibrationTrainer
    
    # Load trained model
    trainer = LightGBMCalibrationTrainer()
    trainer.load_model(model_path)
    
    # Prepare features and predict
    # This would need the full pipeline setup for proper feature engineering
    raise NotImplementedError("Direct calibration requires full pipeline - use run_calibration_pipeline instead")