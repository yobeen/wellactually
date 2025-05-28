# src/uncertainty_calibration/__init__.py
"""
LightGBM Uncertainty Calibration Framework

A framework for calibrating uncertainty estimates from heterogeneous LLMs 
using gradient boosting to learn the mapping from raw uncertainty + metadata 
to meaningful confidence scores.
"""

from .calibration_pipeline import UncertaintyCalibrationPipeline
from .lightgbm_trainer import LightGBMCalibrationTrainer
from .feature_engineering import CalibrationFeatureEngineer
from .data_collection import CalibrationDataCollector
from .evaluation import CalibrationEvaluator
from .model_metadata import get_model_params, get_model_category

__version__ = "0.1.0"

__all__ = [
    "UncertaintyCalibrationPipeline",
    "LightGBMCalibrationTrainer", 
    "CalibrationFeatureEngineer",
    "CalibrationDataCollector",
    "CalibrationEvaluator",
    "get_model_params",
    "get_model_category"
]