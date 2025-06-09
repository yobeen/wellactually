# src/uncertainty_calibration/l1_utils.py
"""
Utility functions for L1 validation analysis.
Shared helper functions for data processing, file operations, and metadata management.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def group_data_by_model(calibration_data_points) -> Dict[str, List]:
    """
    Group calibration data points by model ID.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        
    Returns:
        Dictionary mapping model_id to list of data points
    """
    models_data = defaultdict(list)
    
    for dp in calibration_data_points:
        model_id = getattr(dp, 'model_id', 'unknown_model')
        models_data[model_id].append(dp)
    
    return dict(models_data)

def sanitize_model_name(model_id: str) -> str:
    """
    Convert model ID to filesystem-safe name.
    
    Args:
        model_id: Original model identifier
        
    Returns:
        Filesystem-safe model name
    """
    return model_id.replace('/', '-').replace(':', '-').replace('@', '-')

def generate_timestamp() -> str:
    """Generate filesystem-safe timestamp."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def create_results_directory(timestamp: str, model_names: List[str], 
                           base_save_dir: Optional[str] = None) -> Path:
    """
    Create structured results directory.
    
    Args:
        timestamp: Timestamp string
        model_names: List of model names for subdirectories
        base_save_dir: Optional base directory override
        
    Returns:
        Path to base results directory
    """
    if base_save_dir:
        base_dir = Path(base_save_dir) / timestamp
    else:
        base_dir = Path("results") / timestamp
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model subdirectories
    for model_name in model_names:
        clean_name = sanitize_model_name(model_name)
        model_dir = base_dir / clean_name
        model_dir.mkdir(exist_ok=True)
    
    # Create comparison directory if multiple models
    if len(model_names) > 1:
        comparison_dir = base_dir / "comparison"
        comparison_dir.mkdir(exist_ok=True)
    
    return base_dir

def save_single_model_metadata(base_dir: Path, model_id: str, 
                             calibration_data_points: List, timestamp: str):
    """Save metadata for single model analysis."""
    metadata = {
        "timestamp": timestamp,
        "analysis_type": "single_model",
        "model_analyzed": model_id,
        "total_data_points": len(calibration_data_points),
        "temperature_used": getattr(calibration_data_points[0], 'temperature', 'unknown') if calibration_data_points else 'unknown'
    }
    
    with open(base_dir / "run_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

def save_multi_model_metadata(base_dir: Path, models_data: Dict, 
                            timestamp: str, comparison_results: Dict):
    """Save metadata for multi-model analysis."""
    metadata = {
        "timestamp": timestamp,
        "analysis_type": "multi_model",
        "models_analyzed": list(models_data.keys()),
        "total_data_points": sum(len(data_points) for data_points in models_data.values()),
        "data_points_per_model": {model: len(data_points) for model, data_points in models_data.items()},
        "best_model_by_accuracy": comparison_results.get('best_model_overall', 'unknown'),
        "model_count": len(models_data),
        "temperature_used": getattr(list(models_data.values())[0][0], 'temperature', 'unknown') if models_data else 'unknown'
    }
    
    # Save run metadata
    with open(base_dir / "run_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save analysis summary
    analysis_summary = {
        "overall_summary": {
            "best_model_by_accuracy": comparison_results.get('best_model_overall', 'unknown'),
            "total_models_compared": comparison_results.get('model_count', 0),
            "total_statistical_comparisons": comparison_results.get('total_comparisons', 0)
        },
        "comparison_results": comparison_results
    }
    
    with open(base_dir / "analysis_summary.json", 'w') as f:
        json.dump(analysis_summary, f, indent=2, default=str)

def save_voting_metadata(base_dir: Path, models_data: Dict, timestamp: str, 
                        voting_results: Dict, rejection_rates: List[float]):
    """Save metadata for voting analysis."""
    metadata = {
        "timestamp": timestamp,
        "analysis_type": "voting_analysis",
        "models_analyzed": list(models_data.keys()),
        "total_data_points": sum(len(data_points) for data_points in models_data.values()),
        "data_points_per_model": {model: len(data_points) for model, data_points in models_data.items()},
        "rejection_rates_tested": rejection_rates,
        "voting_strategy": "majority_with_uncertainty_tiebreak",
        "evaluation_modes": ["penalize_missing", "exclude_missing"],
        "model_count": len(models_data)
    }
    
    # Save run metadata
    with open(base_dir / "run_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save voting analysis summary
    if voting_results:
        # Get best performing rejection rate for each mode
        mode1_best = max(voting_results, key=lambda x: x['mode1_accuracy'])
        mode2_best = max(voting_results, key=lambda x: x['mode2_accuracy'])
        
        voting_summary = {
            "best_mode1_performance": {
                "rejection_rate": mode1_best['rejection_rate'],
                "accuracy": mode1_best['mode1_accuracy'],
                "samples_remaining": mode1_best['samples_remaining']
            },
            "best_mode2_performance": {
                "rejection_rate": mode2_best['rejection_rate'], 
                "accuracy": mode2_best['mode2_accuracy'],
                "samples_remaining": mode2_best['samples_remaining']
            },
            "baseline_performance": {
                "rejection_rate": 0.0,
                "mode1_accuracy": voting_results[0]['mode1_accuracy'],
                "mode2_accuracy": voting_results[0]['mode2_accuracy'],
                "samples_remaining": voting_results[0]['samples_remaining']
            }
        }
        
        with open(base_dir / "voting_summary.json", 'w') as f:
            json.dump(voting_summary, f, indent=2, default=str)

def convert_calibration_points_to_dataframe(calibration_data_points) -> pd.DataFrame:
    """
    Convert CalibrationDataPoint objects to pandas DataFrame for analysis.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        
    Returns:
        DataFrame with analysis-ready data
    """
    if not calibration_data_points:
        return pd.DataFrame()
    
    data = []
    for dp in calibration_data_points:
        # Convert human choice back to A/B/Equal format
        if dp.human_multiplier <= 1.2:
            human_label = "Equal"
        elif dp.human_choice == 1.0:
            human_label = "A"
        elif dp.human_choice == 2.0:
            human_label = "B"
        else:
            human_label = "Equal"
            
        data.append({
            'question_id': dp.question_id,
            'model_id': dp.model_id,
            'human_label': human_label,
            'model_prediction': dp.model_prediction,
            'is_correct': dp.is_correct,
            'raw_uncertainty': dp.raw_uncertainty,
            'human_multiplier': dp.human_multiplier,
            'temperature': dp.temperature,
            'param_count': dp.param_count
        })
    
    return pd.DataFrame(data)

def validate_calibration_data(calibration_data_points) -> Dict[str, Any]:
    """
    Validate calibration data and return summary statistics.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        
    Returns:
        Dictionary with validation results and statistics
    """
    if not calibration_data_points:
        return {
            "valid": False,
            "error": "No calibration data points provided",
            "stats": {}
        }
    
    df = convert_calibration_points_to_dataframe(calibration_data_points)
    
    validation_result = {
        "valid": True,
        "error": None,
        "stats": {
            "total_points": len(df),
            "unique_models": df['model_id'].nunique(),
            "unique_questions": df['question_id'].nunique(),
            "model_list": df['model_id'].unique().tolist(),
            "accuracy_distribution": df.groupby('model_id')['is_correct'].mean().to_dict(),
            "uncertainty_stats": {
                "mean": df['raw_uncertainty'].mean(),
                "std": df['raw_uncertainty'].std(),
                "min": df['raw_uncertainty'].min(),
                "max": df['raw_uncertainty'].max()
            },
            "human_label_distribution": df['human_label'].value_counts().to_dict(),
            "temperature_distribution": df['temperature'].value_counts().to_dict()
        }
    }
    
    return validation_result