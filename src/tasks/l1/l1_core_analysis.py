# src/uncertainty_calibration/l1_core_analysis.py
"""
Core L1 validation analysis functions.
Main orchestration and basic accuracy calculations for single and multi-model analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from .l1_utils import (
    group_data_by_model,
    group_data_by_model_and_temperature,
    sanitize_model_name,
    generate_timestamp,
    create_results_directory,
    create_temperature_results_directory,
    save_single_model_metadata,
    save_multi_model_metadata,
    convert_calibration_points_to_dataframe,
    validate_calibration_data
)
from .l1_visualization import create_single_model_plots
from .l1_multi_model import generate_cross_model_analysis

logger = logging.getLogger(__name__)

def run_analysis(calibration_data_points, base_save_dir: Optional[str] = None):
    """
    Main analysis function with automatic single/multi-model detection.
    Now supports temperature-specific result organization.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        base_save_dir: Optional base directory override
        
    Returns:
        Path to results directory
    """
    if not calibration_data_points:
        print("No calibration data points provided!")
        return None
    
    # Validate input data
    validation_result = validate_calibration_data(calibration_data_points)
    if not validation_result['valid']:
        print(f"Data validation failed: {validation_result['error']}")
        return None
    
    # Check if multiple temperatures are present
    temperatures = set(getattr(dp, 'temperature', 0.0) for dp in calibration_data_points)
    
    if len(temperatures) > 1:
        print(f"Multiple temperatures detected: {sorted(temperatures)}")
        print("Running temperature-specific analysis...")
        return run_temperature_sweep_analysis(calibration_data_points, base_save_dir)
    else:
        # Original behavior for single temperature
        models_data = group_data_by_model(calibration_data_points)
        
        if len(models_data) == 1:
            return run_single_model_analysis(calibration_data_points, base_save_dir)
        else:
            return run_multi_model_analysis(calibration_data_points, base_save_dir)

def run_single_model_analysis(calibration_data_points, base_save_dir: Optional[str] = None) -> Path:
    """
    Run analysis for a single model.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        base_save_dir: Optional base directory override
        
    Returns:
        Path to results directory
    """
    print("\n" + "="*50)
    print("L1 VALIDATION ANALYSIS RESULTS (SINGLE MODEL)")
    print("="*50)
    
    # Extract model info
    models_data = group_data_by_model(calibration_data_points)
    model_id = list(models_data.keys())[0]
    
    # Create directory structure
    timestamp = generate_timestamp()
    base_dir = create_results_directory(timestamp, [model_id], base_save_dir)
    model_dir = base_dir / sanitize_model_name(model_id)
    
    # Run analysis
    overall_acc, class_acc, results_df = analyze_accuracy(calibration_data_points, model_dir)
    precision_data = analyze_precision_rejection(calibration_data_points, model_dir)
    create_single_model_plots(results_df, precision_data, overall_acc, model_dir, model_id)
    
    # Save metadata
    save_single_model_metadata(base_dir, model_id, calibration_data_points, timestamp)
    
    print(f"\nSingle model analysis complete!")
    print(f"Results saved to: {base_dir}")
    
    return base_dir

def run_multi_model_analysis(calibration_data_points, base_save_dir: Optional[str] = None) -> Path:
    """
    Run analysis for multiple models with cross-model comparisons.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        base_save_dir: Optional base directory override
        
    Returns:
        Path to results directory
    """
    print("\n" + "="*60)
    print("L1 VALIDATION ANALYSIS RESULTS (MULTI-MODEL)")
    print("="*60)
    
    # Group data by model
    models_data = group_data_by_model(calibration_data_points)
    model_ids = list(models_data.keys())
    
    print(f"Analyzing {len(model_ids)} models: {', '.join(model_ids)}")
    
    # Create directory structure
    timestamp = generate_timestamp()
    base_dir = create_results_directory(timestamp, model_ids, base_save_dir)
    
    # Individual model analysis
    individual_results = {}
    
    for model_id, model_data_points in models_data.items():
        print(f"\nAnalyzing model: {model_id}")
        
        model_dir = base_dir / sanitize_model_name(model_id)
        
        # Run individual analysis
        overall_acc, class_acc, results_df = analyze_accuracy(model_data_points, model_dir)
        precision_data = analyze_precision_rejection(model_data_points, model_dir)
        create_single_model_plots(results_df, precision_data, overall_acc, model_dir, model_id)
        
        # Store results for comparison
        individual_results[model_id] = {
            'overall_accuracy': overall_acc,
            'class_accuracy': class_acc,
            'results_df': results_df,
            'precision_data': precision_data,
            'data_points': model_data_points
        }
        
        print(f"  Overall accuracy: {overall_acc:.3f}")
    
    # Cross-model comparison analysis
    comparison_dir = base_dir / "comparison"
    comparison_results = generate_cross_model_analysis(individual_results, comparison_dir)
    
    # Save metadata
    save_multi_model_metadata(base_dir, models_data, timestamp, comparison_results)
    
    print(f"\nMulti-model analysis complete!")
    print(f"Results saved to: {base_dir}")
    print(f"Cross-model comparisons saved to: {comparison_dir}")
    
    return base_dir

def run_temperature_sweep_analysis(calibration_data_points, base_save_dir: Optional[str] = None) -> Path:
    """
    Run analysis with results organized by model and temperature.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        base_save_dir: Optional base directory override
        
    Returns:
        Path to results directory
    """
    print("\n" + "="*60)
    print("L1 VALIDATION ANALYSIS RESULTS (TEMPERATURE SWEEP)")
    print("="*60)
    
    # Group data by model and temperature
    model_temps_data = group_data_by_model_and_temperature(calibration_data_points)
    
    # Create directory structure
    timestamp = generate_timestamp()
    base_dir = create_temperature_results_directory(timestamp, model_temps_data, base_save_dir)
    
    # Analysis summary
    total_models = len(model_temps_data)
    total_temps = sum(len(temp_data) for temp_data in model_temps_data.values())
    
    print(f"Processing {total_models} model(s) across {total_temps} temperature combinations...")
    
    # Process each model and temperature combination
    for model_id, temp_data in model_temps_data.items():
        clean_model_name = sanitize_model_name(model_id)
        model_dir = base_dir / clean_model_name
        
        print(f"\nModel: {model_id}")
        print(f"Temperatures: {sorted(temp_data.keys())}")
        
        for temperature, temp_calibration_points in temp_data.items():
            temp_str = f"temp_{temperature:.1f}".replace('.', '_')
            temp_dir = model_dir / temp_str
            
            print(f"  Processing temperature {temperature}...")
            
            # Run analysis for this temperature
            overall_acc, class_acc, results_df = analyze_accuracy(temp_calibration_points, temp_dir)
            precision_data = analyze_precision_rejection(temp_calibration_points, temp_dir)
            create_single_model_plots(results_df, precision_data, overall_acc, temp_dir, 
                                     f"{model_id} (T={temperature})")
            
            # Save temperature-specific metadata
            temp_metadata = {
                "timestamp": timestamp,
                "analysis_type": "temperature_specific",
                "model_id": model_id,
                "temperature": temperature,
                "total_data_points": len(temp_calibration_points),
                "overall_accuracy": overall_acc,
                "class_accuracies": class_acc
            }
            
            import json
            with open(temp_dir / "metadata.json", 'w') as f:
                json.dump(temp_metadata, f, indent=2)
            
            print(f"    Overall accuracy: {overall_acc:.3f}")
            print(f"    Results saved to: {temp_dir}")
    
    # Save overall metadata
    overall_metadata = {
        "timestamp": timestamp,
        "analysis_type": "temperature_sweep",
        "models_analyzed": list(model_temps_data.keys()),
        "temperatures_tested": {
            model_id: sorted(temp_data.keys()) 
            for model_id, temp_data in model_temps_data.items()
        },
        "total_combinations": sum(len(temp_data) for temp_data in model_temps_data.values())
    }
    
    import json
    with open(base_dir / "sweep_metadata.json", 'w') as f:
        json.dump(overall_metadata, f, indent=2)
    
    print(f"\nTemperature sweep analysis complete!")
    print(f"Results saved to: {base_dir}")
    print("Directory structure:")
    print(f"  results/{timestamp}/")
    for model_id, temp_data in model_temps_data.items():
        clean_name = sanitize_model_name(model_id)
        print(f"    ├── {clean_name}/")
        for temperature in sorted(temp_data.keys()):
            temp_str = f"temp_{temperature:.1f}".replace('.', '_')
            print(f"    │   ├── {temp_str}/")
            print(f"    │   │   ├── l1_validation_results.csv")
            print(f"    │   │   ├── l1_precision_rejection.csv")
            print(f"    │   │   ├── plots/")
            print(f"    │   │   └── metadata.json")
    
    return base_dir

def analyze_accuracy(calibration_data_points, save_dir: Path) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """
    Calculate overall and per-class accuracy from CalibrationDataPoint objects.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        save_dir: Directory to save results
        
    Returns:
        tuple: (overall_accuracy, class_accuracies_dict, results_dataframe)
    """
    if not calibration_data_points:
        return 0.0, {}, pd.DataFrame()
    
    # Convert to DataFrame for easier analysis
    df = convert_calibration_points_to_dataframe(calibration_data_points)
    
    # Overall accuracy
    overall_acc = df['is_correct'].mean()
    
    # Per-class accuracy
    class_acc = {}
    for class_name in ['A', 'B', 'Equal']:
        class_data = df[df['human_label'] == class_name]
        if len(class_data) > 0:
            class_acc[class_name] = class_data['is_correct'].mean()
        else:
            class_acc[class_name] = 0.0
    
    # Save detailed results
    df.to_csv(save_dir / "l1_validation_results.csv", index=False)
    
    print(f"Accuracy Analysis:")
    print(f"  Overall Accuracy: {overall_acc:.3f}")
    print("  Per-class Accuracy:")
    for class_name, acc in class_acc.items():
        count = len(df[df['human_label'] == class_name])
        print(f"    {class_name}: {acc:.3f} (n={count})")
    
    return overall_acc, class_acc, df

def analyze_precision_rejection(calibration_data_points, save_dir: Path) -> pd.DataFrame:
    """
    Generate precision-rejection curve data with uncertainty thresholds.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        save_dir: Directory to save results
        
    Returns:
        DataFrame with rejection_rate, accuracy, samples_remaining, uncertainty_threshold columns
    """
    if not calibration_data_points:
        return pd.DataFrame()
    
    # Convert to DataFrame and sort by uncertainty descending
    df = convert_calibration_points_to_dataframe(calibration_data_points)
    sorted_df = df.sort_values('raw_uncertainty', ascending=False)
    
    rejection_rates = np.arange(0, 95, 5)  # 0%, 5%, 10%, ..., 90%
    precision_data = []
    thresholds = {}
    
    for rate in rejection_rates:
        # Calculate number of samples to remove
        n_total = len(sorted_df)
        n_remove = int(n_total * rate / 100)
        
        # Keep remaining samples
        remaining_df = sorted_df.iloc[n_remove:]
        
        if len(remaining_df) > 0:
            accuracy = remaining_df['is_correct'].mean()
        else:
            accuracy = 0.0
        
        # Calculate uncertainty threshold
        if n_remove == 0:
            threshold = float('inf')  # Keep all samples
        elif n_remove >= n_total:
            threshold = 0.0  # Reject all samples
        else:
            # Threshold is the uncertainty of the last rejected sample
            threshold = sorted_df.iloc[n_remove - 1]['raw_uncertainty']
        
        precision_data.append({
            'rejection_rate': rate,
            'accuracy': accuracy,
            'samples_remaining': len(remaining_df),
            'uncertainty_threshold': threshold
        })
        
        thresholds[rate] = threshold
    
    precision_df = pd.DataFrame(precision_data)
    
    # Display thresholds during run
    model_id = getattr(calibration_data_points[0], 'model_id', 'unknown_model') if calibration_data_points else 'unknown'
    print(f"\nUncertainty Thresholds for {model_id}:")
    print("-" * 50)
    for rate in rejection_rates:
        threshold = thresholds[rate]
        if threshold == float('inf'):
            threshold_str = "∞"
        else:
            threshold_str = f"{threshold:.8f}"
        print(f"  {rate:2d}%: {threshold_str}")
    
    # Save results
    precision_df.to_csv(save_dir / "l1_precision_rejection.csv", index=False)
    
    return precision_df

# Backward compatibility function
def run_analysis_legacy(calibration_data_points):
    """
    Legacy function for backward compatibility.
    Uses the original behavior but with new structured output.
    """
    return run_analysis(calibration_data_points)