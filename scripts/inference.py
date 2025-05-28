#!/usr/bin/env python3
# scripts/inference.py

"""
Apply trained calibration model to new uncertainty scores.
Demonstrates production usage of the calibration framework.
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.uncertainty_calibration import (
    LightGBMCalibrationTrainer,
    CalibrationFeatureEngineer,
    get_model_metadata,
    validate_model_id
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main inference function."""
    
    parser = argparse.ArgumentParser(description="Calibrate uncertainty scores using trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained calibration model (.lgb file)"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        help="CSV file with columns: raw_uncertainty, model_name, temperature"
    )
    parser.add_argument(
        "--output_csv", 
        type=str,
        help="Output CSV file for calibrated results"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Load trained model
    logger.info(f"Loading calibration model from {args.model_path}")
    
    if not Path(args.model_path).exists():
        logger.error(f"Model file not found: {args.model_path}")
        return 1
    
    try:
        trainer = LightGBMCalibrationTrainer()
        trainer.load_model(args.model_path)
        
        # Load feature engineer metadata
        metadata_path = Path(args.model_path).with_suffix('.metadata.joblib')
        if not metadata_path.exists():
            logger.error(f"Model metadata not found: {metadata_path}")
            return 1
            
        feature_engineer = CalibrationFeatureEngineer()
        # Note: In production, you'd need to load the fitted encoders
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    if args.interactive:
        # Interactive mode
        print("Interactive Uncertainty Calibration")
        print("="*40)
        
        while True:
            try:
                print("\nEnter uncertainty information (or 'quit' to exit):")
                
                raw_uncertainty = input("Raw uncertainty (0.0-1.0): ")
                if raw_uncertainty.lower() == 'quit':
                    break
                raw_uncertainty = float(raw_uncertainty)
                
                model_name = input("Model name (e.g., openai/gpt-4o): ")
                if not validate_model_id(model_name):
                    print(f"Warning: Unknown model {model_name}")
                
                temperature = input("Temperature (0.0-1.0): ")
                temperature = float(temperature)
                
                # Prepare features
                model_metadata = get_model_metadata(model_name)
                
                inference_data = {
                    'raw_uncertainty': [raw_uncertainty],
                    'model_name': [model_name],
                    'param_count': [model_metadata['param_count'] or 100.0],
                    'temperature': [temperature],
                    'level': [1],  # Default
                    'provider': [model_metadata['provider'] or 'unknown'],
                    'architecture': [model_metadata['architecture'] or 'transformer']
                }
                
                inference_df = pd.DataFrame(inference_data)
                
                # Apply feature engineering (simplified)
                inference_df['log_param_count'] = np.log10(inference_df['param_count'] + 1)
                inference_df['is_zero_temp'] = (inference_df['temperature'] == 0.0).astype(int)
                inference_df['temp_squared'] = inference_df['temperature'] ** 2
                
                # Simple encoding (in production, use fitted encoders)
                inference_df['model_name_encoded'] = 0  # Placeholder
                inference_df['provider_encoded'] = 0    # Placeholder
                inference_df['architecture_encoded'] = 0 # Placeholder
                
                # Core features only for this demo
                feature_cols = ['raw_uncertainty', 'param_count', 'temperature', 'log_param_count', 'is_zero_temp', 'temp_squared']
                available_features = [f for f in feature_cols if f in inference_df.columns]
                
                # Get calibrated confidence
                calibrated_confidence = trainer.predict_calibrated_confidence(inference_df[available_features])
                calibrated_uncertainty = 1 - calibrated_confidence[0]
                
                print(f"\nResults:")
                print(f"  Raw uncertainty: {raw_uncertainty:.4f}")
                print(f"  Calibrated confidence: {calibrated_confidence[0]:.4f}")
                print(f"  Calibrated uncertainty: {calibrated_uncertainty:.4f}")
                
                # Interpretation
                if calibrated_confidence[0] > 0.8:
                    interpretation = "High confidence - likely correct"
                elif calibrated_confidence[0] > 0.6:
                    interpretation = "Moderate confidence"
                elif calibrated_confidence[0] > 0.4:
                    interpretation = "Low confidence - uncertain"
                else:
                    interpretation = "Very low confidence - likely incorrect"
                
                print(f"  Interpretation: {interpretation}")
                
            except ValueError as e:
                print(f"Invalid input: {e}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.input_csv:
        # Batch processing mode
        logger.info(f"Processing batch from {args.input_csv}")
        
        if not Path(args.input_csv).exists():
            logger.error(f"Input file not found: {args.input_csv}")
            return 1
        
        try:
            # Load input data
            input_df = pd.read_csv(args.input_csv)
            
            # Validate required columns
            required_cols = ['raw_uncertainty', 'model_name', 'temperature']
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return 1
            
            # Process each row
            results = []
            
            for idx, row in input_df.iterrows():
                try:
                    # Get model metadata
                    model_metadata = get_model_metadata(row['model_name'])
                    
                    # Prepare features (simplified for demo)
                    features = {
                        'raw_uncertainty': row['raw_uncertainty'],
                        'param_count': model_metadata['param_count'] or 100.0,
                        'temperature': row['temperature'],
                        'log_param_count': np.log10((model_metadata['param_count'] or 100.0) + 1),
                        'is_zero_temp': 1 if row['temperature'] == 0.0 else 0,
                        'temp_squared': row['temperature'] ** 2
                    }
                    
                    feature_df = pd.DataFrame([features])
                    
                    # Get calibrated confidence
                    calibrated_confidence = trainer.predict_calibrated_confidence(feature_df)
                    
                    results.append({
                        'raw_uncertainty': row['raw_uncertainty'],
                        'model_name': row['model_name'],
                        'temperature': row['temperature'],
                        'calibrated_confidence': calibrated_confidence[0],
                        'calibrated_uncertainty': 1 - calibrated_confidence[0]
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to process row {idx}: {e}")
                    results.append({
                        'raw_uncertainty': row['raw_uncertainty'],
                        'model_name': row['model_name'],
                        'temperature': row['temperature'],
                        'calibrated_confidence': np.nan,
                        'calibrated_uncertainty': np.nan
                    })
            
            # Save results
            results_df = pd.DataFrame(results)
            
            if args.output_csv:
                results_df.to_csv(args.output_csv, index=False)
                logger.info(f"Results saved to {args.output_csv}")
            else:
                print(results_df)
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return 1
    
    else:
        print("Please specify either --interactive or --input_csv")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)