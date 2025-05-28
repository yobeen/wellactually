# scripts/train.py
#!/usr/bin/env python3
"""
Complete training pipeline for LightGBM uncertainty calibration.
"""

import argparse
import sys
import logging
from pathlib import Path
import yaml
import pandas as pd

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.uncertainty_calibration import (
    UncertaintyCalibrationPipeline,
    CalibrationPipelineConfig,
    create_pipeline_from_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('uncertainty_calibration.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description="Train LightGBM uncertainty calibration model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/uncertainty_calibration/llm.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="train.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to use (overrides config)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum samples per level (overrides config)"
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run quick test with minimal data"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        return 1
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Override config with command line arguments
    if args.models:
        logger.info(f"Overriding models with: {args.models}")
        # Validate models exist in config
        all_models = {**config['models']['primary_models'], **config['models']['secondary_models']}
        valid_models = [m for m in args.models if m in all_models.values()]
        if not valid_models:
            logger.error(f"No valid models found in: {args.models}")
            return 1
        config['models']['primary_models'] = {f"model_{i}": m for i, m in enumerate(valid_models)}
    
    if args.max_samples:
        logger.info(f"Overriding max samples per level: {args.max_samples}")
        config['data_collection']['max_samples_per_level'] = args.max_samples
    
    if args.quick_test:
        logger.info("Running in quick test mode")
        config['data_collection']['max_samples_per_level'] = 5
        config['temperature_sweep']['temperatures'] = [0.0, 0.5]
        config['models']['primary_models'] = dict(list(config['models']['primary_models'].items())[:2])
        config['lightgbm']['training']['num_boost_round'] = 10
        config['lightgbm']['training']['early_stopping_rounds'] = 5
    
    # Verify training data exists
    if not Path(args.train_data).exists():
        logger.error(f"Training data not found: {args.train_data}")
        return 1
    
    # Create pipeline configuration
    pipeline_config = CalibrationPipelineConfig(
        train_data_path=args.train_data,
        output_dir=config['output_dir'],
        model_output_dir=config['model_output_dir'],
        models_to_use=list(config['models']['primary_models'].values()),
        temperatures=config['temperature_sweep']['temperatures'],
        max_samples_per_level=config['data_collection']['max_samples_per_level'],
        validation_fraction=config['lightgbm']['training']['validation_fraction'],
        cross_validation=config['evaluation']['cross_validation']['enabled'],
        cv_folds=config['evaluation']['cross_validation']['folds'],
        max_ece=config['quality_thresholds']['max_ece'],
        min_roc_auc=config['quality_thresholds']['min_roc_auc']
    )
    
    # Initialize and run pipeline
    logger.info("Initializing uncertainty calibration pipeline...")
    
    try:
        pipeline = UncertaintyCalibrationPipeline(pipeline_config, config)
        
        logger.info("Running complete calibration pipeline...")
        results = pipeline.run_complete_pipeline()
        
        # Print results summary
        print("\n" + "="*60)
        print("UNCERTAINTY CALIBRATION RESULTS")
        print("="*60)
        
        if results['status'] == 'success':
            print(f"✅ Training completed successfully!")
            print(f"\nData Collection:")
            print(f"  - Total data points: {results['data_collection']['total_data_points']}")
            print(f"  - Models used: {len(results['data_collection']['models_used'])}")
            print(f"  - Temperatures: {results['data_collection']['temperatures']}")
            
            print(f"\nFeature Engineering:")
            print(f"  - Total features: {results['feature_engineering']['total_features']}")
            print(f"  - Training samples: {results['feature_engineering']['training_samples']}")
            print(f"  - Validation samples: {results['feature_engineering']['validation_samples']}")
            
            print(f"\nModel Training:")
            print(f"  - Best iteration: {results['training']['best_iteration']}")
            print(f"  - Best score: {results['training']['best_score']:.4f}")
            
            print(f"\nCalibration Quality:")
            eval_results = results['evaluation']['overall_metrics']
            print(f"  - Expected Calibration Error: {eval_results['ECE']:.4f}")
            print(f"  - ROC AUC: {eval_results['ROC_AUC']:.4f}")
            print(f"  - Brier Score: {eval_results['Brier_Score']:.4f}")
            print(f"  - Log Loss: {eval_results['Log_Loss']:.4f}")
            
            quality_check = results['quality_check']
            print(f"\nQuality Thresholds:")
            print(f"  - ECE threshold met: {'✅' if quality_check['ece_threshold'] else '❌'}")
            print(f"  - ROC AUC threshold met: {'✅' if quality_check['roc_auc_threshold'] else '❌'}")
            print(f"  - Overall quality: {'✅' if quality_check['overall_quality'] else '❌'}")
            
            print(f"\nOutputs saved to:")
            print(f"  - Data: {pipeline_config.output_dir}")
            print(f"  - Model: {pipeline_config.model_output_dir}")
            
        else:
            print(f"❌ Training failed: {results['error']}")
            return 1
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1
    
    logger.info("Uncertainty calibration training completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)