#!/usr/bin/env python3
# scripts/collect_responses.py
"""
Script to collect LLM responses for uncertainty calibration training.
This is the critical Phase 1 implementation.
"""
import argparse
import sys
import logging
import yaml
from pathlib import Path
import pandas as pd
import os

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uncertainty_calibration.data_collection import UncertaintyDataCollector
from uncertainty_calibration.feature_engineering import TwoModelFeatureEngineer
from uncertainty_calibration.two_model_trainer import TwoModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main data collection function."""

    parser = argparse.ArgumentParser(description="Collect LLM responses for uncertainty calibration")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/uncertainty_calibration/llm.yaml",
        help="Path to configuration file")
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/raw/train.csv",
        help="Path to training data CSV")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/uncertainty_calibration",
        help="Output directory for collected data")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum samples to process (for testing)")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to use")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Convert to object for attribute access
    class Config:
        def __init__(self, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, self._sanitize_key(key), value)
        def _sanitize_key(self, key):
            # Replace hyphens with underscores for valid attribute names
            return key.replace('-', '_')

    config = Config(config)

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set")
        logger.info("Please set your OpenRouter API key: export OPENROUTER_API_KEY=your_key_here")
        return 1

    # Load training data
    logger.info(f"Loading training data from {args.train_data}")
    if not Path(args.train_data).exists():
        logger.error(f"Training data not found: {args.train_data}")
        return 1
    train_df = pd.read_csv(args.train_data)
    logger.info(f"Loaded {len(train_df)} training samples")

    # Initialize data collector
    models_to_use = args.models if args.models else None
    logger.info("Initializing uncertainty data collector...")
    try:
        collector = UncertaintyDataCollector(config, models_to_use)
    except Exception as e:
        logger.error(f"Failed to initialize data collector: {e}")
        return 1

    # Collect responses
    logger.info("Starting LLM data collection...")
    logger.info(f"Processing up to {args.max_samples} samples per level")
    try:
        data_points = collector.collect_training_data(
            train_df=train_df,
            temperatures=config.temperature_sweep.temperatures,
            max_samples_per_level=args.max_samples
        )

        if not data_points:
            logger.error("No data points collected")
            return 1

        logger.info(f"Successfully collected {len(data_points)} data points")
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return 1

    # Save collected data
    collected_data_path = output_dir / "collected_responses.csv"
    logger.info(f"Saving collected data to {collected_data_path}")
    try:
        collected_df = collector.save_collected_data(data_points, str(collected_data_path))
        logger.info(f"Saved {len(collected_df)} responses")

        # Print summary statistics
        print("\n" + "="*60)
        print("DATA COLLECTION SUMMARY")
        print("="*60)
        print(f"Total responses: {len(collected_df)}")
        print(f"Models used: {collected_df['model_id'].nunique()}")
        print(f"Temperatures: {sorted(collected_df['temperature'].unique())}")
        print(f"Levels: {sorted(collected_df['level'].unique())}")
        print(f"Success rate: {(collected_df['is_correct'].sum() / len(collected_df)):.2%}")
        print(f"Average uncertainty: {collected_df['raw_uncertainty'].mean():.3f}")

        # Per-model summary
        print("\nPer-model summary:")
        model_summary = collected_df.groupby('model_id').agg({
            'raw_uncertainty': ['count', 'mean'],
            'is_correct': 'mean'
        }).round(3)
        print(model_summary)
    except Exception as e:
        logger.error(f"Failed to save collected data: {e}")
        return 1

    # Prepare features for training
    logger.info("Preparing features for two-model training...")
    try:
        feature_engineer = TwoModelFeatureEngineer()

        # Prepare choice features
        choice_df = feature_engineer.prepare_choice_features(data_points)
        choice_features_path = output_dir / "choice_features.csv"
        choice_df.to_csv(choice_features_path, index=False)
        logger.info(f"Saved choice features to {choice_features_path}")

        # Prepare confidence features
        confidence_df = feature_engineer.prepare_confidence_features(data_points)
        confidence_features_path = output_dir / "confidence_features.csv"
        confidence_df.to_csv(confidence_features_path, index=False)
        logger.info(f"Saved confidence features to {confidence_features_path}")

        print(f"\nFeature preparation completed:")
        print(f"Choice features: {len(choice_df)} rows, {len(choice_df.columns)} columns")
        print(f"Confidence features: {len(confidence_df)} rows, {len(confidence_df.columns)} columns")
    except Exception as e:
        logger.error(f"Feature preparation failed: {e}")
        return 1

    # Train models (optional)
    train_models = input("\nTrain models now? (y/n): ").lower().startswith('y')
    if train_models:
        logger.info("Training two-model system...")

        try:
            trainer = TwoModelTrainer(config)

            # Split data
            choice_train, choice_val = feature_engineer.create_train_val_split(choice_df)
            conf_train, conf_val = feature_engineer.create_train_val_split(confidence_df)

            # Train choice model
            choice_features = feature_engineer.get_choice_training_features(choice_train)
            trainer.train_choice_model(choice_train, choice_val, choice_features)

            # Train confidence model
            conf_features = feature_engineer.get_confidence_training_features(conf_train)
            trainer.train_confidence_model(conf_train, conf_val, conf_features)

            # Save models
            model_dir = output_dir / "models"
            trainer.save_models(str(model_dir))

            print(f"\nModels trained and saved to {model_dir}")
            print("Training history:")
            for model_name, history in trainer.training_history.items():
                print(f"  {model_name}: {history}")

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return 1

    logger.info("Data collection and processing completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)