#!/usr/bin/env python3
# scripts/validate_l1.py
"""
L1 validation script using existing uncertainty calibration infrastructure.
"""

import sys
import os
import yaml
import logging
from pathlib import Path

from src.uncertainty_calibration.data_collection import UncertaintyDataCollector
from src.utils.data_loader import load_l1_data
from src.uncertainty_calibration.l1_analysis import run_analysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    config_path = "configs/uncertainty_calibration/llm.yaml"
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to object for attribute access
    class Config:
        def __init__(self, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key.replace('-', '_'), value)
    
    return Config(config_dict)

def main():
    """Main validation function."""
    print("L1 Prediction Validation Analysis")
    print("=" * 40)
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key: export OPENROUTER_API_KEY=your_key_here")
        return 1
    
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    
    # Load Level 1 data
    print("\n1. Loading Level 1 data...")
    l1_data = load_l1_data()
    
    if len(l1_data) == 0:
        print("No Level 1 data found. Exiting.")
        return 1
    
    # Initialize data collector with single model
    print("\n2. Initializing data collector...")
    test_models = ["openai/gpt-4o"]  # Single model only
    
    try:
        collector = UncertaintyDataCollector(config, models_subset=test_models)
    except Exception as e:
        print(f"Failed to initialize data collector: {e}")
        return 1
    
    # Collect predictions with temperature 0.7
    print("\n3. Querying LLM predictions...")
    print(f"Model: {test_models[0]}")
    print(f"Temperature: 0.7")
    print(f"Processing {len(l1_data)} comparisons...")
    
    try:
        calibration_data_points = collector.collect_training_data(
            train_df=l1_data,
            temperatures=[0.7],
            max_samples_per_level=len(l1_data)  # Process all L1 data
        )
        
        if not calibration_data_points:
            print("No calibration data points collected. Exiting.")
            return 1
            
        print(f"Successfully collected {len(calibration_data_points)} data points")
        
    except Exception as e:
        print(f"Error during data collection: {e}")
        return 1
    
    # Run analysis
    print("\n4. Running analysis...")
    try:
        run_analysis(calibration_data_points)
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    print("\nValidation complete!")
    return 0

if __name__ == "__main__":
    main()