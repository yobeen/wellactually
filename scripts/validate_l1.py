#!/usr/bin/env python3
# scripts/validate_l1.py
"""
L1 validation script using existing uncertainty calibration infrastructure.
Fixed to use OmegaConf for configuration compatibility.
"""

import sys
import os
import yaml
import logging
from pathlib import Path
from omegaconf import OmegaConf

from src.uncertainty_calibration.data_collection import UncertaintyDataCollector
from src.utils.data_loader import load_l1_data
from src.uncertainty_calibration.l1_analysis import run_analysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file using OmegaConf."""
    config_path = "configs/uncertainty_calibration/llm.yaml"
    
    # Load with OmegaConf for both dict and attribute access
    config = OmegaConf.load(config_path)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config

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
    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return 1
    
    # Load Level 1 data
    print("\n1. Loading Level 1 data...")
    try:
        l1_data = load_l1_data()
        
        if len(l1_data) == 0:
            print("No Level 1 data found. Exiting.")
            return 1
            
        print(f"Loaded {len(l1_data)} Level 1 comparisons")
        
    except Exception as e:
        print(f"Failed to load L1 data: {e}")
        return 1
    
    # Initialize data collector with single model
    print("\n2. Initializing data collector...")
    test_models = ["openai/gpt-4o",
            "meta-llama/llama-4-maverick",
            "deepseek/deepseek-chat",
            "x-ai/grok-3-beta",
            "meta-llama/llama-3.1-405b-instruct",
            "qwen/qwq-32b-preview",
            "mistralai/mixtral-8x22b-instruct",
           # "deepseek/deepseek-coder",
            "google/gemma-3-27b-it"
        ] 
    
    try:
        collector = UncertaintyDataCollector(config, models_subset=test_models)
        print(f"Successfully initialized collector with models: {test_models}")
        
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
        import traceback
        traceback.print_exc()
        return 1
    
    # Run analysis
    print("\n4. Running analysis...")
    try:
        run_analysis(calibration_data_points)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nValidation complete!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)