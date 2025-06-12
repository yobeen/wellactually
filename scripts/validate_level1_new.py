#!/usr/bin/env python3
# scripts/validate_l1.py
"""
L1 validation script for individual model analysis.
Runs temperature sweep on gemma model with limited dataset.
"""

import sys
import os
import yaml
import logging
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv

from src.calibration.data_collection import UncertaintyDataCollector
from src.utils.data_loader import load_l1_data
from src.tasks.l1.l1_core_analysis import run_analysis
from src.tasks.l1.l1_voting_analysis import run_voting_analysis

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
    """Main validation function for individual model analysis."""
    print("L1 Prediction Validation - Individual Model Analysis")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
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
        
        # Limit to 30 random examples for debugging
        import random
        random.seed(42) 
        n_examples = 138
        if len(l1_data) > n_examples:
            l1_data = l1_data.sample(n=n_examples, random_state=42).reset_index(drop=True)
            print(f"Limited to {len(l1_data)} random examples for debugging (seed=42)")
        
    except Exception as e:
        print(f"Failed to load L1 data: {e}")
        return 1
    
    # Initialize data collector with multiple models for voting
    print("\n2. Initializing data collector...")
    test_models = [
            "openai/gpt-4.1",
            "openai/gpt-4o",
            "meta-llama/llama-4-maverick",
           # "mistralai/mixtral-8x22b-instruct",
            "google/gemma-3-27b-it",
            "x-ai/grok-3-beta",
            ] 
    
    try:
        collector = UncertaintyDataCollector(config, models_subset=test_models)
        print(f"Successfully initialized collector with models: {test_models}")
        
    except Exception as e:
        print(f"Failed to initialize data collector: {e}")
        return 1
    
    # Collect predictions
    temperatures = [0.4]
    print(f"\n4. Querying LLM predictions...")
    print(f"Models: {test_models}")
    print(f"Temperatures: {temperatures}")
    print(f"Processing {len(l1_data)} comparisons...")
    
    try:
        calibration_data_points = collector.collect_training_data(
            train_df=l1_data,
            temperatures=temperatures,
            max_samples_per_level=len(l1_data)  # Process selected L1 data
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
    
    # Run analysis based on choice
    try:
        shared_results_dir = None
        
        print("\n5a. Running individual model analysis...")
        individual_results_dir = run_analysis(calibration_data_points)
        shared_results_dir = individual_results_dir
        print(f"Individual analysis results saved to: {individual_results_dir}")
        
        print("\n5b. Running voting analysis...")
        # Use default rejection rates
        rejection_rates = None
        
        voting_results_dir = run_voting_analysis(
            calibration_data_points, 
            rejection_rates=rejection_rates,
            base_save_dir=str(shared_results_dir)
        )
        print(f"Voting analysis results saved to: {voting_results_dir}")
        if shared_results_dir is None:
            shared_results_dir = voting_results_dir
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nValidation complete!")
    
    # Summary of what was done
    print("\nSummary:")
    print("✓ Individual model analysis completed")
    print("✓ Voting analysis with per-model rejection completed")
    print(f"✓ Temperature sweep: {temperatures}")
    print(f"✓ Models: {test_models}")
    print(f"✓ Samples: {len(l1_data)} (random subset with seed=42)")
    
    if shared_results_dir:
        print(f"\n📁 All results saved to: {shared_results_dir}")
        print("   ├── {model_dirs}/     # Individual model results")
        print("   ├── comparison/      # Cross-model comparisons") 
        print("   └── voting/          # Voting analysis results")
    
    print("\nGenerated outputs:")
    print("- Accuracy analysis plots")
    print("- Precision-rejection curves") 
    print("- Uncertainty distribution analysis")
    print("- Voting accuracy curves (dual-mode evaluation)")
    print("- Voting efficiency analysis")
    print("- Model contribution statistics")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)