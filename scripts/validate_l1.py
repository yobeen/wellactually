#!/usr/bin/env python3
# scripts/validate_l1.py
"""
Enhanced L1 validation script with voting analysis support.
Fixed to use OmegaConf for configuration compatibility.
"""

import sys
import os
import yaml
import logging
from pathlib import Path
from omegaconf import OmegaConf

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
    """Main validation function with voting analysis support."""
    print("L1 Prediction Validation Analysis with Voting")
    print("=" * 50)
    
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
    
    # Initialize data collector with multiple models for voting
    print("\n2. Initializing data collector...")
    test_models = [
            "openai/gpt-4o",
            "meta-llama/llama-4-maverick",
            "x-ai/grok-3-beta",
            "mistralai/mixtral-8x22b-instruct",
            "google/gemma-3-27b-it",
            "deepseek/deepseek-chat-v3-0324"
        ] 
    # test_models = [
    #         #"deepseek/deepseek-r1-0528",
    #         #"qwen/qwen3-235b-a22b",
    #     ] 
    
    try:
        collector = UncertaintyDataCollector(config, models_subset=test_models)
        print(f"Successfully initialized collector with models: {test_models}")
        
    except Exception as e:
        print(f"Failed to initialize data collector: {e}")
        return 1
    
    # Collect predictions
    print(f"\n4. Querying LLM predictions...")
    print(f"Models: {test_models}")
    print(f"Temperature: 0.7")
    print(f"Processing {len(l1_data)} comparisons...")
    
    try:
        calibration_data_points = collector.collect_training_data(
            train_df=l1_data,
            temperatures=[0.2],
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
    
    # Run analysis based on choice
    try:
        shared_results_dir = None
        
        print("\n5a. Running individual model analysis...")
        individual_results_dir = run_analysis(calibration_data_points)
        shared_results_dir = individual_results_dir
        print(f"Individual analysis results saved to: {individual_results_dir}")
        
        print("\n5b. Running voting analysis...")
        # Ask for rejection rates
        print("2. Custom range")
        
        rejection_rates = None  # Use default
        
        voting_results_dir = run_voting_analysis(
            calibration_data_points, 
            rejection_rates=rejection_rates
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