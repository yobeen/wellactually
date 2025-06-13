#!/usr/bin/env python3
"""
Level 3 validation script for dependency comparison perplexity analysis.
Randomly selects parent repos and their dependencies from test.csv for L3 comparisons.
Uses existing L3 comparison handler which already calculates choice_uncertainty.
"""

import sys
import os
import pandas as pd
import random
import numpy as np
import logging
import asyncio
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple

from src.api.core.llm_orchestrator import LLMOrchestrator
from src.api.comparison.comparison_handler import ComparisonHandler
from src.api.core.requests import ComparisonRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file using OmegaConf."""
    config_path = "configs/uncertainty_calibration/llm.yaml"
    config = OmegaConf.load(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    return config

def load_test_data():
    """Load test.csv data and parse parent/repo relationships."""
    test_csv_path = "/home/ubuntu/shillscore/data/raw/test.csv"
    
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test data not found at {test_csv_path}")
    
    df = pd.read_csv(test_csv_path)
    logger.info(f"Loaded {len(df)} rows from test.csv")
    
    return df

def select_random_l3_pairs(df: pd.DataFrame, n_pairs: int = 200) -> List[Tuple[str, str, str]]:
    """
    Randomly select pairs for Level 3 comparisons.
    
    Args:
        df: DataFrame with repo, parent columns
        n_pairs: Number of comparison pairs to generate
        
    Returns:
        List of (parent_repo, dep_a, dep_b) tuples
    """
    random.seed(42)
    pairs = []
    
    # Get ethereum repos (potential parents)
    ethereum_repos = df[df['parent'] == 'ethereum']['repo'].tolist()
    
    for _ in range(n_pairs):
        # Step 1: Select random parent repo from ethereum ecosystem
        parent_repo = random.choice(ethereum_repos)
        
        # Step 2: Get dependencies for this parent
        dependencies = df[df['parent'] == parent_repo]['repo'].tolist()
        
        # If we have at least 2 dependencies, select 2 random ones
        if len(dependencies) >= 2:
            dep_a, dep_b = random.sample(dependencies, 2)
            pairs.append((parent_repo, dep_a, dep_b))
        else:
            # If not enough deps, try again (don't count this iteration)
            continue
    
    logger.info(f"Generated {len(pairs)} L3 comparison pairs")
    return pairs

async def run_l3_comparison(handler: ComparisonHandler, parent_repo: str, dep_a: str, dep_b: str, 
                           model_id: str, temperature: float = 0.4) -> Dict[str, Any]:
    """
    Run a single L3 comparison using the existing handler.
    
    Args:
        handler: ComparisonHandler instance
        parent_repo: Parent repository URL
        dep_a: First dependency URL
        dep_b: Second dependency URL
        model_id: Model to use for comparison
        temperature: Temperature parameter
        
    Returns:
        Dictionary with comparison results and choice_uncertainty
    """
    try:
        # Create comparison request
        request = ComparisonRequest(
            repo_a=dep_a,
            repo_b=dep_b,
            parent=parent_repo,
            parameters={
                'model_id': model_id,
                'temperature': temperature,
                'simplified': True,  # Use simplified mode for max_tokens=20
                'include_cost': True,
                'include_tokens': True
            }
        )
        
        # Run L3 comparison using the handler
        response = await handler.handle_l3_comparison(request)
        
        return {
            'parent': parent_repo,
            'dep_a': dep_a,
            'dep_b': dep_b,
            'model': model_id,
            'success': True,
            'choice': response.choice,
            'choice_uncertainty': response.choice_uncertainty,  # This is what we want for perplexity analysis
            'explanation': response.explanation,
            'tokens_used': getattr(response, 'tokens_used', 0),
            'cost_usd': getattr(response, 'cost_usd', 0.0)
        }
        
    except Exception as e:
        logger.error(f"Error in L3 comparison: {e}")
        return {
            'parent': parent_repo,
            'dep_a': dep_a,
            'dep_b': dep_b,
            'model': model_id,
            'success': False,
            'error': str(e),
            'choice_uncertainty': float('inf')
        }

async def main():
    """Main function for L3 choice_uncertainty analysis."""
    print("Level 3 Dependency Comparison - Choice Uncertainty Analysis")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return 1
    
    # Load configuration
    print("Loading configuration...")
    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return 1
    
    # Load test data
    print("Loading test data...")
    try:
        df = load_test_data()
    except Exception as e:
        print(f"Failed to load test data: {e}")
        return 1
    
    # Select random L3 pairs
    print("Selecting random L3 comparison pairs...")
    random.seed(42)  # For reproducibility
    pairs = select_random_l3_pairs(df, n_pairs=200)
    
    if len(pairs) < 200:
        print(f"Warning: Only generated {len(pairs)} pairs (requested 200)")
    
    # Initialize components
    print("Initializing components...")
    try:
        llm_orchestrator = LLMOrchestrator(config)
        comparison_handler = ComparisonHandler(llm_orchestrator)
    except Exception as e:
        print(f"Failed to initialize components: {e}")
        return 1
    
    # Test models as requested: gpt-4o and llama4
    models = ["openai/gpt-4o", "meta-llama/llama-4-maverick"]
    temperature = 0.4
    
    print(f"Running L3 comparisons...")
    print(f"Models: {models}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: 20 (simplified mode)")
    print(f"Total comparisons: {len(pairs)} × {len(models)} = {len(pairs) * len(models)}")
    
    results = []
    
    for i, (parent, dep_a, dep_b) in enumerate(pairs):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(pairs)} pairs completed")
        
        for model_id in models:
            result = await run_l3_comparison(
                comparison_handler, parent, dep_a, dep_b, 
                model_id, temperature
            )
            results.append(result)
    
    # Calculate median choice_uncertainty for each model
    print("\nCalculating median choice_uncertainty by model...")
    
    model_uncertainties = {}
    for model_id in models:
        model_results = [r for r in results if r['model'] == model_id and r['success']]
        uncertainties = [r['choice_uncertainty'] for r in model_results if r['choice_uncertainty'] != float('inf')]
        
        if uncertainties:
            median_uncertainty = np.median(uncertainties)
            model_uncertainties[model_id] = {
                'median_choice_uncertainty': median_uncertainty,
                'total_comparisons': len(model_results),
                'valid_uncertainties': len(uncertainties),
                'mean_choice_uncertainty': np.mean(uncertainties),
                'std_choice_uncertainty': np.std(uncertainties),
                'min_choice_uncertainty': np.min(uncertainties),
                'max_choice_uncertainty': np.max(uncertainties)
            }
        else:
            model_uncertainties[model_id] = {
                'median_choice_uncertainty': float('inf'),
                'total_comparisons': 0,
                'valid_uncertainties': 0,
                'mean_choice_uncertainty': float('inf'),
                'std_choice_uncertainty': 0,
                'min_choice_uncertainty': float('inf'),
                'max_choice_uncertainty': float('inf')
            }
    
    # Print results
    print("\n" + "=" * 60)
    print("CHOICE UNCERTAINTY ANALYSIS RESULTS")
    print("=" * 60)
    
    for model_id, stats in model_uncertainties.items():
        print(f"\nModel: {model_id}")
        print(f"  Median Choice Uncertainty: {stats['median_choice_uncertainty']:.6f}")
        print(f"  Mean Choice Uncertainty: {stats['mean_choice_uncertainty']:.6f}")
        print(f"  Std Choice Uncertainty: {stats['std_choice_uncertainty']:.6f}")
        print(f"  Min Choice Uncertainty: {stats['min_choice_uncertainty']:.6f}")
        print(f"  Max Choice Uncertainty: {stats['max_choice_uncertainty']:.6f}")
        print(f"  Valid Comparisons: {stats['valid_uncertainties']}/{stats['total_comparisons']}")
    
    # Summary statistics
    total_success = sum(1 for r in results if r['success'])
    total_requests = len(results)
    total_cost = sum(r.get('cost_usd', 0) for r in results if r['success'])
    total_tokens = sum(r.get('tokens_used', 0) for r in results if r['success'])
    
    print(f"\nSummary:")
    print(f"  Total requests: {total_requests}")
    print(f"  Successful requests: {total_success}")
    print(f"  Success rate: {total_success/total_requests*100:.1f}%")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Total tokens: {total_tokens}")
    
    print(f"\nNote: choice_uncertainty is calculated from answer token logprobs")
    print(f"      using the existing L3 comparison infrastructure.")
    print(f"      Lower uncertainty = higher confidence in the choice.")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)