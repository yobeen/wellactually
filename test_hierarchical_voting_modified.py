#!/usr/bin/env python3
"""
Test script for the modified hierarchical voting approach.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tasks.l1.l1_voting_analysis import analyze_hierarchical_voting
from src.calibration.data_collection import UncertaintyDataCollector
from src.utils.data_loader import load_l1_data
from omegaconf import OmegaConf
import os
from dotenv import load_dotenv

def test_modified_hierarchical_voting():
    """Test the modified hierarchical voting approach."""
    
    # Load environment variables
    load_dotenv()
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return
    
    print("Loading validation data...")
    
    # Load configuration
    config = OmegaConf.load("configs/uncertainty_calibration/llm.yaml")
    
    # Create synthetic test data
    import random
    import numpy as np
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("Creating synthetic test data...")
    
    class MockDataPoint:
        def __init__(self, question_id, model_id, human_choice, human_multiplier):
            self.question_id = question_id
            self.model_id = model_id
            self.human_choice = human_choice
            self.human_multiplier = human_multiplier
            
            # Generate realistic model predictions and uncertainties
            choices = ["A", "B", "Equal"]
            if random.random() < 0.7:  # 70% chance of being correct
                if human_multiplier <= 1.2:
                    self.model_prediction = "Equal"
                elif human_choice == 1.0:
                    self.model_prediction = "A"
                else:
                    self.model_prediction = "B"
            else:
                self.model_prediction = random.choice(choices)
            
            # Generate uncertainty - lower for correct predictions
            is_correct = (
                (human_multiplier <= 1.2 and self.model_prediction == "Equal") or
                (human_choice == 1.0 and human_multiplier > 1.2 and self.model_prediction == "A") or
                (human_choice == 2.0 and human_multiplier > 1.2 and self.model_prediction == "B")
            )
            
            if is_correct:
                self.raw_uncertainty = random.uniform(0.0, 0.3)
            else:
                self.raw_uncertainty = random.uniform(0.2, 0.8)
    
    # Create test dataset
    models = ["openai/gpt-4o", "meta-llama/llama-4-maverick", 
              "google/gemma-3-27b-it", "mistralai/mixtral-8x22b-instruct"] 
    
    n_questions = 35  # Small test set
    calibration_data_points = []
    
    for q_id in range(n_questions):
        question_id = f"q_{q_id}"
        
        # Generate human labels
        label_type = random.choice(["equal", "a", "b"])
        if label_type == "equal":
            human_choice = 1.0
            human_multiplier = 1.0
        elif label_type == "a":
            human_choice = 1.0
            human_multiplier = 2.0
        else:
            human_choice = 2.0
            human_multiplier = 2.0
        
        # Create data points for each model on this question
        for model_id in models:
            dp = MockDataPoint(question_id, model_id, human_choice, human_multiplier)
            calibration_data_points.append(dp)
    
    if not calibration_data_points:
        print("No data loaded!")
        return
    
    print(f"Loaded {len(calibration_data_points)} data points")
    
    # Group by model to see what we have
    from src.tasks.l1.l1_utils import group_data_by_model
    models_data = group_data_by_model(calibration_data_points)
    
    print(f"Available models: {list(models_data.keys())}")
    for model_id, data_points in models_data.items():
        print(f"  {model_id}: {len(data_points)} data points")
    
    # Test the modified hierarchical voting
    print("\nTesting modified hierarchical voting...")
    
    # Create a temporary directory for results
    test_dir = Path("test_results") / "hierarchical_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Run hierarchical voting analysis
    result = analyze_hierarchical_voting(calibration_data_points, test_dir)
    
    if result:
        print(f"\nHierarchical voting completed!")
        print(f"Overall accuracy: {result['overall_accuracy']:.3f}")
        print(f"Questions decided: {result['questions_decided']}/{result['total_questions']}")
        
        print("\nStage breakdown:")
        for stage, data in result['stage_results'].items():
            print(f"  {stage} ({data['model']}): {data['questions_decided']} decisions, {data['stage_accuracy']:.3f} accuracy")
        
        print(f"\nResults saved to: {test_dir}")
    else:
        print("Hierarchical voting failed!")

if __name__ == "__main__":
    test_modified_hierarchical_voting()