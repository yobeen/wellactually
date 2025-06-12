#!/usr/bin/env python3
"""
Quick test of hierarchical voting functionality without API calls.
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.tasks.l1.l1_voting_analysis import analyze_hierarchical_voting, apply_hierarchical_voting

@dataclass
class MockCalibrationDataPoint:
    """Mock calibration data point for testing."""
    question_id: str
    model_id: str
    model_prediction: str
    raw_uncertainty: float
    human_choice: float
    human_multiplier: float

def create_test_data():
    """Create mock test data for hierarchical voting."""
    test_data = []
    
    # Create 20 test questions with predictions from 4 models
    models = [
        "openai/gpt-4o",
        "meta-llama/llama-4-maverick", 
        "google/gemma-3-27b-it",
        "mistralai/mixtral-8x22b-instruct"
    ]
    
    for q_id in range(1, 21):
        # Human labels (for comparison)
        human_choice = 1.0 if q_id % 3 == 0 else 2.0  # Mix of A and B
        human_multiplier = 1.0 if q_id % 5 == 0 else 2.0  # Some Equal labels
        
        for i, model in enumerate(models):
            # Create varying predictions and uncertainties
            if q_id <= 10:  # First 10 questions - models mostly agree
                prediction = "A" if q_id % 2 == 0 else "B"
                # GPT-4 gets high confidence on first few, then medium
                if model == "openai/gpt-4o":
                    uncertainty = 0.2 if q_id <= 5 else 0.6  # Only confident on first 5
                else:
                    uncertainty = 0.4 + (i * 0.2)  # Others less confident
            else:  # Last 10 questions - more disagreement
                predictions = ["A", "B", "Equal", "A"]
                prediction = predictions[i]
                # Make later models more confident on remaining questions
                if model == "meta-llama/llama-4-maverick":
                    uncertainty = 0.3 if q_id <= 15 else 0.7
                elif model == "google/gemma-3-27b-it":
                    uncertainty = 0.35 if q_id <= 18 else 0.8
                else:
                    uncertainty = 0.5 + (i * 0.15)  # Higher uncertainty
            
            test_data.append(MockCalibrationDataPoint(
                question_id=f"q_{q_id}",
                model_id=model,
                model_prediction=prediction,
                raw_uncertainty=uncertainty,
                human_choice=human_choice,
                human_multiplier=human_multiplier
            ))
    
    return test_data

def test_hierarchical_voting():
    """Test the hierarchical voting functionality."""
    print("Testing Hierarchical Voting Functionality")
    print("=" * 50)
    
    # Create test data
    test_data = create_test_data()
    print(f"Created {len(test_data)} test data points")
    
    # Create temporary save directory
    save_dir = Path("/tmp/hierarchical_test")
    save_dir.mkdir(exist_ok=True)
    
    try:
        # Test hierarchical voting
        result = analyze_hierarchical_voting(test_data, save_dir)
        
        if result:
            print("\n✅ Hierarchical voting analysis completed!")
            
            # Test the summary function
            from src.tasks.l1.l1_voting_analysis import print_hierarchical_summary
            print_hierarchical_summary(result)
            
            # Check if files were created
            expected_files = ["hierarchical_voting_results.csv", "hierarchical_decisions.csv"]
            for file in expected_files:
                if (save_dir / file).exists():
                    print(f"✅ Created {file}")
                else:
                    print(f"❌ Missing {file}")
            
            return True
        else:
            print("❌ Hierarchical voting analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hierarchical_voting()
    sys.exit(0 if success else 1)