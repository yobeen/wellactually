# scripts/test_originality_pipeline.py
"""
Test script for originality assessment pipeline.
Validates the pipeline works correctly with a small subset of repositories.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.uncertainty_calibration.originality_assessment.originality_assessment_pipeline import OriginalityAssessmentPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_multiple_repositories():
    """Test assessment of multiple repositories."""
    print("\nTesting multiple repository assessment...")
    
    # Load configuration
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/uncertainty_calibration/llm.yaml")
    
    # Initialize pipeline
    pipeline = OriginalityAssessmentPipeline(config)
    
    # Test with a few repositories from different categories
    test_repos = [
        "https://github.com/ethereum/solidity",      # F: Smart Contract Languages
        "https://github.com/foundry-rs/foundry",     # E: Development Frameworks
    ]
    
    try:
        results = pipeline.run_full_assessment(
            model_id="deepseek/deepseek-r1-0528",
            temperature=0.7,
            repositories=test_repos,
            output_dir="test_results/originality_test"
        )
        
        print(f"\n✅ Multiple repository test successful!")
        print(f"Repositories assessed: {results['successful_assessments']}/{results['total_repositories']}")
        print(f"Failed assessments: {results['failed_assessments']}")
        print(f"Results saved to: {results['output_path']}")
        
        if results['originality_scores']:
            print(f"\nScores:")
            for repo_url, score in results['originality_scores'].items():
                repo_name = repo_url.split('/')[-1]
                print(f"  {repo_name}: {score:.3f}")
        
        # Verify files were created
        output_path = Path(results['output_path'])
        expected_files = [
            "detailed_originality_assessments.json",
            "originality_scores.json", 
            "originality_scores.csv"
        ]
        
        for filename in expected_files:
            file_path = output_path / filename
            if file_path.exists():
                print(f"  ✓ Created: {filename}")
                
                # Show file contents for JSON files
                if filename.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        print(f"    Records: {len(data) if isinstance(data, list) else 'N/A'}")
            else:
                print(f"  ❌ Missing: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multiple repository test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_repository_config_loading():
    """Test loading repository configurations from seed_repositories.yaml."""
    print("\nTesting repository configuration loading...")
    
    try:
        from src.uncertainty_calibration.originality_assessment.originality_prompt_generator import OriginalityPromptGenerator
        
        generator = OriginalityPromptGenerator()
        
        # Test loading a specific repository config
        test_repo = "https://github.com/ethereum/solidity"
        config = generator.get_repo_originality_config(test_repo)
        
        print(f"✅ Configuration loading successful!")
        print(f"Repository: {config['name']}")
        print(f"Category: {config['category']}")  # FIXED: use 'category', not 'originality_category'
        print(f"Primary Language: {config['primary_language']}")
        print(f"Domain: {config['domain']}")
        
        print(f"\nWeights:")
        for criterion, weight in config['weights'].items():  # FIXED: use 'weights', not 'originality_weights'
            print(f"  {criterion}: {weight:.3f}")
        
        # Verify weights sum to 1.0
        total_weight = sum(config['weights'].values())  # FIXED: use 'weights', not 'originality_weights'
        print(f"\nTotal weight: {total_weight:.3f}")
        if abs(total_weight - 1.0) < 0.001:
            print("✓ Weights sum to 1.0")
        else:
            print("❌ Weights do not sum to 1.0")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Originality Assessment Pipeline Test")
    print("=" * 50)
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key:")
        print("export OPENROUTER_API_KEY='your-api-key-here'")
        return 1
    
    tests = [
        ("Repository Config Loading", test_repository_config_loading),
        ("Multiple Repository Assessment", test_multiple_repositories)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nOverall: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("🎉 All tests passed! Pipeline is ready for use.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())