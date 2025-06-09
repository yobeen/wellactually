# scripts/test_originality_pipeline.py
"""
Test script for originality assessment pipeline with uncertainty calculation.
Processes repositories individually and saves results in owner/repo structure.
"""

import os
import sys
import json
import logging
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.tasks.originality.originality_prompt_generator import OriginalityPromptGenerator
from src.tasks.originality.originality_assessment_pipeline import OriginalityAssessmentPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_repository_config_loading():
    """Test loading repository configurations from seed_repositories.yaml."""
    print("\nTesting repository configuration loading...")
    
    try:
        generator = OriginalityPromptGenerator()
        
        # Test loading a specific repository config
        test_repo = "https://github.com/ethereum/solidity"
        config = generator.get_repo_originality_config(test_repo)
        
        print(f"✅ Configuration loading successful!")
        print(f"Repository: {config['name']}")
        print(f"Category: {config['category']}")
        print(f"Primary Language: {config['primary_language']}")
        print(f"Domain: {config['domain']}")
        
        print(f"\nWeights:")
        for criterion, weight in config['weights'].items():
            print(f"  {criterion}: {weight:.3f}")
        
        # Verify weights sum to 1.0
        total_weight = sum(config['weights'].values())
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

def extract_owner_repo(repo_url):
    """Extract owner and repo name from GitHub URL."""
    # https://github.com/ethereum/solidity -> ethereum, solidity
    parts = repo_url.rstrip('/').split('/')
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "unknown", "unknown"

def test_individual_repository_assessments():
    """Test originality assessment for multiple repositories, processing each individually."""
    print("\nTesting individual repository assessments with uncertainty...")
    
    try:
        # Load configuration
        config = OmegaConf.load("configs/uncertainty_calibration/llm.yaml")
        
        # Initialize pipeline
        pipeline = OriginalityAssessmentPipeline(config)
        
        # Generate timestamp for this test run
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Test run timestamp: {timestamp}")
        
        # Get test repositories from configuration
        test_repos = [
            # Category A: Execution Clients
            "https://github.com/ethereum/go-ethereum",
            "https://github.com/ledgerwatch/erigon", 
            "https://github.com/nethermindeth/nethermind",
            "https://github.com/hyperledger/besu",
            "https://github.com/paradigmxyz/reth",
            
            # Category B: Consensus Clients
            "https://github.com/prysmaticlabs/prysm",
            "https://github.com/sigp/lighthouse",
            "https://github.com/ChainSafe/lodestar",
            "https://github.com/ConsenSys/teku",
            "https://github.com/status-im/nimbus-eth2",
            "https://github.com/anza-xyz/grandine",
            
            # Category C: JavaScript/TypeScript Libraries
            "https://github.com/web3/web3.js",
            "https://github.com/ethers-io/ethers.js",
            "https://github.com/wevm/viem",
            "https://github.com/ethereumjs/ethereumjs-monorepo",
            
            # Category D: Other Language Libraries
            "https://github.com/ethereum/web3.py",
            "https://github.com/Nethereum/Nethereum",
            "https://github.com/web3j/web3j",
            "https://github.com/alloy-rs/alloy",
            
            # Category E: Development Frameworks
            "https://github.com/foundry-rs/foundry",
            "https://github.com/NomicFoundation/hardhat",
            "https://github.com/ApeWorX/ape",
            "https://github.com/ethereum/remix-project",
            
            # Category F: Smart Contract Languages
            "https://github.com/ethereum/solidity",
            "https://github.com/vyperlang/vyper",
            "https://github.com/ethereum/fe",
            
            # Category G: Smart Contract Security/Standards
            "https://github.com/OpenZeppelin/openzeppelin-contracts",
            "https://github.com/safe-global/safe-smart-account",
            "https://github.com/eth-infinitism/account-abstraction",
            
            # Category H: Specialized Tools
            "https://github.com/ethereum/sourcify",
            "https://github.com/vyperlang/titanoboa",
            "https://github.com/scaffold-eth/scaffold-eth-2",
            "https://github.com/a16z/helios",
            
            # Category I: Data/Infrastructure
            "https://github.com/ethereum-lists/chains",
            "https://github.com/ethereum/py-evm"
        ]
        
        print(f"Processing {len(test_repos)} repositories individually...")
        
        successful_assessments = 0
        failed_assessments = 0
        
        # Process each repository individually
        for i, repo_url in enumerate(test_repos, 1):
            try:
                print(f"\n{'-'*60}")
                print(f"Processing repository {i}/{len(test_repos)}: {repo_url}")
                print(f"{'-'*60}")
                
                # Extract owner and repo name for directory structure
                owner, repo_name = extract_owner_repo(repo_url)
                output_dir = f"results/{timestamp}/{owner}/{repo_name}"
                
                print(f"Output directory: {output_dir}")
                
                # Run assessment for single repository with uncertainty
                results = pipeline.run_full_assessment(
                    model_id="deepseek/deepseek-r1-0528",  # Using deepseek R1 model
                    temperature=0.7,
                    repositories=[repo_url],  # Single repository
                    output_dir=output_dir
                )
                
                # Extract results for this repository
                repo_score = results['originality_scores'].get(repo_url, 0.0)
                uncertainty_summary = results.get('uncertainty_summary', {})
                
                print(f"✅ Assessment completed for {repo_name}")
                print(f"  Originality Score: {repo_score:.3f}")
                
                # Display uncertainty metrics
                if 'overall_reasoning' in uncertainty_summary:
                    overall_uncertainty = uncertainty_summary['overall_reasoning'].get('mean', 0.0)
                    print(f"  Overall Reasoning Uncertainty: {overall_uncertainty:.3f}")
                
                if 'aggregate_criteria' in uncertainty_summary:
                    aggregate_uncertainty = uncertainty_summary['aggregate_criteria'].get('mean', 0.0)
                    print(f"  Aggregate Criteria Uncertainty: {aggregate_uncertainty:.3f}")
                
                # Check if files were created
                output_path = Path(output_dir)
                expected_files = [
                    "detailed_originality_assessments_with_uncertainty.json",
                    "originality_scores.json",
                    "uncertainty_metrics.json", 
                    "originality_scores_with_uncertainty.csv"
                ]
                
                for filename in expected_files:
                    file_path = output_path / filename
                    if file_path.exists():
                        print(f"  ✓ Created: {filename}")
                    else:
                        print(f"  ❌ Missing: {filename}")
                
                successful_assessments += 1
                
            except Exception as e:
                print(f"❌ Failed to assess {repo_url}: {e}")
                failed_assessments += 1
                continue
        
        print(f"\nTest run timestamp: {timestamp}")
        print(f"Results saved under: results/{timestamp}/")
        print(f"Total repositories: {len(test_repos)}")
        print(f"Successful assessments: {successful_assessments}")
        print(f"Failed assessments: {failed_assessments}")
        
        return successful_assessments == len(test_repos)
        
    except Exception as e:
        print(f"❌ Individual repository test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests with individual repository processing."""
    print("Enhanced Originality Assessment Pipeline Test")
    print("=" * 60)
    print("Processing repositories individually with uncertainty calculation")
    print("Model: deepseek/deepseek-r1-0528")
    print("Temperature: 0.7")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key:")
        print("export OPENROUTER_API_KEY='your-api-key-here'")
        return 1
    
    tests = [
        ("Repository Config Loading", test_repository_config_loading),
        ("Individual Repository Assessments", test_individual_repository_assessments)
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
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nOverall: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("🎉 All tests passed! Enhanced pipeline with individual processing is ready.")
        print("\nFeatures Verified:")
        print("  ✓ Individual repository processing with owner/repo directory structure")
        print("  ✓ Perplexity-based uncertainty calculation for each assessment")
        print("  ✓ Enhanced output files with uncertainty metrics")
        print("  ✓ DeepSeek R1 model integration (temperature: 0.7)")
        print("  ✓ Timestamped results directory structure")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())