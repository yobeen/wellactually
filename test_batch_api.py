#!/usr/bin/env python3
"""
Test script for batch comparison logic.
Calls the batch comparison handler directly for easier debugging.
"""

import asyncio
import json
import subprocess
import time
import requests
import sys
import os
from typing import Dict, Any
import logging
from hydra import initialize, compose

# Set up environment
os.environ['PYTHONPATH'] = 'src'
sys.path.insert(0, 'src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchAPITester:
    def __init__(self):
        self.comparison_handler = None
        
    async def initialize(self):
        """Initialize the comparison handler directly."""
        try:
            logger.info("Initializing comparison handler...")
            
            # Create a minimal config directly
            from omegaconf import OmegaConf
            
            cfg = OmegaConf.create({
                'api': {
                    'openrouter': {
                        'base_url': 'https://openrouter.ai/api/v1/chat/completions',
                        'api_key_env': 'OPENROUTER_API_KEY'
                    }
                },
                'models': {
                    'primary_models': {
                        'deepseek_r1': {
                            'model_id': 'deepseek/deepseek-r1-0528',
                            'provider': 'deepseek'
                        }
                    }
                },
                'response_parsing': {
                    'uncertainty_calculation': {
                        'method': 'answer_specific'
                    }
                }
            })
            
            # Import and initialize the LLM orchestrator first
            from src.api.core.llm_orchestrator import LLMOrchestrator
            llm_orchestrator = LLMOrchestrator(cfg)
            
            # Import and initialize the comparison handler
            from src.api.comparison.comparison_handler import ComparisonHandler
            self.comparison_handler = ComparisonHandler(llm_orchestrator)
            
            logger.info("Comparison handler initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize comparison handler: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_api_batch_comparison(self) -> bool:
        """Test the batch comparison API endpoint."""
        try:
            logger.info("Testing batch comparison API...")
            
            # The exact same request as the curl command
            request_data = {
                "pairs": [
                    {
                        "repo_a": "https://github.com/ethereum/go-ethereum",
                        "repo_b": "https://github.com/ethereum/consensus-specs"
                    },
                    {
                        "repo_a": "https://github.com/ethereum/solidity", 
                        "repo_b": "https://github.com/ethereum/web3.py"
                    }
                ],
                "parent": "ethereum",
                "parameters": {
                    "model_id": "meta-llama/llama-4-maverick",
                    "temperature": 0.4,
                    "simplified": True
                }
            }
            
            logger.info(f"Sending request with {len(request_data['pairs'])} pairs...")
            
            # Make the API request
            response = requests.post(
                "http://localhost:8000/compare/batch",
                json=request_data,
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
            
            result = response.json()
            
            logger.info("API batch comparison completed!")
            logger.info(f"Response summary:")
            logger.info(f"  - Total input pairs: {result.get('total_input_pairs', 'N/A')}")
            logger.info(f"  - Total successful: {result.get('total_successful', 'N/A')}")
            logger.info(f"  - Total filtered: {result.get('total_filtered', 'N/A')}")
            
            # Check for expected response format
            expected_substring = '],"total_input_pairs":2,"total_successful":1,"total_filtered":1'
            response_text = response.text
            
            if expected_substring in response_text:
                logger.info("✓ Expected response format found!")
            else:
                logger.warning("✗ Expected response format not found")
                logger.info(f"Looking for: {expected_substring}")
                logger.info(f"Full response: {response_text}")
            
            # Log first successful comparison for inspection
            successful = result.get('successful_comparisons', [])
            if successful:
                first = successful[0]
                logger.info(f"First comparison result:")
                logger.info(f"  - Pair: {first.get('repo_a', 'N/A')} vs {first.get('repo_b', 'N/A')}")
                logger.info(f"  - Choice: {first.get('choice', 'N/A')}")
                logger.info(f"  - Model used: {first.get('model_used', 'N/A')}")
                logger.info(f"  - Uncertainty: {first.get('choice_uncertainty', 'N/A')}")
            
            # Log filtered comparisons to understand why they were filtered
            filtered = result.get('filtered_comparisons', [])
            if filtered:
                logger.info(f"Filtered comparisons ({len(filtered)}):")
                for i, filt in enumerate(filtered[:2], 1):  # Show first 2
                    logger.info(f"  Filtered {i}: {filt.get('repo_a', 'N/A')} vs {filt.get('repo_b', 'N/A')}")
                    logger.info(f"    Reason: {filt.get('reason', 'N/A')}")
                    if 'llama_uncertainty' in filt:
                        logger.info(f"    Llama uncertainty: {filt.get('llama_uncertainty', 'N/A')}")
                    if 'gpt4o_uncertainty' in filt:
                        logger.info(f"    GPT-4o uncertainty: {filt.get('gpt4o_uncertainty', 'N/A')}")
            
            return True
            
        except requests.exceptions.Timeout:
            logger.error("API request timed out after 5 minutes")
            return False
        except Exception as e:
            logger.error(f"API batch comparison error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_tests(self):
        """Run all tests."""
        logger.info("Starting batch API tests...")
        
        try:
            # Run API test (no initialization needed for API calls)
            if self.test_api_batch_comparison():
                logger.info("API batch test passed!")
                return True
            else:
                logger.error("API batch test failed!")
                return False
            
        except Exception as e:
            logger.error(f"Unexpected error during tests: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Main function."""
    tester = BatchAPITester()
    
    try:
        success = await tester.run_tests()
        if success:
            logger.info("All tests passed!")
            sys.exit(0)
        else:
            logger.error("Some tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())