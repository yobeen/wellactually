#!/usr/bin/env python3
"""
Test script for the new batch comparison endpoint.
"""

import requests
import json
import time

# Test data
test_pairs = [
    {
        "repo_a": "https://github.com/ethereum/go-ethereum",
        "repo_b": "https://github.com/ethereum/solidity"
    },
    {
        "repo_a": "https://github.com/ethereum/solidity", 
        "repo_b": "https://github.com/foundry-rs/foundry"
    },
    {
        "repo_a": "https://github.com/ethereum/go-ethereum",
        "repo_b": "https://github.com/foundry-rs/foundry"
    }
]

def test_batch_comparison():
    """Test the batch comparison endpoint."""
    url = "http://localhost:8000/compare/batch"
    
    payload = {
        "pairs": test_pairs,
        "parent": "ethereum",
        "parameters": {
            "include_model_metadata": True
        }
    }
    
    print("Testing batch comparison endpoint...")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=300)
        duration = time.time() - start_time
        
        print(f"\nResponse Status: {response.status_code}")
        print(f"Duration: {duration:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nSUCCESS!")
            print(f"Total input pairs: {result['total_input_pairs']}")
            print(f"Successful comparisons: {result['total_successful']}")
            print(f"Filtered comparisons: {result['total_filtered']}")
            
            if result['successful_comparisons']:
                print(f"\nFirst successful comparison:")
                first = result['successful_comparisons'][0]
                print(f"  {first['repo_a']} vs {first['repo_b']}")
                print(f"  Choice: {first['choice']}")
                print(f"  Uncertainty: {first['choice_uncertainty']}")
                print(f"  Model: {first['model_used']}")
            
            if result['filtered_comparisons']:
                print(f"\nFirst filtered comparison:")
                first_filtered = result['filtered_comparisons'][0]
                print(f"  {first_filtered['repo_a']} vs {first_filtered['repo_b']}")
                print(f"  Reason: {first_filtered['reason']}")
            
            print(f"\nProcessing summary:")
            summary = result['processing_summary']
            print(f"  Total time: {summary['total_processing_time_ms']:.2f}ms")
            print(f"  Llama queries: {summary['llama_queries']}")
            print(f"  GPT-4o queries: {summary['gpt4o_queries']}")
            
        else:
            print(f"ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API server. Is it running on localhost:8000?")
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out after 5 minutes")
    except Exception as e:
        print(f"ERROR: {e}")

def test_health_check():
    """Test if the API server is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("✓ API server is running")
            return True
        else:
            print(f"✗ API server returned {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot reach API server: {e}")
        return False

if __name__ == "__main__":
    print("=== Batch Comparison Endpoint Test ===\n")
    
    # Check if server is running
    if test_health_check():
        print()
        test_batch_comparison()
    else:
        print("\nPlease start the API server first:")
        print("  python src/api/main.py")
        print("  # or")
        print("  uvicorn src.api.main:app --reload")