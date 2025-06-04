# scripts/test_criteria_api.py
"""
Example script to test the new criteria assessment API endpoint.
"""

import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_criteria_endpoint():
    """Test the /criteria endpoint with a sample repository."""
    
    api_base_url = "http://localhost:8000"
    
    # Test request
    test_request = {
        "repo": "https://github.com/ethereum/solidity",
        "parameters": {
            "model_id": "openai/gpt-4o",
            "temperature": 0.0,
            "include_model_metadata": True,
            "include_cost": True
        }
    }
    
    try:
        print("Testing /criteria endpoint...")
        print(f"Request: {json.dumps(test_request, indent=2)}")
        
        # Make request
        response = requests.post(
            f"{api_base_url}/criteria",
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        print(f"\nResponse Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nSuccess! Response:")
            print(f"Repository: {result['repository_name']}")
            print(f"Target Score: {result['target_score']:.2f}")
            print(f"Raw Target Score: {result['raw_target_score']:.2f}")
            print(f"Parsing Method: {result['parsing_method']}")
            print(f"Processing Time: {result.get('processing_time_ms', 'N/A')} ms")
            
            print(f"\nCriteria Scores:")
            for criterion, score_data in result['criteria_scores'].items():
                print(f"  {criterion}:")
                print(f"    Score: {score_data['score']}/10")
                print(f"    Weight: {score_data['weight']:.3f}")
                print(f"    Uncertainty: {score_data['raw_uncertainty']:.3f}")
                print(f"    Reasoning: {score_data['reasoning'][:100]}...")
            
            if result.get('parsing_warnings'):
                print(f"\nWarnings: {result['parsing_warnings']}")
            
            return True
            
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception during API test: {e}")
        return False

def test_validation_errors():
    """Test API validation with invalid requests."""
    
    api_base_url = "http://localhost:8000"
    
    # Test invalid repository URL
    invalid_requests = [
        {
            "repo": "not-a-valid-url",
            "parameters": {}
        },
        {
            "repo": "https://gitlab.com/some/repo",  # Not GitHub
            "parameters": {}
        },
        {
            "repo": "",  # Empty
            "parameters": {}
        }
    ]
    
    print("\n" + "="*50)
    print("Testing validation errors...")
    
    for i, test_request in enumerate(invalid_requests, 1):
        try:
            print(f"\nTest {i}: {test_request['repo']}")
            
            response = requests.post(
                f"{api_base_url}/criteria",
                json=test_request,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            print(f"Status: {response.status_code}")
            if response.status_code != 200:
                error_data = response.json()
                print(f"Error: {error_data.get('detail', 'Unknown error')}")
            else:
                print("Unexpected success!")
                
        except Exception as e:
            print(f"Exception: {e}")

def main():
    """Run all tests."""
    print("Criteria Assessment API Test")
    print("="*50)
    
    # Test successful request
    success = test_criteria_endpoint()
    
    if success:
        # Test validation errors
        test_validation_errors()
        print(f"\n✅ All tests completed!")
    else:
        print(f"\n❌ Main test failed!")

if __name__ == "__main__":
    main()