#!/usr/bin/env python3
"""
Simple test script for bulk API endpoints.
Quick validation of /cached-comparisons/bulk and /cached-originality/bulk.
"""

import requests
import json
import time

def test_endpoint(url: str, name: str) -> dict:
    """Test a single endpoint and return results."""
    print(f"\n🔍 Testing {name} endpoint: {url}")
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=30)
        response_time = (time.time() - start_time) * 1000
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {response_time:.2f}ms")
        
        if response.status_code == 200:
            data = response.json()
            
            if "error" in data:
                print(f"   ❌ API Error: {data['error']}")
                return {"success": False, "error": data["error"]}
            
            # Print key metrics based on endpoint type
            if "total_comparisons" in data:
                # Bulk comparisons endpoint
                print(f"   ✅ Success - Bulk Comparisons")
                print(f"   Total Repositories: {data.get('total_repositories', 0)}")
                print(f"   Total Comparisons: {data.get('total_comparisons', 0)}")
                
                comparisons = data.get('comparisons', [])
                if comparisons:
                    sample = comparisons[0]
                    print(f"   Sample: {sample.get('repo_a', '').split('/')[-1]} vs {sample.get('repo_b', '').split('/')[-1]} → {sample.get('choice', '')}")
                
            elif "assessments" in data:
                # Bulk originality endpoint
                print(f"   ✅ Success - Bulk Originality")
                print(f"   Total Repositories: {data.get('total_repositories', 0)}")
                
                assessments = data.get('assessments', [])
                if assessments:
                    sample = assessments[0]
                    print(f"   Sample: {sample.get('repository_name', '')} → Score: {sample.get('originality_score', 0):.3f} (Category {sample.get('originality_category', '')})")
                
                # Show categories distribution
                categories = {}
                for assessment in assessments:
                    cat = assessment.get('originality_category', 'Unknown')
                    categories[cat] = categories.get(cat, 0) + 1
                
                print(f"   Categories: {dict(sorted(categories.items()))}")
            
            return {"success": True, "data": data, "response_time_ms": response_time}
        
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return {"success": False, "error": f"HTTP {response.status_code}"}
    
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return {"success": False, "error": str(e)}

def main():
    """Main test function."""
    base_url = "http://localhost:8000"
    
    print("🚀 Simple Bulk API Endpoints Test")
    print("=" * 50)
    
    # Test health first
    health_result = test_endpoint(f"{base_url}/health", "Health Check")
    if not health_result.get("success"):
        print("\n❌ API is not healthy, aborting tests")
        return
    
    # Test bulk endpoints
    comparison_result = test_endpoint(f"{base_url}/cached-comparisons/bulk", "Bulk Comparisons")
    originality_result = test_endpoint(f"{base_url}/cached-originality/bulk", "Bulk Originality")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 2
    
    if comparison_result.get("success"):
        print("✅ Bulk Comparisons: PASS")
        tests_passed += 1
    else:
        print("❌ Bulk Comparisons: FAIL")
    
    if originality_result.get("success"):
        print("✅ Bulk Originality: PASS")
        tests_passed += 1
    else:
        print("❌ Bulk Originality: FAIL")
    
    print(f"\nResult: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")

if __name__ == "__main__":
    main()