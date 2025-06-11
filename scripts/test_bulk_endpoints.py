#!/usr/bin/env python3
"""
Test script for bulk API endpoints.
Tests both /cached-comparisons/bulk and /cached-originality/bulk endpoints.
"""

import json
import requests
import time
from typing import Dict, Any, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class BulkEndpointTester:
    """Test class for bulk API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize tester with API base URL."""
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {}
        
    def test_api_health(self) -> bool:
        """Test if API is healthy and responsive."""
        print("🔍 Testing API health...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API is healthy: {health_data.get('status', 'unknown')}")
                print(f"   Version: {health_data.get('version', 'unknown')}")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API health check error: {e}")
            return False
    
    def test_bulk_comparisons(self) -> Dict[str, Any]:
        """Test bulk L1 comparisons endpoint."""
        print("\n🔍 Testing bulk L1 comparisons endpoint...")
        test_result = {
            "endpoint": "/cached-comparisons/bulk",
            "success": False,
            "response_time_ms": 0,
            "total_repositories": 0,
            "total_comparisons": 0,
            "sample_comparison": None,
            "errors": []
        }
        
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/cached-comparisons/bulk", timeout=30)
            response_time = (time.time() - start_time) * 1000
            test_result["response_time_ms"] = round(response_time, 2)
            
            if response.status_code != 200:
                test_result["errors"].append(f"HTTP {response.status_code}: {response.text}")
                print(f"❌ Bulk comparisons request failed: {response.status_code}")
                return test_result
            
            data = response.json()
            
            # Validate response structure
            if "error" in data:
                test_result["errors"].append(f"API error: {data['error']}")
                print(f"❌ API returned error: {data['error']}")
                return test_result
            
            # Extract metrics
            test_result["total_repositories"] = data.get("total_repositories", 0)
            test_result["total_comparisons"] = data.get("total_comparisons", 0)
            
            # Validate required fields
            required_fields = ["total_repositories", "total_comparisons", "repositories", "comparisons", "metadata"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                test_result["errors"].append(f"Missing required fields: {missing_fields}")
                print(f"❌ Missing required fields: {missing_fields}")
                return test_result
            
            # Validate comparisons structure
            comparisons = data.get("comparisons", [])
            if not comparisons:
                test_result["errors"].append("No comparisons found in response")
                print("❌ No comparisons found in response")
                return test_result
            
            # Validate first comparison structure
            sample_comparison = comparisons[0]
            required_comparison_fields = [
                "repo_a", "repo_b", "choice", "multiplier", "explanation", 
                "method", "score_a", "score_b", "comparison_level"
            ]
            missing_comparison_fields = [
                field for field in required_comparison_fields 
                if field not in sample_comparison
            ]
            
            if missing_comparison_fields:
                test_result["errors"].append(f"Missing comparison fields: {missing_comparison_fields}")
                print(f"❌ Missing comparison fields: {missing_comparison_fields}")
                return test_result
            
            test_result["sample_comparison"] = {
                "repo_a": sample_comparison.get("repo_a", ""),
                "repo_b": sample_comparison.get("repo_b", ""),
                "choice": sample_comparison.get("choice", ""),
                "multiplier": sample_comparison.get("multiplier", 0),
                "method": sample_comparison.get("method", ""),
                "comparison_level": sample_comparison.get("comparison_level", "")
            }
            
            # Validate metadata
            metadata = data.get("metadata", {})
            if metadata.get("comparison_level") != "L1":
                test_result["errors"].append(f"Expected L1 comparison level, got: {metadata.get('comparison_level')}")
            
            test_result["success"] = True
            print(f"✅ Bulk comparisons endpoint successful")
            print(f"   Repositories: {test_result['total_repositories']}")
            print(f"   Comparisons: {test_result['total_comparisons']}")
            print(f"   Response time: {test_result['response_time_ms']:.2f}ms")
            print(f"   Sample comparison: {sample_comparison.get('repo_a', '')} vs {sample_comparison.get('repo_b', '')} → {sample_comparison.get('choice', '')}")
            
        except Exception as e:
            test_result["errors"].append(f"Exception: {str(e)}")
            print(f"❌ Bulk comparisons test error: {e}")
        
        return test_result
    
    def test_bulk_originality(self) -> Dict[str, Any]:
        """Test bulk originality assessments endpoint."""
        print("\n🔍 Testing bulk originality assessments endpoint...")
        test_result = {
            "endpoint": "/cached-originality/bulk", 
            "success": False,
            "response_time_ms": 0,
            "total_repositories": 0,
            "sample_assessment": None,
            "categories_found": [],
            "score_range": {"min": 1.0, "max": 0.0},
            "errors": []
        }
        
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/cached-originality/bulk", timeout=30)
            response_time = (time.time() - start_time) * 1000
            test_result["response_time_ms"] = round(response_time, 2)
            
            if response.status_code != 200:
                test_result["errors"].append(f"HTTP {response.status_code}: {response.text}")
                print(f"❌ Bulk originality request failed: {response.status_code}")
                return test_result
            
            data = response.json()
            
            # Validate response structure
            if "error" in data:
                test_result["errors"].append(f"API error: {data['error']}")
                print(f"❌ API returned error: {data['error']}")
                return test_result
            
            # Extract metrics
            test_result["total_repositories"] = data.get("total_repositories", 0)
            
            # Validate required fields
            required_fields = ["total_repositories", "repositories", "assessments", "metadata"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                test_result["errors"].append(f"Missing required fields: {missing_fields}")
                print(f"❌ Missing required fields: {missing_fields}")
                return test_result
            
            # Validate assessments structure
            assessments = data.get("assessments", [])
            if not assessments:
                test_result["errors"].append("No assessments found in response")
                print("❌ No assessments found in response")
                return test_result
            
            # Validate first assessment structure
            sample_assessment = assessments[0]
            required_assessment_fields = [
                "repository_url", "repository_name", "owner", "repo",
                "originality_score", "originality_category", "criteria_scores", "method"
            ]
            missing_assessment_fields = [
                field for field in required_assessment_fields 
                if field not in sample_assessment
            ]
            
            if missing_assessment_fields:
                test_result["errors"].append(f"Missing assessment fields: {missing_assessment_fields}")
                print(f"❌ Missing assessment fields: {missing_assessment_fields}")
                return test_result
            
            # Analyze assessment data
            categories = set()
            scores = []
            
            for assessment in assessments:
                category = assessment.get("originality_category")
                if category:
                    categories.add(category)
                
                score = assessment.get("originality_score")
                if score is not None:
                    scores.append(score)
            
            test_result["categories_found"] = sorted(list(categories))
            if scores:
                test_result["score_range"] = {
                    "min": round(min(scores), 3),
                    "max": round(max(scores), 3)
                }
            
            test_result["sample_assessment"] = {
                "repository_url": sample_assessment.get("repository_url", ""),
                "repository_name": sample_assessment.get("repository_name", ""),
                "originality_score": sample_assessment.get("originality_score", 0),
                "originality_category": sample_assessment.get("originality_category", ""),
                "criteria_count": len(sample_assessment.get("criteria_scores", {})),
                "method": sample_assessment.get("method", "")
            }
            
            # Validate metadata
            metadata = data.get("metadata", {})
            if metadata.get("assessment_type") != "originality":
                test_result["errors"].append(f"Expected originality assessment type, got: {metadata.get('assessment_type')}")
            
            test_result["success"] = True
            print(f"✅ Bulk originality endpoint successful")
            print(f"   Repositories: {test_result['total_repositories']}")
            print(f"   Response time: {test_result['response_time_ms']:.2f}ms")
            print(f"   Categories found: {test_result['categories_found']}")
            print(f"   Score range: {test_result['score_range']['min']:.3f} - {test_result['score_range']['max']:.3f}")
            print(f"   Sample: {sample_assessment.get('repository_name', '')} → {sample_assessment.get('originality_score', 0):.3f} (Category {sample_assessment.get('originality_category', '')})")
            
        except Exception as e:
            test_result["errors"].append(f"Exception: {str(e)}")
            print(f"❌ Bulk originality test error: {e}")
        
        return test_result
    
    def test_data_consistency(self, comparison_data: Dict, originality_data: Dict) -> Dict[str, Any]:
        """Test data consistency between bulk endpoints."""
        print("\n🔍 Testing data consistency between endpoints...")
        consistency_result = {
            "success": False,
            "comparison_repos": set(),
            "originality_repos": set(),
            "common_repos": set(),
            "comparison_only": set(),
            "originality_only": set(),
            "errors": []
        }
        
        try:
            # Extract repository sets
            if comparison_data.get("success") and comparison_data.get("total_repositories") > 0:
                comparison_repos = set()
                for comp in comparison_data.get("comparisons", []):
                    comparison_repos.add(comp.get("repo_a", ""))
                    comparison_repos.add(comp.get("repo_b", ""))
                comparison_repos.discard("")  # Remove empty strings
                consistency_result["comparison_repos"] = comparison_repos
            
            if originality_data.get("success") and originality_data.get("total_repositories") > 0:
                originality_repos = set(originality_data.get("repositories", []))
                consistency_result["originality_repos"] = originality_repos
            
            # Analyze overlap
            if consistency_result["comparison_repos"] and consistency_result["originality_repos"]:
                comparison_repos = consistency_result["comparison_repos"]
                originality_repos = consistency_result["originality_repos"]
                
                consistency_result["common_repos"] = comparison_repos & originality_repos
                consistency_result["comparison_only"] = comparison_repos - originality_repos
                consistency_result["originality_only"] = originality_repos - comparison_repos
                
                # Convert sets to lists for JSON serialization
                consistency_result["comparison_repos"] = list(comparison_repos)
                consistency_result["originality_repos"] = list(originality_repos)
                consistency_result["common_repos"] = list(consistency_result["common_repos"])
                consistency_result["comparison_only"] = list(consistency_result["comparison_only"])
                consistency_result["originality_only"] = list(consistency_result["originality_only"])
                
                overlap_percentage = len(consistency_result["common_repos"]) / len(comparison_repos | originality_repos) * 100
                
                print(f"✅ Data consistency analysis complete")
                print(f"   Comparison repos: {len(comparison_repos)}")
                print(f"   Originality repos: {len(originality_repos)}")
                print(f"   Common repos: {len(consistency_result['common_repos'])}")
                print(f"   Overlap percentage: {overlap_percentage:.1f}%")
                
                if consistency_result["comparison_only"]:
                    print(f"   Comparison-only repos: {len(consistency_result['comparison_only'])}")
                
                if consistency_result["originality_only"]:
                    print(f"   Originality-only repos: {len(consistency_result['originality_only'])}")
                
                consistency_result["success"] = True
            else:
                consistency_result["errors"].append("Insufficient data for consistency analysis")
                print("❌ Insufficient data for consistency analysis")
        
        except Exception as e:
            consistency_result["errors"].append(f"Exception: {str(e)}")
            print(f"❌ Data consistency test error: {e}")
        
        return consistency_result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all bulk endpoint tests."""
        print("🚀 Starting bulk API endpoints test suite...")
        print("=" * 60)
        
        # Test API health first
        if not self.test_api_health():
            print("\n❌ API is not healthy, skipping bulk endpoint tests")
            return {"success": False, "error": "API health check failed"}
        
        # Test bulk comparisons
        comparison_results = self.test_bulk_comparisons()
        
        # Test bulk originality
        originality_results = self.test_bulk_originality()
        
        # Test data consistency
        consistency_results = self.test_data_consistency(comparison_results, originality_results)
        
        # Compile overall results
        overall_results = {
            "test_suite": "bulk_api_endpoints",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_success": (
                comparison_results.get("success", False) and 
                originality_results.get("success", False) and
                consistency_results.get("success", False)
            ),
            "tests": {
                "bulk_comparisons": comparison_results,
                "bulk_originality": originality_results,
                "data_consistency": consistency_results
            }
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 BULK API ENDPOINTS TEST SUMMARY")
        print("=" * 60)
        
        if overall_results["overall_success"]:
            print("✅ ALL TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED")
        
        print(f"   Bulk Comparisons: {'✅ PASS' if comparison_results.get('success') else '❌ FAIL'}")
        print(f"   Bulk Originality: {'✅ PASS' if originality_results.get('success') else '❌ FAIL'}")
        print(f"   Data Consistency: {'✅ PASS' if consistency_results.get('success') else '❌ FAIL'}")
        
        # Show key metrics
        if comparison_results.get("success"):
            print(f"\n📈 Bulk Comparisons Metrics:")
            print(f"   Repositories: {comparison_results.get('total_repositories', 0)}")
            print(f"   Comparisons: {comparison_results.get('total_comparisons', 0)}")
            print(f"   Response time: {comparison_results.get('response_time_ms', 0):.2f}ms")
        
        if originality_results.get("success"):
            print(f"\n📈 Bulk Originality Metrics:")
            print(f"   Repositories: {originality_results.get('total_repositories', 0)}")
            print(f"   Categories: {len(originality_results.get('categories_found', []))}")
            print(f"   Response time: {originality_results.get('response_time_ms', 0):.2f}ms")
        
        return overall_results
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> None:
        """Save test results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"bulk_api_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Test results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")

def main():
    """Main test execution function."""
    # Parse command line arguments
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    # Create tester and run tests
    tester = BulkEndpointTester(base_url)
    results = tester.run_all_tests()
    
    # Save results
    tester.save_results(results)
    
    # Exit with appropriate code
    sys.exit(0 if results.get("overall_success", False) else 1)

if __name__ == "__main__":
    main()