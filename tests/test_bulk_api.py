"""
Unit tests for bulk API endpoints.
Tests the bulk comparison and originality endpoints.
"""

import pytest
import requests
import json
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30

class TestBulkAPI:
    """Test class for bulk API endpoints."""
    
    @pytest.fixture(scope="class")
    def api_session(self):
        """Create a requests session for API calls."""
        session = requests.Session()
        # Test API health first
        response = session.get(f"{BASE_URL}/health", timeout=10)
        assert response.status_code == 200, "API is not healthy"
        yield session
        session.close()
    
    def test_bulk_comparisons_endpoint_exists(self, api_session):
        """Test that the bulk comparisons endpoint exists and responds."""
        response = api_session.get(f"{BASE_URL}/cached-comparisons/bulk", timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    def test_bulk_comparisons_response_structure(self, api_session):
        """Test bulk comparisons response has correct structure."""
        response = api_session.get(f"{BASE_URL}/cached-comparisons/bulk", timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        
        # Should not have error field
        assert "error" not in data, f"API returned error: {data.get('error')}"
        
        # Required top-level fields
        required_fields = ["total_repositories", "total_comparisons", "repositories", "comparisons", "metadata"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Validate data types
        assert isinstance(data["total_repositories"], int), "total_repositories should be int"
        assert isinstance(data["total_comparisons"], int), "total_comparisons should be int"
        assert isinstance(data["repositories"], list), "repositories should be list"
        assert isinstance(data["comparisons"], list), "comparisons should be list"
        assert isinstance(data["metadata"], dict), "metadata should be dict"
        
        # Should have some data
        assert data["total_repositories"] > 0, "Should have repositories"
        assert data["total_comparisons"] > 0, "Should have comparisons"
        assert len(data["repositories"]) > 0, "Repositories list should not be empty"
        assert len(data["comparisons"]) > 0, "Comparisons list should not be empty"
    
    def test_bulk_comparisons_data_validity(self, api_session):
        """Test bulk comparisons data validity."""
        response = api_session.get(f"{BASE_URL}/cached-comparisons/bulk", timeout=TIMEOUT)
        data = response.json()
        
        comparisons = data["comparisons"]
        
        # Test first comparison structure
        if comparisons:
            comparison = comparisons[0]
            required_comparison_fields = [
                "repo_a", "repo_b", "choice", "multiplier", "explanation", 
                "method", "comparison_level"
            ]
            
            for field in required_comparison_fields:
                assert field in comparison, f"Missing comparison field: {field}"
            
            # Validate choice values
            assert comparison["choice"] in ["A", "B", "Equal"], f"Invalid choice: {comparison['choice']}"
            
            # Validate multiplier
            assert isinstance(comparison["multiplier"], (int, float)), "Multiplier should be numeric"
            assert comparison["multiplier"] > 0, "Multiplier should be positive"
            
            # Validate comparison level
            assert comparison["comparison_level"] == "L1", f"Expected L1, got {comparison['comparison_level']}"
            
            # Validate URLs
            assert comparison["repo_a"].startswith("https://github.com/"), "repo_a should be GitHub URL"
            assert comparison["repo_b"].startswith("https://github.com/"), "repo_b should be GitHub URL"
    
    def test_bulk_originality_endpoint_exists(self, api_session):
        """Test that the bulk originality endpoint exists and responds."""
        response = api_session.get(f"{BASE_URL}/cached-originality/bulk", timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    def test_bulk_originality_response_structure(self, api_session):
        """Test bulk originality response has correct structure."""
        response = api_session.get(f"{BASE_URL}/cached-originality/bulk", timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        
        # Should not have error field
        assert "error" not in data, f"API returned error: {data.get('error')}"
        
        # Required top-level fields
        required_fields = ["total_repositories", "repositories", "assessments", "metadata"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Validate data types
        assert isinstance(data["total_repositories"], int), "total_repositories should be int"
        assert isinstance(data["repositories"], list), "repositories should be list"
        assert isinstance(data["assessments"], list), "assessments should be list"
        assert isinstance(data["metadata"], dict), "metadata should be dict"
        
        # Should have some data
        assert data["total_repositories"] > 0, "Should have repositories"
        assert len(data["repositories"]) > 0, "Repositories list should not be empty"
        assert len(data["assessments"]) > 0, "Assessments list should not be empty"
    
    def test_bulk_originality_data_validity(self, api_session):
        """Test bulk originality data validity."""
        response = api_session.get(f"{BASE_URL}/cached-originality/bulk", timeout=TIMEOUT)
        data = response.json()
        
        assessments = data["assessments"]
        
        # Test first assessment structure
        if assessments:
            assessment = assessments[0]
            required_assessment_fields = [
                "repository_url", "repository_name", "owner", "repo",
                "originality_score", "originality_category", "criteria_scores", "method"
            ]
            
            for field in required_assessment_fields:
                assert field in assessment, f"Missing assessment field: {field}"
            
            # Validate originality score
            score = assessment["originality_score"]
            assert isinstance(score, (int, float)), "Originality score should be numeric"
            assert 0 <= score <= 1, f"Originality score should be 0-1, got {score}"
            
            # Validate category
            category = assessment["originality_category"]
            valid_categories = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
            assert category in valid_categories, f"Invalid category: {category}"
            
            # Validate URL
            assert assessment["repository_url"].startswith("https://github.com/"), "Should be GitHub URL"
            
            # Validate criteria scores structure
            criteria_scores = assessment["criteria_scores"]
            assert isinstance(criteria_scores, dict), "criteria_scores should be dict"
            
            # Check at least one criterion exists
            assert len(criteria_scores) > 0, "Should have at least one criterion score"
            
            # Validate first criterion structure
            if criteria_scores:
                criterion_name, criterion_data = next(iter(criteria_scores.items()))
                assert isinstance(criterion_data, dict), f"Criterion {criterion_name} should be dict"
                
                required_criterion_fields = ["score", "weight", "reasoning"]
                for field in required_criterion_fields:
                    assert field in criterion_data, f"Missing criterion field {field} in {criterion_name}"
    
    def test_bulk_metadata_validity(self, api_session):
        """Test that metadata in both endpoints is valid."""
        # Test comparisons metadata
        response = api_session.get(f"{BASE_URL}/cached-comparisons/bulk", timeout=TIMEOUT)
        comp_data = response.json()
        comp_metadata = comp_data["metadata"]
        
        assert comp_metadata.get("comparison_level") == "L1", "Comparisons should be L1"
        assert "data_source" in comp_metadata, "Should have data_source"
        assert "method" in comp_metadata, "Should have method"
        
        # Test originality metadata
        response = api_session.get(f"{BASE_URL}/cached-originality/bulk", timeout=TIMEOUT)
        orig_data = response.json()
        orig_metadata = orig_data["metadata"]
        
        assert orig_metadata.get("assessment_type") == "originality", "Should be originality assessment"
        assert "data_source" in orig_metadata, "Should have data_source"
        assert "method" in orig_metadata, "Should have method"
        assert "files_used" in orig_metadata, "Should have files_used"
        assert isinstance(orig_metadata["files_used"], list), "files_used should be list"
    
    def test_response_performance(self, api_session):
        """Test that bulk endpoints respond within reasonable time."""
        import time
        
        # Test comparisons endpoint performance
        start_time = time.time()
        response = api_session.get(f"{BASE_URL}/cached-comparisons/bulk", timeout=TIMEOUT)
        comp_time = time.time() - start_time
        
        assert response.status_code == 200
        assert comp_time < 10.0, f"Comparisons endpoint too slow: {comp_time:.2f}s"
        
        # Test originality endpoint performance
        start_time = time.time()
        response = api_session.get(f"{BASE_URL}/cached-originality/bulk", timeout=TIMEOUT)
        orig_time = time.time() - start_time
        
        assert response.status_code == 200
        assert orig_time < 10.0, f"Originality endpoint too slow: {orig_time:.2f}s"
    
    def test_data_consistency_between_endpoints(self, api_session):
        """Test that data is consistent between endpoints where applicable."""
        # Get data from both endpoints
        comp_response = api_session.get(f"{BASE_URL}/cached-comparisons/bulk", timeout=TIMEOUT)
        orig_response = api_session.get(f"{BASE_URL}/cached-originality/bulk", timeout=TIMEOUT)
        
        comp_data = comp_response.json()
        orig_data = orig_response.json()
        
        # Extract repository URLs from comparisons
        comp_repos = set()
        for comparison in comp_data["comparisons"]:
            comp_repos.add(comparison["repo_a"])
            comp_repos.add(comparison["repo_b"])
        
        # Extract repository URLs from originality
        orig_repos = set(orig_data["repositories"])
        
        # There should be significant overlap
        common_repos = comp_repos & orig_repos
        total_unique_repos = len(comp_repos | orig_repos)
        
        if total_unique_repos > 0:
            overlap_percentage = len(common_repos) / total_unique_repos * 100
            # Allow for some differences but expect significant overlap
            assert overlap_percentage > 50, f"Low repository overlap: {overlap_percentage:.1f}%"

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])