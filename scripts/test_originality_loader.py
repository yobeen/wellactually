#!/usr/bin/env python3
"""
Test script for OriginalityAssessmentLoader.
Validates that the loader can read originality assessment data correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from api.originality.originality_assessment_loader import OriginalityAssessmentLoader

def main():
    """Test the originality assessment loader."""
    print("Testing OriginalityAssessmentLoader...")
    
    # Initialize loader
    loader = OriginalityAssessmentLoader()
    
    # Test loading assessments
    print("\n1. Loading assessments...")
    try:
        assessments = loader.load_assessments()
        print(f"✓ Successfully loaded {len(assessments)} assessments")
        
        # Show some basic stats
        categories = {}
        for assessment in assessments.values():
            category = assessment.originality_category
            categories[category] = categories.get(category, 0) + 1
        
        print(f"Categories found: {dict(sorted(categories.items()))}")
        
    except Exception as e:
        print(f"✗ Error loading assessments: {e}")
        return 1
    
    # Test getting available repositories
    print("\n2. Getting available repositories...")
    try:
        repos = loader.get_available_repositories()
        print(f"✓ Found {len(repos)} repositories")
        print(f"First 5 repositories:")
        for i, repo in enumerate(sorted(repos)[:5]):
            print(f"  {i+1}. {repo}")
            
    except Exception as e:
        print(f"✗ Error getting repositories: {e}")
        return 1
    
    # Test getting specific assessment
    print("\n3. Testing specific repository lookup...")
    test_url = "https://github.com/ethereum/go-ethereum"
    try:
        assessment = loader.get_assessment_by_url(test_url)
        if assessment:
            print(f"✓ Found assessment for {test_url}")
            print(f"  Repository name: {assessment.repository_name}")
            print(f"  Category: {assessment.originality_category}")
            print(f"  Originality score: {assessment.final_originality_score:.4f}")
            print(f"  Confidence: {assessment.assessment_confidence}")
            print(f"  Parsing success: {assessment.parsing_success}")
            print(f"  Number of criteria: {len(assessment.criteria_scores)}")
        else:
            print(f"✗ No assessment found for {test_url}")
            
    except Exception as e:
        print(f"✗ Error getting specific assessment: {e}")
        return 1
    
    # Test getting repositories by category
    print("\n4. Testing category filtering...")
    try:
        category_a_repos = loader.get_repositories_by_category("A")
        print(f"✓ Found {len(category_a_repos)} repositories in category A")
        for repo in category_a_repos[:3]:  # Show first 3
            print(f"  - {repo.repository_name} ({repo.repository_url})")
            
    except Exception as e:
        print(f"✗ Error getting category repositories: {e}")
        return 1
    
    # Test validation
    print("\n5. Validating assessment data...")
    try:
        validation = loader.validate_assessment_data()
        print(f"✓ Validation completed")
        print(f"  Valid: {validation['valid']}")
        print(f"  Total assessments: {validation['total_assessments']}")
        print(f"  Errors: {len(validation['errors'])}")
        print(f"  Warnings: {len(validation['warnings'])}")
        print(f"  Categories: {validation['categories']}")
        
        if validation['errors']:
            print("  Errors found:")
            for error in validation['errors'][:5]:  # Show first 5
                print(f"    - {error}")
                
        if validation['warnings']:
            print("  Warnings found:")
            for warning in validation['warnings'][:5]:  # Show first 5
                print(f"    - {warning}")
                
    except Exception as e:
        print(f"✗ Error during validation: {e}")
        return 1
    
    print("\n✓ All tests completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())