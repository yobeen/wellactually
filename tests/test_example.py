"""
Example test file to verify test setup works.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_example():
    """Example test to verify pytest works."""
    assert True

def test_imports():
    """Test that main packages can be imported."""
    try:
        import src.llm_augmentation
        import src.uncertainty_aggregation
        import src.utils
    except ImportError as e:
        pytest.fail(f"Failed to import packages: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
