# src/uncertainty_calibration/originality_assessment/__init__.py
"""
Originality Assessment Package

This package provides a comprehensive framework for assessing repository originality
using the Ethereum Ecosystem Repository Originality Assessment Framework.

The framework evaluates repositories across 8 criteria with category-specific weights:
1. Protocol Implementation Depth and Originality
2. Algorithmic and Performance Innovation  
3. Developer Experience and API Innovation
4. Architectural Innovation and Modularity
5. Security and Formal Verification Innovation
6. Standards Leadership and Protocol Extension
7. Cross-Client Compatibility and Interoperability
8. Domain-Specific Problem Solving

Repositories are categorized into 9 categories (A-I) with specific weight matrices:
- A: Execution Clients
- B: Consensus Clients  
- C: JavaScript/TypeScript Libraries
- D: Other Language Libraries
- E: Development Frameworks
- F: Smart Contract Languages
- G: Smart Contract Security/Standards
- H: Specialized Tools
- I: Data/Infrastructure
"""

from .originality_prompt_generator import OriginalityPromptGenerator
from .originality_response_parser import OriginalityResponseParser, ParsedOriginalityResponse
from .originality_assessment_pipeline import OriginalityAssessmentPipeline

__all__ = [
    # Core components
    'OriginalityPromptGenerator',
    'OriginalityResponseParser', 
    'OriginalityAssessmentPipeline',
    
    # Enhanced query engine
    'EnhancedQueryEngine',
    
    # Data classes
    'ParsedOriginalityResponse'
]

__version__ = "1.0.0"
__author__ = "Originality Assessment Team"
__description__ = "Repository originality assessment framework for Ethereum ecosystem"

# Framework configuration
ORIGINALITY_CRITERIA = [
    'protocol_implementation',
    'algorithmic_innovation', 
    'developer_experience',
    'architectural_innovation',
    'security_innovation',
    'standards_leadership',
    'cross_client_compatibility',
]

ORIGINALITY_CATEGORIES = {
    'A': 'Execution Clients',
    'B': 'Consensus Clients',
    'C': 'JavaScript/TypeScript Libraries', 
    'D': 'Other Language Libraries',
    'E': 'Development Frameworks',
    'F': 'Smart Contract Languages',
    'G': 'Smart Contract Security/Standards',
    'H': 'Specialized Tools',
    'I': 'Data/Infrastructure'
}

SCORE_RANGE = (0.1, 0.9)  # Framework score range

def get_framework_info():
    """Get information about the originality assessment framework."""
    return {
        'criteria': ORIGINALITY_CRITERIA,
        'categories': ORIGINALITY_CATEGORIES,
        'score_range': SCORE_RANGE,
        'version': __version__,
        'description': __description__
    }