# src/api/dependency_models.py
"""
Pydantic models for dependency comparison API requests and responses.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
import re

class DependencyComparisonRequest(BaseModel):
    """Request model for dependency comparison assessment."""
    
    parent_repo: str = Field(
        ..., 
        description="Parent repository URL for the dependency comparison",
        example="https://github.com/ethereum/web3.py"
    )
    dependency_a: str = Field(
        ...,
        description="First dependency repository URL to compare",
        example="https://github.com/psf/requests"
    )
    dependency_b: str = Field(
        ...,
        description="Second dependency repository URL to compare", 
        example="https://github.com/pydantic/pydantic"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional parameters for the assessment",
        example={"model_id": "openai/gpt-4o", "temperature": 0.0}
    )
    
    @validator('parent_repo', 'dependency_a', 'dependency_b')
    def validate_repository_url(cls, v):
        """Validate that repository URLs are properly formatted."""
        if not v or not isinstance(v, str):
            raise ValueError("Repository URL must be a non-empty string")
        
        v = v.strip()
        if not v:
            raise ValueError("Repository URL cannot be empty")
        
        # Basic GitHub URL validation
        github_pattern = r'https://github\.com/[^/]+/[^/]+'
        if not re.match(github_pattern, v):
            raise ValueError("Repository URL must be a valid GitHub URL (https://github.com/owner/repo)")
        
        return v
    
    @validator('parameters')
    def validate_parameters(cls, v):
        """Validate parameters dictionary."""
        if v is None:
            return {}
        
        if not isinstance(v, dict):
            raise ValueError("Parameters must be a dictionary")
        
        return v
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "parent_repo": "https://github.com/ethereum/web3.py",
                "dependency_a": "https://github.com/psf/requests",
                "dependency_b": "https://github.com/pydantic/pydantic",
                "parameters": {
                    "model_id": "openai/gpt-4o",
                    "temperature": 0.0,
                    "include_model_metadata": True
                }
            }
        }

class DimensionAssessment(BaseModel):
    """Model for individual dimension assessment."""
    
    score_a: int = Field(
        ...,
        description="Score for dependency A on this dimension (1-10)",
        ge=1, le=10,
        example=7
    )
    score_b: int = Field(
        ...,
        description="Score for dependency B on this dimension (1-10)",
        ge=1, le=10,
        example=5
    )
    weight: float = Field(
        ...,
        description="Weight for this dimension (should sum to ~1.0 across all dimensions)",
        ge=0.0, le=1.0,
        example=0.4
    )
    reasoning: str = Field(
        ...,
        description="Reasoning comparing both dependencies on this dimension",
        example="Dependency A is more critical for core functionality because..."
    )
    raw_uncertainty: float = Field(
        ...,
        description="Perplexity-based uncertainty for this dimension's reasoning [0,1]",
        ge=0.0, le=1.0,
        example=0.15
    )

class OverallAssessment(BaseModel):
    """Model for overall dependency comparison assessment."""
    
    choice: str = Field(
        ...,
        description="Which dependency is more important (A, B, or Equal)",
        regex="^(A|B|Equal)$",
        example="A"
    )
    confidence: float = Field(
        ...,
        description="Confidence in the choice [0,1]",
        ge=0.0, le=1.0,
        example=0.75
    )
    weighted_score_a: float = Field(
        ...,
        description="Weighted sum of scores for dependency A",
        example=6.8
    )
    weighted_score_b: float = Field(
        ...,
        description="Weighted sum of scores for dependency B",
        example=5.2
    )
    reasoning: str = Field(
        ...,
        description="Overall reasoning for the comparison decision",
        example="Dependency A is more important overall because it provides critical functionality..."
    )

class DependencyComparisonResponse(BaseModel):
    """Response model for dependency comparison assessment."""
    
    parent_url: str = Field(
        ...,
        description="The parent repository URL",
        example="https://github.com/ethereum/web3.py"
    )
    parent_name: str = Field(
        ...,
        description="Extracted parent repository name",
        example="web3.py"
    )
    dependency_a_url: str = Field(
        ...,
        description="First dependency URL",
        example="https://github.com/psf/requests"
    )
    dependency_a_name: str = Field(
        ...,
        description="First dependency name",
        example="requests"
    )
    dependency_b_url: str = Field(
        ...,
        description="Second dependency URL", 
        example="https://github.com/pydantic/pydantic"
    )
    dependency_b_name: str = Field(
        ...,
        description="Second dependency name",
        example="pydantic"
    )
    dimension_assessments: Dict[str, DimensionAssessment] = Field(
        ...,
        description="Assessment scores for each dimension"
    )
    overall_assessment: OverallAssessment = Field(
        ...,
        description="Overall comparison assessment"
    )
    parsing_method: str = Field(
        ...,
        description="Method used to parse the LLM response",
        example="perfect_json"
    )
    parsing_success: bool = Field(
        ...,
        description="Whether parsing was successful",
        example=True
    )
    parsing_warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings from the parsing process"
    )
    
    # Additional metadata fields
    processing_time_ms: Optional[float] = Field(
        None,
        description="Processing time in milliseconds"
    )
    model_metadata: Optional[str] = Field(
        None,
        description="Model metadata if requested"
    )
    cost_usd: Optional[float] = Field(
        None,
        description="Cost in USD if requested"
    )
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "parent_url": "https://github.com/ethereum/web3.py",
                "parent_name": "web3.py",
                "dependency_a_url": "https://github.com/psf/requests",
                "dependency_a_name": "requests",
                "dependency_b_url": "https://github.com/pydantic/pydantic", 
                "dependency_b_name": "pydantic",
                "dimension_assessments": {
                    "functional_necessity": {
                        "score_a": 8,
                        "score_b": 4,
                        "weight": 0.4,
                        "reasoning": "Requests is essential for HTTP communication while pydantic is for validation",
                        "raw_uncertainty": 0.12
                    },
                    "performance_impact": {
                        "score_a": 6,
                        "score_b": 5,
                        "weight": 0.3,
                        "reasoning": "Both have moderate performance impact, requests slightly higher due to network I/O",
                        "raw_uncertainty": 0.18
                    },
                    "replaceability": {
                        "score_a": 4,
                        "score_b": 7,
                        "weight": 0.2,
                        "reasoning": "Pydantic is more easily replaceable with other validation libraries",
                        "raw_uncertainty": 0.15
                    },
                    "integration_depth": {
                        "score_a": 7,
                        "score_b": 3,
                        "weight": 0.1,
                        "reasoning": "Requests is deeply integrated into web3.py's architecture",
                        "raw_uncertainty": 0.10
                    }
                },
                "overall_assessment": {
                    "choice": "A",
                    "confidence": 0.78,
                    "weighted_score_a": 6.8,
                    "weighted_score_b": 4.7,
                    "reasoning": "Requests is more critical overall due to its functional necessity and deep integration"
                },
                "parsing_method": "perfect_json",
                "parsing_success": True,
                "parsing_warnings": [],
                "processing_time_ms": 2350.7,
                "model_metadata": "gpt-4o_temp_0.0"
            }
        }