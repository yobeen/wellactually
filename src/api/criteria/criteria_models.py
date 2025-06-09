# src/api/criteria_models.py
"""
Pydantic models for criteria assessment API requests and responses.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
import re

class CriteriaRequest(BaseModel):
    """Request model for repository criteria assessment."""
    
    repo: str = Field(
        ..., 
        description="Repository URL to assess against criteria",
        example="https://github.com/ethereum/solidity"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional parameters for the assessment",
        example={"model_id": "openai/gpt-4o", "temperature": 0.0}
    )
    
    @validator('repo')
    def validate_repository_url(cls, v):
        """Validate that repository URL is properly formatted."""
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
                "repo": "https://github.com/ethereum/solidity",
                "parameters": {
                    "model_id": "openai/gpt-4o",
                    "temperature": 0.0,
                    "include_model_metadata": True
                }
            }
        }

class CriterionScore(BaseModel):
    """Model for individual criterion assessment."""
    
    score: int = Field(
        ...,
        description="Score for this criterion (1-10)",
        ge=1, le=10,
        example=8
    )
    weight: float = Field(
        ...,
        description="Weight for this criterion (should sum to ~1.0 across all criteria)",
        ge=0.0, le=1.0,
        example=0.25
    )
    reasoning: str = Field(
        ...,
        description="Reasoning for the score assignment",
        example="This repository provides critical infrastructure for Ethereum execution..."
    )
    raw_uncertainty: float = Field(
        ...,
        description="Perplexity-based uncertainty for this criterion's reasoning [0,1]",
        ge=0.0, le=1.0,
        example=0.15
    )

class CriteriaResponse(BaseModel):
    """Response model for repository criteria assessment."""
    
    repository_url: str = Field(
        ...,
        description="The repository URL that was assessed",
        example="https://github.com/ethereum/solidity"
    )
    repository_name: str = Field(
        ...,
        description="Extracted repository name",
        example="solidity"
    )
    criteria_scores: Dict[str, CriterionScore] = Field(
        ...,
        description="Assessment scores for each criterion"
    )
    target_score: Optional[float] = Field(
        None,
        description="Overall weighted target score",
        example=7.85
    )
    raw_target_score: Optional[float] = Field(
        None,
        description="Raw target score from assessment summary",
        example=7.85
    )
    total_weight: Optional[float] = Field(
        None,
        description="Sum of all criterion weights",
        example=1.0
    )
    overall_reasoning: Optional[str] = Field(
        None,
        description="Overall assessment reasoning"
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
                "repository_url": "https://github.com/ethereum/solidity",
                "repository_name": "solidity",
                "criteria_scores": {
                    "core_protocol": {
                        "score": 9,
                        "weight": 0.25,
                        "reasoning": "Solidity is the primary smart contract language for Ethereum...",
                        "raw_uncertainty": 0.12
                    },
                    "developer_ecosystem": {
                        "score": 8,
                        "weight": 0.15,
                        "reasoning": "Essential tool for Ethereum developers...",
                        "raw_uncertainty": 0.18
                    }
                },
                "target_score": 7.85,
                "raw_target_score": 7.85,
                "total_weight": 1.0,
                "overall_reasoning": "Solidity is a foundational component of the Ethereum ecosystem...",
                "parsing_method": "perfect_json",
                "parsing_success": True,
                "parsing_warnings": [],
                "processing_time_ms": 2150.5,
                "model_metadata": "gpt-4o_temp_0.0"
            }
        }