# src/api/models/responses.py
"""
Pydantic models for API response validation.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator

class ComparisonResponse(BaseModel):
    """Response model for repository comparison."""
    
    choice: int = Field(
        ..., 
        description="Which repository is better (1 for repo_a, 2 for repo_b)",
        ge=1, le=2,
        example=1
    )
    multiplier: float = Field(
        ..., 
        description="How many times better the chosen repository is",
        ge=1.0, le=999.0,
        example=2.5
    )
    choice_uncertainty: float = Field(
        ..., 
        description="Uncertainty in the choice decision [0,1]",
        ge=0.0, le=1.0,
        example=0.15
    )
    multiplier_uncertainty: float = Field(
        ..., 
        description="Uncertainty in the multiplier magnitude [0,1]",
        ge=0.0, le=1.0,
        example=0.25
    )
    explanation: str = Field(
        ..., 
        description="Reasoning behind the comparison decision",
        example="Repository A provides more critical infrastructure..."
    )
    
    # Additional flexible fields for extra metadata
    model_metadata: Optional[str] = Field(
        None,
        description="Model metadata if requested in parameters"
    )
    processing_time_ms: Optional[float] = Field(
        None,
        description="Processing time in milliseconds"
    )
    cache_hit: Optional[bool] = Field(
        None,
        description="Whether response was served from cache"
    )
    
    @validator('choice')
    def validate_choice(cls, v):
        """Validate choice is 1 or 2."""
        if v not in [1, 2]:
            raise ValueError("Choice must be 1 or 2")
        return v
    
    @validator('multiplier')
    def validate_multiplier(cls, v):
        """Validate multiplier is in valid range."""
        if not (1.0 <= v <= 999.0):
            raise ValueError("Multiplier must be between 1.0 and 999.0")
        return v
    
    @validator('choice_uncertainty', 'multiplier_uncertainty')
    def validate_uncertainty(cls, v):
        """Validate uncertainty values are in [0,1] range."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Uncertainty must be between 0.0 and 1.0")
        return v
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "choice": 1,
                "multiplier": 3.2,
                "choice_uncertainty": 0.12,
                "multiplier_uncertainty": 0.18,
                "explanation": "Repository A (go-ethereum) is significantly more important to the Ethereum ecosystem as it serves as the primary execution client...",
                "model_metadata": "gpt-4o_temp_0.7",
                "processing_time_ms": 1250.5,
                "cache_hit": False
            }
        }

class OriginalityResponse(BaseModel):
    """Response model for repository originality assessment."""
    
    originality: float = Field(
        ..., 
        description="Originality score [0.1,0.9] where higher means more original",
        ge=0.1, le=0.9,
        example=0.75
    )
    uncertainty: float = Field(
        ..., 
        description="Uncertainty in the originality assessment [0,1]",
        ge=0.0, le=1.0,
        example=0.15
    )
    explanation: str = Field(
        ..., 
        description="Reasoning behind the originality assessment",
        example="This repository demonstrates high originality with novel approaches..."
    )
    
    # Additional flexible fields for extra metadata
    model_metadata: Optional[str] = Field(
        None,
        description="Model metadata if requested in parameters"
    )
    processing_time_ms: Optional[float] = Field(
        None,
        description="Processing time in milliseconds"
    )
    cache_hit: Optional[bool] = Field(
        None,
        description="Whether response was served from cache"
    )
    dependency_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional dependency analysis if requested"
    )
    repository_category: Optional[str] = Field(
        None,
        description="Repository originality category (A-I)"
    )
    method: Optional[str] = Field(
        None,
        description="Assessment method used (special_case_originality, llm_assessment)"
    )
    
    @validator('originality')
    def validate_originality(cls, v):
        """Validate originality is in valid range."""
        if not (0.1 <= v <= 0.9):
            raise ValueError("Originality must be between 0.1 and 0.9")
        return v
    
    @validator('uncertainty')
    def validate_uncertainty(cls, v):
        """Validate uncertainty is in [0,1] range."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Uncertainty must be between 0.0 and 1.0")
        return v
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "originality": 0.78,
                "uncertainty": 0.12,
                "explanation": "Solidity demonstrates high originality as a purpose-built smart contract language with novel features for blockchain development...",
                "model_metadata": "gpt-4o_temp_0.7",
                "processing_time_ms": 980.2,
                "cache_hit": True,
                "dependency_analysis": {
                    "total_dependencies": 5,
                    "core_dependencies": 2,
                    "analysis_method": "static_analysis"
                }
            }
        }

class ErrorResponse(BaseModel):
    """Response model for error cases."""
    
    error: str = Field(
        ..., 
        description="Error type or category",
        example="validation_error"
    )
    detail: str = Field(
        ..., 
        description="Detailed error message",
        example="Repository URL must be a valid GitHub URL"
    )
    timestamp: Optional[str] = Field(
        None,
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID for tracking"
    )
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "error": "validation_error",
                "detail": "Repository URL must be a valid GitHub URL (https://github.com/owner/repo)",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_123456789"
            }
        }