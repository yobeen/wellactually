# src/api/models/requests.py
"""
Pydantic models for API request validation.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator, HttpUrl
import re

class ComparisonRequest(BaseModel):
    """Request model for repository comparison."""
    
    repo_a: str = Field(
        ..., 
        description="First repository URL to compare",
        example="https://github.com/ethereum/go-ethereum"
    )
    repo_b: str = Field(
        ..., 
        description="Second repository URL to compare",
        example="https://github.com/hyperledger/besu"
    )
    parent: str = Field(
        ..., 
        description="Parent repository or ecosystem context",
        example="ethereum"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional parameters for the comparison",
        example={"include_model_metadata": True, "temperature": 0.4}
    )
    
    @validator('repo_a', 'repo_b')
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
    
    @validator('parent')
    def validate_parent(cls, v):
        """Validate parent field."""
        if not v or not isinstance(v, str):
            raise ValueError("Parent must be a non-empty string")
        
        v = v.strip()
        if not v:
            raise ValueError("Parent cannot be empty")
        
        return v
    
    @validator('parameters')
    def validate_parameters(cls, v):
        """Validate parameters dictionary."""
        if v is None:
            return {}
        
        if not isinstance(v, dict):
            raise ValueError("Parameters must be a dictionary")
        
        return v

class BatchComparisonRequest(BaseModel):
    """Request model for confidence-based batch repository comparison."""
    
    pairs: List[Dict[str, str]] = Field(
        ..., 
        description="List of repository pairs to compare",
        example=[
            {"repo_a": "https://github.com/ethereum/go-ethereum", "repo_b": "https://github.com/ethereum/solidity"},
            {"repo_a": "https://github.com/ethereum/solidity", "repo_b": "https://github.com/foundry-rs/foundry"}
        ]
    )
    parent: str = Field(
        ..., 
        description="Parent repository or ecosystem context",
        example="ethereum"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional parameters for the comparison",
        example={"include_model_metadata": True}
    )
    
    @validator('pairs')
    def validate_pairs(cls, v):
        """Validate pairs list."""
        if not v or not isinstance(v, list):
            raise ValueError("Pairs must be a non-empty list")
        
        if len(v) == 0:
            raise ValueError("At least one repository pair is required")
        
        for i, pair in enumerate(v):
            if not isinstance(pair, dict):
                raise ValueError(f"Pair {i} must be a dictionary")
            
            if 'repo_a' not in pair or 'repo_b' not in pair:
                raise ValueError(f"Pair {i} must contain 'repo_a' and 'repo_b' keys")
            
            # Validate repository URLs using same logic as ComparisonRequest
            for key in ['repo_a', 'repo_b']:
                repo_url = pair[key]
                if not repo_url or not isinstance(repo_url, str):
                    raise ValueError(f"Pair {i} {key} must be a non-empty string")
                
                repo_url = repo_url.strip()
                if not repo_url:
                    raise ValueError(f"Pair {i} {key} cannot be empty")
                
                # Basic GitHub URL validation
                github_pattern = r'https://github\.com/[^/]+/[^/]+'
                if not re.match(github_pattern, repo_url):
                    raise ValueError(f"Pair {i} {key} must be a valid GitHub URL (https://github.com/owner/repo)")
        
        return v

class OriginalityRequest(BaseModel):
    """Request model for repository originality assessment."""
    
    repo: str = Field(
        ..., 
        description="Repository URL to assess for originality",
        example="https://github.com/ethereum/solidity"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional parameters for the assessment",
        example={"dependency_count": 15, "include_model_metadata": True}
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
                    "dependency_count": 15,
                    "include_model_metadata": True,
                    "analysis_depth": "standard"
                }
            }
        }