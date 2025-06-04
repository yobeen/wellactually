# src/api/__init__.py
"""
FastAPI LLM Repository Assessment API.
"""

__version__ = "1.0.0"
__author__ = "LLM Assessment Team"
__description__ = "API for repository comparison and originality assessment using LLMs"

# src/api/models/__init__.py
"""
Pydantic models for API requests and responses.
"""

from .requests import ComparisonRequest, OriginalityRequest
from .responses import ComparisonResponse, OriginalityResponse, ErrorResponse

__all__ = [
    'ComparisonRequest',
    'OriginalityRequest', 
    'ComparisonResponse',
    'OriginalityResponse',
    'ErrorResponse'
]

# src/api/services/__init__.py
"""
Business logic services for the API.
"""

from .llm_orchestrator import LLMOrchestrator

__all__ = [
    'LLMOrchestrator'
]

# src/api/handlers/__init__.py
"""
Request handlers for different API endpoints.
"""

from .comparison_handler import ComparisonHandler
from .originality_handler import OriginalityHandler

__all__ = [
    'ComparisonHandler',
    'OriginalityHandler'
]

# src/api/config/__init__.py
"""
Configuration management for the API.
"""

from .settings import APISettings

__all__ = [
    'APISettings'
]