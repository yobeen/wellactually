# src/tasks/l3/__init__.py
"""
Level 3 Dependency Assessment Package

This package provides components for dependency context extraction and response parsing
for Level 3 dependency importance assessments.
"""

from .dependency_context_extractor import DependencyContextExtractor
from .dependency_response_parser import DependencyResponseParser
from .level3_prompts import Level3PromptGenerator

__all__ = [
    'DependencyContextExtractor',
    'DependencyResponseParser', 
    'Level3PromptGenerator'
]

__version__ = "1.0.0"
__author__ = "L3 Assessment Team"
__description__ = "Level 3 dependency importance assessment framework"