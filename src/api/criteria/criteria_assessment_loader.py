# src/api/criteria/criteria_assessment_loader.py
"""
Loader for criteria assessment data from JSON file.
Provides repository assessment lookup and validation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CriteriaAssessment:
    """Container for repository criteria assessment data."""
    repository_url: str
    repository_name: str
    target_score: float
    total_weight: float
    criteria_scores: Dict[str, Dict[str, Any]]
    overall_reasoning: str
    parsing_method: str
    parsing_success: bool
    parsing_warnings: List[str]

class CriteriaAssessmentLoader:
    """
    Loads and manages criteria assessment data from JSON file.
    """
    
    def __init__(self, file_path: str = "data/processed/criteria_assessment/detailed_assessments.json"):
        """
        Initialize the loader.
        
        Args:
            file_path: Path to the JSON assessment file
        """
        self.file_path = Path(file_path)
        self._assessments_cache: Optional[Dict[str, CriteriaAssessment]] = None
        self._last_modified: Optional[float] = None
        
    def load_assessments(self, force_reload: bool = False) -> Dict[str, CriteriaAssessment]:
        """
        Load assessments from JSON file with caching.
        
        Args:
            force_reload: Force reload even if cache is valid
            
        Returns:
            Dictionary mapping repository URLs to assessment objects
            
        Raises:
            FileNotFoundError: If assessment file doesn't exist
            ValueError: If JSON is invalid or malformed
        """
        # Check if we need to reload
        if not force_reload and self._is_cache_valid():
            logger.debug("Using cached assessment data")
            return self._assessments_cache
        
        logger.info(f"Loading criteria assessments from {self.file_path}")
        
        try:
            # Check file exists
            if not self.file_path.exists():
                raise FileNotFoundError(f"Assessment file not found: {self.file_path}")
            
            # Load and parse JSON
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate and convert to assessment objects
            assessments = self._parse_assessments(data)
            
            # Cache results
            self._assessments_cache = assessments
            self._last_modified = self.file_path.stat().st_mtime
            
            logger.info(f"Loaded {len(assessments)} criteria assessments")
            return assessments
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in assessment file: {e}")
        except Exception as e:
            logger.error(f"Error loading assessments: {e}")
            raise
    
    def get_assessment_by_url(self, repository_url: str) -> Optional[CriteriaAssessment]:
        """
        Get assessment for a specific repository URL.
        
        Args:
            repository_url: Repository URL to look up
            
        Returns:
            CriteriaAssessment object or None if not found
        """
        assessments = self.load_assessments()
        
        # Try exact URL match first
        if repository_url in assessments:
            return assessments[repository_url]
        
        # Try normalized URL matching (handle trailing slashes, etc.)
        normalized_url = self._normalize_url(repository_url)
        for url, assessment in assessments.items():
            if self._normalize_url(url) == normalized_url:
                return assessment
        
        logger.warning(f"No assessment found for repository: {repository_url}")
        return None
    
    def get_available_repositories(self) -> List[str]:
        """
        Get list of all available repository URLs.
        
        Returns:
            List of repository URLs with assessments
        """
        assessments = self.load_assessments()
        return list(assessments.keys())
    
    def validate_assessment_data(self) -> Dict[str, Any]:
        """
        Validate the loaded assessment data.
        
        Returns:
            Validation results dictionary
        """
        try:
            assessments = self.load_assessments()
            
            validation_results = {
                "valid": True,
                "total_assessments": len(assessments),
                "errors": [],
                "warnings": []
            }
            
            for url, assessment in assessments.items():
                # Check required fields
                if not assessment.repository_url:
                    validation_results["errors"].append(f"Missing repository_url for {url}")
                
                if assessment.target_score is None or assessment.target_score < 0:
                    validation_results["errors"].append(f"Invalid target_score for {url}")
                
                if not assessment.criteria_scores:
                    validation_results["errors"].append(f"Missing criteria_scores for {url}")
                
                # Check for failed parsing indicators (approximate equality for floating point)
                all_uncertainties_half = all(
                    abs(criterion.get("raw_uncertainty", 0) - 0.5) < 1e-10
                    for criterion in assessment.criteria_scores.values()
                )
                
                if all_uncertainties_half:
                    validation_results["warnings"].append(
                        f"All uncertainties are 0.5 for {url} - possible parsing failure"
                    )
            
            validation_results["valid"] = len(validation_results["errors"]) == 0
            
            return validation_results
            
        except Exception as e:
            return {
                "valid": False,
                "total_assessments": 0,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": []
            }
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if self._assessments_cache is None or self._last_modified is None:
            return False
        
        if not self.file_path.exists():
            return False
        
        current_modified = self.file_path.stat().st_mtime
        return current_modified == self._last_modified
    
    def _parse_assessments(self, data: List[Dict[str, Any]]) -> Dict[str, CriteriaAssessment]:
        """
        Parse raw JSON data into assessment objects.
        
        Args:
            data: Raw JSON data (list of dictionaries)
            
        Returns:
            Dictionary mapping URLs to CriteriaAssessment objects
        """
        if not isinstance(data, list):
            raise ValueError("Assessment data must be a list of objects")
        
        assessments = {}
        
        for item in data:
            try:
                assessment = CriteriaAssessment(
                    repository_url=item.get("repository_url", ""),
                    repository_name=item.get("repository_name", ""),
                    target_score=float(item.get("target_score", 0.0)),
                    total_weight=float(item.get("total_weight", 1.0)),
                    criteria_scores=item.get("criteria_scores", {}),
                    overall_reasoning=item.get("overall_reasoning", ""),
                    parsing_method=item.get("parsing_method", "unknown"),
                    parsing_success=item.get("parsing_success", False),
                    parsing_warnings=item.get("parsing_warnings", [])
                )
                
                if assessment.repository_url:
                    assessments[assessment.repository_url] = assessment
                else:
                    logger.warning(f"Skipping assessment with missing repository_url")
                    
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error parsing assessment item: {e}")
                continue
        
        return assessments
    
    def _normalize_url(self, url: str) -> str:
        """Normalize repository URL for comparison."""
        if not url:
            return ""
        
        # Remove trailing slash and convert to lowercase
        normalized = url.rstrip('/').lower()
        
        # Ensure https:// prefix
        if not normalized.startswith(('http://', 'https://')):
            normalized = 'https://' + normalized
        
        return normalized