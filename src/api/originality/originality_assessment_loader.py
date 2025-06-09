# src/api/originality/originality_assessment_loader.py
"""
Loader for originality assessment data from directory structure.
Provides repository originality lookup and validation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OriginalityAssessment:
    """Container for repository originality assessment data."""
    repository_url: str
    repository_name: str
    originality_category: str
    final_originality_score: float
    assessment_confidence: float
    overall_reasoning_uncertainty: float
    aggregate_uncertainty: float
    criteria_uncertainties: Dict[str, float]
    criteria_scores: Dict[str, Dict[str, Any]]
    overall_reasoning: str
    parsing_method: str
    parsing_success: bool
    parsing_warnings: List[str]

class OriginalityAssessmentLoader:
    """
    Loads and manages originality assessment data from directory structure.
    Data is organized as: data/processed/originality/owner/repo/detailed_originality_assessments_with_uncertainty.json
    """
    
    def __init__(self, base_path: str = "data/processed/originality"):
        """
        Initialize the loader.
        
        Args:
            base_path: Base path to the originality assessment directory
        """
        self.base_path = Path(base_path)
        self._assessments_cache: Optional[Dict[str, OriginalityAssessment]] = None
        self._last_scan_time: Optional[float] = None
        
    def load_assessments(self, force_reload: bool = False) -> Dict[str, OriginalityAssessment]:
        """
        Load assessments from directory structure with caching.
        
        Args:
            force_reload: Force reload even if cache is valid
            
        Returns:
            Dictionary mapping repository URLs to assessment objects
            
        Raises:
            FileNotFoundError: If base directory doesn't exist
            ValueError: If JSON files are invalid or malformed
        """
        # Check if we need to reload
        if not force_reload and self._is_cache_valid():
            logger.debug("Using cached originality assessment data")
            return self._assessments_cache
        
        logger.info(f"Loading originality assessments from {self.base_path}")
        
        try:
            # Check base directory exists
            if not self.base_path.exists():
                raise FileNotFoundError(f"Assessment directory not found: {self.base_path}")
            
            # Scan directory structure and load assessments
            assessments = self._scan_and_load_assessments()
            
            # Cache results
            self._assessments_cache = assessments
            self._last_scan_time = self.base_path.stat().st_mtime
            
            logger.info(f"Loaded {len(assessments)} originality assessments")
            return assessments
            
        except Exception as e:
            logger.error(f"Error loading originality assessments: {e}")
            raise
    
    def get_assessment_by_url(self, repository_url: str) -> Optional[OriginalityAssessment]:
        """
        Get assessment for a specific repository URL.
        
        Args:
            repository_url: Repository URL to look up
            
        Returns:
            OriginalityAssessment object or None if not found
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
        
        logger.warning(f"No originality assessment found for repository: {repository_url}")
        return None
    
    def get_available_repositories(self) -> List[str]:
        """
        Get list of all available repository URLs.
        
        Returns:
            List of repository URLs with assessments
        """
        assessments = self.load_assessments()
        return list(assessments.keys())
    
    def get_repositories_by_category(self, category: str) -> List[OriginalityAssessment]:
        """
        Get all repositories in a specific originality category.
        
        Args:
            category: Originality category (A, B, C, etc.)
            
        Returns:
            List of assessments for the specified category
        """
        assessments = self.load_assessments()
        return [
            assessment for assessment in assessments.values()
            if assessment.originality_category == category
        ]
    
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
                "warnings": [],
                "categories": {}
            }
            
            # Count by category
            for assessment in assessments.values():
                category = assessment.originality_category
                if category not in validation_results["categories"]:
                    validation_results["categories"][category] = 0
                validation_results["categories"][category] += 1
            
            for url, assessment in assessments.items():
                # Check required fields
                if not assessment.repository_url:
                    validation_results["errors"].append(f"Missing repository_url for {url}")
                
                if assessment.final_originality_score is None or not (0 <= assessment.final_originality_score <= 1):
                    validation_results["errors"].append(f"Invalid final_originality_score for {url}")
                
                if not assessment.criteria_scores:
                    validation_results["errors"].append(f"Missing criteria_scores for {url}")
                
                # Check for failed parsing indicators
                if not assessment.parsing_success:
                    validation_results["warnings"].append(f"Parsing failed for {url}")
                
                # Check for suspiciously uniform uncertainties
                if assessment.criteria_uncertainties:
                    unique_uncertainties = set(assessment.criteria_uncertainties.values())
                    if len(unique_uncertainties) == 1 and 0.5 in unique_uncertainties:
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
                "warnings": [],
                "categories": {}
            }
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if self._assessments_cache is None or self._last_scan_time is None:
            return False
        
        if not self.base_path.exists():
            return False
        
        # Check if directory has been modified since last scan
        current_modified = self.base_path.stat().st_mtime
        return current_modified == self._last_scan_time
    
    def _scan_and_load_assessments(self) -> Dict[str, OriginalityAssessment]:
        """
        Scan directory structure and load all assessments.
        
        Returns:
            Dictionary mapping URLs to OriginalityAssessment objects
        """
        assessments = {}
        
        # Walk through owner/repo directory structure
        for owner_dir in self.base_path.iterdir():
            if not owner_dir.is_dir():
                continue
            
            for repo_dir in owner_dir.iterdir():
                if not repo_dir.is_dir():
                    continue
                
                # Look for the detailed assessment file
                assessment_file = repo_dir / "detailed_originality_assessments_with_uncertainty.json"
                if not assessment_file.exists():
                    logger.debug(f"No assessment file found for {owner_dir.name}/{repo_dir.name}")
                    continue
                
                try:
                    # Load and parse the assessment file
                    with open(assessment_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Handle both list and single object formats
                    if isinstance(data, list):
                        if len(data) == 0:
                            logger.debug(f"Empty assessment file for {owner_dir.name}/{repo_dir.name}")
                            continue
                        assessment_data = data[0]  # Take first assessment
                    else:
                        assessment_data = data
                    
                    # Parse into OriginalityAssessment object
                    assessment = self._parse_assessment(assessment_data)
                    if assessment and assessment.repository_url:
                        assessments[assessment.repository_url] = assessment
                        logger.debug(f"Loaded assessment for {assessment.repository_url}")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {assessment_file}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error loading {assessment_file}: {e}")
                    continue
        
        return assessments
    
    def _parse_assessment(self, data: Dict[str, Any]) -> Optional[OriginalityAssessment]:
        """
        Parse raw JSON data into an OriginalityAssessment object.
        
        Args:
            data: Raw JSON data dictionary
            
        Returns:
            OriginalityAssessment object or None if parsing fails
        """
        try:
            assessment = OriginalityAssessment(
                repository_url=data.get("repository_url", ""),
                repository_name=data.get("repository_name", ""),
                originality_category=data.get("originality_category", ""),
                final_originality_score=float(data.get("final_originality_score", 0.0)),
                assessment_confidence=float(data.get("assessment_confidence", 0.0)),
                overall_reasoning_uncertainty=float(data.get("overall_reasoning_uncertainty", 0.0)),
                aggregate_uncertainty=float(data.get("aggregate_uncertainty", 0.0)),
                criteria_uncertainties=data.get("criteria_uncertainties", {}),
                criteria_scores=data.get("criteria_scores", {}),
                overall_reasoning=data.get("overall_reasoning", ""),
                parsing_method=data.get("parsing_method", "unknown"),
                parsing_success=data.get("parsing_success", False),
                parsing_warnings=data.get("parsing_warnings", [])
            )
            
            return assessment
            
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Error parsing assessment data: {e}")
            return None
    
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