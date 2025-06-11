# src/api/handlers/originality_handler.py
"""
Handler for repository originality assessment requests.
Skeleton implementation for future development.
"""

import logging
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional
import json

from src.api.core.requests import OriginalityRequest
from src.api.core.responses import OriginalityResponse
from src.api.core.llm_orchestrator import LLMOrchestrator

logger = logging.getLogger(__name__)

class OriginalityHandler:
    """
    Handles repository originality assessment requests.
    This is a skeleton implementation for future development.
    """
    
    def __init__(self, llm_orchestrator: LLMOrchestrator,
                 originality_data_dir: str = "data/processed/originality"):
        """
        Initialize originality handler.
        
        Args:
            llm_orchestrator: LLM orchestrator service
            originality_data_dir: Directory containing originality assessment data
        """
        self.llm_orchestrator = llm_orchestrator
        self.originality_data_dir = Path(originality_data_dir)
        self._special_case_enabled = True
        logger.info("OriginalityHandler initialized with special case support")
    
    async def handle_originality_assessment(self, request: OriginalityRequest) -> OriginalityResponse:
        """
        Handle repository originality assessment.
        Uses pre-computed originality data if available, falls back to LLM otherwise.
        
        Args:
            request: Originality request
            
        Returns:
            OriginalityResponse with assessment results
        """
        try:
            start_time = time.time()
            
            # Extract owner/repo from URL
            owner, repo = self._extract_owner_repo_from_url(request.repo)
            
            logger.info(f"Processing originality assessment for: {owner}/{repo}")
            
            # Try special case first
            if self._special_case_enabled:
                special_case_response = self._try_special_case_assessment(
                    owner, repo, request, start_time
                )
                if special_case_response:
                    logger.info(f"Originality assessment completed via special case: "
                               f"score={special_case_response.originality:.2f}")
                    return special_case_response
            
            # Fallback to LLM assessment
            logger.info(f"Falling back to LLM assessment for {owner}/{repo}")
            return await self._llm_assessment(request, start_time)
            
        except Exception as e:
            logger.error(f"Error in originality assessment: {e}")
            raise
    
    def _transform_to_originality_response(self, model_response, request: OriginalityRequest,
                                         start_time: float) -> OriginalityResponse:
        """
        Transform ModelResponse to OriginalityResponse API format.
        
        Args:
            model_response: ModelResponse from LLM orchestrator
            request: Original request
            start_time: Request start time for calculating processing time
            
        Returns:
            OriginalityResponse in expected API format
        """
        try:
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Extract originality score (mock implementation)
            originality = self._extract_originality_score(model_response)
            
            # Get uncertainty
            uncertainty = model_response.uncertainty
            
            # Get explanation
            explanation = f"[SKELETON IMPLEMENTATION] {model_response.content}"
            
            # Build additional fields based on request parameters
            additional_fields = self._build_additional_fields(
                model_response, request.parameters, processing_time_ms
            )
            
            return OriginalityResponse(
                originality=originality,
                uncertainty=uncertainty,
                explanation=explanation,
                method="llm_assessment",
                model_used=getattr(model_response, 'model_id', 'unknown'),
                **additional_fields
            )
            
        except Exception as e:
            logger.error(f"Error transforming originality response: {e}")
            raise
    
    def _extract_owner_repo_from_url(self, url: str) -> tuple[str, str]:
        """
        Extract owner and repo from GitHub URL.
        Handles URLs like:
        - https://github.com/a16z/helios
        - https://github.com/originality/a16z/helios (strips 'originality')
        
        Args:
            url: GitHub repository URL
            
        Returns:
            Tuple of (owner, repo)
        """
        try:
            # Remove protocol and domain
            path = url.replace('https://github.com/', '').replace('http://github.com/', '')
            path = path.strip('/')
            
            parts = path.split('/')
            
            # Handle URLs with 'originality' prefix
            if len(parts) >= 3 and parts[0] == 'originality':
                return parts[1], parts[2]
            elif len(parts) >= 2:
                return parts[0], parts[1]
            else:
                raise ValueError(f"Invalid GitHub URL format: {url}")
                
        except Exception as e:
            logger.error(f"Failed to extract owner/repo from URL {url}: {e}")
            raise ValueError(f"Invalid repository URL: {url}")
    
    def _try_special_case_assessment(self, owner: str, repo: str, 
                                   request: OriginalityRequest, start_time: float) -> Optional[OriginalityResponse]:
        """
        Try to get originality assessment from pre-computed data.
        
        Args:
            owner: Repository owner
            repo: Repository name
            request: Original request
            start_time: Request start time
            
        Returns:
            OriginalityResponse if data available, None otherwise
        """
        try:
            # Look for assessment file
            assessment_file = self.originality_data_dir / owner / repo / "detailed_originality_assessments_with_uncertainty.json"
            
            if not assessment_file.exists():
                logger.info(f"No pre-computed originality data found for {owner}/{repo}")
                return None
            
            # Load assessment data
            with open(assessment_file, 'r') as f:
                assessment_data = json.load(f)
            
            # Handle list format (take first item)
            if isinstance(assessment_data, list) and len(assessment_data) > 0:
                assessment_data = assessment_data[0]
            elif isinstance(assessment_data, list):
                logger.warning(f"Empty assessment data list for {owner}/{repo}")
                return None
            
            # Extract required fields
            originality_score = assessment_data.get('final_originality_score', 0.5)
            uncertainty = assessment_data.get('overall_reasoning_uncertainty', 0.3)
            reasoning = assessment_data.get('overall_reasoning', 'Pre-computed assessment')
            category = assessment_data.get('originality_category', 'Unknown')
            
            # Extract criteria scores (convert from 1-10 scale to 0-1 scale for API consistency)
            criteria_scores_raw = assessment_data.get('criteria_scores', {})
            criteria_scores = {}
            for criterion, details in criteria_scores_raw.items():
                if isinstance(details, dict) and 'score' in details:
                    # API docs show 1-10 scale, but data is stored on 1-10 scale
                    criteria_scores[criterion] = float(details['score'])
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Loaded pre-computed originality data for {owner}/{repo}: score={originality_score:.3f}")
            
            return OriginalityResponse(
                originality=originality_score,
                uncertainty=uncertainty,
                explanation=f"Assessment-based originality evaluation. {reasoning}",
                repository_category=category,
                processing_time_ms=round(processing_time_ms, 2),
                cache_hit=True,
                method="special_case_originality",
                criteria_scores=criteria_scores if criteria_scores else None,
                model_used="pre-computed"
            )
            
        except Exception as e:
            logger.warning(f"Failed to load pre-computed originality data for {owner}/{repo}: {e}")
            return None
    
    async def _llm_assessment(self, request: OriginalityRequest, start_time: float) -> OriginalityResponse:
        """
        Perform LLM-based originality assessment as fallback.
        
        Args:
            request: Originality request
            start_time: Request start time
            
        Returns:
            OriginalityResponse from LLM assessment
        """
        # Extract repository information using LLM orchestrator
        repo_info = self.llm_orchestrator.extract_repo_info(request.repo)
        
        # Get model and temperature from parameters
        model_id = request.parameters.get('model_id')
        temperature = request.parameters.get('temperature', 0.7)
        
        # Query LLM using orchestrator
        model_response = await self.llm_orchestrator.query_originality_assessment(
            repo_info=repo_info,
            model_id=model_id,
            temperature=temperature
        )
        
        # Transform to API response format
        return self._transform_to_originality_response(
            model_response, request, start_time
        )
    
    def _extract_originality_score(self, model_response) -> float:
        """
        Extract originality score from model response.
        This is a skeleton implementation using mock data.
        
        Args:
            model_response: ModelResponse from LLM
            
        Returns:
            Originality score [0.1, 0.9]
        """
        try:
            # Check if mock response has originality score
            if hasattr(model_response, 'originality_score'):
                score = model_response.originality_score
            else:
                # Parse from content (skeleton implementation)
                content = model_response.content.lower()
                
                # Simple heuristic based on keywords (mock logic)
                if any(word in content for word in ['high', 'novel', 'innovative', 'original']):
                    score = random.uniform(0.6, 0.9)
                elif any(word in content for word in ['medium', 'moderate', 'some']):
                    score = random.uniform(0.4, 0.7)
                else:
                    score = random.uniform(0.1, 0.5)
            
            # Ensure score is in valid range
            return max(0.1, min(0.9, score))
            
        except Exception as e:
            logger.warning(f"Error extracting originality score: {e}")
            return 0.5  # Safe default
    
    def _build_additional_fields(self, model_response, parameters: Dict[str, Any],
                               processing_time_ms: float) -> Dict[str, Any]:
        """
        Build additional response fields based on request parameters.
        
        Args:
            model_response: ModelResponse from LLM
            parameters: Request parameters
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            Dictionary of additional fields
        """
        additional_fields = {}
        
        # Add processing time
        additional_fields['processing_time_ms'] = round(processing_time_ms, 2)
        
        # Add cache hit information if available
        if hasattr(model_response, 'cache_hit'):
            additional_fields['cache_hit'] = model_response.cache_hit
        else:
            additional_fields['cache_hit'] = False
        
        # Add model metadata if requested
        if parameters.get('include_model_metadata', False):
            model_id = getattr(model_response, 'model_id', 'unknown')
            temperature = getattr(model_response, 'temperature', 0.7)
            additional_fields['model_metadata'] = self.llm_orchestrator.get_model_metadata_string(
                model_id, temperature
            )
        
        # Add dependency analysis if requested (skeleton)
        if parameters.get('include_dependency_analysis', False):
            additional_fields['dependency_analysis'] = self._generate_mock_dependency_analysis(
                parameters
            )
        
        # Add any other requested fields
        if parameters.get('include_cost', False):
            additional_fields['cost_usd'] = getattr(model_response, 'cost_usd', 0.0)
        
        if parameters.get('include_tokens', False):
            additional_fields['tokens_used'] = getattr(model_response, 'tokens_used', 0)
        
        return additional_fields
    
    def enable_special_case(self, enabled: bool):
        """
        Enable or disable special case handling.
        
        Args:
            enabled: Whether to enable special case handling
        """
        self._special_case_enabled = enabled
        logger.info(f"Special case originality assessment {'enabled' if enabled else 'disabled'}")
    
    def get_special_case_stats(self) -> Dict[str, Any]:
        """
        Get statistics about special case originality data.
        
        Returns:
            Dictionary with statistics about available originality data
        """
        try:
            if not self.originality_data_dir.exists():
                return {
                    "enabled": self._special_case_enabled,
                    "data_valid": False,
                    "available_repositories": 0,
                    "error": "Originality data directory not found"
                }
            
            # Count available repositories
            repo_count = 0
            for owner_dir in self.originality_data_dir.iterdir():
                if owner_dir.is_dir():
                    for repo_dir in owner_dir.iterdir():
                        if repo_dir.is_dir():
                            assessment_file = repo_dir / "detailed_originality_assessments_with_uncertainty.json"
                            if assessment_file.exists():
                                repo_count += 1
            
            return {
                "enabled": self._special_case_enabled,
                "data_valid": repo_count > 0,
                "available_repositories": repo_count,
                "data_directory": str(self.originality_data_dir)
            }
            
        except Exception as e:
            logger.error(f"Error getting originality special case stats: {e}")
            return {
                "enabled": self._special_case_enabled,
                "data_valid": False,
                "available_repositories": 0,
                "error": str(e)
            }
    
    def get_bulk_cached_originality(self) -> Dict[str, Any]:
        """
        Load and return bulk cached originality assessment results for all processed repositories.
        
        Returns:
            Dictionary with cached originality assessment results
        """
        try:
            logger.info("Loading bulk cached originality assessments")
            
            if not self.originality_data_dir.exists():
                return {
                    "error": "Originality data directory not found",
                    "total_repositories": 0,
                    "assessments": []
                }
            
            assessments = []
            
            # Traverse all owner/repo directories
            for owner_dir in self.originality_data_dir.iterdir():
                if not owner_dir.is_dir() or owner_dir.name == "originality_list.csv":
                    continue
                    
                for repo_dir in owner_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue
                    
                    try:
                        # Load detailed assessment data
                        assessment_file = repo_dir / "detailed_originality_assessments_with_uncertainty.json"
                        scores_file = repo_dir / "originality_scores.json"
                        uncertainty_file = repo_dir / "uncertainty_metrics.json"
                        
                        if not assessment_file.exists():
                            logger.warning(f"Missing assessment file for {owner_dir.name}/{repo_dir.name}")
                            continue
                        
                        # Load assessment data
                        with open(assessment_file, 'r') as f:
                            assessment_data = json.load(f)
                        
                        # Handle list format (take first item)
                        if isinstance(assessment_data, list) and len(assessment_data) > 0:
                            assessment_data = assessment_data[0]
                        elif isinstance(assessment_data, list):
                            logger.warning(f"Empty assessment data for {owner_dir.name}/{repo_dir.name}")
                            continue
                        
                        # Load scores data if available
                        scores_data = {}
                        if scores_file.exists():
                            with open(scores_file, 'r') as f:
                                scores_data = json.load(f)
                        
                        # Load uncertainty metrics if available
                        uncertainty_data = {}
                        if uncertainty_file.exists():
                            with open(uncertainty_file, 'r') as f:
                                uncertainty_data = json.load(f)
                        
                        # Extract key information
                        repository_url = assessment_data.get('repository_url', f"https://github.com/{owner_dir.name}/{repo_dir.name}")
                        repository_name = assessment_data.get('repository_name', repo_dir.name)
                        originality_score = assessment_data.get('final_originality_score', 0.5)
                        category = assessment_data.get('originality_category', 'Unknown')
                        confidence = assessment_data.get('assessment_confidence', 0.7)
                        overall_uncertainty = assessment_data.get('overall_reasoning_uncertainty', 0.3)
                        aggregate_uncertainty = assessment_data.get('aggregate_uncertainty', 0.3)
                        
                        # Extract criteria scores
                        criteria_scores = {}
                        criteria_raw = assessment_data.get('criteria_scores', {})
                        for criterion, details in criteria_raw.items():
                            if isinstance(details, dict):
                                criteria_scores[criterion] = {
                                    'score': details.get('score', 0),
                                    'weight': details.get('weight', 0),
                                    'reasoning': details.get('reasoning', ''),
                                    'uncertainty': details.get('raw_uncertainty', 0.3)
                                }
                        
                        # Extract criteria uncertainties
                        criteria_uncertainties = assessment_data.get('criteria_uncertainties', {})
                        
                        # Create assessment entry
                        assessment_entry = {
                            "repository_url": repository_url,
                            "repository_name": repository_name,
                            "owner": owner_dir.name,
                            "repo": repo_dir.name,
                            "originality_score": originality_score,
                            "originality_category": category,
                            "assessment_confidence": confidence,
                            "overall_uncertainty": overall_uncertainty,
                            "aggregate_uncertainty": aggregate_uncertainty,
                            "criteria_scores": criteria_scores,
                            "criteria_uncertainties": criteria_uncertainties,
                            "method": "pre_computed_originality"
                        }
                        
                        # Add uncertainty metrics if available
                        if uncertainty_data:
                            assessment_entry["uncertainty_metrics"] = uncertainty_data
                        
                        # Add simple score mapping if available
                        if repository_url in scores_data:
                            assessment_entry["simple_score"] = scores_data[repository_url]
                        
                        assessments.append(assessment_entry)
                        
                    except Exception as e:
                        logger.warning(f"Error processing originality data for {owner_dir.name}/{repo_dir.name}: {e}")
                        continue
            
            logger.info(f"Loaded {len(assessments)} bulk cached originality assessments")
            
            # Extract repository URLs for summary
            repository_urls = [a["repository_url"] for a in assessments]
            
            return {
                "total_repositories": len(assessments),
                "repositories": repository_urls,
                "assessments": assessments,
                "metadata": {
                    "data_source": "data/processed/originality/*/",
                    "method": "pre_computed_cached",
                    "assessment_type": "originality",
                    "files_used": [
                        "detailed_originality_assessments_with_uncertainty.json",
                        "originality_scores.json",
                        "uncertainty_metrics.json"
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading bulk cached originality: {e}", exc_info=True)
            return {
                "error": str(e),
                "total_repositories": 0,
                "assessments": []
            }

    def _generate_mock_dependency_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate mock dependency analysis for skeleton implementation.
        
        Args:
            parameters: Request parameters
            
        Returns:
            Mock dependency analysis data
        """
        # Get dependency count from parameters or generate random
        dependency_count = parameters.get('dependency_count', random.randint(5, 50))
        
        return {
            'total_dependencies': dependency_count,
            'core_dependencies': random.randint(1, max(1, dependency_count // 3)),
            'dev_dependencies': random.randint(0, dependency_count // 2),
            'optional_dependencies': random.randint(0, dependency_count // 4),
            'analysis_method': 'mock_static_analysis',
            'confidence': random.uniform(0.7, 0.95),
            'note': 'This is mock data from skeleton implementation'
        }