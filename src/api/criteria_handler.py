# src/api/criteria_handler.py
"""
Handler for repository criteria assessment requests.
Uses existing criteria assessment pipeline components.
"""

import logging
import time
from typing import Dict, Any, Optional

from src.api.criteria_models import CriteriaRequest, CriteriaResponse, CriterionScore
from src.api.llm_orchestrator import LLMOrchestrator
from src.uncertainty_calibration.criteria_assessment.repo_extractor import RepositoryExtractor
from src.uncertainty_calibration.criteria_assessment.criteria_prompt_generator import CriteriaPromptGenerator
from src.uncertainty_calibration.criteria_assessment.fuzzy_response_parser import FuzzyCriteriaResponseParser

logger = logging.getLogger(__name__)

class CriteriaHandler:
    """
    Handles repository criteria assessment requests.
    """
    
    def __init__(self, llm_orchestrator: LLMOrchestrator):
        """
        Initialize criteria handler.
        
        Args:
            llm_orchestrator: LLM orchestrator service
        """
        self.llm_orchestrator = llm_orchestrator
        
        # Initialize criteria assessment components
        self.repo_extractor = RepositoryExtractor()
        self.prompt_generator = CriteriaPromptGenerator()
        self.response_parser = FuzzyCriteriaResponseParser()
        
        logger.info("CriteriaHandler initialized")
    
    async def handle_criteria_assessment(self, request: CriteriaRequest) -> CriteriaResponse:
        """
        Handle repository criteria assessment request.
        
        Args:
            request: Criteria assessment request
            
        Returns:
            CriteriaResponse with detailed criteria evaluation
        """
        try:
            start_time = time.time()
            
            # Extract repository information
            repo_info = self.repo_extractor.get_repo_info(request.repo)
            
            logger.info(f"Processing criteria assessment for: {repo_info['name']}")
            
            # Get model and temperature from parameters
            model_id = request.parameters.get('model_id')
            temperature = request.parameters.get('temperature', 0.0)
            
            # Generate criteria assessment prompt
            prompt_messages = self.prompt_generator.create_criteria_assessment_prompt(repo_info)
            
            # Query LLM using orchestrator (reuse existing infrastructure)
            model_response = await self._query_llm_with_logprobs(
                model_id, prompt_messages, temperature
            )
            
            # Parse response with enhanced uncertainty calculation
            parsed_response = self.response_parser.parse_response(
                raw_response=model_response.content,
                repo_url=request.repo,
                repo_name=repo_info.get('name', 'unknown'),
                logprobs_data=self._extract_logprobs_data(model_response)
            )
            
            # Transform to API response format
            api_response = self._transform_to_criteria_response(
                parsed_response, request, start_time, model_response
            )
            
            logger.info(f"Criteria assessment completed: target_score={api_response.target_score:.2f}")
            
            return api_response
            
        except Exception as e:
            logger.error(f"Error in criteria assessment: {e}")
            raise
    
    async def _query_llm_with_logprobs(self, model_id: Optional[str], 
                                     prompt_messages: list, temperature: float):
        """
        Query LLM with logprobs enabled, reusing existing MultiModelEngine.
        
        Args:
            model_id: Model identifier
            prompt_messages: Prompt in OpenAI format
            temperature: Sampling temperature
            
        Returns:
            ModelResponse with logprobs data
        """
        # Use default model if not specified
        if model_id is None:
            model_id = self.llm_orchestrator._get_default_model()
        
        # Query using existing MultiModelEngine (includes caching and logprobs)
        model_response = self.llm_orchestrator.multi_model_engine.query_single_model_with_temperature(
            model_id=model_id,
            prompt=prompt_messages,
            temperature=temperature
        )
        
        if not model_response.success:
            raise ValueError(f"LLM query failed: {model_response.error}")
        
        return model_response
    
    def _extract_logprobs_data(self, model_response) -> Optional[list]:
        """
        Extract logprobs data from ModelResponse for perplexity calculation.
        
        Args:
            model_response: ModelResponse from MultiModelEngine
            
        Returns:
            List of token logprob dictionaries or None
        """
        try:
            # Access the enhanced full_content_logprobs from ModelResponse
            if hasattr(model_response, 'full_content_logprobs') and model_response.full_content_logprobs:
                return model_response.full_content_logprobs
            
            # Fallback: try to extract from answer_token_info if available
            if hasattr(model_response, 'answer_token_info') and model_response.answer_token_info:
                # This won't have full content, but better than nothing
                logger.warning("Using limited logprobs data from answer_token_info")
                return None
            
            logger.warning("No logprobs data available for perplexity calculation")
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting logprobs data: {e}")
            return None
    
    def _transform_to_criteria_response(self, parsed_response, request: CriteriaRequest,
                                      start_time: float, model_response) -> CriteriaResponse:
        """
        Transform ParsedCriteriaResponse to CriteriaResponse API format.
        
        Args:
            parsed_response: ParsedCriteriaResponse from parser
            request: Original request
            start_time: Request start time
            model_response: Original ModelResponse
            
        Returns:
            CriteriaResponse in API format
        """
        try:
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Transform criteria_scores to match API model
            criteria_scores = {}
            for criterion_id, assessment in parsed_response.criteria_scores.items():
                criteria_scores[criterion_id] = CriterionScore(
                    score=assessment["score"],
                    weight=assessment["weight"],
                    reasoning=assessment["reasoning"],
                    raw_uncertainty=assessment.get("raw_uncertainty", 0.5)
                )
            
            # Build additional fields based on request parameters
            additional_fields = self._build_additional_fields(
                model_response, request.parameters, processing_time_ms
            )
            
            return CriteriaResponse(
                repository_url=parsed_response.repository_url,
                repository_name=parsed_response.repository_name,
                criteria_scores=criteria_scores,
                target_score=parsed_response.target_score,
                raw_target_score=parsed_response.raw_target_score,
                total_weight=parsed_response.total_weight,
                overall_reasoning=parsed_response.overall_reasoning,
                parsing_method=parsed_response.parsing_method,
                parsing_success=parsed_response.parsing_success,
                parsing_warnings=parsed_response.parsing_warnings,
                **additional_fields
            )
            
        except Exception as e:
            logger.error(f"Error transforming criteria response: {e}")
            raise
    
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
        
        # Add model metadata if requested
        if parameters.get('include_model_metadata', False):
            model_id = getattr(model_response, 'model_id', 'unknown')
            temperature = getattr(model_response, 'temperature', 0.0)
            additional_fields['model_metadata'] = self.llm_orchestrator.get_model_metadata_string(
                model_id, temperature
            )
        
        # Add cost if requested
        if parameters.get('include_cost', False):
            additional_fields['cost_usd'] = getattr(model_response, 'cost_usd', 0.0)
        
        return additional_fields