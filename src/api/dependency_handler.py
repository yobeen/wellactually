# src/api/dependency_handler.py
"""
Handler for dependency comparison requests.
Uses dependency comparison pipeline components for Level 3 assessments.
"""

import logging
import time
from typing import Dict, Any, Optional

from src.api.dependency_models import DependencyComparisonRequest, DependencyComparisonResponse, DimensionAssessment, OverallAssessment
from src.api.llm_orchestrator import LLMOrchestrator
from src.uncertainty_calibration.dependency_context_extractor import DependencyContextExtractor
from src.uncertainty_calibration.level3_prompts import Level3PromptGenerator
from src.uncertainty_calibration.dependency_response_parser import DependencyResponseParser

logger = logging.getLogger(__name__)

class DependencyHandler:
    """
    Handles dependency comparison requests using Level 3 assessment framework.
    """
    
    def __init__(self, llm_orchestrator: LLMOrchestrator,
                 parent_csv_path: str = "data/external/parent_repos.csv",
                 dependencies_csv_path: str = "data/external/dependencies.csv"):
        """
        Initialize dependency handler.
        
        Args:
            llm_orchestrator: LLM orchestrator service
            parent_csv_path: Path to parent repositories CSV
            dependencies_csv_path: Path to dependencies CSV
        """
        self.llm_orchestrator = llm_orchestrator
        
        # Initialize dependency comparison components
        self.context_extractor = DependencyContextExtractor(parent_csv_path, dependencies_csv_path)
        self.prompt_generator = Level3PromptGenerator()
        self.response_parser = DependencyResponseParser()
        
        logger.info("DependencyHandler initialized")
    
    async def handle_dependency_comparison(self, request: DependencyComparisonRequest) -> DependencyComparisonResponse:
        """
        Handle dependency comparison request.
        
        Args:
            request: Dependency comparison request
            
        Returns:
            DependencyComparisonResponse with detailed dependency evaluation
        """
        try:
            start_time = time.time()
            
            # Extract repository contexts from CSV data
            comparison_context = self.context_extractor.extract_comparison_context(
                parent_url=request.parent_repo,
                dep_a_url=request.dependency_a,
                dep_b_url=request.dependency_b
            )
            
            logger.info(f"Processing dependency comparison: {comparison_context['dependency_a']['name']} vs "
                       f"{comparison_context['dependency_b']['name']} for parent {comparison_context['parent']['name']}")
            
            # Get model and temperature from parameters
            model_id = request.parameters.get('model_id')
            temperature = request.parameters.get('temperature', 0.0)
            
            # Generate dependency comparison prompt
            prompt_messages = self.prompt_generator.create_dependency_comparison_prompt(
                parent_context=comparison_context['parent'],
                dep_a_context=comparison_context['dependency_a'],
                dep_b_context=comparison_context['dependency_b']
            )
            
            # Query LLM using orchestrator (reuse existing infrastructure)
            model_response = await self._query_llm_with_logprobs(
                model_id, prompt_messages, temperature
            )
            
            # Parse response with enhanced uncertainty calculation
            parsed_response = self.response_parser.parse_response(
                raw_response=model_response.content,
                parent_url=request.parent_repo,
                dep_a_url=request.dependency_a,
                dep_b_url=request.dependency_b,
                logprobs_data=self._extract_logprobs_data(model_response)
            )
            
            # Transform to API response format
            api_response = self._transform_to_dependency_response(
                parsed_response, request, start_time, model_response
            )
            
            logger.info(f"Dependency comparison completed: choice={api_response.overall_assessment.choice}, "
                       f"confidence={api_response.overall_assessment.confidence:.2f}")
            
            return api_response
            
        except ValueError as e:
            # Handle validation errors (missing repos, etc.)
            logger.warning(f"Validation error in dependency comparison: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in dependency comparison: {e}")
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
                logger.warning("Using limited logprobs data from answer_token_info")
                return None
            
            logger.warning("No logprobs data available for perplexity calculation")
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting logprobs data: {e}")
            return None
    
    def _transform_to_dependency_response(self, parsed_response, request: DependencyComparisonRequest,
                                        start_time: float, model_response) -> DependencyComparisonResponse:
        """
        Transform ParsedDependencyResponse to DependencyComparisonResponse API format.
        
        Args:
            parsed_response: ParsedDependencyResponse from parser
            request: Original request
            start_time: Request start time
            model_response: Original ModelResponse
            
        Returns:
            DependencyComparisonResponse in API format
        """
        try:
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Transform dimension_assessments to match API model
            dimension_assessments = {}
            for dimension_id, assessment in parsed_response.dimension_assessments.items():
                dimension_assessments[dimension_id] = DimensionAssessment(
                    score_a=assessment["score_a"],
                    score_b=assessment["score_b"],
                    weight=assessment["weight"],
                    reasoning=assessment["reasoning"],
                    raw_uncertainty=assessment.get("raw_uncertainty", 0.5)
                )
            
            # Transform overall_assessment
            overall_assessment_data = parsed_response.overall_assessment
            overall_assessment = OverallAssessment(
                choice=overall_assessment_data.get("choice", "Equal"),
                confidence=overall_assessment_data.get("confidence", 0.5),
                weighted_score_a=overall_assessment_data.get("weighted_score_a", 0.0),
                weighted_score_b=overall_assessment_data.get("weighted_score_b", 0.0),
                reasoning=overall_assessment_data.get("reasoning", "No reasoning provided")
            )
            
            # Build additional fields based on request parameters
            additional_fields = self._build_additional_fields(
                model_response, request.parameters, processing_time_ms
            )
            
            return DependencyComparisonResponse(
                parent_url=parsed_response.parent_url,
                parent_name=parsed_response.parent_name,
                dependency_a_url=parsed_response.dependency_a_url,
                dependency_a_name=parsed_response.dependency_a_name,
                dependency_b_url=parsed_response.dependency_b_url,
                dependency_b_name=parsed_response.dependency_b_name,
                dimension_assessments=dimension_assessments,
                overall_assessment=overall_assessment,
                parsing_method=parsed_response.parsing_method,
                parsing_success=parsed_response.parsing_success,
                parsing_warnings=parsed_response.parsing_warnings,
                **additional_fields
            )
            
        except Exception as e:
            logger.error(f"Error transforming dependency response: {e}")
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
    
    def validate_csv_data(self) -> Dict[str, Any]:
        """
        Validate that required CSV files are available and properly formatted.
        
        Returns:
            Validation results dictionary
        """
        try:
            return self.context_extractor.validate_csv_schemas()
        except Exception as e:
            logger.error(f"Error validating CSV data: {e}")
            return {
                "error": str(e),
                "parent_csv": {"valid": False, "errors": ["Validation failed"]},
                "dependencies_csv": {"valid": False, "errors": ["Validation failed"]}
            }
    
    def get_available_comparisons(self, parent_url: str) -> Dict[str, Any]:
        """
        Get available dependencies for a parent repository.
        
        Args:
            parent_url: Parent repository URL
            
        Returns:
            Dictionary with available dependencies
        """
        try:
            dependencies_df = self.context_extractor.get_dependencies_for_parent(parent_url)
            
            return {
                "parent_url": parent_url,
                "available_dependencies": dependencies_df.to_dict('records'),
                "count": len(dependencies_df)
            }
            
        except Exception as e:
            logger.error(f"Error getting available comparisons: {e}")
            return {
                "parent_url": parent_url,
                "error": str(e),
                "available_dependencies": [],
                "count": 0
            }