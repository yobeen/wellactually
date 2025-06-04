# src/api/handlers/comparison_handler.py
"""
Handler for repository comparison requests.
Routes between L1 (ethereum ecosystem) and L3 (dependency) comparisons.
"""

import logging
import random
import time
from typing import Dict, Any, Optional

from src.api.requests import ComparisonRequest
from src.api.responses import ComparisonResponse
from src.api.llm_orchestrator import LLMOrchestrator

logger = logging.getLogger(__name__)

class ComparisonHandler:
    """
    Handles repository comparison requests with L1/L3 routing.
    """
    
    def __init__(self, llm_orchestrator: LLMOrchestrator):
        """
        Initialize comparison handler.
        
        Args:
            llm_orchestrator: LLM orchestrator service
        """
        self.llm_orchestrator = llm_orchestrator
        logger.info("ComparisonHandler initialized")
    
    async def handle_l1_comparison(self, request: ComparisonRequest) -> ComparisonResponse:
        """
        Handle Level 1 comparison (seed repositories to ethereum ecosystem).
        
        Args:
            request: Comparison request
            
        Returns:
            ComparisonResponse with analysis results
        """
        try:
            start_time = time.time()
            
            # Extract repository information
            repo_a_info = self.llm_orchestrator.extract_repo_info(request.repo_a)
            repo_b_info = self.llm_orchestrator.extract_repo_info(request.repo_b)
            
            logger.info(f"Processing L1 comparison: {repo_a_info['name']} vs {repo_b_info['name']}")
            
            # Get model and temperature from parameters
            model_id = request.parameters.get('model_id')
            temperature = request.parameters.get('temperature', 0.7)
            
            # Query LLM using orchestrator
            model_response = await self.llm_orchestrator.query_l1_comparison(
                repo_a_info=repo_a_info,
                repo_b_info=repo_b_info,
                model_id=model_id,
                temperature=temperature
            )
            
            # Transform to API response format
            if model_response.success:
                api_response = self._transform_to_comparison_response(
                    model_response, request, start_time
                )
                
                logger.info(f"L1 comparison completed: choice={api_response.choice}, "
                          f"multiplier={api_response.multiplier:.2f}")
                
                return api_response
            else:
                raise ValueError(f"LLM query failed: {model_response.error}")
            
        except Exception as e:
            logger.error(f"Error in L1 comparison: {e}")
            raise
    
    async def handle_l3_comparison(self, request: ComparisonRequest) -> ComparisonResponse:
        """
        Handle Level 3 comparison (dependencies within parent repository).
        This is a skeleton implementation.
        
        Args:
            request: Comparison request
            
        Returns:
            ComparisonResponse with skeleton results
        """
        try:
            start_time = time.time()
            
            # Extract repository information
            dep_a_info = self.llm_orchestrator.extract_repo_info(request.repo_a)
            dep_b_info = self.llm_orchestrator.extract_repo_info(request.repo_b)
            parent_info = self.llm_orchestrator.extract_repo_info(request.parent)
            
            logger.info(f"Processing L3 comparison: {dep_a_info['name']} vs {dep_b_info['name']} "
                       f"for parent {parent_info['name']} (SKELETON)")
            
            # Get model and temperature from parameters
            model_id = request.parameters.get('model_id')
            temperature = request.parameters.get('temperature', 0.7)
            
            # Query LLM using orchestrator (skeleton implementation)
            model_response = await self.llm_orchestrator.query_l3_comparison(
                dep_a_info=dep_a_info,
                dep_b_info=dep_b_info,
                parent_info=parent_info,
                model_id=model_id,
                temperature=temperature
            )
            
            # Transform to API response format
            api_response = self._transform_to_comparison_response(
                model_response, request, start_time, is_skeleton=True
            )
            
            logger.info(f"L3 comparison completed (SKELETON): choice={api_response.choice}, "
                       f"multiplier={api_response.multiplier:.2f}")
            
            return api_response
            
        except Exception as e:
            logger.error(f"Error in L3 comparison: {e}")
            raise
    
    def _transform_to_comparison_response(self, model_response, request: ComparisonRequest,
                                        start_time: float, is_skeleton: bool = False) -> ComparisonResponse:
        """
        Transform ModelResponse to ComparisonResponse API format.
        
        Args:
            model_response: ModelResponse from LLM orchestrator
            request: Original request
            start_time: Request start time for calculating processing time
            is_skeleton: Whether this is a skeleton implementation
            
        Returns:
            ComparisonResponse in expected API format
        """
        try:
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Map choice from A/B/Equal to 1/2 numeric format
            choice = self._map_choice_to_numeric(model_response.raw_choice)
            
            # Calculate multiplier from uncertainty (inverse relationship)
            multiplier = self._calculate_multiplier_from_uncertainty(model_response.uncertainty)
            
            # Get uncertainty measures
            choice_uncertainty = model_response.uncertainty
            multiplier_uncertainty = self._calculate_multiplier_uncertainty(model_response.uncertainty)
            
            # Get explanation
            explanation = model_response.content
            if is_skeleton:
                explanation = f"[SKELETON IMPLEMENTATION] {explanation}"
            
            # Build additional fields based on request parameters
            additional_fields = self._build_additional_fields(
                model_response, request.parameters, processing_time_ms
            )
            
            return ComparisonResponse(
                choice=choice,
                multiplier=multiplier,
                choice_uncertainty=choice_uncertainty,
                multiplier_uncertainty=multiplier_uncertainty,
                explanation=explanation,
                **additional_fields
            )
            
        except Exception as e:
            logger.error(f"Error transforming response: {e}")
            raise
    
    def _map_choice_to_numeric(self, raw_choice: str) -> int:
        """
        Map A/B/Equal choice to 1/2 numeric format.
        
        Args:
            raw_choice: Raw choice from LLM (A, B, Equal, etc.)
            
        Returns:
            1 for A, 2 for B, random for Equal
        """
        if not raw_choice:
            return random.choice([1, 2])
        
        choice_upper = str(raw_choice).upper().strip()
        
        if choice_upper in ['A', '1']:
            return 1
        elif choice_upper in ['B', '2']:
            return 2
        elif choice_upper in ['EQUAL', 'EQUALS', 'TIE']:
            # For equal, randomly choose 1 or 2
            return random.choice([1, 2])
        else:
            # Default fallback
            logger.warning(f"Unknown choice format: {raw_choice}, defaulting to random")
            return random.choice([1, 2])
    
    def _calculate_multiplier_from_uncertainty(self, uncertainty: float) -> float:
        """
        Calculate multiplier from uncertainty score.
        Lower uncertainty = higher confidence = higher multiplier.
        
        Args:
            uncertainty: Uncertainty score [0, 1]
            
        Returns:
            Multiplier value [1.0, 999.0]
        """
        try:
            # Clamp uncertainty to valid range
            uncertainty = max(0.0, min(1.0, uncertainty))
            
            # Convert uncertainty to confidence
            confidence = 1.0 - uncertainty
            
            # Map confidence to multiplier using exponential scaling
            # confidence=0.5 -> multiplier=1.0 (no preference)
            # confidence=0.9 -> multiplier~10
            # confidence=0.99 -> multiplier~100
            
            if confidence <= 0.5:
                # Low confidence = low multiplier
                multiplier = 1.0 + (confidence * 2.0)  # [1.0, 2.0]
            else:
                # High confidence = exponential scaling
                normalized_confidence = (confidence - 0.5) * 2.0  # [0, 1]
                multiplier = 1.0 + (normalized_confidence ** 2) * 20.0  # [1.0, 21.0]
            
            # Apply additional scaling for very high confidence
            if confidence > 0.9:
                extra_scaling = (confidence - 0.9) * 10.0  # [0, 1]
                multiplier *= (1.0 + extra_scaling * 4.0)  # Up to 5x additional
            
            # Clamp to valid range and add some randomness for variety
            multiplier = max(1.0, min(999.0, multiplier))
            
            # Add small random factor for variety (±10%)
            random_factor = random.uniform(0.9, 1.1)
            multiplier *= random_factor
            
            return max(1.0, min(999.0, round(multiplier, 2)))
            
        except Exception as e:
            logger.warning(f"Error calculating multiplier: {e}")
            return 2.0  # Safe default
    
    def _calculate_multiplier_uncertainty(self, choice_uncertainty: float) -> float:
        """
        Calculate multiplier uncertainty from choice uncertainty.
        
        Args:
            choice_uncertainty: Uncertainty in choice decision
            
        Returns:
            Multiplier uncertainty [0, 1]
        """
        # Multiplier uncertainty is typically correlated with choice uncertainty
        # but may be slightly higher due to additional complexity in magnitude estimation
        return min(1.0, choice_uncertainty * 1.2 + 0.05)
    
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
        
        # Add any other requested fields
        if parameters.get('include_cost', False):
            additional_fields['cost_usd'] = getattr(model_response, 'cost_usd', 0.0)
        
        if parameters.get('include_tokens', False):
            additional_fields['tokens_used'] = getattr(model_response, 'tokens_used', 0)
        
        return additional_fields