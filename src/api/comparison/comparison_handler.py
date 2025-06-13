# src/api/comparison_handler.py
"""
Handler for repository comparison requests.
Routes between L1 (ethereum ecosystem) and L3 (dependency) comparisons.
Now includes special case for L1 using criteria assessment data.
"""

import logging
import random
import time
from typing import Dict, Any, Optional

from src.api.core.requests import ComparisonRequest
from src.api.core.responses import ComparisonResponse
from src.api.core.llm_orchestrator import LLMOrchestrator
from src.api.criteria.criteria_assessment_loader import CriteriaAssessmentLoader
from src.api.criteria.score_based_comparator import ScoreBasedComparator

logger = logging.getLogger(__name__)

class ComparisonHandler:
    """
    Handles repository comparison requests with L1/L3 routing and special case handling.
    """
    
    def __init__(self, llm_orchestrator: LLMOrchestrator,
                 criteria_assessment_path: str = "data/processed/criteria_assessment/detailed_assessments.json"):
        """
        Initialize comparison handler.
        
        Args:
            llm_orchestrator: LLM orchestrator service
            criteria_assessment_path: Path to criteria assessment JSON file
        """
        self.llm_orchestrator = llm_orchestrator
        
        # Initialize special case components
        self.criteria_loader = CriteriaAssessmentLoader(criteria_assessment_path)
        self.score_comparator = ScoreBasedComparator()
        self._special_case_enabled = True
        
        logger.info("ComparisonHandler initialized with special case support")
    
    async def handle_l1_comparison(self, request: ComparisonRequest) -> ComparisonResponse:
        """
        Handle Level 1 comparison (seed repositories to ethereum ecosystem).
        Uses special case logic if criteria assessments are available.
        
        Args:
            request: Comparison request
            
        Returns:
            ComparisonResponse with analysis results
        """
        try:
            start_time = time.time()
            
            # Check for special case conditions
            if self._should_use_special_case(request):
                logger.info(f"Using special case (criteria-based) comparison for L1")
                return await self._handle_l1_special_case(request, start_time)
            else:
                logger.info(f"Using standard LLM-based comparison for L1")
                return await self._handle_l1_standard(request, start_time)
                
        except Exception as e:
            logger.error(f"Error in L1 comparison: {e}")
            # Fallback to standard comparison on any special case error
            if self._should_use_special_case(request):
                logger.warning("Special case failed, falling back to standard LLM comparison")
                return await self._handle_l1_standard(request, start_time)
            raise
    
    async def handle_l3_comparison(self, request: ComparisonRequest) -> ComparisonResponse:
        """
        Handle Level 3 comparison (dependencies within parent repository).
        
        Args:
            request: Comparison request
            
        Returns:
            ComparisonResponse with dependency comparison results
        """
        try:
            start_time = time.time()
            
            logger.info(f"Starting L3 comparison processing...")
            logger.info(f"  Repo A: {request.repo_a}")
            logger.info(f"  Repo B: {request.repo_b}")
            logger.info(f"  Parent: {request.parent}")
            logger.info(f"  Parameters: {request.parameters}")
            
            # Extract repository information
            logger.info("Extracting repository information...")
            try:
                dep_a_info = self.llm_orchestrator.extract_repo_info(request.repo_a)
                logger.info(f"  Dep A info extracted: {dep_a_info}")
            except Exception as e:
                logger.error(f"Failed to extract repo A info: {e}")
                raise
            
            try:
                dep_b_info = self.llm_orchestrator.extract_repo_info(request.repo_b)
                logger.info(f"  Dep B info extracted: {dep_b_info}")
            except Exception as e:
                logger.error(f"Failed to extract repo B info: {e}")
                raise
            
            try:
                parent_info = self.llm_orchestrator.extract_repo_info(request.parent)
                logger.info(f"  Parent info extracted: {parent_info}")
            except Exception as e:
                logger.error(f"Failed to extract parent info: {e}")
                raise
            
            logger.info(f"Processing L3 comparison: {dep_a_info['name']} vs {dep_b_info['name']} "
                       f"for parent {parent_info['name']}")
            
            # Get model, temperature, and simplified option from parameters
            model_id = request.parameters.get('model_id')
            temperature = request.parameters.get('temperature', 0.4)
            simplified = request.parameters.get('simplified', False)
            
            # Determine actual model that will be used
            if simplified:
                actual_model_id = model_id
                logger.info(f"Using simplified mode: model={actual_model_id}, temperature={temperature}, max_tokens=20")
            else:
                actual_model_id = model_id
                logger.info(f"Using full mode: model={actual_model_id}, temperature={temperature}, simplified={simplified}")
            
            # Query LLM using orchestrator
            logger.info("Querying LLM orchestrator for L3 comparison...")
            try:
                model_response = await self.llm_orchestrator.query_l3_comparison(
                    dep_a_info=dep_a_info,
                    dep_b_info=dep_b_info,
                    parent_info=parent_info,
                    model_id=model_id,
                    temperature=temperature,
                    simplified=simplified
                )
                logger.info(f"LLM query completed: {type(model_response)}")
            except Exception as e:
                logger.error(f"LLM query failed: {e}")
                raise
            
            # Transform to API response format
            logger.info("Transforming to API response format...")
            try:
                api_response = self._transform_to_comparison_response(
                    model_response, request, start_time, is_skeleton=False, simplified=simplified
                )
                logger.info(f"Response transformation completed")
            except Exception as e:
                logger.error(f"Response transformation failed: {e}")
                raise
            
            # Log completion with conditional multiplier formatting
            if hasattr(api_response, 'multiplier') and api_response.multiplier is not None:
                logger.info(f"L3 comparison completed: choice={api_response.choice}, "
                           f"multiplier={api_response.multiplier:.2f}")
            else:
                logger.info(f"L3 comparison completed: choice={api_response.choice} (simplified mode)")
            
            return api_response
            
        except Exception as e:
            logger.error(f"Error in L3 comparison: {e}", exc_info=True)
            raise
    
    def _should_use_special_case(self, request: ComparisonRequest) -> bool:
        """
        Check if we should use special case (criteria-based) comparison.
        
        Args:
            request: Comparison request
            
        Returns:
            True if special case should be used, False otherwise
        """
        if not self._special_case_enabled:
            return False
        
        # Only for Level 1 comparisons (parent == "ethereum")
        if request.parent.lower() != "ethereum":
            return False
        
        # Check if criteria assessments are available for both repositories
        try:
            assessment_a = self.criteria_loader.get_assessment_by_url(request.repo_a)
            assessment_b = self.criteria_loader.get_assessment_by_url(request.repo_b)
            
            if assessment_a is None or assessment_b is None:
                missing_repos = []
                if assessment_a is None:
                    missing_repos.append(request.repo_a)
                if assessment_b is None:
                    missing_repos.append(request.repo_b)
                
                logger.info(f"Criteria assessments not available for: {missing_repos}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking special case availability: {e}")
            return False
    
    async def _handle_l1_special_case(self, request: ComparisonRequest, start_time: float) -> ComparisonResponse:
        """
        Handle L1 comparison using criteria assessment data.
        
        Args:
            request: Comparison request
            start_time: Request start time for timing
            
        Returns:
            ComparisonResponse based on criteria scores
        """
        # Load assessments
        assessment_a = self.criteria_loader.get_assessment_by_url(request.repo_a)
        assessment_b = self.criteria_loader.get_assessment_by_url(request.repo_b)
        
        if assessment_a is None or assessment_b is None:
            raise ValueError("Criteria assessments not available for comparison")
        
        logger.debug(f"Special case comparison: {assessment_a.repository_name} vs {assessment_b.repository_name}")
        
        # Perform score-based comparison
        comparison_result = self.score_comparator.compare_repositories(assessment_a, assessment_b)
        
        # Transform to API response format
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Build additional fields
        additional_fields = self._build_additional_fields_special(
            request.parameters, processing_time_ms, comparison_result
        )
        
        response = ComparisonResponse(
            choice=comparison_result.choice,
            multiplier=comparison_result.multiplier,
            choice_uncertainty=comparison_result.choice_uncertainty,
            multiplier_uncertainty=comparison_result.multiplier_uncertainty,
            explanation=comparison_result.explanation,
            **additional_fields
        )
        
        logger.info(f"Special case L1 comparison completed: choice={response.choice}, "
                   f"multiplier={response.multiplier:.2f}")
        
        return response
    
    async def _handle_l1_standard(self, request: ComparisonRequest, start_time: float) -> ComparisonResponse:
        """
        Handle L1 comparison using standard LLM approach.
        
        Args:
            request: Comparison request
            start_time: Request start time for timing
            
        Returns:
            ComparisonResponse from LLM analysis
        """
        # Extract repository information
        repo_a_info = self.llm_orchestrator.extract_repo_info(request.repo_a)
        repo_b_info = self.llm_orchestrator.extract_repo_info(request.repo_b)
        
        logger.info(f"Processing standard L1 comparison: {repo_a_info['name']} vs {repo_b_info['name']}")
        
        # Get model and temperature from parameters
        model_id = request.parameters.get('model_id')
        temperature = request.parameters.get('temperature', 0.4)
        
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
            
            logger.info(f"Standard L1 comparison completed: choice={api_response.choice}, "
                      f"multiplier={api_response.multiplier:.2f}")
            
            return api_response
        else:
            raise ValueError(f"LLM query failed: {model_response.error}")
    
    def _build_additional_fields_special(self, parameters: Dict[str, Any], 
                                       processing_time_ms: float,
                                       comparison_result) -> Dict[str, Any]:
        """
        Build additional response fields for special case comparisons.
        
        Args:
            parameters: Request parameters
            processing_time_ms: Processing time in milliseconds
            comparison_result: ComparisonResult from score-based comparison
            
        Returns:
            Dictionary of additional fields
        """
        additional_fields = {}
        
        # Add processing time
        additional_fields['processing_time_ms'] = round(processing_time_ms, 2)
        
        # Add cache hit information (special case is always "cached" since it's file-based)
        additional_fields['cache_hit'] = True
        
        # Add model metadata if requested (mark as criteria-based)
        if parameters.get('include_model_metadata', False):
            additional_fields['model_metadata'] = "criteria_based_comparison"
        
        # Add cost (special case has no API cost)
        if parameters.get('include_cost', False):
            additional_fields['cost_usd'] = 0.0
        
        # Add tokens (no tokens used for special case)
        if parameters.get('include_tokens', False):
            additional_fields['tokens_used'] = 0
        
        # Add special case debugging info if requested
        if parameters.get('include_debug_info', False):
            additional_fields['debug_info'] = {
                'method': 'criteria_based',
                'score_a': comparison_result.score_a,
                'score_b': comparison_result.score_b,
                'ratio': comparison_result.ratio,
                'reasoning_uncertainty': comparison_result.reasoning_uncertainty
            }
        
        return additional_fields
    
    def _transform_to_comparison_response(self, model_response, request: ComparisonRequest,
                                        start_time: float, is_skeleton: bool = False, simplified: bool = False) -> ComparisonResponse:
        """
        Transform ModelResponse to ComparisonResponse API format.
        
        Args:
            model_response: ModelResponse from LLM orchestrator
            request: Original request
            start_time: Request start time for calculating processing time
            is_skeleton: Whether this is a skeleton implementation
            simplified: Whether this is a simplified response (affects fields included)
            
        Returns:
            ComparisonResponse in expected API format
        """
        try:
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Map choice from A/B/Equal to 1/2 numeric format
            print(f"DEBUG: raw_choice before mapping = '{model_response.raw_choice}'")
            choice = self._map_choice_to_numeric(model_response.raw_choice)
            print(f"DEBUG: mapped choice = {choice}")
            
            # Get uncertainty measures with defensive check
            choice_uncertainty = model_response.uncertainty
            if choice_uncertainty is None:
                logger.warning("model_response.uncertainty is None for choice_uncertainty, using default 0.5")
                choice_uncertainty = 0.5
            
            # Get explanation
            explanation = model_response.content
            if is_skeleton:
                explanation = f"[SKELETON IMPLEMENTATION] {explanation}"
            
            # Build additional fields based on request parameters
            additional_fields = self._build_additional_fields(
                model_response, request.parameters, processing_time_ms
            )
            
            # Build response fields conditionally based on simplified mode
            response_fields = {
                "choice": choice,
                "choice_uncertainty": choice_uncertainty,
                "explanation": explanation,
                **additional_fields
            }
            
            # Only add multiplier fields for non-simplified responses
            if not simplified:
                # Defensive check for None uncertainty
                uncertainty = model_response.uncertainty
                if uncertainty is None:
                    logger.warning("model_response.uncertainty is None, using default 0.5")
                    uncertainty = 0.5
                
                multiplier = self._calculate_multiplier_from_uncertainty(uncertainty)
                multiplier_uncertainty = self._calculate_multiplier_uncertainty(uncertainty)
                
                # Ensure multiplier_uncertainty is never None
                if multiplier_uncertainty is None:
                    logger.warning("multiplier_uncertainty is None, using default 0.5")
                    multiplier_uncertainty = 0.5
                
                response_fields["multiplier"] = multiplier
                response_fields["multiplier_uncertainty"] = multiplier_uncertainty
            
            return ComparisonResponse(**response_fields)
            
        except Exception as e:
            logger.error(f"Error transforming response: {e}")
            raise
    
    def _map_choice_to_numeric(self, raw_choice: str):
        """
        Map A/B/Equal choice to appropriate format.
        
        Args:
            raw_choice: Raw choice from LLM (A, B, Equal, etc.)
            
        Returns:
            1 for A, 2 for B, 0 for Equal, or "E" for Equal (depending on context)
        """
        if not raw_choice:
            return random.choice([1, 2])
        
        choice_upper = str(raw_choice).upper().strip()
        
        if choice_upper in ['A', '1']:
            return 1
        elif choice_upper in ['B', '2']:
            return 2
        elif choice_upper in ['EQUAL', 'EQUALS', 'TIE']:
            # For equal, return 0 (or "E" - using 0 for now as it's more API-friendly)
            return 0
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
            temperature = getattr(model_response, 'temperature', 0.4)
            additional_fields['model_metadata'] = self.llm_orchestrator.get_model_metadata_string(
                model_id, temperature
            )
        
        # Add any other requested fields
        if parameters.get('include_cost', False):
            additional_fields['cost_usd'] = getattr(model_response, 'cost_usd', 0.0)
        
        if parameters.get('include_tokens', False):
            additional_fields['tokens_used'] = getattr(model_response, 'tokens_used', 0)
        
        return additional_fields
    
    def enable_special_case(self, enabled: bool = True):
        """
        Enable or disable special case handling.
        
        Args:
            enabled: Whether to enable special case handling
        """
        self._special_case_enabled = enabled
        logger.info(f"Special case handling {'enabled' if enabled else 'disabled'}")
    
    def validate_special_case_data(self) -> Dict[str, Any]:
        """
        Validate that special case data is available and properly formatted.
        
        Returns:
            Validation results dictionary
        """
        try:
            return self.criteria_loader.validate_assessment_data()
        except Exception as e:
            logger.error(f"Error validating special case data: {e}")
            return {
                "valid": False,
                "error": str(e),
                "total_assessments": 0,
                "errors": ["Validation failed"],
                "warnings": []
            }
    
    def get_special_case_stats(self) -> Dict[str, Any]:
        """
        Get statistics about special case availability.
        
        Returns:
            Statistics dictionary
        """
        try:
            available_repos = self.criteria_loader.get_available_repositories()
            validation_results = self.validate_special_case_data()
            
            return {
                "enabled": self._special_case_enabled,
                "available_repositories": len(available_repos),
                "data_valid": validation_results.get("valid", False),
                "sample_repositories": available_repos[:5],  # First 5 as examples
                "validation_summary": {
                    "errors": len(validation_results.get("errors", [])),
                    "warnings": len(validation_results.get("warnings", []))
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting special case stats: {e}")
            return {
                "enabled": self._special_case_enabled,
                "error": str(e),
                "available_repositories": 0,
                "data_valid": False
            }
    
    def get_bulk_cached_comparisons(self) -> Dict[str, Any]:
        """
        Load and return bulk cached L1 comparison results from pre-computed file.
        
        Returns:
            Dictionary with cached L1 comparison results
        """
        import json
        from pathlib import Path
        
        try:
            logger.info("Loading bulk cached L1 comparisons from file")
            
            comparison_file = Path("data/processed/l1_comparison_results.json")
            
            if not comparison_file.exists():
                return {
                    "error": "L1 comparison results file not found",
                    "total_comparisons": 0,
                    "comparisons": []
                }
            
            # Load the cached comparison results
            with open(comparison_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            comparisons = data.get("comparisons", [])
            
            # Extract unique repositories from the comparisons
            repositories = set()
            for comp in comparisons:
                repositories.add(comp.get("repo_a", ""))
                repositories.add(comp.get("repo_b", ""))
            repositories.discard("")  # Remove empty strings
            
            logger.info(f"Loaded {len(comparisons)} cached L1 comparisons for {len(repositories)} repositories")
            
            return {
                "total_repositories": len(repositories),
                "total_comparisons": len(comparisons),
                "repositories": sorted(list(repositories)),
                "comparisons": comparisons,
                "metadata": {
                    "data_source": "data/processed/l1_comparison_results.json",
                    "method": "pre_computed_cached",
                    "all_pairwise_combinations": True,
                    "comparison_level": "L1"
                }
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in L1 comparison results file: {e}")
            return {
                "error": f"Invalid JSON format: {str(e)}",
                "total_comparisons": 0,
                "comparisons": []
            }
        except Exception as e:
            logger.error(f"Error loading bulk cached comparisons: {e}", exc_info=True)
            return {
                "error": str(e),
                "total_comparisons": 0,
                "comparisons": []
            }
    
    async def handle_batch_comparison(self, pairs, parent, parameters=None) -> Dict[str, Any]:
        """
        Handle confidence-based batch comparison with uncertainty filtering.
        
        Args:
            pairs: List of dictionaries with repo_a and repo_b keys
            parent: Parent context (usually "ethereum")
            parameters: Optional parameters dictionary
            
        Returns:
            Dictionary with successful and filtered comparisons
        """
        self._special_case_enabled_tmp = self._special_case_enabled
        self._special_case_enabled = False
        # Uncertainty thresholds for models
        LLAMA_THRESHOLD = 0.00000034
        GPT4O_THRESHOLD = 0.00077255
        
        successful_comparisons = []
        filtered_comparisons = []
        
        llama_queries = 0
        gpt4o_queries = 0
        start_time = time.time()
        
        logger.info(f"Starting batch comparison processing for {len(pairs)} pairs")
        
        for i, pair in enumerate(pairs):
            try:
                logger.info(f"Processing pair {i+1}/{len(pairs)}: {pair['repo_a']} vs {pair['repo_b']}")
                
                # Create individual comparison request for llama-4-maverick first
                from src.api.core.requests import ComparisonRequest
                llama_request = ComparisonRequest(
                    repo_a=pair['repo_a'],
                    repo_b=pair['repo_b'],
                    parent=parent,
                    parameters={**(parameters or {}), "model_id": "meta-llama/llama-4-maverick", "temperature": 0.4, "simplified": True}
                )
                
                # Query llama-4-maverick first using proper routing
                if parent.lower() == "ethereum":
                    llama_response = await self.handle_l1_comparison(llama_request)
                else:
                    llama_response = await self.handle_l3_comparison(llama_request)
                
                llama_queries += 1
                logger.info(f"  Llama uncertainty: {llama_response.choice_uncertainty}")
                
                # Check if llama uncertainty is above threshold
                if llama_response.choice_uncertainty > LLAMA_THRESHOLD:
                    # Query gpt-4o next
                    logger.info(f"  Llama uncertainty {llama_response.choice_uncertainty} > {LLAMA_THRESHOLD}, trying GPT-4o")
                    
                    gpt4o_request = ComparisonRequest(
                        repo_a=pair['repo_a'],
                        repo_b=pair['repo_b'],
                        parent=parent,
                        parameters={**(parameters or {}), "model_id": "openai/gpt-4o", "temperature": 0.4, "simplified": True}
                    )
                    
                    # Query gpt-4o using proper routing
                    if parent.lower() == "ethereum":
                        gpt4o_response = await self.handle_l1_comparison(gpt4o_request)
                    else:
                        gpt4o_response = await self.handle_l3_comparison(gpt4o_request)
                    
                    gpt4o_queries += 1
                    logger.info(f"  GPT-4o uncertainty: {gpt4o_response.choice_uncertainty}")
                    
                    # Check if gpt-4o uncertainty is above threshold
                    if gpt4o_response.choice_uncertainty > GPT4O_THRESHOLD:
                        # Filter out this pair
                        logger.info(f"  Both models exceed thresholds, filtering out pair")
                        filtered_comparisons.append({
                            "repo_a": pair['repo_a'],
                            "repo_b": pair['repo_b'],
                            "reason": "High uncertainty on both models",
                            "llama_uncertainty": llama_response.choice_uncertainty,
                            "gpt4o_uncertainty": gpt4o_response.choice_uncertainty
                        })
                    else:
                        # Use gpt-4o result
                        logger.info(f"  GPT-4o uncertainty acceptable, using GPT-4o result")
                        successful_comparisons.append({
                            "repo_a": pair['repo_a'],
                            "repo_b": pair['repo_b'],
                            "choice": gpt4o_response.choice,
                            "multiplier": gpt4o_response.multiplier,
                            "choice_uncertainty": gpt4o_response.choice_uncertainty,
                            "multiplier_uncertainty": gpt4o_response.multiplier_uncertainty,
                            "explanation": gpt4o_response.explanation,
                            "model_used": "openai/gpt-4o"
                        })
                else:
                    # Use llama result
                    logger.info(f"  Llama uncertainty acceptable, using Llama result")
                    successful_comparisons.append({
                        "repo_a": pair['repo_a'],
                        "repo_b": pair['repo_b'],
                        "choice": llama_response.choice,
                        "multiplier": llama_response.multiplier,
                        "choice_uncertainty": llama_response.choice_uncertainty,
                        "multiplier_uncertainty": llama_response.multiplier_uncertainty,
                        "explanation": llama_response.explanation,
                        "model_used": "meta-llama/llama-4-maverick"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing pair {i+1}: {e}")
                filtered_comparisons.append({
                    "repo_a": pair['repo_a'],
                    "repo_b": pair['repo_b'],
                    "reason": f"Processing error: {str(e)}",
                    "error": str(e)
                })
        
        total_processing_time = (time.time() - start_time) * 1000
        
        result = {
            "successful_comparisons": successful_comparisons,
            "filtered_comparisons": filtered_comparisons,
            "total_input_pairs": len(pairs),
            "total_successful": len(successful_comparisons),
            "total_filtered": len(filtered_comparisons),
            "processing_summary": {
                "total_processing_time_ms": total_processing_time,
                "llama_queries": llama_queries,
                "gpt4o_queries": gpt4o_queries,
                "uncertainty_thresholds": {
                    "llama-4-maverick": LLAMA_THRESHOLD,
                    "gpt-4o": GPT4O_THRESHOLD
                }
            }
        }
        
        logger.info(f"Batch comparison completed: {len(successful_comparisons)} successful, {len(filtered_comparisons)} filtered")
        self._special_case_enabled = self._special_case_enabled_tmp
        return result