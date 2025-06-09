# src/api/core/llm_orchestrator.py
"""
LLM Orchestrator service that coordinates existing uncertainty calibration components.
Handles prompt generation, LLM querying, and response parsing.
"""

import logging
import time
import requests
from typing import Dict, Any, List, Optional, Tuple
from omegaconf import DictConfig

from src.shared.multi_model_engine import MultiModelEngine
from src.tasks.l1.level1_prompts import Level1PromptGenerator
from src.shared.response_parser import ResponseParser, ModelResponse
from src.shared.model_metadata import get_model_metadata
from src.tasks.originality.originality_prompt_generator import OriginalityPromptGenerator

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    """
    Orchestrates LLM operations using existing uncertainty calibration components.
    """
    
    def __init__(self, llm_config: DictConfig):
        """
        Initialize the LLM orchestrator.
        
        Args:
            llm_config: LLM configuration from loaded YAML
        """
        self.config = llm_config
        
        # Initialize existing components
        try:
            self.multi_model_engine = MultiModelEngine(llm_config)
            self.level1_prompt_generator = Level1PromptGenerator(llm_config)
            self.response_parser = ResponseParser()
            
            # Initialize originality prompt generator
            self.originality_prompt_generator = OriginalityPromptGenerator()
            
            logger.info("LLM Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM Orchestrator: {e}")
            raise
    
    async def query_l1_comparison(self, repo_a_info: Dict[str, Any], repo_b_info: Dict[str, Any],
                                 model_id: Optional[str] = None, temperature: float = 0.7) -> ModelResponse:
        """
        Query LLM for Level 1 repository comparison.
        
        Args:
            repo_a_info: Information about first repository
            repo_b_info: Information about second repository
            model_id: Model to use (default from config)
            temperature: Sampling temperature
            
        Returns:
            ModelResponse with parsed results
        """
        try:
            # Use default model if not specified
            if model_id is None:
                model_id = self._get_default_model()
            
            # Generate comparison prompt using existing Level1PromptGenerator
            prompt_messages = self.level1_prompt_generator.create_comparison_prompt(
                repo_a_info, repo_b_info
            )
            
            logger.debug(f"Generated L1 prompt for {repo_a_info.get('name')} vs {repo_b_info.get('name')}")
            
            # Query LLM with existing MultiModelEngine (includes caching)
            start_time = time.time()
            model_response = self.multi_model_engine.query_single_model_with_temperature(
                model_id=model_id,
                prompt=prompt_messages,
                temperature=temperature
            )
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Add processing time to response
            model_response.processing_time_ms = processing_time
            
            logger.debug(f"L1 comparison query completed in {processing_time:.1f}ms")
            
            return model_response
            
        except Exception as e:
            logger.error(f"Error in L1 comparison query: {e}")
            raise
    
    async def query_l3_comparison(self, dep_a_info: Dict[str, Any], dep_b_info: Dict[str, Any],
                                 parent_info: Dict[str, Any], model_id: Optional[str] = None,
                                 temperature: float = 0.7) -> ModelResponse:
        """
        Query LLM for Level 3 dependency comparison (skeleton implementation).
        
        Args:
            dep_a_info: Information about first dependency
            dep_b_info: Information about second dependency
            parent_info: Information about parent repository
            model_id: Model to use
            temperature: Sampling temperature
            
        Returns:
            ModelResponse with mock results for now
        """
        logger.info("L3 comparison query (skeleton implementation)")
        
        # TODO: Implement actual L3 comparison logic
        # For now, return a mock response
        mock_response = self._create_mock_comparison_response(
            dep_a_info, dep_b_info, parent_info, "L3 comparison not yet implemented"
        )
        
        return mock_response
    
    async def query_originality_assessment(self, repo_info: Dict[str, Any],
                                         model_id: Optional[str] = None,
                                         temperature: float = 0.7) -> ModelResponse:
        """
        Query LLM for repository originality assessment using real implementation.
        
        Args:
            repo_info: Information about repository to assess
            model_id: Model to use
            temperature: Sampling temperature
            
        Returns:
            ModelResponse with real originality assessment
        """
        try:
            # Use default model if not specified
            if model_id is None:
                model_id = self._get_default_model()
            
            repo_url = repo_info.get('url')
            if not repo_url:
                raise ValueError("Repository URL is required for originality assessment")
            
            logger.info(f"Starting originality assessment for {repo_url}")
            
            # Generate originality assessment prompt
            messages = self.originality_prompt_generator.create_originality_assessment_prompt(repo_url)
            
            logger.debug(f"Generated originality prompt for {repo_info.get('name', repo_url)}")
            
            # Query LLM using the proven method from OriginalityAssessmentPipeline
            start_time = time.time()
            model_response = self._query_originality_assessment_internal(
                model_id=model_id,
                messages=messages,
                temperature=temperature
            )
            processing_time = (time.time() - start_time) * 1000
            
            # Add processing time to response
            model_response.processing_time_ms = processing_time
            
            logger.info(f"Originality assessment completed in {processing_time:.1f}ms")
            
            return model_response
            
        except Exception as e:
            logger.error(f"Error in originality assessment query: {e}")
            raise
    
    def _query_originality_assessment_internal(self, model_id: str, messages: List[Dict[str, str]], 
                                             temperature: float) -> ModelResponse:
        """
        Internal method for originality assessments that need longer responses.
        Adapted from OriginalityAssessmentPipeline._query_originality_assessment().
        
        Args:
            model_id: Model identifier
            messages: Message list in OpenAI format
            temperature: Sampling temperature
            
        Returns:
            ModelResponse with full originality assessment
        """
        try:
            # Create custom payload for originality assessment with higher max_tokens
            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": 4000,  # Much higher for detailed originality assessments
                "temperature": temperature,
                "logprobs": True,
                "top_logprobs": 5
            }
            
            # Add provider filtering if configured (copied from MultiModelEngine)
            if hasattr(self.multi_model_engine.api_config.openrouter, 'providers') and self.multi_model_engine.api_config.openrouter.providers:
                providers = list(self.multi_model_engine.api_config.openrouter.providers)
                filtered_providers = self.multi_model_engine._filter_providers_for_model(model_id, providers)
                if filtered_providers:
                    payload["provider"] = {
                        "only": filtered_providers
                    }
            
            # Apply rate limiting
            self.multi_model_engine._wait_if_needed()
            
            # Make API request with retry logic
            api_response = None
            for attempt in range(self.multi_model_engine.max_retries + 1):
                try:
                    response = requests.post(
                        self.multi_model_engine.base_url,
                        headers=self.multi_model_engine.headers,
                        json=payload,
                        timeout=self.multi_model_engine.timeout
                    )
                    
                    if response.status_code == 200:
                        api_response = response.json()
                        break
                    elif response.status_code == 429:  # Rate limited
                        if attempt < self.multi_model_engine.max_retries:
                            wait_time = (self.multi_model_engine.backoff_factor ** attempt) * 60
                            logger.warning(f"Rate limited for {model_id}, waiting {wait_time:.1f}s")
                            time.sleep(wait_time)
                            continue
                        else:
                            api_response = self.multi_model_engine._create_api_error_response(
                                f"Rate limit exceeded after {self.multi_model_engine.max_retries} retries"
                            )
                            break
                    else:  # Other HTTP error
                        error_msg = f"HTTP {response.status_code}: {response.text}"
                        if attempt < self.multi_model_engine.max_retries:
                            logger.warning(f"HTTP error for {model_id}: {error_msg}, retrying...")
                            time.sleep(self.multi_model_engine.backoff_factor ** attempt)
                            continue
                        else:
                            api_response = self.multi_model_engine._create_api_error_response(error_msg)
                            break
                            
                except Exception as e:
                    error_msg = f"Request exception: {str(e)}"
                    if attempt < self.multi_model_engine.max_retries:
                        logger.warning(f"Exception for {model_id}: {error_msg}, retrying...")
                        time.sleep(self.multi_model_engine.backoff_factor ** attempt)
                        continue
                    else:
                        api_response = self.multi_model_engine._create_api_error_response(error_msg)
                        break
            
            if api_response is None:
                api_response = self.multi_model_engine._create_api_error_response("Unexpected error in retry loop")
            
            # Parse response using the engine's response parser
            parsed_response = self.multi_model_engine.response_parser.parse_response(
                model_id, api_response, temperature
            )
            
            # Update cost tracking
            self.multi_model_engine.total_cost += parsed_response.cost_usd
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error in originality assessment query: {e}")
            # Return error response
            return ModelResponse(
                model_id=model_id,
                success=False,
                content="",
                logprobs=None,
                uncertainty=1.0,
                raw_choice="",
                cost_usd=0.0,
                tokens_used=0,
                temperature=temperature,
                error=str(e),
                timestamp="",
                answer_token_info={"extraction_success": False, "method": "error"},
                full_content_logprobs=None
            )
    
    def extract_repo_info(self, repo_url: str) -> Dict[str, Any]:
        """
        Extract repository information from URL.
        
        Args:
            repo_url: Repository URL
            
        Returns:
            Dictionary with repository information
        """
        try:
            # Extract owner and name from GitHub URL
            parts = repo_url.strip('/').split('/')
            if len(parts) >= 2:
                owner = parts[-2]
                name = parts[-1]
            else:
                owner = "unknown"
                name = repo_url.split('/')[-1] if '/' in repo_url else repo_url
            
            return {
                'url': repo_url,
                'name': name,
                'owner': owner,
                'full_name': f"{owner}/{name}"
            }
            
        except Exception as e:
            logger.warning(f"Error extracting repo info from {repo_url}: {e}")
            return {
                'url': repo_url,
                'name': 'unknown',
                'owner': 'unknown',
                'full_name': 'unknown/unknown'
            }
    
    def get_model_metadata_string(self, model_id: str, temperature: float) -> str:
        """
        Get model metadata string for responses.
        
        Args:
            model_id: Model identifier
            temperature: Temperature used
            
        Returns:
            Formatted metadata string
        """
        try:
            metadata = get_model_metadata(model_id)
            model_name = model_id.split('/')[-1] if '/' in model_id else model_id
            return f"{model_name}_temp_{temperature}"
        except Exception:
            return f"unknown_temp_{temperature}"
    
    def _get_default_model(self) -> str:
        """Get default model from configuration."""
        try:
            primary_models = self.config.get('models', {}).get('primary_models', {})
            if primary_models:
                # Use first primary model as default
                return list(primary_models.values())[0]
            else:
                return "openai/gpt-4o"  # Fallback default
        except Exception:
            return "openai/gpt-4o"
    
    def _create_mock_comparison_response(self, repo_a_info: Dict, repo_b_info: Dict,
                                       context_info: Dict, explanation: str) -> ModelResponse:
        """Create mock comparison response for skeleton implementations."""
        import random
        
        # Generate reasonable mock data
        choice = random.choice(['A', 'B'])
        uncertainty = random.uniform(0.1, 0.3)  # Low uncertainty for mock
        
        mock_response = ModelResponse(
            model_id="mock/skeleton",
            success=True,
            content=f"Mock comparison: {choice}. {explanation}",
            logprobs={'A': 0.6, 'B': 0.4} if choice == 'A' else {'A': 0.4, 'B': 0.6},
            uncertainty=uncertainty,
            raw_choice=choice,
            cost_usd=0.0,
            tokens_used=0,
            temperature=0.7,
            processing_time_ms=100.0
        )
        
        return mock_response
    
    def _create_mock_originality_response(self, repo_info: Dict, explanation: str) -> ModelResponse:
        """Create mock originality response for skeleton implementation."""
        import random
        
        # Generate reasonable mock data
        originality_score = random.uniform(0.3, 0.8)
        uncertainty = random.uniform(0.1, 0.3)
        
        mock_response = ModelResponse(
            model_id="mock/skeleton",
            success=True,
            content=f"Mock originality: {originality_score:.2f}. {explanation}",
            logprobs={'high': 0.5, 'medium': 0.3, 'low': 0.2},
            uncertainty=uncertainty,
            raw_choice=f"{originality_score:.2f}",
            cost_usd=0.0,
            tokens_used=0,
            temperature=0.7,
            processing_time_ms=80.0
        )
        
        # Add custom field for originality score
        mock_response.originality_score = originality_score
        
        return mock_response