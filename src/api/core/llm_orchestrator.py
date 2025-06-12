# src/api/core/llm_orchestrator.py
"""
LLM Orchestrator service that coordinates existing uncertainty calibration components.
Handles prompt generation, LLM querying, and response parsing.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
from omegaconf import DictConfig

from src.shared.multi_model_engine import MultiModelEngine
from src.tasks.l1.level1_prompts import Level1PromptGenerator
from src.shared.response_parser import ResponseParser, ModelResponse
from src.shared.model_metadata import get_model_metadata
from src.tasks.originality.originality_prompt_generator import OriginalityPromptGenerator
from src.tasks.l3.level3_prompts import Level3PromptGenerator
from src.tasks.l3.dependency_response_parser import DependencyResponseParser
from src.tasks.l3.dependency_context_extractor import DependencyContextExtractor

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    """
    Orchestrates LLM operations using existing uncertainty calibration components.
    """
    
    def __init__(self, llm_config: DictConfig, default_model: Optional[str] = None):
        """
        Initialize the LLM orchestrator.
        
        Args:
            llm_config: LLM configuration from loaded YAML
            default_model: Default model to use (overrides config default)
        """
        self.config = llm_config
        self.default_model = default_model
        
        # Initialize existing components
        try:
            self.multi_model_engine = MultiModelEngine(llm_config)
            self.level1_prompt_generator = Level1PromptGenerator(llm_config)
            self.response_parser = ResponseParser()
            
            # Initialize originality prompt generator
            self.originality_prompt_generator = OriginalityPromptGenerator()
            
            # Initialize L3 components
            self.level3_prompt_generator = Level3PromptGenerator()
            self.dependency_response_parser = DependencyResponseParser()
            self.dependency_context_extractor = DependencyContextExtractor(
                parent_csv_path="data/raw/DeepFunding Repos Enhanced via OpenQ - ENHANCED TEAMS.csv",
                dependencies_csv_path="data/raw/DeepFunding Repos Enhanced via OpenQ - ENHANCED TEAMS.csv"
            )
            
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
                                 temperature: float = 0.7, simplified: bool = False) -> ModelResponse:
        """
        Query LLM for Level 3 dependency comparison using real implementation.
        
        Args:
            dep_a_info: Information about first dependency
            dep_b_info: Information about second dependency
            parent_info: Information about parent repository
            model_id: Model to use
            temperature: Sampling temperature
            simplified: If True, returns only overall assessment choice without detailed reasoning/dimensions
            
        Returns:
            ModelResponse with dependency comparison results
        """
        try:
            # Use default model if not specified
            if model_id is None:
                model_id = self._get_default_model()
            
            parent_url = parent_info.get('url')
            dep_a_url = dep_a_info.get('url')
            dep_b_url = dep_b_info.get('url')
            
            if not all([parent_url, dep_a_url, dep_b_url]):
                raise ValueError("Parent and dependency URLs are required for L3 comparison")
            
            logger.info(f"Starting L3 dependency comparison: {dep_a_url} vs {dep_b_url} for {parent_url}")
            
            # Extract detailed context using DependencyContextExtractor
            try:
                comparison_context = self.dependency_context_extractor.extract_comparison_context(
                    parent_url, dep_a_url, dep_b_url
                )
                parent_context = comparison_context["parent"]
                dep_a_context = comparison_context["dependency_a"]
                dep_b_context = comparison_context["dependency_b"]
            except Exception as e:
                logger.warning(f"Could not extract detailed context, using basic info: {e}")
                # Fallback to basic info provided
                parent_context = parent_info
                dep_a_context = dep_a_info
                dep_b_context = dep_b_info
            
            # Generate L3 comparison prompt using Level3PromptGenerator
            messages = self.level3_prompt_generator.create_dependency_comparison_prompt(
                parent_context, dep_a_context, dep_b_context, simplified
            )
            
            logger.debug(f"Generated L3 prompt for {dep_a_info.get('name', dep_a_url)} vs {dep_b_info.get('name', dep_b_url)}")
            
            # Query LLM using MultiModelEngine interface
            start_time = time.time()
            
            # Use special model and token limit for simplified responses
            if simplified:
                actual_model_id = model_id
                max_tokens = 20
                logger.info(f"SIMPLIFIED MODE ACTIVATED: Using model={actual_model_id}, max_tokens={max_tokens}")
            else:
                actual_model_id = model_id
                max_tokens = None
                logger.info(f"FULL MODE: Using model={actual_model_id}, unlimited tokens")
            
            model_response = self.multi_model_engine.query_single_model_with_temperature(
                model_id=actual_model_id,
                prompt=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            processing_time = (time.time() - start_time) * 1000
            
            # Parse the response using DependencyResponseParser
            try:
                parsed_response = self.dependency_response_parser.parse_response(
                    model_response.content,
                    parent_url,
                    dep_a_url,
                    dep_b_url,
                    model_response.full_content_logprobs
                )
                
                # Convert parsed response to match ModelResponse format
                overall_assessment = parsed_response.overall_assessment
                choice = overall_assessment.get('choice', 'Equal')
                confidence = overall_assessment.get('confidence', 0.5)
                reasoning = overall_assessment.get('reasoning', '')
                
                # Update ModelResponse with parsed data
                model_response.raw_choice = choice
                model_response.uncertainty = 1.0 - confidence
                model_response.content = f"Choice: {choice}. Reasoning: {reasoning}"
                
                # Store parsed response in a custom field for API response
                model_response.parsed_dependency_response = parsed_response
                
            except Exception as e:
                logger.warning(f"Could not parse L3 response, using raw content: {e}")
                logger.warning(f"RAW LLM OUTPUT FOR DEBUG:")
                logger.warning(f"Content: {repr(model_response.content)}")
                logger.warning(f"Full content: {repr(getattr(model_response, 'full_content', 'N/A'))}")
                logger.warning(f"Logprobs: {repr(getattr(model_response, 'full_content_logprobs', 'N/A'))}")
                
                # Fallback to basic choice extraction
                content_lower = model_response.content.lower()
                if 'choice: a' in content_lower or 'dependency a' in content_lower:
                    model_response.raw_choice = 'A'
                elif 'choice: b' in content_lower or 'dependency b' in content_lower:
                    model_response.raw_choice = 'B'
                else:
                    model_response.raw_choice = 'Equal'
            
            # Add processing time to response
            model_response.processing_time_ms = processing_time
            
            logger.info(f"L3 dependency comparison completed in {processing_time:.1f}ms")
            
            return model_response
            
        except Exception as e:
            logger.error(f"Error in L3 dependency comparison query: {e}")
            raise
    
    async def query_originality_assessment(self, repo_info: Dict[str, Any],
                                         model_id: Optional[str] = None,
                                         temperature: float = 0.7) -> ModelResponse:
        """
        Query LLM for repository originality assessment using MultiModelEngine interface.
        
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
            
            # Query LLM using MultiModelEngine interface with higher max_tokens for detailed assessments
            start_time = time.time()
            model_response = self.multi_model_engine.query_single_model_with_temperature(
                model_id=model_id,
                prompt=messages,
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
            # Use explicitly set default model if provided
            if self.default_model:
                logger.debug(f"Using configured default model: {self.default_model}")
                return self.default_model
                
            # Fall back to first primary model from config
            primary_models = self.config.get('models', {}).get('primary_models', {})
            if primary_models:
                first_model = list(primary_models.values())[0]
                logger.debug(f"Using first primary model from config: {first_model}")
                return first_model
            else:
                logger.debug("Using hardcoded fallback model: deepseek/deepseek-r1-0528")
                return "deepseek/deepseek-r1-0528"  # Updated fallback default
        except Exception:
            logger.debug("Exception occurred, using hardcoded fallback model: deepseek/deepseek-r1-0528")
            return "deepseek/deepseek-r1-0528"
    
