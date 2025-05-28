# src/uncertainty_calibration/multi_model_engine.py
"""
Multi-Model Engine for LLM Data Augmentation

Handles querying multiple LLM models through OpenRouter API and calculating consensus
from their responses using logprobs extraction.
"""

import os
import time
import logging
import requests
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Container for model response data."""
    model_id: str
    success: bool
    content: str
    logprobs: Optional[Dict]
    cost_usd: float
    tokens_used: int
    error: Optional[str] = None
    timestamp: str = ""

class MultiModelEngine:
    """
    Engine for querying multiple LLM models and extracting consensus from their responses.
    
    Uses OpenRouter API to query multiple models in parallel/sequence and extracts
    probability distributions from logprobs to calculate reliable consensus.
    """
    
    def __init__(self, config):
        """
        Initialize the multi-model engine.
        
        Args:
            config: LLM augmentation configuration
        """
        self.config = config
        self.api_config = config.api
        self.models_config = config.models
        self.cost_config = config.cost_management
        
        # API setup
        self.api_key = os.getenv(self.api_config.openrouter.api_key_env)
        if not self.api_key:
            raise ValueError(f"API key not found in environment variable {self.api_config.openrouter.api_key_env}")
        
        self.base_url = self.api_config.openrouter.base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add optional headers
        if hasattr(self.api_config.openrouter, 'headers'):
            if hasattr(self.api_config.openrouter.headers, 'http_referer'):
                self.headers["HTTP-Referer"] = self.api_config.openrouter.headers.http_referer
            if hasattr(self.api_config.openrouter.headers, 'x_title'):
                self.headers["X-Title"] = self.api_config.openrouter.headers.x_title
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.api_config.rate_limiting.requests_per_minute
        )
        
        # Cost tracking
        self.total_cost = 0.0
        
        logger.info("MultiModelEngine initialized with OpenRouter API")
    
    def query_models(self, prompt: List[Dict[str, str]], 
                    iteration: int = 1) -> List[ModelResponse]:
        """
        Query multiple models with the same prompt and return responses.
        
        Args:
            prompt: List of message dictionaries (OpenAI format)
            iteration: Current iteration number (affects which models to use)
            
        Returns:
            List of ModelResponse objects
        """
        # Determine which models to use for this iteration
        iteration_key = f"iteration_{iteration}"
        model_names = self.models_config.iteration_models.get(
            iteration_key, 
            list(self.models_config.primary_models.keys())
        )
        
        responses = []
        
        for model_name in model_names:
            # Get model ID
            model_id = self._get_model_id(model_name)
            if not model_id:
                logger.warning(f"Model {model_name} not found in configuration")
                continue
            
            # Query model
            response = self._query_single_model(model_id, prompt)
            responses.append(response)
            
            # Update cost tracking
            self.total_cost += response.cost_usd
            
            # Check budget constraints
            if self.cost_config.track_costs and self.total_cost > self.cost_config.total_budget_usd:
                logger.error(f"Budget exceeded: ${self.total_cost:.2f} > ${self.cost_config.total_budget_usd}")
                break
            
            # Rate limiting between models
            self.rate_limiter.wait_if_needed()
        
        logger.info(f"Queried {len(responses)} models, total cost: ${self.total_cost:.4f}")
        return responses
    
    def _query_single_model(self, model_id: str, prompt: List[Dict[str, str]]) -> ModelResponse:
        """
        Query a single model and return structured response.
        
        Args:
            model_id: OpenRouter model identifier
            prompt: Message list in OpenAI format
            
        Returns:
            ModelResponse object
        """
        payload = {
            "model": model_id,
            "messages": prompt,
            "max_tokens": self.config.prompts.level_1.max_tokens,  # Default, may be overridden
            "temperature": self.config.prompts.level_1.temperature,
            "logprobs": True,
            "top_logprobs": 5
        }
        
        max_retries = self.api_config.rate_limiting.max_retries
        backoff_factor = self.api_config.rate_limiting.backoff_factor
        timeout = self.api_config.rate_limiting.timeout_seconds
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    self.base_url, 
                    headers=self.headers, 
                    json=payload,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_successful_response(model_id, data)
                    
                elif response.status_code == 429:  # Rate limited
                    if attempt < max_retries:
                        wait_time = (backoff_factor ** attempt) * 60  # Wait in seconds
                        logger.warning(f"Rate limited for {model_id}, waiting {wait_time:.1f}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        return ModelResponse(
                            model_id=model_id,
                            success=False,
                            content="",
                            logprobs=None,
                            cost_usd=0.0,
                            tokens_used=0,
                            error=f"Rate limit exceeded after {max_retries} retries",
                            timestamp=datetime.now().isoformat()
                        )
                        
                else:  # Other HTTP error
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    if attempt < max_retries:
                        logger.warning(f"HTTP error for {model_id}: {error_msg}, retrying...")
                        time.sleep(backoff_factor ** attempt)
                        continue
                    else:
                        return ModelResponse(
                            model_id=model_id,
                            success=False,
                            content="",
                            logprobs=None,
                            cost_usd=0.0,
                            tokens_used=0,
                            error=error_msg,
                            timestamp=datetime.now().isoformat()
                        )
                        
            except Exception as e:
                error_msg = f"Exception: {str(e)}"
                if attempt < max_retries:
                    logger.warning(f"Exception for {model_id}: {error_msg}, retrying...")
                    time.sleep(backoff_factor ** attempt)
                    continue
                else:
                    return ModelResponse(
                        model_id=model_id,
                        success=False,
                        content="",
                        logprobs=None,
                        cost_usd=0.0,
                        tokens_used=0,
                        error=error_msg,
                        timestamp=datetime.now().isoformat()
                    )
        
        # Should never reach here
        return ModelResponse(
            model_id=model_id,
            success=False,
            content="",
            logprobs=None,
            cost_usd=0.0,
            tokens_used=0,
            error="Unexpected error in retry loop",
            timestamp=datetime.now().isoformat()
        )
    
    def _parse_successful_response(self, model_id: str, data: Dict) -> ModelResponse:
        """Parse successful API response into ModelResponse."""
        try:
            choice = data.get('choices', [{}])[0]
            message = choice.get('message', {})
            content = message.get('content', '').strip()
            
            # Extract logprobs
            logprobs_data = choice.get('logprobs')
            parsed_logprobs = None
            
            if logprobs_data:
                parsed_logprobs = self._extract_logprobs(logprobs_data)
            
            # Calculate cost
            usage = data.get('usage', {})
            total_tokens = usage.get('total_tokens', 0)
            cost_usd = total_tokens * self.cost_config.cost_per_1k_tokens / 1000.0
            
            return ModelResponse(
                model_id=model_id,
                success=True,
                content=content,
                logprobs=parsed_logprobs,
                cost_usd=cost_usd,
                tokens_used=total_tokens,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return ModelResponse(
                model_id=model_id,
                success=False,
                content="",
                logprobs=None,
                cost_usd=0.0,
                tokens_used=0,
                error=f"Response parsing error: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def _extract_logprobs(self, logprobs_data: Dict) -> Dict[str, float]:
        """
        Extract probability distribution from logprobs data.
        
        Args:
            logprobs_data: Raw logprobs from API response
            
        Returns:
            Dictionary mapping tokens to linear probabilities
        """
        try:
            # Handle different logprobs formats
            token_logprobs = logprobs_data.get('content', [])
            
            if not token_logprobs:
                return {}
            
            # Get first token's logprobs (usually the choice token)
            first_token_data = token_logprobs[0]
            top_logprobs = first_token_data.get('top_logprobs', [])
            
            # Convert log probabilities to linear probabilities
            probabilities = {}
            for token_data in top_logprobs:
                token = token_data.get('token', '').strip()
                logprob = token_data.get('logprob', -float('inf'))
                
                # Convert to linear probability
                linear_prob = np.exp(logprob) if logprob > -float('inf') else 0.0
                probabilities[token] = linear_prob
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error extracting logprobs: {e}")
            return {}
    
    def calculate_consensus(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """
        Calculate consensus from Level 1 or Level 3 responses (A vs B choices).
        
        Args:
            responses: List of valid ModelResponse objects
            
        Returns:
            Dictionary with consensus metrics
        """
        if not responses:
            return {"error": "No valid responses provided"}
        
        # Valid choice tokens
        valid_choices = self.config.prompts.level_1.choice_options  # ["A", "B", "Equal"]
        
        # Aggregate probabilities across models
        aggregated_probs = {choice: [] for choice in valid_choices}
        
        for response in responses:
            if not response.success or not response.logprobs:
                continue
            
            # Normalize probabilities for this response
            total_prob = sum(response.logprobs.get(choice, 0.0) for choice in valid_choices)
            
            if total_prob > 0:
                for choice in valid_choices:
                    normalized_prob = response.logprobs.get(choice, 0.0) / total_prob
                    aggregated_probs[choice].append(normalized_prob)
        
        if not any(aggregated_probs.values()):
            return {"error": "No valid probability distributions found"}
        
        # Calculate consensus metrics
        consensus_probs = {}
        for choice in valid_choices:
            if aggregated_probs[choice]:
                consensus_probs[choice] = np.mean(aggregated_probs[choice])
            else:
                consensus_probs[choice] = 0.0
        
        # Normalize consensus probabilities
        total_consensus = sum(consensus_probs.values())
        if total_consensus > 0:
            consensus_probs = {k: v/total_consensus for k, v in consensus_probs.items()}
        
        # Determine chosen option
        chosen_option = max(consensus_probs.keys(), key=lambda k: consensus_probs[k])
        
        # Calculate multiplier (ratio between chosen and other option)
        if chosen_option in ["A", "B"]:
            other_option = "B" if chosen_option == "A" else "A"
            chosen_prob = consensus_probs[chosen_option]
            other_prob = consensus_probs[other_option]
            
            if other_prob > 0:
                consensus_multiplier = chosen_prob / other_prob
            else:
                consensus_multiplier = 10.0  # Default high multiplier
        else:  # "Equal" chosen
            consensus_multiplier = 1.0
        
        # Calculate agreement rate (how many models agree with consensus)
        agreement_count = 0
        total_responses = 0
        
        for response in responses:
            if response.success and response.logprobs:
                response_choice = max(response.logprobs.keys(), 
                                    key=lambda k: response.logprobs.get(k, 0))
                if response_choice == chosen_option:
                    agreement_count += 1
                total_responses += 1
        
        agreement_rate = agreement_count / total_responses if total_responses > 0 else 0.0
        
        return {
            "consensus_probabilities": consensus_probs,
            "chosen_option": chosen_option,
            "consensus_multiplier": min(consensus_multiplier, 25.0),  # Cap at 25x
            "consensus_confidence": consensus_probs[chosen_option],
            "equal_probability": consensus_probs.get("Equal", 0.0),
            "agreement_rate": agreement_rate,
            "valid_responses": len([r for r in responses if r.success and r.logprobs])
        }
    
    def calculate_level2_consensus(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """
        Calculate consensus from Level 2 responses (originality bucket assessment).
        
        Args:
            responses: List of valid ModelResponse objects
            
        Returns:
            Dictionary with originality consensus metrics
        """
        if not responses:
            return {"error": "No valid responses provided"}
        
        # Bucket values (0.1 to 0.9 in 0.1 increments, with 10 being 0.9)
        bucket_values = {str(i): i * 0.1 for i in range(1, 10)}
        bucket_values["10"] = 0.9  # Bucket 10 maps to 0.9
        
        # Aggregate bucket probabilities across models
        aggregated_bucket_probs = {bucket: [] for bucket in bucket_values.keys()}
        
        for response in responses:
            if not response.success or not response.logprobs:
                continue
            
            # Normalize probabilities for this response
            valid_buckets = [b for b in bucket_values.keys() if b in response.logprobs]
            total_prob = sum(response.logprobs.get(bucket, 0.0) for bucket in valid_buckets)
            
            if total_prob > 0:
                for bucket in bucket_values.keys():
                    normalized_prob = response.logprobs.get(bucket, 0.0) / total_prob
                    aggregated_bucket_probs[bucket].append(normalized_prob)
        
        if not any(aggregated_bucket_probs.values()):
            return {"error": "No valid bucket probability distributions found"}
        
        # Calculate consensus bucket probabilities
        consensus_bucket_probs = {}
        for bucket in bucket_values.keys():
            if aggregated_bucket_probs[bucket]:
                consensus_bucket_probs[bucket] = np.mean(aggregated_bucket_probs[bucket])
            else:
                consensus_bucket_probs[bucket] = 0.0
        
        # Normalize consensus probabilities
        total_consensus = sum(consensus_bucket_probs.values())
        if total_consensus > 0:
            consensus_bucket_probs = {k: v/total_consensus for k, v in consensus_bucket_probs.items()}
        
        # Calculate expected originality score
        expected_originality = sum(
            bucket_values[bucket] * prob 
            for bucket, prob in consensus_bucket_probs.items()
        )
        
        # Calculate bucket confidence (highest bucket probability)
        bucket_confidence = max(consensus_bucket_probs.values())
        
        return {
            "bucket_distribution": consensus_bucket_probs,
            "expected_originality": expected_originality,
            "bucket_confidence": bucket_confidence,
            "valid_responses": len([r for r in responses if r.success and r.logprobs])
        }
    
    def _get_model_id(self, model_name: str) -> Optional[str]:
        """Get model ID from configuration."""
        # Check primary models first
        if hasattr(self.models_config.primary_models, model_name):
            return getattr(self.models_config.primary_models, model_name)
        
        # Check secondary models
        if hasattr(self.models_config.secondary_models, model_name):
            return getattr(self.models_config.secondary_models, model_name)
        
        return None
    
    def get_total_cost(self) -> float:
        """Get total cost incurred so far."""
        return self.total_cost


class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute  # Minimum seconds between requests
        self.last_request_time = 0.0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            time.sleep(wait_time)
        
        self.last_request_time = time.time()