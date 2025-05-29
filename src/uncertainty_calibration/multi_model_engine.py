# src/uncertainty_calibration/multi_model_engine.py
"""
Multi-Model Engine for LLM Data Augmentation - FIXED VERSIONHandles querying multiple LLM models through OpenRouter API and extracting
uncertainty from logprobs for calibration training.
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
logger = logging.getLogger(__name__) # Fixed: __name__ instead of name
@dataclass
class ModelResponse:
    """Container for model response data."""
    model_id: str
    success: bool
    content: str
    logprobs: Optional[Dict]
    uncertainty: float
    raw_choice: str
    cost_usd: float
    tokens_used: int
    temperature: float
    error: Optional[str] = None
    timestamp: str = ""
class MultiModelEngine:
    """
    Fixed engine for querying multiple LLM models and extracting uncertainty.
    """


    def __init__(self, config):
        """Initialize the multi-model engine."""
        self.config = config
        self.api_config = config.api

        # API setup
        self.api_key = os.getenv(self.api_config.openrouter.api_key_env,
                                 os.getenv("OPENROUTER_API_KEY"))
        if not self.api_key:
            raise ValueError(f"API key not found. Set {self.api_config.openrouter.api_key_env} environment variable")

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
        self.min_interval = 60.0 / self.api_config.rate_limiting.requests_per_minute
        self.last_request_time = 0.0

        # Cost tracking
        self.total_cost = 0.0

        logger.info("MultiModelEngine initialized with OpenRouter API")

    def query_single_model_with_temperature(self, model_id: str, prompt: List[Dict[str, str]],
                                            temperature: float = 0.0) -> ModelResponse:
        """
        Query a single model with specific temperature and extract uncertainty.

        Args:
            model_id: OpenRouter model identifier
            prompt: Message list in OpenAI format
            temperature: Sampling temperature

        Returns:
            ModelResponse with uncertainty extracted
        """

        payload = {
            "model": model_id,
            "messages": prompt,
            "max_tokens": 10,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": 10
        }

        max_retries = self.api_config.rate_limiting.max_retries
        backoff_factor = self.api_config.rate_limiting.backoff_factor
        timeout = self.api_config.rate_limiting.timeout_seconds

        # Rate limiting
        self._wait_if_needed()

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
                    return self._parse_successful_response(model_id, data, temperature)

                elif response.status_code == 429:  # Rate limited
                    if attempt < max_retries:
                        wait_time = (backoff_factor ** attempt) * 60
                        logger.warning(f"Rate limited for {model_id}, waiting {wait_time:.1f}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        return self._create_error_response(model_id, temperature,
                                                          f"Rate limit exceeded after {max_retries} retries")

                else:  # Other HTTP error
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    if attempt < max_retries:
                        logger.warning(f"HTTP error for {model_id}: {error_msg}, retrying...")
                        time.sleep(backoff_factor ** attempt)
                        continue
                    else:
                        return self._create_error_response(model_id, temperature, error_msg)

            except Exception as e:
                error_msg = f"Exception: {str(e)}"
                if attempt < max_retries:
                    logger.warning(f"Exception for {model_id}: {error_msg}, retrying...")
                    time.sleep(backoff_factor ** attempt)
                    continue
                else:
                    return self._create_error_response(model_id, temperature, error_msg)
        return self._create_error_response(model_id, temperature, "Unexpected error in retry loop")

    def query_multiple_models(self, model_ids: List[str], prompt: List[Dict[str, str]],
                              temperatures: List[float] = None) -> List[ModelResponse]:
        """
        Query multiple models across temperature sweep.

        Args:
            model_ids: List of model identifiers
            prompt: Message list
            temperatures: Temperature values to test

        Returns:
            List of ModelResponse objects
        """
        if temperatures is None:
            temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        responses = []
        for model_id in model_ids:
            for temperature in temperatures:
                response = self.query_single_model_with_temperature(model_id, prompt, temperature)
                responses.append(response)

                # Update cost tracking
                self.total_cost += response.cost_usd

                # Small delay between requests
                time.sleep(0.1)

        logger.info(f"Queried {len(model_ids)} models x {len(temperatures)} temperatures = {len(responses)} total responses")
        return responses

    def _parse_successful_response(self, model_id: str, data: Dict, temperature: float) -> ModelResponse:
        """Parse successful API response into ModelResponse with uncertainty."""
        try:
            choice = data.get('choices', [{}])[0]
            message = choice.get('message', {})
            content = message.get('content', '').strip()

            # Extract logprobs
            logprobs_data = choice.get('logprobs')
            parsed_logprobs = {}
            uncertainty = 1.0  # Default high uncertainty
            raw_choice = content

            if logprobs_data and 'content' in logprobs_data:
                token_logprobs = logprobs_data['content']
                if token_logprobs:
                    # Get first token logprobs (usually the answer)
                    first_token = token_logprobs[0]
                    top_logprobs = first_token.get('top_logprobs', [])

                    # Convert to linear probabilities
                    for token_data in top_logprobs:
                        token = token_data.get('token', '').strip()
                        logprob = token_data.get('logprob', -float('inf'))
                        if logprob > -float('inf'):
                            parsed_logprobs[token] = np.exp(logprob)

                    # Calculate uncertainty as entropy
                    uncertainty = self._calculate_uncertainty(parsed_logprobs)

                    # Extract raw choice from highest probability token
                    if parsed_logprobs:
                        raw_choice = max(parsed_logprobs.keys(), key=lambda k: parsed_logprobs[k])

            # Calculate cost
            usage = data.get('usage', {})
            total_tokens = usage.get('total_tokens', 0)
            cost_usd = total_tokens * 0.01 / 1000.0  # Approximate cost

            return ModelResponse(
                model_id=model_id,
                success=True,
                content=content,
                logprobs=parsed_logprobs,
                uncertainty=uncertainty,
                raw_choice=raw_choice,
                cost_usd=cost_usd,
                tokens_used=total_tokens,
                temperature=temperature,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return self._create_error_response(model_id, temperature, f"Response parsing error: {str(e)}")

    def _calculate_uncertainty(self, probabilities: Dict[str, float]) -> float:
        """Calculate uncertainty as normalized entropy."""
        if not probabilities:
            return 1.0

        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob == 0:
            return 1.0

        normalized_probs = [p / total_prob for p in probabilities.values()]

        # Calculate entropy
        entropy = -sum(p * np.log2(p + 1e-10) for p in normalized_probs if p > 0)

        # Normalize by max possible entropy
        max_entropy = np.log2(len(probabilities))
        if max_entropy == 0:
            return 1.0

        normalized_entropy = entropy / max_entropy
        return min(1.0, max(0.0, normalized_entropy))

    def _create_error_response(self, model_id: str, temperature: float, error_msg: str) -> ModelResponse:
        """Create error response."""
        return ModelResponse(
            model_id=model_id,
            success=False,
            content="",
            logprobs=None,
            uncertainty=1.0,  # Maximum uncertainty for errors
            raw_choice="",
            cost_usd=0.0,
            tokens_used=0,
            temperature=temperature,
            error=error_msg,
            timestamp=datetime.now().isoformat()
        )

    def _wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def get_total_cost(self) -> float:
        """Get total cost incurred so far."""
        return self.total_cost