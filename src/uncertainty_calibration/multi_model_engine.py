# src/uncertainty_calibration/multi_model_engine.py
"""
Multi-Model Engine for LLM Data Augmentation - REFACTORED VERSION
Handles querying multiple LLM models through OpenRouter API.
Response parsing is now handled by the ResponseParser module.
"""
import os
import time
import logging
import requests
from typing import Dict, List, Any, Optional
from uncertainty_calibration.response_parser import ResponseParser, ModelResponse

logger = logging.getLogger(__name__)

class MultiModelEngine:
    """
    Refactored engine for querying multiple LLM models.
    Focuses on API communication while delegating parsing to ResponseParser.
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

        # Rate limiting configuration
        self.min_interval = 60.0 / self.api_config.rate_limiting.requests_per_minute
        self.last_request_time = 0.0
        self.max_retries = self.api_config.rate_limiting.max_retries
        self.backoff_factor = self.api_config.rate_limiting.backoff_factor
        self.timeout = self.api_config.rate_limiting.timeout_seconds

        # Initialize response parser
        cost_per_1k = getattr(self.config, 'cost_management', {}).get('cost_per_1k_tokens', 0.01)
        self.response_parser = ResponseParser(cost_per_1k_tokens=cost_per_1k)

        # Cost tracking
        self.total_cost = 0.0

        logger.info("MultiModelEngine initialized with OpenRouter API")

    def query_single_model_with_temperature(self, model_id: str, prompt: List[Dict[str, str]],
                                            temperature: float = 0.0) -> ModelResponse:
        """
        Query a single model with specific temperature.

        Args:
            model_id: OpenRouter model identifier
            prompt: Message list in OpenAI format
            temperature: Sampling temperature

        Returns:
            ModelResponse with parsed data and uncertainty
        """
        
        # Make API request
        api_response = self._make_api_request(model_id, prompt, temperature)
        
        # Parse response using ResponseParser
        parsed_response = self.response_parser.parse_response(model_id, api_response, temperature)
        
        # Update cost tracking
        self.total_cost += parsed_response.cost_usd
        
        return parsed_response

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

                # Small delay between requests to be respectful
                time.sleep(0.1)

        logger.info(f"Queried {len(model_ids)} models x {len(temperatures)} temperatures = {len(responses)} total responses")
        return responses

    def _make_api_request(self, model_id: str, prompt: List[Dict[str, str]], 
                         temperature: float) -> Dict:
        """
        Make API request with retry logic and rate limiting.
        
        Args:
            model_id: Model identifier
            prompt: Message list
            temperature: Temperature
            
        Returns:
            Raw API response dictionary or error dictionary
        """
        
        payload = {
            "model": model_id,
            "messages": prompt,
            "max_tokens": 10,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": 10
        }

        # Rate limiting
        self._wait_if_needed()

        # Retry loop
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return response.json()

                elif response.status_code == 429:  # Rate limited
                    if attempt < self.max_retries:
                        wait_time = (self.backoff_factor ** attempt) * 60
                        logger.warning(f"Rate limited for {model_id}, waiting {wait_time:.1f}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        return self._create_api_error_response(
                            f"Rate limit exceeded after {self.max_retries} retries"
                        )

                else:  # Other HTTP error
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    if attempt < self.max_retries:
                        logger.warning(f"HTTP error for {model_id}: {error_msg}, retrying...")
                        time.sleep(self.backoff_factor ** attempt)
                        continue
                    else:
                        return self._create_api_error_response(error_msg)

            except Exception as e:
                error_msg = f"Request exception: {str(e)}"
                if attempt < self.max_retries:
                    logger.warning(f"Exception for {model_id}: {error_msg}, retrying...")
                    time.sleep(self.backoff_factor ** attempt)
                    continue
                else:
                    return self._create_api_error_response(error_msg)

        # Should not reach here, but just in case
        return self._create_api_error_response("Unexpected error in retry loop")

    def _wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def _create_api_error_response(self, error_message: str) -> Dict:
        """
        Create error response dictionary for API failures.
        
        Args:
            error_message: Error description
            
        Returns:
            Error response dictionary
        """
        return {
            "error": {"message": error_message},
            "choices": [],
            "usage": {"total_tokens": 0}
        }

    def get_total_cost(self) -> float:
        """Get total cost incurred so far."""
        return self.total_cost

    def reset_cost_tracking(self):
        """Reset cost tracking to zero."""
        self.total_cost = 0.0
        logger.info("Cost tracking reset to $0.00")

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limiting status.
        
        Returns:
            Dictionary with rate limiting information
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        return {
            "requests_per_minute": 60.0 / self.min_interval,
            "min_interval_seconds": self.min_interval,
            "time_since_last_request": time_since_last,
            "ready_for_next_request": time_since_last >= self.min_interval,
            "seconds_until_ready": max(0, self.min_interval - time_since_last)
        }