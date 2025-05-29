# src/uncertainty_calibration/response_parser.py
"""
Response parser for LLM API responses.
Handles parsing, uncertainty calculation, and ModelResponse creation.
"""
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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

# Export ModelResponse for backward compatibility
__all__ = ['ModelResponse', 'ResponseParser', 'parse_response', 'calculate_uncertainty']

class ResponseParser:
    """Handles parsing of LLM API responses and uncertainty calculation."""
    
    def __init__(self, cost_per_1k_tokens: float = 0.01):
        """
        Initialize response parser.
        
        Args:
            cost_per_1k_tokens: Approximate cost per 1000 tokens
        """
        self.cost_per_1k_tokens = cost_per_1k_tokens
    
    def parse_response(self, model_id: str, api_response: Dict, 
                      temperature: float) -> ModelResponse:
        """
        Parse API response into ModelResponse object.
        
        Args:
            model_id: Model identifier
            api_response: Raw API response dictionary
            temperature: Temperature used for request
            
        Returns:
            ModelResponse object with parsed data
        """
        try:
            if self._is_successful_response(api_response):
                return self._parse_successful_response(model_id, api_response, temperature)
            else:
                error_msg = self._extract_error_message(api_response)
                return self._create_error_response(model_id, temperature, error_msg)
        except Exception as e:
            return self._create_error_response(
                model_id, temperature, f"Response parsing error: {str(e)}"
            )
    
    def _is_successful_response(self, api_response: Dict) -> bool:
        """Check if API response indicates success."""
        return (
            isinstance(api_response, dict) and
            'choices' in api_response and
            len(api_response['choices']) > 0 and
            'message' in api_response['choices'][0]
        )
    
    def _extract_error_message(self, api_response: Dict) -> str:
        """Extract error message from failed API response."""
        if isinstance(api_response, dict):
            if 'error' in api_response:
                error = api_response['error']
                if isinstance(error, dict) and 'message' in error:
                    return error['message']
                else:
                    return str(error)
            else:
                return "Unknown API error - no error field in response"
        else:
            return f"Invalid API response format: {type(api_response)}"
    
    def _parse_successful_response(self, model_id: str, data: Dict, 
                                 temperature: float) -> ModelResponse:
        """Parse successful API response into ModelResponse."""
        
        # Extract basic response data
        choice = data.get('choices', [{}])[0]
        message = choice.get('message', {})
        content = message.get('content', '').strip()
        
        # Extract and process logprobs
        logprobs_data = choice.get('logprobs')
        parsed_logprobs = self._extract_logprobs(logprobs_data)
        uncertainty = self._calculate_uncertainty(parsed_logprobs)
        raw_choice = self._extract_raw_choice(parsed_logprobs, content)
        
        # Extract usage and calculate cost
        usage_info = self._extract_usage_info(data)
        cost_usd = self._calculate_cost(usage_info['total_tokens'])
        
        return ModelResponse(
            model_id=model_id,
            success=True,
            content=content,
            logprobs=parsed_logprobs,
            uncertainty=uncertainty,
            raw_choice=raw_choice,
            cost_usd=cost_usd,
            tokens_used=usage_info['total_tokens'],
            temperature=temperature,
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_logprobs(self, logprobs_data: Optional[Dict]) -> Dict[str, float]:
        """
        Extract and convert logprobs to linear probabilities.
        
        Args:
            logprobs_data: Logprobs section from API response
            
        Returns:
            Dictionary mapping tokens to probabilities
        """
        parsed_logprobs = {}
        
        if not logprobs_data or 'content' not in logprobs_data:
            return parsed_logprobs
        
        token_logprobs = logprobs_data['content']
        if not token_logprobs:
            return parsed_logprobs
        
        # Get first token logprobs (usually contains the answer)
        first_token = token_logprobs[0]
        top_logprobs = first_token.get('top_logprobs', [])
        
        # Convert logprobs to linear probabilities
        for token_data in top_logprobs:
            token = token_data.get('token', '').strip()
            logprob = token_data.get('logprob', -float('inf'))
            
            if logprob > -float('inf') and token:
                parsed_logprobs[token] = np.exp(logprob)
        
        return parsed_logprobs
    
    def _calculate_uncertainty(self, probabilities: Dict[str, float]) -> float:
        """
        Calculate uncertainty as normalized entropy.
        
        Args:
            probabilities: Dictionary mapping tokens to probabilities
            
        Returns:
            Uncertainty score between 0.0 and 1.0
        """
        if not probabilities:
            return 1.0  # Maximum uncertainty for empty probabilities
        
        # Normalize probabilities to sum to 1
        total_prob = sum(probabilities.values())
        if total_prob == 0:
            return 1.0
        
        normalized_probs = [p / total_prob for p in probabilities.values()]
        
        # Calculate entropy
        entropy = 0.0
        for p in normalized_probs:
            if p > 0:
                entropy -= p * np.log2(p + 1e-10)  # Add small epsilon to avoid log(0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(probabilities))
        if max_entropy == 0:
            return 1.0
        
        normalized_entropy = entropy / max_entropy
        return min(1.0, max(0.0, normalized_entropy))
    
    def _extract_raw_choice(self, probabilities: Dict[str, float], 
                          content: str) -> str:
        """
        Extract the raw choice from probabilities or content.
        
        Args:
            probabilities: Token probabilities
            content: Response content as fallback
            
        Returns:
            Raw choice string
        """
        if probabilities:
            # Return token with highest probability
            return max(probabilities.keys(), key=lambda k: probabilities[k])
        else:
            # Fallback to content
            return content
    
    def _extract_usage_info(self, data: Dict) -> Dict[str, int]:
        """
        Extract token usage information from API response.
        
        Args:
            data: API response data
            
        Returns:
            Dictionary with usage information
        """
        usage = data.get('usage', {})
        return {
            'total_tokens': usage.get('total_tokens', 0),
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0)
        }
    
    def _calculate_cost(self, total_tokens: int) -> float:
        """
        Calculate cost based on token usage.
        
        Args:
            total_tokens: Total tokens used
            
        Returns:
            Cost in USD
        """
        return total_tokens * self.cost_per_1k_tokens / 1000.0
    
    def _create_error_response(self, model_id: str, temperature: float, 
                             error_msg: str) -> ModelResponse:
        """
        Create error response for failed requests.
        
        Args:
            model_id: Model identifier
            temperature: Temperature used
            error_msg: Error message
            
        Returns:
            ModelResponse with error information
        """
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

# Convenience functions for backward compatibility
def parse_response(model_id: str, api_response: Dict, temperature: float) -> ModelResponse:
    """
    Convenience function to parse a single response.
    
    Args:
        model_id: Model identifier
        api_response: Raw API response
        temperature: Temperature used
        
    Returns:
        Parsed ModelResponse
    """
    parser = ResponseParser()
    return parser.parse_response(model_id, api_response, temperature)

def calculate_uncertainty(probabilities: Dict[str, float]) -> float:
    """
    Convenience function to calculate uncertainty from probabilities.
    
    Args:
        probabilities: Token probabilities
        
    Returns:
        Uncertainty score
    """
    parser = ResponseParser()
    return parser._calculate_uncertainty(probabilities)