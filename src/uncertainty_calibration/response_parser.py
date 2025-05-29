# src/uncertainty_calibration/response_parser.py
"""
Enhanced response parser with smart answer token extraction and cache cleaning.
Fixes critical bug where uncertainty was extracted from wrong token.
"""
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
import copy

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
    answer_token_info: Optional[Dict] = None  # New: extraction metadata

class AnswerTokenExtractor:
    """
    Extracts the actual answer token from logprobs instead of using first token.
    Handles multiple answer formats and provides robust fallback mechanisms.
    """
    
    def __init__(self):
        # Answer patterns by level
        self.choice_answers = ["A", "B", "EQUAL", "1", "2"]  # Level 1 & 3
        self.numeric_answers = [str(i) for i in range(1, 11)]  # Level 2: 1-10 scale
        
        # Pattern indicators that precede answers
        self.answer_indicators = [
            "Answer:",
            "Answer",
            "ANSWER:",
            "ANSWER",
            "The answer is",
            "My answer is",
            "Response:",
            "Choice:"
        ]
    
    def extract_answer_token(self, logprobs_content: List[Dict], 
                           expected_format: str = "choice") -> Tuple[Optional[Dict], Dict]:
        """
        Extract the actual answer token from logprobs content.
        
        Args:
            logprobs_content: List of token logprob objects from API
            expected_format: "choice" (A/B/Equal), "numeric" (1-10), or "auto"
            
        Returns:
            (answer_token_dict, extraction_info)
        """
        extraction_info = {
            "method": None,
            "pattern_found": None,
            "token_index": None,
            "fallback_used": False,
            "extraction_confidence": 0.0
        }
        
        if not logprobs_content:
            extraction_info["method"] = "failed_no_content"
            return None, extraction_info
        
        # Strategy 1: Look for "Answer:" pattern followed by answer
        answer_token, info = self._find_answer_after_indicator(logprobs_content, expected_format)
        if answer_token:
            extraction_info.update(info)
            extraction_info["method"] = "answer_indicator"
            return answer_token, extraction_info
        
        # Strategy 2: Look for bracketed answers [A], [B], etc.
        answer_token, info = self._find_bracketed_answer(logprobs_content, expected_format)
        if answer_token:
            extraction_info.update(info)
            extraction_info["method"] = "bracketed_answer"
            return answer_token, extraction_info
        
        # Strategy 3: Find direct answer tokens with meaningful alternatives
        answer_token, info = self._find_direct_answer_token(logprobs_content, expected_format)
        if answer_token:
            extraction_info.update(info)
            extraction_info["method"] = "direct_answer"
            return answer_token, extraction_info
        
        # Strategy 4: Fallback - last resort heuristics
        answer_token, info = self._fallback_extraction(logprobs_content, expected_format)
        extraction_info.update(info)
        extraction_info["method"] = "fallback"
        extraction_info["fallback_used"] = True
        
        return answer_token, extraction_info
    
    def _find_answer_after_indicator(self, logprobs_content: List[Dict], 
                                   expected_format: str) -> Tuple[Optional[Dict], Dict]:
        """Find answer token that comes after 'Answer:' or similar indicators."""
        info = {"pattern_found": None, "token_index": None}
        
        for i, token_obj in enumerate(logprobs_content):
            token_text = token_obj.get('token', '').strip()
            
            # Check if this token is an answer indicator
            for indicator in self.answer_indicators:
                if self._matches_indicator(token_text, indicator):
                    info["pattern_found"] = indicator
                    
                    # Look for answer in next few tokens
                    for j in range(i + 1, min(i + 4, len(logprobs_content))):
                        next_token = logprobs_content[j]
                        if self._is_valid_answer_token(next_token, expected_format):
                            info["token_index"] = j
                            info["extraction_confidence"] = self._calculate_confidence(next_token)
                            return next_token, info
        
        return None, info
    
    def _find_bracketed_answer(self, logprobs_content: List[Dict], 
                             expected_format: str) -> Tuple[Optional[Dict], Dict]:
        """Find answers in brackets like [A], [B], [Equal]."""
        info = {"pattern_found": None, "token_index": None}
        
        bracket_patterns = [r'\[([AB]|EQUAL|[1-9]|10)\]', r'\{([AB]|EQUAL|[1-9]|10)\}']
        
        for i, token_obj in enumerate(logprobs_content):
            token_text = token_obj.get('token', '').strip().upper()
            
            for pattern in bracket_patterns:
                match = re.match(pattern, token_text)
                if match:
                    answer = match.group(1)
                    if self._is_valid_answer(answer, expected_format):
                        info["pattern_found"] = f"bracketed_{pattern}"
                        info["token_index"] = i
                        info["extraction_confidence"] = self._calculate_confidence(token_obj)
                        return token_obj, info
        
        return None, info
    
    def _find_direct_answer_token(self, logprobs_content: List[Dict], 
                                expected_format: str) -> Tuple[Optional[Dict], Dict]:
        """Find answer tokens directly (A, B, Equal) with meaningful alternatives."""
        info = {"pattern_found": "direct", "token_index": None}
        
        candidates = []
        
        for i, token_obj in enumerate(logprobs_content):
            if self._is_valid_answer_token(token_obj, expected_format):
                # Check if this token has meaningful alternatives in top_logprobs
                confidence = self._calculate_confidence(token_obj)
                alternatives_quality = self._assess_alternatives_quality(token_obj, expected_format)
                
                score = confidence * alternatives_quality
                candidates.append((token_obj, i, score))
        
        if candidates:
            # Take the candidate with the best score
            best_token, best_index, best_score = max(candidates, key=lambda x: x[2])
            info["token_index"] = best_index
            info["extraction_confidence"] = best_score
            return best_token, info
        
        return None, info
    
    def _fallback_extraction(self, logprobs_content: List[Dict], 
                           expected_format: str) -> Tuple[Optional[Dict], Dict]:
        """Last resort: find any token that might be an answer."""
        info = {"pattern_found": "fallback", "token_index": None, "extraction_confidence": 0.1}
        
        # Take the last token that looks like an answer
        for i in reversed(range(len(logprobs_content))):
            token_obj = logprobs_content[i]
            if self._is_valid_answer_token(token_obj, expected_format):
                info["token_index"] = i
                return token_obj, info
        
        # Absolute fallback: take last meaningful token
        if logprobs_content:
            last_token = logprobs_content[-1]
            info["token_index"] = len(logprobs_content) - 1
            return last_token, info
        
        return None, info
    
    def _matches_indicator(self, token_text: str, indicator: str) -> bool:
        """Check if token matches an answer indicator."""
        token_clean = token_text.strip().upper().rstrip(':')
        indicator_clean = indicator.strip().upper().rstrip(':')
        return token_clean == indicator_clean
    
    def _is_valid_answer_token(self, token_obj: Dict, expected_format: str) -> bool:
        """Check if token object represents a valid answer."""
        token_text = token_obj.get('token', '').strip().upper()
        return self._is_valid_answer(token_text, expected_format)
    
    def _is_valid_answer(self, token_text: str, expected_format: str) -> bool:
        """Check if token text is a valid answer for the expected format."""
        token_clean = token_text.strip().upper().rstrip('.,!?:')
        
        if expected_format == "choice":
            return token_clean in ["A", "B", "EQUAL", "1", "2"]
        elif expected_format == "numeric":
            return token_clean in self.numeric_answers
        else:  # auto
            return (token_clean in self.choice_answers or 
                   token_clean in self.numeric_answers)
    
    def _calculate_confidence(self, token_obj: Dict) -> float:
        """Calculate confidence score for a token based on its logprob."""
        logprob = token_obj.get('logprob', -10.0)
        # Convert logprob to confidence (higher logprob = higher confidence)
        confidence = min(1.0, max(0.0, (logprob + 5.0) / 5.0))  # Normalize roughly
        return confidence
    
    def _assess_alternatives_quality(self, token_obj: Dict, expected_format: str) -> float:
        """Assess quality of alternatives in top_logprobs."""
        top_logprobs = token_obj.get('top_logprobs', [])
        
        if not top_logprobs:
            return 0.1  # Low quality if no alternatives
        
        # Count how many alternatives are valid answers
        valid_alternatives = 0
        for alt in top_logprobs:
            alt_token = alt.get('token', '').strip().upper()
            if self._is_valid_answer(alt_token, expected_format):
                valid_alternatives += 1
        
        # Quality score based on number of valid alternatives
        if len(top_logprobs) == 0:
            return 0.1
        
        quality = valid_alternatives / len(top_logprobs)
        return max(0.2, quality)  # Minimum quality threshold


class ResponseParser:
    """
    Enhanced response parser with smart answer token extraction and cache cleaning.
    """
    
    def __init__(self, cost_per_1k_tokens: float = 0.01):
        """
        Initialize response parser.
        
        Args:
            cost_per_1k_tokens: Approximate cost per 1000 tokens
        """
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.answer_extractor = AnswerTokenExtractor()
    
    def parse_response(self, model_id: str, api_response: Dict, 
                      temperature: float, expected_answer_format: str = "choice") -> ModelResponse:
        """
        Parse API response into ModelResponse object with enhanced answer extraction.
        
        Args:
            model_id: Model identifier
            api_response: Raw API response dictionary
            temperature: Temperature used for request
            expected_answer_format: "choice", "numeric", or "auto"
            
        Returns:
            ModelResponse object with parsed data
        """
        try:
            if self._is_successful_response(api_response):
                return self._parse_successful_response(
                    model_id, api_response, temperature, expected_answer_format
                )
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
                                 temperature: float, expected_answer_format: str) -> ModelResponse:
        """Parse successful API response into ModelResponse."""
        
        # Extract basic response data
        choice = data.get('choices', [{}])[0]
        message = choice.get('message', {})
        content = message.get('content', '').strip()
        
        # Extract and process logprobs with enhanced answer extraction
        logprobs_data = choice.get('logprobs')
        parsed_logprobs, answer_token_info = self._extract_logprobs_enhanced(
            logprobs_data, expected_answer_format
        )
        uncertainty = self._calculate_uncertainty(parsed_logprobs)
        raw_choice = self._extract_raw_choice(parsed_logprobs, content, answer_token_info)
        
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
            timestamp=datetime.now().isoformat(),
            answer_token_info=answer_token_info
        )
    
    def _extract_logprobs_enhanced(self, logprobs_data: Optional[Dict], 
                                 expected_answer_format: str) -> Tuple[Dict[str, float], Dict]:
        """
        Enhanced logprobs extraction that finds the actual answer token.
        
        Args:
            logprobs_data: Logprobs section from API response
            expected_answer_format: Expected format of the answer
            
        Returns:
            (parsed_logprobs, answer_token_info)
        """
        parsed_logprobs = {}
        answer_token_info = {
            "extraction_success": False,
            "method": None,
            "token_found": None,
            "alternatives_count": 0
        }
        
        if not logprobs_data or 'content' not in logprobs_data:
            return parsed_logprobs, answer_token_info
        
        token_logprobs = logprobs_data['content']
        if not token_logprobs:
            return parsed_logprobs, answer_token_info
        
        # Use enhanced answer token extraction
        answer_token, extraction_info = self.answer_extractor.extract_answer_token(
            token_logprobs, expected_answer_format
        )
        
        answer_token_info.update(extraction_info)
        
        if answer_token:
            answer_token_info["extraction_success"] = True
            answer_token_info["token_found"] = answer_token.get('token', '')
            
            # Get top logprobs from the answer token
            top_logprobs = answer_token.get('top_logprobs', [])
            answer_token_info["alternatives_count"] = len(top_logprobs)
            
            # Convert answer token logprobs to linear probabilities
            for token_data in top_logprobs:
                token = token_data.get('token', '').strip()
                logprob = token_data.get('logprob', -float('inf'))
                
                if logprob > -float('inf') and token:
                    parsed_logprobs[token] = np.exp(logprob)
            
            # Include the main answer token if not in top_logprobs
            main_token = answer_token.get('token', '').strip()
            main_logprob = answer_token.get('logprob', -float('inf'))
            if main_token and main_logprob > -float('inf'):
                parsed_logprobs[main_token] = np.exp(main_logprob)
        
        else:
            # Fallback to old behavior if extraction completely fails
            logger.warning(f"Answer token extraction failed, falling back to first token")
            answer_token_info["method"] = "fallback_first_token"
            
            first_token = token_logprobs[0]
            top_logprobs = first_token.get('top_logprobs', [])
            
            for token_data in top_logprobs:
                token = token_data.get('token', '').strip()
                logprob = token_data.get('logprob', -float('inf'))
                
                if logprob > -float('inf') and token:
                    parsed_logprobs[token] = np.exp(logprob)
        
        return parsed_logprobs, answer_token_info
    
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
                          content: str, answer_token_info: Dict) -> str:
        """
        Extract the raw choice from probabilities or content.
        
        Args:
            probabilities: Token probabilities
            content: Response content as fallback
            answer_token_info: Information about answer token extraction
            
        Returns:
            Raw choice string
        """
        # Use extracted answer token if available
        if answer_token_info.get("extraction_success") and answer_token_info.get("token_found"):
            return answer_token_info["token_found"]
        
        # Fall back to highest probability token
        if probabilities:
            return max(probabilities.keys(), key=lambda k: probabilities[k])
        
        # Final fallback to content parsing
        return self._parse_choice_from_content(content)
    
    def _parse_choice_from_content(self, content: str) -> str:
        """Parse choice from response content as final fallback."""
        content_upper = content.upper()
        
        # Look for common answer patterns
        patterns = [
            r"ANSWER:\s*([AB]|EQUAL|[1-9]|10)",
            r"ANSWER\s*([AB]|EQUAL|[1-9]|10)",
            r"\[([AB]|EQUAL|[1-9]|10)\]",
            r"\{([AB]|EQUAL|[1-9]|10)\}",
            r"THE ANSWER IS\s*([AB]|EQUAL|[1-9]|10)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content_upper)
            if match:
                return match.group(1)
        
        # Last resort: look for standalone A, B, Equal at end of response
        if content_upper.strip().endswith((' A', ' B', ' EQUAL')):
            return content_upper.strip()[-1] if content_upper.strip().endswith((' A', ' B')) else 'EQUAL'
        
        return content  # Return full content if nothing else works
    
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
            timestamp=datetime.now().isoformat(),
            answer_token_info={"extraction_success": False, "method": "error"}
        )


# Convenience functions for backward compatibility
def parse_response(model_id: str, api_response: Dict, temperature: float, 
                  expected_answer_format: str = "choice") -> ModelResponse:
    """
    Convenience function to parse a single response.
    
    Args:
        model_id: Model identifier
        api_response: Raw API response
        temperature: Temperature used
        expected_answer_format: Expected format of the answer
        
    Returns:
        Parsed ModelResponse
    """
    parser = ResponseParser()
    return parser.parse_response(model_id, api_response, temperature, expected_answer_format)

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