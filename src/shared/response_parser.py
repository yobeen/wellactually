# src/shared/response_parser.py
"""
Enhanced response parser with model-specific answer postprocessing integration.
Updated to provide access to full content logprobs for criteria assessment.
"""
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
import copy

# Import the new postprocessor
from src.shared.model_answer_postprocessor import ModelAnswerPostprocessor

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
    answer_token_info: Optional[Dict] = None
    normalized_answer: Optional[str] = None
    # NEW: Full content logprobs for criteria assessment
    full_content_logprobs: Optional[List[Dict]] = None

class AnswerTokenExtractor:
    """
    Enhanced answer token extractor with model-specific postprocessing.
    """
    
    def __init__(self):
        # Answer patterns by level (before normalization)
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
        
        # Initialize postprocessor
        self.postprocessor = ModelAnswerPostprocessor()
    
    def extract_answer_token(self, logprobs_content: List[Dict], model_id: str,
                           expected_format: str = "choice") -> Tuple[Optional[Dict], Dict]:
        """
        Extract answer token from PREPROCESSED logprobs content.
        
        Args:
            logprobs_content: List of ALREADY CLEANED token logprob objects
            model_id: Model identifier for family-specific processing
            expected_format: "choice" (A/B/Equal), "numeric" (1-10), or "auto"
            
        Returns:
            (answer_token_dict, extraction_info)
        """
        extraction_info = {
            "method": None,
            "pattern_found": None,
            "token_index": None,
            "fallback_used": False,
            "extraction_confidence": 0.0,
            "original_token": None,
            "normalized_token": None
        }
        
        if not logprobs_content:
            extraction_info["method"] = "failed_no_content"
            return None, extraction_info
        
        # Strategy 1: Look for "Answer:" pattern followed by answer
        answer_token, info = self._find_answer_after_indicator(logprobs_content, expected_format)
        if answer_token:
            extraction_info.update(info)
            extraction_info["method"] = "answer_indicator"
        else:
            # Strategy 2: Look for bracketed answers [A], [B], etc.
            answer_token, info = self._find_bracketed_answer(logprobs_content, expected_format)
            if answer_token:
                extraction_info.update(info)
                extraction_info["method"] = "bracketed_answer"
            else:
                # Strategy 3: Find direct answer tokens with meaningful alternatives
                answer_token, info = self._find_direct_answer_token(logprobs_content, expected_format)
                if answer_token:
                    extraction_info.update(info)
                    extraction_info["method"] = "direct_answer"
                else:
                    # Strategy 4: Fallback - last resort heuristics
                    answer_token, info = self._fallback_extraction(logprobs_content, expected_format)
                    extraction_info.update(info)
                    extraction_info["method"] = "fallback"
                    extraction_info["fallback_used"] = True
        
        # Apply final answer normalization rules if token found
        if answer_token:
            raw_token = answer_token.get('token', '')
            extraction_info["original_token"] = answer_token.get('original_token', raw_token)
            
            # Apply final normalization rules (equ → Equal, etc.)
            try:
                model_family = self.postprocessor.detect_model_family(model_id)
                normalized_answer = self.postprocessor.apply_family_rules(
                    raw_token, model_family, expected_format
                )
                
                # Create normalized token object
                normalized_token = answer_token.copy()
                normalized_token['token'] = normalized_answer
                extraction_info["normalized_token"] = normalized_answer
                
                # Validate that normalized answer is valid
                if self.postprocessor.is_valid_answer(normalized_answer, expected_format):
                    extraction_info["extraction_confidence"] = max(extraction_info.get("extraction_confidence", 0), 0.8)
                    return normalized_token, extraction_info
                else:
                    # Fallback to original if normalization produces invalid answer
                    logger.warning(f"Normalization produced invalid answer: {normalized_answer} for {model_id}")
                    return answer_token, extraction_info
                    
            except Exception as e:
                logger.warning(f"Final normalization failed for {model_id}: {e}")
                return answer_token, extraction_info
        
        return None, extraction_info
    
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
                        if self._is_potential_answer_token(next_token, expected_format):
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
                    if self._is_potential_answer(answer, expected_format):
                        info["pattern_found"] = f"bracketed_{pattern}"
                        info["token_index"] = i
                        info["extraction_confidence"] = self._calculate_confidence(token_obj)
                        return token_obj, info
        
        return None, info
    
    def _find_direct_answer_token(self, logprobs_content: List[Dict], 
                                expected_format: str) -> Tuple[Optional[Dict], Dict]:
        """Find answer tokens directly with meaningful alternatives."""
        info = {"pattern_found": "direct", "token_index": None}
        
        candidates = []
        
        for i, token_obj in enumerate(logprobs_content):
            if self._is_potential_answer_token(token_obj, expected_format):
                confidence = self._calculate_confidence(token_obj)
                alternatives_quality = self._assess_alternatives_quality(token_obj, expected_format)
                
                score = confidence * alternatives_quality
                candidates.append((token_obj, i, score))
        
        if candidates:
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
            if self._is_potential_answer_token(token_obj, expected_format):
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
    
    def _is_potential_answer_token(self, token_obj: Dict, expected_format: str) -> bool:
        """Check if token object represents a potential answer (works on CLEANED tokens)."""
        token_text = token_obj.get('token', '').strip()
        return self._is_potential_answer(token_text, expected_format)
    
    def _is_potential_answer(self, token_text: str, expected_format: str) -> bool:
        """Check if token text could be an answer (works on CLEANED tokens)."""
        token_clean = token_text.strip().upper().rstrip('.,!?:')
        
        if expected_format == "choice":
            # Now works on cleaned tokens - should find "A", "B", "Equal" directly
            # Also include "EQU" for the equ → Equal rule
            return token_clean in ["A", "B", "EQUAL", "1", "2", "EQU", "EQ"]
        elif expected_format == "numeric":
            return token_clean in [str(i) for i in range(1, 11)]
        else:  # auto
            return token_clean in (["A", "B", "EQUAL", "1", "2", "EQU", "EQ"] + 
                                 [str(i) for i in range(1, 11)])
    
    def _calculate_confidence(self, token_obj: Dict) -> float:
        """Calculate confidence score for a token based on its logprob."""
        logprob = token_obj.get('logprob', -10.0)
        confidence = min(1.0, max(0.0, (logprob + 5.0) / 5.0))
        return confidence
    
    def _assess_alternatives_quality(self, token_obj: Dict, expected_format: str) -> float:
        """Assess quality of alternatives in top_logprobs."""
        top_logprobs = token_obj.get('top_logprobs', [])
        
        if not top_logprobs:
            return 0.1
        
        valid_alternatives = 0
        for alt in top_logprobs:
            alt_token = alt.get('token', '').strip().upper()
            if self._is_potential_answer(alt_token, expected_format):
                valid_alternatives += 1
        
        if len(top_logprobs) == 0:
            return 0.1
        
        quality = valid_alternatives / len(top_logprobs)
        return max(0.2, quality)


class ResponseParser:
    """
    Enhanced response parser with model-specific answer postprocessing and full logprobs access.
    """
    
    def __init__(self, cost_per_1k_tokens: float = 0.01):
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.answer_extractor = AnswerTokenExtractor()
    
    def parse_response(self, model_id: str, api_response: Dict, 
                      temperature: float, expected_answer_format: str = "choice") -> ModelResponse:
        """Parse API response with enhanced answer extraction and normalization."""
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
        """Parse successful API response with enhanced extraction."""
        
        # Extract basic response data
        choice = data.get('choices', [{}])[0]
        message = choice.get('message', {})
        content = message.get('content', '').strip()
        
        # Extract and process logprobs with enhanced answer extraction
        logprobs_data = choice.get('logprobs')
        parsed_logprobs, answer_token_info, full_content_logprobs = self._extract_logprobs_enhanced(
            logprobs_data, model_id, expected_answer_format
        )
        # Calculate uncertainty from perplexity (answer token only for L1, all tokens for others)
        uncertainty = self._calculate_uncertainty_from_perplexity(
            full_content_logprobs, answer_token_info, expected_answer_format
        )
        raw_choice = self._extract_raw_choice(parsed_logprobs, content, answer_token_info)
        
        # Extract normalized answer if available
        normalized_answer = answer_token_info.get("normalized_token")
        
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
            answer_token_info=answer_token_info,
            normalized_answer=normalized_answer,
            full_content_logprobs=full_content_logprobs  # NEW
        )
    
    def _extract_logprobs_enhanced(self, logprobs_data: Optional[Dict], model_id: str,
                                 expected_answer_format: str) -> Tuple[Dict[str, float], Dict, Optional[List[Dict]]]:
        """Enhanced logprobs extraction with model-specific postprocessing and full content access."""
        parsed_logprobs = {}
        answer_token_info = {
            "extraction_success": False,
            "method": None,
            "token_found": None,
            "alternatives_count": 0,
            "model_family": None,
            "preprocessing_applied": False
        }
        full_content_logprobs = None
        
        if not logprobs_data or 'content' not in logprobs_data:
            return parsed_logprobs, answer_token_info, full_content_logprobs
        
        raw_token_logprobs = logprobs_data['content']
        if not raw_token_logprobs:
            return parsed_logprobs, answer_token_info, full_content_logprobs
        
        # CRITICAL: Apply preprocessing to ALL tokens BEFORE extraction
        preprocessed_token_logprobs = self._preprocess_all_tokens(raw_token_logprobs, model_id)
        answer_token_info["preprocessing_applied"] = True
        answer_token_info["model_family"] = self.answer_extractor.postprocessor.detect_model_family(model_id)
        
        # NEW: Store full content logprobs for criteria assessment
        full_content_logprobs = preprocessed_token_logprobs
        
        # Use enhanced answer token extraction on CLEANED tokens
        answer_token, extraction_info = self.answer_extractor.extract_answer_token(
            preprocessed_token_logprobs, model_id, expected_answer_format
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
            logger.warning(f"Answer token extraction failed for {model_id}, falling back to first token")
            answer_token_info["method"] = "fallback_first_token"
            
            if preprocessed_token_logprobs:
                first_token = preprocessed_token_logprobs[0]
                top_logprobs = first_token.get('top_logprobs', [])
                
                for token_data in top_logprobs:
                    token = token_data.get('token', '').strip()
                    logprob = token_data.get('logprob', -float('inf'))
                    
                    if logprob > -float('inf') and token:
                        parsed_logprobs[token] = np.exp(logprob)
        
        return parsed_logprobs, answer_token_info, full_content_logprobs
    
    def _preprocess_all_tokens(self, token_logprobs: List[Dict], model_id: str) -> List[Dict]:
        """
        Apply model-family preprocessing to ALL tokens before extraction.
        This is the critical fix - clean tokens BEFORE trying to find answers.
        """
        model_family = self.answer_extractor.postprocessor.detect_model_family(model_id)
        preprocessed_tokens = []
        
        for token_obj in token_logprobs:
            if not isinstance(token_obj, dict):
                preprocessed_tokens.append(token_obj)
                continue
            
            # Create copy to avoid modifying original
            cleaned_token_obj = token_obj.copy()
            raw_token = token_obj.get('token', '')
            
            # Apply family-specific preprocessing
            cleaned_token = self.answer_extractor.postprocessor.preprocess_token_by_family(
                raw_token, model_family
            )
            
            # Update token while preserving all other fields (logprob, top_logprobs, etc.)
            cleaned_token_obj['token'] = cleaned_token
            cleaned_token_obj['original_token'] = raw_token  # Keep for debugging
            
            # Also clean tokens in top_logprobs
            if 'top_logprobs' in cleaned_token_obj:
                cleaned_top_logprobs = []
                for top_token_data in cleaned_token_obj['top_logprobs']:
                    if isinstance(top_token_data, dict):
                        cleaned_top_token = top_token_data.copy()
                        raw_top_token = top_token_data.get('token', '')
                        cleaned_top_token['token'] = self.answer_extractor.postprocessor.preprocess_token_by_family(
                            raw_top_token, model_family
                        )
                        cleaned_top_token['original_token'] = raw_top_token
                        cleaned_top_logprobs.append(cleaned_top_token)
                    else:
                        cleaned_top_logprobs.append(top_token_data)
                cleaned_token_obj['top_logprobs'] = cleaned_top_logprobs
            
            preprocessed_tokens.append(cleaned_token_obj)
        
        return preprocessed_tokens
    
    def _calculate_uncertainty_from_perplexity(self, full_content_logprobs: Optional[List[Dict]], 
                                             answer_token_info: Optional[Dict] = None, 
                                             expected_format: str = "choice") -> float:
        """
        Calculate uncertainty based on perplexity of answer tokens only for Level 1,
        or all tokens for other levels.
        
        Args:
            full_content_logprobs: List of token logprob dictionaries
            answer_token_info: Information about the extracted answer token
            expected_format: Expected answer format ("choice" for L1/L3, "numeric" for L2)
            
        Returns:
            Uncertainty score between 0.0 and 1.0
        """
        if not full_content_logprobs:
            return 1.0  # Maximum uncertainty if no logprobs available
        
        try:
            # For Level 1 comparisons (choice format), use only answer token logprob
            if expected_format == "choice" and answer_token_info and answer_token_info.get("token_index") is not None:
                token_index = answer_token_info["token_index"]
                
                # Extract logprob from the answer token only
                if 0 <= token_index < len(full_content_logprobs):
                    answer_token_data = full_content_logprobs[token_index]
                    if isinstance(answer_token_data, dict) and 'logprob' in answer_token_data:
                        answer_logprob = answer_token_data['logprob']
                        if answer_logprob is not None and not np.isnan(answer_logprob) and not np.isinf(answer_logprob):
                            # Convert single token logprob to perplexity
                            perplexity = 2 ** (-answer_logprob)
                            
                            # Clamp perplexity to reasonable range
                            perplexity = max(1.0, min(100.0, perplexity))
                            
                            # Transform to uncertainty: higher perplexity = higher uncertainty
                            uncertainty = np.log(perplexity) / np.log(100.0)
                            
                            # Ensure uncertainty is in [0, 1] range
                            uncertainty = max(0.0, min(1.0, uncertainty))
                            
                            return uncertainty
            
            # Fallback: Extract logprobs for all tokens (original behavior)
            token_logprobs = []
            for token_data in full_content_logprobs:
                if isinstance(token_data, dict) and 'logprob' in token_data:
                    logprob = token_data['logprob']
                    if logprob is not None and not np.isnan(logprob) and not np.isinf(logprob):
                        token_logprobs.append(logprob)
            
            if not token_logprobs:
                return 1.0  # Maximum uncertainty if no valid logprobs
            
            # Calculate average logprob (log probability)
            avg_logprob = np.mean(token_logprobs)
            
            # Convert to perplexity: perplexity = 2^(-avg_logprob)
            perplexity = 2 ** (-avg_logprob)
            
            # Normalize perplexity to uncertainty range [0, 1]
            # Lower perplexity = higher confidence = lower uncertainty
            # We use a sigmoid-like transformation to map perplexity to [0, 1]
            # Typical perplexity ranges from 1 (perfect) to 100+ (very uncertain)
            
            # Clamp perplexity to reasonable range
            perplexity = max(1.0, min(100.0, perplexity))
            
            # Transform to uncertainty: higher perplexity = higher uncertainty
            # Using logarithmic scaling: uncertainty = log(perplexity) / log(100)
            uncertainty = np.log(perplexity) / np.log(100.0)
            
            # Ensure uncertainty is in [0, 1] range
            uncertainty = max(0.0, min(1.0, uncertainty))
            
            return uncertainty
            
        except Exception as e:
            logger.warning(f"Error calculating uncertainty from perplexity: {e}")
            return 0.5  # Return neutral uncertainty on error

    def _calculate_uncertainty(self, probabilities: Dict[str, float]) -> float:
        """Calculate uncertainty as normalized entropy (legacy method)."""
        if not probabilities:
            return 1.0
        
        total_prob = sum(probabilities.values())
        if total_prob == 0:
            return 1.0
        
        normalized_probs = [p / total_prob for p in probabilities.values()]
        
        entropy = 0.0
        for p in normalized_probs:
            if p > 0:
                entropy -= p * np.log2(p + 1e-10)
        
        max_entropy = np.log2(len(probabilities))
        if max_entropy == 0:
            return 1.0
        
        normalized_entropy = entropy / max_entropy
        return min(1.0, max(0.0, normalized_entropy))
    
    def _extract_raw_choice(self, probabilities: Dict[str, float], 
                          content: str, answer_token_info: Dict) -> str:
        """Extract the raw choice with preference for normalized answer."""
        
        # Use normalized answer if available and valid
        if answer_token_info.get("normalized_token"):
            return answer_token_info["normalized_token"]
        
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
        
        if content_upper.strip().endswith((' A', ' B', ' EQUAL')):
            return content_upper.strip()[-1] if content_upper.strip().endswith((' A', ' B')) else 'EQUAL'
        
        return content
    
    def _extract_usage_info(self, data: Dict) -> Dict[str, int]:
        """Extract token usage information from API response."""
        usage = data.get('usage', {})
        return {
            'total_tokens': usage.get('total_tokens', 0),
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0)
        }
    
    def _calculate_cost(self, total_tokens: int) -> float:
        """Calculate cost based on token usage."""
        return total_tokens * self.cost_per_1k_tokens / 1000.0
    
    def _create_error_response(self, model_id: str, temperature: float, 
                             error_msg: str) -> ModelResponse:
        """Create error response for failed requests."""
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
            error=error_msg,
            timestamp=datetime.now().isoformat(),
            answer_token_info={"extraction_success": False, "method": "error"},
            full_content_logprobs=None
        )