# src/uncertainty_calibration/dependency_response_parser.py
"""
Enhanced dependency response parser for Level 3 comparisons.
Robust parsing of LLM responses with 4-dimension framework and perplexity-based uncertainty.
"""

import json
import re
import logging
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ParsedDependencyResponse:
    """Container for parsed dependency comparison response."""
    parent_url: str
    parent_name: str
    dependency_a_url: str
    dependency_a_name: str
    dependency_b_url: str
    dependency_b_name: str
    dimension_assessments: Dict[str, Dict[str, Any]]  # dimension_id -> {score_a, score_b, weight, reasoning, raw_uncertainty}
    overall_assessment: Dict[str, Any]  # choice, confidence, weighted_scores, reasoning
    parsing_method: str
    parsing_success: bool
    parsing_warnings: List[str]

class DependencyResponseParser:
    """
    Enhanced parser for dependency comparison responses using 4-dimension framework.
    Extends the fuzzy parsing approach from criteria assessment.
    """
    
    def __init__(self):
        """Initialize the dependency response parser."""
        # Default dimension IDs (in order)
        self.default_dimensions = [
            "functional_necessity", "performance_impact", 
            "replacement_difficulty", "integration_depth"
        ]
        
        # Default weights
        self.default_weights = {
            "functional_necessity": 0.4,
            "performance_impact": 0.3,
            "replacement_difficulty": 0.2,
            "integration_depth": 0.1
        }
    
    def parse_response(self, raw_response: str, parent_url: str, dep_a_url: str, 
                      dep_b_url: str, logprobs_data: Optional[List[Dict]] = None) -> ParsedDependencyResponse:
        """
        Parse LLM response using multiple strategies with enhanced uncertainty calculation.
        
        Args:
            raw_response: Raw text response from LLM
            parent_url: Parent repository URL for context
            dep_a_url: First dependency URL
            dep_b_url: Second dependency URL
            logprobs_data: Token logprobs data from MultiModelEngine
            
        Returns:
            ParsedDependencyResponse object with enhanced fields
        """
        warnings = []
        
        # Strategy 1: Try perfect JSON parsing
        parsed_data, method = self._try_json_parsing(raw_response)
        if parsed_data:
            logger.debug(f"Successfully parsed response using {method}")
            return self._create_response_from_json(
                parsed_data, parent_url, dep_a_url, dep_b_url, method, warnings, logprobs_data
            )
        
        # Strategy 2: Try cleaned JSON parsing
        parsed_data, method = self._try_cleaned_json_parsing(raw_response)
        if parsed_data:
            logger.debug(f"Successfully parsed response using {method}")
            warnings.append("Required JSON cleaning for parsing")
            return self._create_response_from_json(
                parsed_data, parent_url, dep_a_url, dep_b_url, method, warnings, logprobs_data
            )
        
        # Strategy 3: Try regex extraction
        parsed_data, method = self._try_regex_extraction(raw_response)
        if parsed_data:
            logger.debug(f"Successfully parsed response using {method}")
            warnings.append("Used regex extraction instead of JSON parsing")
            return self._create_response_from_extracted_data(
                parsed_data, parent_url, dep_a_url, dep_b_url, method, warnings, logprobs_data
            )
        
        # Strategy 4: Try line-by-line parsing
        parsed_data, method = self._try_line_parsing(raw_response)
        if parsed_data:
            logger.debug(f"Successfully parsed response using {method}")
            warnings.append("Used line-by-line parsing as fallback")
            return self._create_response_from_extracted_data(
                parsed_data, parent_url, dep_a_url, dep_b_url, method, warnings, logprobs_data
            )
        
        # All parsing failed - raise error
        error_msg = f"All parsing strategies failed for dependency comparison"
        logger.error(error_msg)
        raise ValueError(f"Failed to parse dependency response: {error_msg}")
    
    def _calculate_reasoning_perplexity(self, reasoning_text: str, 
                                      logprobs_data: Optional[List[Dict]]) -> float:
        """
        Calculate perplexity-based uncertainty for reasoning text.
        Reuses the logic from criteria assessment parser.
        """
        if not logprobs_data or not reasoning_text:
            return 0.5  # Default uncertainty when no logprobs available
        
        try:
            # Find tokens that correspond to this reasoning text
            relevant_tokens = self._extract_tokens_for_reasoning(reasoning_text, logprobs_data)
            
            if not relevant_tokens:
                return 0.5  # Default when no matching tokens found
            
            # Calculate average log probability
            total_log_prob = 0.0
            token_count = 0
            
            for token_data in relevant_tokens:
                logprob = token_data.get('logprob')
                if logprob is not None and not math.isinf(logprob):
                    total_log_prob += logprob
                    token_count += 1
            
            if token_count == 0:
                return 0.5
            
            # Calculate perplexity and normalize to uncertainty
            avg_log_prob = total_log_prob / token_count
            perplexity = math.exp(-avg_log_prob)
            
            # Normalize perplexity to [0,1] uncertainty range
            uncertainty = self._normalize_perplexity_to_uncertainty(perplexity)
            
            return max(0.0, min(1.0, uncertainty))
            
        except Exception as e:
            logger.warning(f"Error calculating perplexity for reasoning: {e}")
            return 0.5
    
    def _extract_tokens_for_reasoning(self, reasoning_text: str, 
                                    logprobs_data: List[Dict]) -> List[Dict]:
        """Extract logprob tokens that correspond to the reasoning text."""
        if not reasoning_text or not logprobs_data:
            return []
        
        # Clean reasoning text for matching
        reasoning_clean = reasoning_text.lower().strip()
        reasoning_words = set(reasoning_clean.split())
        
        relevant_tokens = []
        
        # Simple approach: find tokens whose text overlaps with reasoning words
        for token_data in logprobs_data:
            token_text = token_data.get('token', '').lower().strip()
            
            # Skip empty or whitespace-only tokens
            if not token_text or token_text.isspace():
                continue
            
            # Check if token text appears in reasoning or shares words
            if (token_text in reasoning_clean or 
                any(word in token_text for word in reasoning_words if len(word) > 2)):
                relevant_tokens.append(token_data)
        
        return relevant_tokens
    
    def _normalize_perplexity_to_uncertainty(self, perplexity: float) -> float:
        """Normalize perplexity value to uncertainty in [0,1] range."""
        if perplexity <= 1.0:
            return 0.1  # Minimum uncertainty
        
        # Sigmoid transformation
        k = 2.0  # Steepness parameter
        offset = 1.0  # Midpoint
        
        log_perplexity = math.log(perplexity)
        uncertainty = 1.0 / (1.0 + math.exp(-k * (log_perplexity - offset)))
        
        return uncertainty
    
    def _try_json_parsing(self, response: str) -> Tuple[Optional[Dict], str]:
        """Try to parse response as perfect JSON."""
        try:
            # Look for JSON block in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed, "perfect_json"
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.debug(f"JSON parsing error: {e}")
        
        return None, ""
    
    def _try_cleaned_json_parsing(self, response: str) -> Tuple[Optional[Dict], str]:
        """Try to parse after cleaning common JSON issues."""
        try:
            # Extract potential JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return None, ""
            
            json_str = json_match.group(0)
            
            # Common fixes
            # Remove trailing commas
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Fix unquoted keys
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            
            # Fix single quotes
            json_str = json_str.replace("'", '"')
            
            # Remove comments
            json_str = re.sub(r'//.*?\n', '\n', json_str)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            
            parsed = json.loads(json_str)
            return parsed, "cleaned_json"
            
        except Exception as e:
            logger.debug(f"Cleaned JSON parsing error: {e}")
        
        return None, ""
    
    def _try_regex_extraction(self, response: str) -> Tuple[Optional[Dict], str]:
        """Try to extract dependency data using regex patterns."""
        try:
            extracted = {}
            
            # Pattern for dimension entries with scores for both dependencies
            dimension_pattern = r'"?(\w+)"?\s*:\s*\{\s*"?score_a"?\s*:\s*(\d+),?\s*"?score_b"?\s*:\s*(\d+),?\s*"?weight"?\s*:\s*([\d.]+),?\s*"?reasoning"?\s*:\s*"([^"]*)"?\s*\}'
            
            matches = re.findall(dimension_pattern, response, re.IGNORECASE | re.DOTALL)
            
            if matches:
                dimension_assessments = {}
                for dimension, score_a, score_b, weight, reasoning in matches:
                    if dimension in self.default_dimensions:
                        dimension_assessments[dimension] = {
                            "score_a": int(score_a),
                            "score_b": int(score_b),
                            "weight": float(weight),
                            "reasoning": reasoning.strip()
                        }
                
                if dimension_assessments:
                    extracted["dimension_assessments"] = dimension_assessments
                    
                    # Try to extract overall assessment
                    choice_match = re.search(r'"?choice"?\s*:\s*"?([AB]|Equal)"?', response, re.IGNORECASE)
                    if choice_match:
                        extracted["overall_assessment"] = {"choice": choice_match.group(1)}
                    
                    return extracted, "regex_extraction"
            
        except Exception as e:
            logger.debug(f"Regex extraction error: {e}")
        
        return None, ""
    
    def _try_line_parsing(self, response: str) -> Tuple[Optional[Dict], str]:
        """Try line-by-line parsing for structured text."""
        try:
            lines = response.split('\n')
            dimension_assessments = {}
            
            current_dimension = None
            current_scores = {}
            current_weight = None
            current_reasoning = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for dimension names
                for dimension in self.default_dimensions:
                    if dimension.replace('_', ' ').lower() in line.lower():
                        current_dimension = dimension
                        break
                
                # Look for scores
                score_a_match = re.search(r'score[_\s]*a[:\s]*(\d+)', line, re.IGNORECASE)
                if score_a_match and current_dimension:
                    current_scores['score_a'] = int(score_a_match.group(1))
                
                score_b_match = re.search(r'score[_\s]*b[:\s]*(\d+)', line, re.IGNORECASE)
                if score_b_match and current_dimension:
                    current_scores['score_b'] = int(score_b_match.group(1))
                
                # Look for weights
                weight_match = re.search(r'weight[:\s]*([\d.]+)', line, re.IGNORECASE)
                if weight_match and current_dimension:
                    current_weight = float(weight_match.group(1))
                
                # Collect reasoning
                if current_dimension and any(word in line.lower() for word in ['reason', 'because', 'explanation']):
                    current_reasoning = line
                
                # If we have enough info, store it
                if (current_dimension and 'score_a' in current_scores and 'score_b' in current_scores):
                    dimension_assessments[current_dimension] = {
                        "score_a": current_scores['score_a'],
                        "score_b": current_scores['score_b'],
                        "weight": current_weight or self.default_weights.get(current_dimension, 0.25),
                        "reasoning": current_reasoning or "No reasoning provided"
                    }
                    current_dimension = None
                    current_scores = {}
                    current_weight = None
                    current_reasoning = ""
            
            if dimension_assessments:
                return {"dimension_assessments": dimension_assessments}, "line_parsing"
            
        except Exception as e:
            logger.debug(f"Line parsing error: {e}")
        
        return None, ""
    
    def _create_response_from_json(self, parsed_data: Dict, parent_url: str, dep_a_url: str, 
                                  dep_b_url: str, method: str, warnings: List[str],
                                  logprobs_data: Optional[List[Dict]]) -> ParsedDependencyResponse:
        """Create response object from successfully parsed JSON."""
        
        # Extract dimension assessments
        dimension_assessments = parsed_data.get("dimension_assessments", {})
        
        # Extract overall assessment
        overall_assessment = parsed_data.get("overall_assessment", {})
        
        # Check if this is an ultra-simplified response (only contains choice)
        is_ultra_simplified = "choice" in parsed_data and len(parsed_data) == 1
        
        if is_ultra_simplified:
            # Ultra-simplified: only {"choice": "A"}
            overall_assessment = {"choice": parsed_data["choice"]}
            is_simplified = True
        else:
            # Check if this is a simplified response (no dimension assessments)
            is_simplified = not dimension_assessments and overall_assessment.get("choice")
        
        if is_simplified:
            # For simplified responses, create empty dimensions
            validated_dimensions = {}
            warnings.append("Simplified response format detected - no dimension assessments")
        else:
            # Validate and fill missing dimensions with enhanced uncertainty calculation
            validated_dimensions = self._validate_and_fill_dimensions_enhanced(
                dimension_assessments, warnings, logprobs_data
            )
        
        # Extract metadata (use unknown for ultra-simplified responses)
        if is_ultra_simplified:
            parent_name = "unknown"
            dep_a_name = "unknown" 
            dep_b_name = "unknown"
        else:
            parent_name = parsed_data.get("parent_name", "unknown")
            dep_a_name = parsed_data.get("dependency_a_name", "unknown") 
            dep_b_name = parsed_data.get("dependency_b_name", "unknown")
        
        return ParsedDependencyResponse(
            parent_url=parent_url,
            parent_name=parent_name,
            dependency_a_url=dep_a_url,
            dependency_a_name=dep_a_name,
            dependency_b_url=dep_b_url,
            dependency_b_name=dep_b_name,
            dimension_assessments=validated_dimensions,
            overall_assessment=overall_assessment,
            parsing_method=method,
            parsing_success=True,
            parsing_warnings=warnings
        )
    
    def _create_response_from_extracted_data(self, extracted_data: Dict, parent_url: str, 
                                           dep_a_url: str, dep_b_url: str, method: str, 
                                           warnings: List[str],
                                           logprobs_data: Optional[List[Dict]]) -> ParsedDependencyResponse:
        """Create response object from extracted data."""
        
        dimension_assessments = extracted_data.get("dimension_assessments", {})
        overall_assessment = extracted_data.get("overall_assessment", {})
        
        # Validate and fill missing dimensions
        validated_dimensions = self._validate_and_fill_dimensions_enhanced(
            dimension_assessments, warnings, logprobs_data
        )
        
        # Calculate overall assessment if missing
        if not overall_assessment.get("choice"):
            overall_assessment = self._calculate_overall_assessment(validated_dimensions)
            warnings.append("Calculated overall assessment from dimension scores")
        
        return ParsedDependencyResponse(
            parent_url=parent_url,
            parent_name="unknown",
            dependency_a_url=dep_a_url,
            dependency_a_name="unknown",
            dependency_b_url=dep_b_url,
            dependency_b_name="unknown",
            dimension_assessments=validated_dimensions,
            overall_assessment=overall_assessment,
            parsing_method=method,
            parsing_success=True,
            parsing_warnings=warnings
        )
    
    def _validate_and_fill_dimensions_enhanced(self, dimension_assessments: Dict, warnings: List[str],
                                             logprobs_data: Optional[List[Dict]]) -> Dict[str, Dict]:
        """Validate dimension assessments and fill with enhanced uncertainty calculation."""
        
        validated_dimensions = {}
        
        for dimension in self.default_dimensions:
            if dimension in dimension_assessments:
                assessment = dimension_assessments[dimension]
                
                # Validate scores
                score_a = assessment.get("score_a", 5)
                score_b = assessment.get("score_b", 5)
                
                if not isinstance(score_a, (int, float)) or not (1 <= score_a <= 10):
                    score_a = 5
                    warnings.append(f"Invalid score_a for {dimension}, using default")
                
                if not isinstance(score_b, (int, float)) or not (1 <= score_b <= 10):
                    score_b = 5
                    warnings.append(f"Invalid score_b for {dimension}, using default")
                
                # Validate weight
                weight = assessment.get("weight", self.default_weights[dimension])
                if not isinstance(weight, (int, float)) or weight < 0:
                    weight = self.default_weights[dimension]
                    warnings.append(f"Invalid weight for {dimension}, using default")
                
                # Get reasoning
                reasoning = assessment.get("reasoning", "No reasoning provided")
                
                # Calculate raw_uncertainty from perplexity
                raw_uncertainty = self._calculate_reasoning_perplexity(reasoning, logprobs_data)
                
                validated_dimensions[dimension] = {
                    "score_a": int(score_a),
                    "score_b": int(score_b),
                    "weight": float(weight),
                    "reasoning": reasoning,
                    "raw_uncertainty": raw_uncertainty
                }
            else:
                # Missing dimension - raise error instead of using defaults
                raise ValueError(f"Missing assessment for required dimension: {dimension}")
        
        return validated_dimensions
    
    def _calculate_overall_assessment(self, dimension_assessments: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate overall assessment from dimension scores."""
        weighted_score_a = 0.0
        weighted_score_b = 0.0
        total_weight = 0.0
        
        for dimension, assessment in dimension_assessments.items():
            score_a = assessment.get("score_a", 5)
            score_b = assessment.get("score_b", 5)
            weight = assessment.get("weight", 0.25)
            
            weighted_score_a += score_a * weight
            weighted_score_b += score_b * weight
            total_weight += weight
        
        # Determine choice
        if abs(weighted_score_a - weighted_score_b) < 0.5:
            choice = "Equal"
            confidence = 0.5
        elif weighted_score_a > weighted_score_b:
            choice = "A"
            confidence = min(0.95, 0.5 + (weighted_score_a - weighted_score_b) / 10.0)
        else:
            choice = "B"
            confidence = min(0.95, 0.5 + (weighted_score_b - weighted_score_a) / 10.0)
        
        return {
            "choice": choice,
            "confidence": confidence,
            "weighted_score_a": weighted_score_a,
            "weighted_score_b": weighted_score_b,
            "reasoning": f"Calculated from dimension scores: A={weighted_score_a:.2f}, B={weighted_score_b:.2f}"
        }