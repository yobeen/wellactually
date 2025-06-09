# src/uncertainty_calibration/originality_assessment/originality_response_parser.py
"""
Originality Response Parser with Perplexity-based Uncertainty

Parses LLM responses for originality assessments with fuzzy matching, error recovery,
and perplexity-based uncertainty calculation for both individual criteria and overall reasoning.
"""

import json
import re
import math
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ParsedOriginalityResponse:
    """Structured representation of a parsed originality assessment response with uncertainty."""
    
    # Core assessment data
    repository_url: str
    repository_name: str
    originality_category: str
    criteria_scores: Dict[str, Dict[str, Any]]  # criterion -> {score, weight, reasoning, raw_uncertainty}
    overall_reasoning: str
    final_originality_score: float
    assessment_confidence: float
    
    # Uncertainty fields
    criteria_uncertainties: Dict[str, float]  # per-criterion uncertainty from perplexity
    overall_reasoning_uncertainty: float      # overall reasoning uncertainty
    aggregate_uncertainty: float              # weighted average uncertainty across criteria
    
    # Parsing metadata
    parsing_method: str
    parsing_success: bool
    parsing_warnings: List[str]

class OriginalityResponseParser:
    """
    Parses LLM responses for originality assessments with robust error handling
    and perplexity-based uncertainty calculation.
    """
    
    def __init__(self):
        """Initialize the parser with default configurations."""
        self.default_criteria = [
            'protocol_implementation',
            'algorithmic_innovation', 
            'developer_experience',
            'architectural_innovation',
            'security_innovation',
            'standards_leadership',
            'cross_client_compatibility',
        ]
        
        # Default weights (will be overridden by repository-specific weights)
        self.default_weights = {
            'protocol_implementation': 0.143,  # 1/7
            'algorithmic_innovation': 0.143,
            'developer_experience': 0.143,
            'architectural_innovation': 0.143,
            'security_innovation': 0.143,
            'standards_leadership': 0.143,
            'cross_client_compatibility': 0.143,
        }
    
    def parse_response(self, response_text: str, expected_weights: Dict[str, float], 
                      repo_url: str, repo_name: str, originality_category: str,
                      logprobs_data: Optional[List[Dict]] = None) -> ParsedOriginalityResponse:
        """
        Parse LLM response for originality assessment with multiple fallback strategies.
        
        Args:
            response_text: Raw LLM response text
            expected_weights: Expected criterion weights for this repository
            repo_url: Repository URL
            repo_name: Repository name
            originality_category: Repository originality category (A-I)
            logprobs_data: Token logprobs data from MultiModelEngine for uncertainty calculation
            
        Returns:
            ParsedOriginalityResponse with extracted information and uncertainty metrics
        """
        warnings = []
        
        # Try parsing methods in order of reliability
        try:
            # Method 1: JSON parsing
            parsed = self._try_json_parsing(response_text)
            if parsed:
                criteria_scores = self._validate_and_normalize_criteria(
                    parsed.get('criteria_scores', {}), expected_weights, warnings, logprobs_data
                )
                overall_reasoning = parsed.get('overall_reasoning', '')
                
                # Calculate overall reasoning uncertainty
                overall_uncertainty = self._calculate_reasoning_perplexity(overall_reasoning, logprobs_data)
                
                # Calculate aggregate uncertainty
                aggregate_uncertainty = self._calculate_aggregate_uncertainty(criteria_scores)
                
                # Extract per-criterion uncertainties
                criteria_uncertainties = {
                    criterion: data.get('raw_uncertainty', 0.5) 
                    for criterion, data in criteria_scores.items()
                }
                
                return ParsedOriginalityResponse(
                    repository_url=repo_url,
                    repository_name=repo_name,
                    originality_category=originality_category,
                    criteria_scores=criteria_scores,
                    overall_reasoning=overall_reasoning,
                    final_originality_score=self._calculate_final_score(criteria_scores),
                    assessment_confidence=float(parsed.get('assessment_confidence', 0.7)),
                    criteria_uncertainties=criteria_uncertainties,
                    overall_reasoning_uncertainty=overall_uncertainty,
                    aggregate_uncertainty=aggregate_uncertainty,
                    parsing_method='json',
                    parsing_success=True,
                    parsing_warnings=warnings
                )
        except Exception as e:
            logger.debug(f"JSON parsing failed: {e}")
            warnings.append(f"JSON parsing failed: {e}")
        
        # Method 2: Regex pattern matching
        try:
            parsed = self._try_regex_parsing(response_text)
            if parsed:
                criteria_scores = self._validate_and_normalize_criteria(
                    parsed.get('criteria_scores', {}), expected_weights, warnings, logprobs_data
                )
                overall_reasoning = parsed.get('overall_reasoning', '')
                overall_uncertainty = self._calculate_reasoning_perplexity(overall_reasoning, logprobs_data)
                aggregate_uncertainty = self._calculate_aggregate_uncertainty(criteria_scores)
                
                criteria_uncertainties = {
                    criterion: data.get('raw_uncertainty', 0.5) 
                    for criterion, data in criteria_scores.items()
                }
                
                return ParsedOriginalityResponse(
                    repository_url=repo_url,
                    repository_name=repo_name,
                    originality_category=originality_category,
                    criteria_scores=criteria_scores,
                    overall_reasoning=overall_reasoning,
                    final_originality_score=self._calculate_final_score(criteria_scores),
                    assessment_confidence=float(parsed.get('assessment_confidence', 0.5)),
                    criteria_uncertainties=criteria_uncertainties,
                    overall_reasoning_uncertainty=overall_uncertainty,
                    aggregate_uncertainty=aggregate_uncertainty,
                    parsing_method='regex',
                    parsing_success=True,
                    parsing_warnings=warnings
                )
        except Exception as e:
            logger.debug(f"Regex parsing failed: {e}")
            warnings.append(f"Regex parsing failed: {e}")
        
        # Method 3: Fuzzy text extraction
        try:
            parsed = self._try_fuzzy_parsing(response_text)
            criteria_scores = self._validate_and_normalize_criteria(
                parsed.get('criteria_scores', {}), expected_weights, warnings, logprobs_data
            )
            overall_reasoning = parsed.get('overall_reasoning', response_text[:500])
            overall_uncertainty = self._calculate_reasoning_perplexity(overall_reasoning, logprobs_data)
            aggregate_uncertainty = self._calculate_aggregate_uncertainty(criteria_scores)
            
            criteria_uncertainties = {
                criterion: data.get('raw_uncertainty', 0.5) 
                for criterion, data in criteria_scores.items()
            }
            
            return ParsedOriginalityResponse(
                repository_url=repo_url,
                repository_name=repo_name,
                originality_category=originality_category,
                criteria_scores=criteria_scores,
                overall_reasoning=overall_reasoning,
                final_originality_score=self._calculate_final_score(criteria_scores),
                assessment_confidence=0.3,  # Low confidence for fuzzy parsing
                criteria_uncertainties=criteria_uncertainties,
                overall_reasoning_uncertainty=overall_uncertainty,
                aggregate_uncertainty=aggregate_uncertainty,
                parsing_method='fuzzy',
                parsing_success=len(criteria_scores) >= len(self.default_criteria) // 2,
                parsing_warnings=warnings
            )
        except Exception as e:
            logger.error(f"All parsing methods failed: {e}")
            warnings.append(f"Fuzzy parsing failed: {e}")
        
        # Fallback: Return default scores with uncertainty
        warnings.append("Using default scores due to parsing failure")
        criteria_scores = self._create_default_criteria_scores(expected_weights)
        
        # Calculate uncertainties for fallback case
        criteria_uncertainties = {
            criterion: 0.8  # High uncertainty for fallback
            for criterion in criteria_scores.keys()
        }
        
        return ParsedOriginalityResponse(
            repository_url=repo_url,
            repository_name=repo_name,
            originality_category=originality_category,
            criteria_scores=criteria_scores,
            overall_reasoning="Failed to parse LLM response",
            final_originality_score=0.5,  # Neutral default
            assessment_confidence=0.1,  # Very low confidence
            criteria_uncertainties=criteria_uncertainties,
            overall_reasoning_uncertainty=0.8,  # High uncertainty
            aggregate_uncertainty=0.8,  # High uncertainty
            parsing_method='fallback',
            parsing_success=False,
            parsing_warnings=warnings
        )
    
    def _calculate_reasoning_perplexity(self, reasoning_text: str, 
                                      logprobs_data: Optional[List[Dict]]) -> float:
        """
        Calculate perplexity-based uncertainty for reasoning text.
        Adapted from criteria assessment parser.
        
        Args:
            reasoning_text: The reasoning text for a criterion or overall assessment
            logprobs_data: Token logprobs from the LLM response
            
        Returns:
            Uncertainty score [0,1] where higher means more uncertain
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
        """
        Extract logprob tokens that correspond to the reasoning text.
        Adapted from criteria assessment parser.
        
        Args:
            reasoning_text: The reasoning text to match
            logprobs_data: List of token logprob dictionaries
            
        Returns:
            List of relevant token dictionaries
        """
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
        """
        Normalize perplexity value to uncertainty in [0,1] range.
        Uses identical normalization as criteria assessment parser.
        
        Args:
            perplexity: Raw perplexity value (1 to infinity)
            
        Returns:
            Uncertainty score [0,1]
        """
        # Use sigmoid normalization with reasonable parameters
        # perplexity of 1 (perfect prediction) -> low uncertainty
        # perplexity of 10+ (poor prediction) -> high uncertainty
        
        if perplexity <= 1.0:
            return 0.1  # Minimum uncertainty
        
        # Sigmoid transformation: 1 / (1 + exp(-k * (log(perplexity) - offset)))
        k = 2.0  # Steepness parameter
        offset = 1.0  # Midpoint (perplexity ~2.7 gives 0.5 uncertainty)
        
        log_perplexity = math.log(perplexity)
        uncertainty = 1.0 / (1.0 + math.exp(-k * (log_perplexity - offset)))
        
        return uncertainty
    
    def _calculate_aggregate_uncertainty(self, criteria_scores: Dict[str, Dict]) -> float:
        """
        Calculate weighted average uncertainty across all criteria.
        
        Args:
            criteria_scores: Dictionary of criteria with scores, weights, and uncertainties
            
        Returns:
            Aggregate uncertainty score [0,1]
        """
        if not criteria_scores:
            return 0.5
        
        total_weighted_uncertainty = 0.0
        total_weight = 0.0
        
        for criterion, data in criteria_scores.items():
            weight = data.get('weight', 0.0)
            uncertainty = data.get('raw_uncertainty', 0.5)
            
            total_weighted_uncertainty += uncertainty * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return total_weighted_uncertainty / total_weight
    
    def _validate_and_normalize_criteria(self, criteria_assessments: Dict[str, Any], 
                                       expected_weights: Dict[str, float],
                                       warnings: List[str],
                                       logprobs_data: Optional[List[Dict]] = None) -> Dict[str, Dict]:
        """
        Validate and normalize criteria assessments with uncertainty calculation.
        Enhanced version that calculates perplexity-based uncertainty for each criterion.
        
        Args:
            criteria_assessments: Raw criteria data from parsed response
            expected_weights: Expected weights for this repository category
            warnings: List to append validation warnings
            logprobs_data: Token logprobs for uncertainty calculation
            
        Returns:
            Validated criteria scores with uncertainty information
        """
        validated_criteria = {}
        
        for criterion in self.default_criteria:
            if criterion in criteria_assessments:
                assessment = criteria_assessments[criterion]
                
                # Validate score
                score = assessment.get("score", 5)
                if not isinstance(score, (int, float)) or not (1 <= score <= 10):
                    score = 5
                    warnings.append(f"Invalid score for {criterion}, using default")
                
                # Validate weight - use expected weight if available
                weight = expected_weights.get(criterion, self.default_weights.get(criterion, 0.143))
                assessment_weight = assessment.get("weight")
                if assessment_weight is not None and isinstance(assessment_weight, (int, float)) and assessment_weight >= 0:
                    weight = float(assessment_weight)
                
                # Get reasoning
                reasoning = assessment.get("reasoning", "No reasoning provided")
                
                # Calculate raw_uncertainty from perplexity
                raw_uncertainty = self._calculate_reasoning_perplexity(reasoning, logprobs_data)
                
                validated_criteria[criterion] = {
                    "score": int(score),
                    "weight": float(weight),
                    "reasoning": reasoning,
                    "raw_uncertainty": raw_uncertainty
                }
            else:
                # Missing criterion - use expected weight and default values
                weight = expected_weights.get(criterion, self.default_weights.get(criterion, 0.143))
                warnings.append(f"Missing assessment for {criterion}, using defaults")
                
                validated_criteria[criterion] = {
                    "score": 5,  # Neutral default
                    "weight": float(weight),
                    "reasoning": "No reasoning provided - missing from response",
                    "raw_uncertainty": 0.7  # High uncertainty for missing data
                }
        
        return validated_criteria
    
    def _calculate_final_score(self, criteria_scores: Dict[str, Dict]) -> float:
        """Calculate weighted final originality score from criteria assessments."""
        if not criteria_scores:
            return 0.5
        
        total_score = 0.0
        total_weight = 0.0
        
        for criterion, assessment in criteria_scores.items():
            score = assessment.get("score", 5)
            weight = assessment.get("weight", 0)
            total_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        # Normalize to [0.1, 0.9] range as per originality framework
        weighted_score = total_score / total_weight  # Range: 1-10
        normalized_score = (weighted_score - 1) / 9  # Range: 0-1
        final_score = 0.1 + (normalized_score * 0.8)  # Range: 0.1-0.9
        
        return max(0.1, min(0.9, final_score))
    
    def _create_default_criteria_scores(self, expected_weights: Dict[str, float]) -> Dict[str, Dict]:
        """Create default criteria scores for fallback cases."""
        default_scores = {}
        
        for criterion in self.default_criteria:
            weight = expected_weights.get(criterion, self.default_weights.get(criterion, 0.143))
            default_scores[criterion] = {
                "score": 5,  # Neutral default
                "weight": float(weight),
                "reasoning": "Default reasoning due to parsing failure",
                "raw_uncertainty": 0.8  # High uncertainty for defaults
            }
        
        return default_scores
    
    # Additional parsing methods (JSON, regex, fuzzy) remain the same as original implementation
    # ... (keeping existing _try_json_parsing, _try_regex_parsing, _try_fuzzy_parsing methods)
    
    def _try_json_parsing(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Try to extract JSON from response text."""
        # Look for JSON blocks
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_match = re.search(json_pattern, response_text, re.DOTALL)
        
        if json_match:
            json_text = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end + 1]
            else:
                return None
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return None
    
    def _try_regex_parsing(self, response_text: str) -> Dict[str, Any]:
        """Try to extract information using regex patterns."""
        parsed = {
            'criteria_scores': {},
            'overall_reasoning': '',
            'assessment_confidence': 0.5
        }
        
        # Extract overall reasoning (implementation depends on prompt format)
        reasoning_patterns = [
            r'overall[_\s]reasoning["\s]*:[\s]*["\']([^"\']+)["\']',
            r'comprehensive[_\s]explanation[^:]*:[\s]*([^\.]+)',
            r'assessment[_\s]summary[^:]*:[\s]*([^\.]+)'
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                parsed['overall_reasoning'] = match.group(1).strip()
                break
        
        # Extract criteria scores (basic implementation)
        for criterion in self.default_criteria:
            score_pattern = rf'{criterion}[^:]*:\s*(\d+)'
            score_match = re.search(score_pattern, response_text, re.IGNORECASE)
            if score_match:
                parsed['criteria_scores'][criterion] = {
                    'score': int(score_match.group(1)),
                    'reasoning': f"Extracted from response for {criterion}"
                }
        
        return parsed
    
    def _try_fuzzy_parsing(self, response_text: str) -> Dict[str, Any]:
        """Try to extract information using fuzzy text matching."""
        parsed = {
            'criteria_scores': {},
            'overall_reasoning': response_text[:500],  # First 500 chars as fallback
            'assessment_confidence': 0.3
        }
        
        # Very basic fuzzy extraction - look for numbers that might be scores
        lines = response_text.split('\n')
        for line in lines:
            for criterion in self.default_criteria:
                if criterion.replace('_', ' ') in line.lower():
                    # Look for numbers in this line
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        score = int(numbers[0])
                        if 1 <= score <= 10:
                            parsed['criteria_scores'][criterion] = {
                                'score': score,
                                'reasoning': line.strip()
                            }
        
        return parsed