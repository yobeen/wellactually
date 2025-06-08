# src/uncertainty_calibration/originality_assessment/originality_response_parser.py
"""
Originality Response Parser

Parses LLM responses for originality assessments with fuzzy matching and error recovery.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ParsedOriginalityResponse:
    """Structured representation of a parsed originality assessment response."""
    
    repository_url: str
    repository_name: str
    originality_category: str
    criteria_scores: Dict[str, Dict[str, Any]]  # criterion -> {score, weight, reasoning, uncertainty}
    overall_reasoning: str
    final_originality_score: float
    assessment_confidence: float
    parsing_method: str
    parsing_success: bool
    parsing_warnings: List[str]

class OriginalityResponseParser:
    """
    Parses LLM responses for originality assessments with robust error handling.
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
            'protocol_implementation': 0.125,
            'algorithmic_innovation': 0.125,
            'developer_experience': 0.125,
            'architectural_innovation': 0.125,
            'security_innovation': 0.125,
            'standards_leadership': 0.125,
            'cross_client_compatibility': 0.125,
        }
    
    def parse_response(self, response_text: str, expected_weights: Dict[str, float], 
                      repo_url: str, repo_name: str, originality_category: str) -> ParsedOriginalityResponse:
        """
        Parse LLM response for originality assessment with multiple fallback strategies.
        
        Args:
            response_text: Raw LLM response text
            expected_weights: Expected criterion weights for this repository
            repo_url: Repository URL
            repo_name: Repository name
            originality_category: Repository originality category (A-I)
            
        Returns:
            ParsedOriginalityResponse with extracted information
        """
        warnings = []
        
        # Try parsing methods in order of reliability
        try:
            # Method 1: JSON parsing
            parsed = self._try_json_parsing(response_text)
            if parsed:
                criteria_scores = self._validate_and_normalize_criteria(
                    parsed.get('criteria_scores', {}), expected_weights, warnings
                )
                
                return ParsedOriginalityResponse(
                    repository_url=repo_url,
                    repository_name=repo_name,
                    originality_category=originality_category,
                    criteria_scores=criteria_scores,
                    overall_reasoning=parsed.get('overall_reasoning', ''),
                    final_originality_score=self._calculate_final_score(criteria_scores),
                    assessment_confidence=float(parsed.get('assessment_confidence', 0.7)),
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
                    parsed.get('criteria_scores', {}), expected_weights, warnings
                )
                
                return ParsedOriginalityResponse(
                    repository_url=repo_url,
                    repository_name=repo_name,
                    originality_category=originality_category,
                    criteria_scores=criteria_scores,
                    overall_reasoning=parsed.get('overall_reasoning', ''),
                    final_originality_score=self._calculate_final_score(criteria_scores),
                    assessment_confidence=float(parsed.get('assessment_confidence', 0.5)),
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
                parsed.get('criteria_scores', {}), expected_weights, warnings
            )
            
            return ParsedOriginalityResponse(
                repository_url=repo_url,
                repository_name=repo_name,
                originality_category=originality_category,
                criteria_scores=criteria_scores,
                overall_reasoning=parsed.get('overall_reasoning', response_text[:500]),
                final_originality_score=self._calculate_final_score(criteria_scores),
                assessment_confidence=0.3,  # Low confidence for fuzzy parsing
                parsing_method='fuzzy',
                parsing_success=len(criteria_scores) >= len(self.default_criteria) // 2,
                parsing_warnings=warnings
            )
        except Exception as e:
            logger.error(f"All parsing methods failed: {e}")
            warnings.append(f"Fuzzy parsing failed: {e}")
        
        # Fallback: Return default scores
        warnings.append("Using default scores due to parsing failure")
        criteria_scores = self._create_default_criteria_scores(expected_weights)
        
        return ParsedOriginalityResponse(
            repository_url=repo_url,
            repository_name=repo_name,
            originality_category=originality_category,
            criteria_scores=criteria_scores,
            overall_reasoning="Failed to parse LLM response",
            final_originality_score=0.5,  # Neutral default
            assessment_confidence=0.1,  # Very low confidence
            parsing_method='fallback',
            parsing_success=False,
            parsing_warnings=warnings
        )
    
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
        
        # Extract overall reasoning
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
        
        # Extract criterion scores
        for criterion in self.default_criteria:
            # Look for score patterns
            score_patterns = [
                rf'{criterion}["\s]*:[\s]*\{{[^}}]*"score"[\s]*:[\s]*(\d+)',
                rf'{criterion}[^:]*score[^:]*:[\s]*(\d+)',
                rf'{criterion}[^:]*:[\s]*(\d+)'
            ]
            
            score = None
            for pattern in score_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    score = int(match.group(1))
                    break
            
            if score is not None and 1 <= score <= 10:
                # Look for reasoning
                reasoning_patterns = [
                    rf'{criterion}[^:]*reasoning["\s]*:[\s]*["\']([^"\']+)["\']',
                    rf'{criterion}[^:]*explanation[^:]*:[\s]*([^\.]+)',
                ]
                
                reasoning = "No reasoning provided"
                for pattern in reasoning_patterns:
                    match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        reasoning = match.group(1).strip()
                        break
                
                parsed['criteria_scores'][criterion] = {
                    'score': score,
                    'reasoning': reasoning,
                    'uncertainty': 0.5  # Default uncertainty
                }
        
        return parsed
    
    def _try_fuzzy_parsing(self, response_text: str) -> Dict[str, Any]:
        """Try to extract scores using fuzzy text matching."""
        parsed = {
            'criteria_scores': {},
            'overall_reasoning': response_text[:500],  # First 500 chars as reasoning
            'assessment_confidence': 0.3
        }
        
        # Look for any numbers that might be scores
        number_pattern = r'(\d+)'
        lines = response_text.split('\n')
        
        criterion_index = 0
        for line in lines:
            if criterion_index >= len(self.default_criteria):
                break
                
            # Check if line mentions a criterion
            current_criterion = None
            for criterion in self.default_criteria:
                if criterion.replace('_', ' ').lower() in line.lower():
                    current_criterion = criterion
                    break
            
            if current_criterion:
                # Look for a number in this line or next few lines
                numbers = re.findall(number_pattern, line)
                score = None
                
                if numbers:
                    for num in numbers:
                        num_val = int(num)
                        if 1 <= num_val <= 10:
                            score = num_val
                            break
                
                if score is not None:
                    parsed['criteria_scores'][current_criterion] = {
                        'score': score,
                        'reasoning': line.strip(),
                        'uncertainty': 0.7  # High uncertainty for fuzzy parsing
                    }
                    criterion_index += 1
        
        return parsed
    
    def _validate_and_normalize_criteria(self, criteria_scores: Dict[str, Any], 
                                       expected_weights: Dict[str, float],
                                       warnings: List[str]) -> Dict[str, Dict[str, Any]]:
        """Validate and normalize criteria scores."""
        validated_criteria = {}
        
        for criterion in self.default_criteria:
            if criterion in criteria_scores:
                assessment = criteria_scores[criterion]
                
                # Validate score
                score = assessment.get('score', 5)
                if not isinstance(score, (int, float)) or not (1 <= score <= 10):
                    score = 5
                    warnings.append(f"Invalid score for {criterion}, using default")
                
                # Get weight from expected weights
                weight = expected_weights.get(criterion, self.default_weights[criterion])
                
                # Get reasoning
                reasoning = assessment.get('reasoning', 'No reasoning provided')
                
                # Get uncertainty
                uncertainty = assessment.get('uncertainty', 0.5)
                if not isinstance(uncertainty, (int, float)) or not (0 <= uncertainty <= 1):
                    uncertainty = 0.5
                    warnings.append(f"Invalid uncertainty for {criterion}, using default")
                
                validated_criteria[criterion] = {
                    'score': float(score),
                    'weight': float(weight),
                    'reasoning': str(reasoning),
                    'uncertainty': float(uncertainty)
                }
            else:
                # Missing criterion - create default
                warnings.append(f"Missing assessment for {criterion}, using default")
                validated_criteria[criterion] = {
                    'score': 5.0,
                    'weight': float(expected_weights.get(criterion, self.default_weights[criterion])),
                    'reasoning': 'No assessment provided',
                    'uncertainty': 0.8  # High uncertainty for missing data
                }
        
        return validated_criteria
    
    def _calculate_final_score(self, criteria_scores: Dict[str, Dict[str, Any]]) -> float:
        """Calculate weighted average originality score."""
        if not criteria_scores:
            return 0.5
        
        total_score = 0.0
        total_weight = 0.0
        
        for criterion, assessment in criteria_scores.items():
            score = assessment.get('score', 5.0) / 10.0  # Convert to 0.1-1.0 scale
            weight = assessment.get('weight', 0.125)
            total_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        # Normalize to [0.1, 0.9] range as per framework
        final_score = total_score / total_weight
        return max(0.1, min(0.9, final_score))
    
    def _create_default_criteria_scores(self, expected_weights: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Create default criteria scores when parsing fails."""
        default_scores = {}
        
        for criterion in self.default_criteria:
            default_scores[criterion] = {
                'score': 5.0,  # Neutral score
                'weight': float(expected_weights.get(criterion, self.default_weights[criterion])),
                'reasoning': 'Default score due to parsing failure',
                'uncertainty': 0.9  # Very high uncertainty
            }
        
        return default_scores