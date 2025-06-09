# src/uncertainty_calibration/criteria_assessment/fuzzy_response_parser.py
"""
Enhanced fuzzy response parser for criteria assessment.
Robust parsing of LLM responses with perplexity-based uncertainty calculation.
"""

import json
import re
import logging
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ParsedCriteriaResponse:
    """Container for parsed criteria assessment response with enhanced fields."""
    repository_url: str
    repository_name: str
    criteria_scores: Dict[str, Dict[str, Any]]  # criterion_id -> {score, weight, reasoning, raw_uncertainty}
    target_score: Optional[float]
    raw_target_score: Optional[float]  # NEW: from assessment_summary
    total_weight: Optional[float]
    overall_reasoning: Optional[str]
    parsing_method: str
    parsing_success: bool
    parsing_warnings: List[str]

class FuzzyCriteriaResponseParser:
    """
    Enhanced parser for criteria assessment responses from LLMs.
    Handles imperfect JSON and calculates perplexity-based uncertainties.
    """
    
    def __init__(self):
        """Initialize the fuzzy response parser."""
        # Default criteria IDs (in order)
        self.default_criteria = [
            "core_protocol", "market_adoption", "developer_ecosystem",
            "general_purpose_tools", "security_infrastructure", "defi_infrastructure",
            "data_analytics", "innovation_research", "ecosystem_coordination",
            "community_trust", "user_applications"
        ]
        
        # Default weights
        self.default_weights = {
            "core_protocol": 0.25,
            "market_adoption": 0.20,
            "developer_ecosystem": 0.15,
            "general_purpose_tools": 0.10,
            "security_infrastructure": 0.10,
            "defi_infrastructure": 0.05,
            "data_analytics": 0.05,
            "innovation_research": 0.03,
            "ecosystem_coordination": 0.03,
            "community_trust": 0.02,
            "user_applications": 0.02
        }
    
    def parse_response(self, raw_response: str, repo_url: str, repo_name: str, 
                      logprobs_data: Optional[List[Dict]] = None) -> ParsedCriteriaResponse:
        """
        Parse LLM response using multiple strategies with enhanced uncertainty calculation.
        
        Args:
            raw_response: Raw text response from LLM
            repo_url: Repository URL for context
            repo_name: Repository name for context
            logprobs_data: Token logprobs data from MultiModelEngine
            
        Returns:
            ParsedCriteriaResponse object with enhanced fields
        """
        warnings = []
        
        # Strategy 1: Try perfect JSON parsing
        parsed_data, method = self._try_json_parsing(raw_response)
        if parsed_data:
            logger.debug(f"Successfully parsed response using {method}")
            return self._create_response_from_json(
                parsed_data, repo_url, repo_name, method, warnings, logprobs_data
            )
        
        # Strategy 2: Try cleaned JSON parsing
        parsed_data, method = self._try_cleaned_json_parsing(raw_response)
        if parsed_data:
            logger.debug(f"Successfully parsed response using {method}")
            warnings.append("Required JSON cleaning for parsing")
            return self._create_response_from_json(
                parsed_data, repo_url, repo_name, method, warnings, logprobs_data
            )
        
        # Strategy 3: Try regex extraction
        parsed_data, method = self._try_regex_extraction(raw_response)
        if parsed_data:
            logger.debug(f"Successfully parsed response using {method}")
            warnings.append("Used regex extraction instead of JSON parsing")
            return self._create_response_from_extracted_data(
                parsed_data, repo_url, repo_name, method, warnings, logprobs_data
            )
        
        # Strategy 4: Try line-by-line parsing
        parsed_data, method = self._try_line_parsing(raw_response)
        if parsed_data:
            logger.debug(f"Successfully parsed response using {method}")
            warnings.append("Used line-by-line parsing as fallback")
            return self._create_response_from_extracted_data(
                parsed_data, repo_url, repo_name, method, warnings, logprobs_data
            )
        
        # All parsing failed - raise error instead of fallback
        error_msg = f"All parsing strategies failed for {repo_url}"
        logger.error(error_msg)
        raise ValueError(f"Failed to parse criteria response: {error_msg}")
    
    def _calculate_reasoning_perplexity(self, reasoning_text: str, 
                                      logprobs_data: Optional[List[Dict]]) -> float:
        """
        Calculate perplexity-based uncertainty for reasoning text.
        
        Args:
            reasoning_text: The reasoning text for a criterion
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
            # Lower perplexity = higher confidence = lower uncertainty
            uncertainty = self._normalize_perplexity_to_uncertainty(perplexity)
            
            return max(0.0, min(1.0, uncertainty))
            
        except Exception as e:
            logger.warning(f"Error calculating perplexity for reasoning: {e}")
            return 0.5
    
    def _extract_tokens_for_reasoning(self, reasoning_text: str, 
                                    logprobs_data: List[Dict]) -> List[Dict]:
        """
        Extract logprob tokens that correspond to the reasoning text.
        
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
    
    def _extract_target_score_from_parsed_data(self, parsed_data: Dict) -> Optional[float]:
        """Extract target score from assessment_summary section."""
        try:
            summary = parsed_data.get("assessment_summary", {})
            target_score = summary.get("target_score")
            
            if target_score is not None:
                return float(target_score)
            
            return None
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Error extracting target score: {e}")
            return None
    
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
        """Try to extract criteria data using regex patterns."""
        try:
            extracted = {}
            
            # Pattern for criterion entries
            # Look for patterns like: "core_protocol": { "score": 8, "weight": 0.25, "reasoning": "..." }
            criterion_pattern = r'"?(\w+)"?\s*:\s*\{\s*"?score"?\s*:\s*(\d+),?\s*"?weight"?\s*:\s*([\d.]+),?\s*"?reasoning"?\s*:\s*"([^"]*)"?\s*\}'
            
            matches = re.findall(criterion_pattern, response, re.IGNORECASE | re.DOTALL)
            
            if matches:
                criteria_assessments = {}
                for criterion, score, weight, reasoning in matches:
                    if criterion in self.default_criteria:
                        criteria_assessments[criterion] = {
                            "score": int(score),
                            "weight": float(weight),
                            "reasoning": reasoning.strip()
                        }
                
                if criteria_assessments:
                    extracted["criteria_assessments"] = criteria_assessments
                    
                    # Try to extract target score
                    target_match = re.search(r'"?target_score"?\s*:\s*([\d.]+)', response)
                    if target_match:
                        extracted["assessment_summary"] = {"target_score": float(target_match.group(1))}
                    
                    return extracted, "regex_extraction"
            
            # Simpler pattern: just look for scores
            simple_pattern = r'(\w+).*?score.*?(\d+)'
            simple_matches = re.findall(simple_pattern, response, re.IGNORECASE)
            
            if simple_matches:
                criteria_assessments = {}
                for criterion, score in simple_matches:
                    if criterion in self.default_criteria:
                        criteria_assessments[criterion] = {
                            "score": int(score),
                            "weight": self.default_weights.get(criterion, 0.05),
                            "reasoning": "Extracted from simple pattern"
                        }
                
                if criteria_assessments:
                    extracted["criteria_assessments"] = criteria_assessments
                    return extracted, "simple_regex"
            
        except Exception as e:
            logger.debug(f"Regex extraction error: {e}")
        
        return None, ""
    
    def _try_line_parsing(self, response: str) -> Tuple[Optional[Dict], str]:
        """Try line-by-line parsing for structured text."""
        try:
            lines = response.split('\n')
            criteria_assessments = {}
            
            current_criterion = None
            current_score = None
            current_weight = None
            current_reasoning = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for criterion names
                for criterion in self.default_criteria:
                    if criterion.replace('_', ' ').lower() in line.lower():
                        current_criterion = criterion
                        break
                
                # Look for scores
                score_match = re.search(r'score[:\s]*(\d+)', line, re.IGNORECASE)
                if score_match and current_criterion:
                    current_score = int(score_match.group(1))
                
                # Look for weights
                weight_match = re.search(r'weight[:\s]*([\d.]+)', line, re.IGNORECASE)
                if weight_match and current_criterion:
                    current_weight = float(weight_match.group(1))
                
                # Collect reasoning
                if current_criterion and any(word in line.lower() for word in ['reason', 'because', 'explanation']):
                    current_reasoning = line
                
                # If we have enough info, store it
                if current_criterion and current_score is not None:
                    criteria_assessments[current_criterion] = {
                        "score": current_score,
                        "weight": current_weight or self.default_weights.get(current_criterion, 0.05),
                        "reasoning": current_reasoning or "No reasoning provided"
                    }
                    current_criterion = None
                    current_score = None
                    current_weight = None
                    current_reasoning = ""
            
            if criteria_assessments:
                return {"criteria_assessments": criteria_assessments}, "line_parsing"
            
        except Exception as e:
            logger.debug(f"Line parsing error: {e}")
        
        return None, ""
    
    def _create_response_from_json(self, parsed_data: Dict, repo_url: str, repo_name: str, 
                                  method: str, warnings: List[str],
                                  logprobs_data: Optional[List[Dict]]) -> ParsedCriteriaResponse:
        """Create response object from successfully parsed JSON with enhanced fields."""
        
        # Extract criteria assessments
        criteria_assessments = parsed_data.get("criteria_assessments", {})
        
        # Extract summary info
        summary = parsed_data.get("assessment_summary", {})
        target_score = summary.get("target_score")
        raw_target_score = target_score  # Store raw target score
        total_weight = summary.get("total_weight")
        overall_reasoning = summary.get("overall_reasoning")
        
        # Validate and fill missing criteria with enhanced uncertainty calculation
        criteria_scores = self._validate_and_fill_criteria_enhanced(
            criteria_assessments, warnings, logprobs_data
        )
        
        return ParsedCriteriaResponse(
            repository_url=repo_url,
            repository_name=repo_name,
            criteria_scores=criteria_scores,
            target_score=target_score,
            raw_target_score=raw_target_score,  # NEW
            total_weight=total_weight,
            overall_reasoning=overall_reasoning,
            parsing_method=method,
            parsing_success=True,
            parsing_warnings=warnings
        )
    
    def _create_response_from_extracted_data(self, extracted_data: Dict, repo_url: str, repo_name: str,
                                           method: str, warnings: List[str],
                                           logprobs_data: Optional[List[Dict]]) -> ParsedCriteriaResponse:
        """Create response object from extracted data with enhanced fields."""
        
        criteria_assessments = extracted_data.get("criteria_assessments", {})
        summary = extracted_data.get("assessment_summary", {})
        
        # Validate and fill missing criteria with enhanced uncertainty
        criteria_scores = self._validate_and_fill_criteria_enhanced(
            criteria_assessments, warnings, logprobs_data
        )
        
        # Calculate target score if not provided
        target_score = summary.get("target_score")
        raw_target_score = target_score
        if target_score is None:
            target_score = self._calculate_target_score(criteria_scores)
            warnings.append("Calculated target score from individual criteria")
        
        return ParsedCriteriaResponse(
            repository_url=repo_url,
            repository_name=repo_name,
            criteria_scores=criteria_scores,
            target_score=target_score,
            raw_target_score=raw_target_score,  # NEW
            total_weight=sum(c.get("weight", 0) for c in criteria_scores.values()),
            overall_reasoning="Extracted from partial parsing",
            parsing_method=method,
            parsing_success=True,
            parsing_warnings=warnings
        )
    
    def _validate_and_fill_criteria_enhanced(self, criteria_assessments: Dict, warnings: List[str],
                                           logprobs_data: Optional[List[Dict]]) -> Dict[str, Dict]:
        """Validate criteria assessments and fill with enhanced uncertainty calculation."""
        
        validated_criteria = {}
        
        for criterion in self.default_criteria:
            if criterion in criteria_assessments:
                assessment = criteria_assessments[criterion]
                
                # Validate score
                score = assessment.get("score", 5)
                if not isinstance(score, (int, float)) or not (1 <= score <= 10):
                    score = 5
                    warnings.append(f"Invalid score for {criterion}, using default")
                
                # Validate weight
                weight = assessment.get("weight", self.default_weights[criterion])
                if not isinstance(weight, (int, float)) or weight < 0:
                    weight = self.default_weights[criterion]
                    warnings.append(f"Invalid weight for {criterion}, using default")
                
                # Get reasoning
                reasoning = assessment.get("reasoning", "No reasoning provided")
                
                # NEW: Calculate raw_uncertainty from perplexity
                raw_uncertainty = self._calculate_reasoning_perplexity(reasoning, logprobs_data)
                
                validated_criteria[criterion] = {
                    "score": int(score),
                    "weight": float(weight),
                    "reasoning": reasoning,
                    "raw_uncertainty": raw_uncertainty  # NEW
                }
            else:
                # Missing criterion - raise error instead of using defaults
                raise ValueError(f"Missing assessment for required criterion: {criterion}")
        
        return validated_criteria
    
    def _calculate_target_score(self, criteria_scores: Dict[str, Dict]) -> float:
        """Calculate weighted target score from criteria assessments."""
        total_score = 0.0
        total_weight = 0.0
        
        for criterion, assessment in criteria_scores.items():
            score = assessment.get("score", 5)
            weight = assessment.get("weight", 0)
            total_score += score * weight
            total_weight += weight
        
        return total_score