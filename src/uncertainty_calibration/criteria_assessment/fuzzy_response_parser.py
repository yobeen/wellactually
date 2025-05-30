# src/uncertainty_calibration/criteria_assessment/fuzzy_response_parser.py
"""
Fuzzy response parser for criteria assessment.
Robust parsing of LLM responses that may not be perfect JSON.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ParsedCriteriaResponse:
    """Container for parsed criteria assessment response."""
    repository_url: str
    repository_name: str
    criteria_scores: Dict[str, Dict[str, Any]]  # criterion_id -> {score, weight, reasoning}
    target_score: Optional[float]
    total_weight: Optional[float]
    overall_reasoning: Optional[str]
    parsing_method: str
    parsing_success: bool
    parsing_warnings: List[str]

class FuzzyCriteriaResponseParser:
    """
    Robust parser for criteria assessment responses from LLMs.
    Handles imperfect JSON and various response formats.
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
    
    def parse_response(self, raw_response: str, repo_url: str, repo_name: str) -> ParsedCriteriaResponse:
        """
        Parse LLM response using multiple strategies.
        
        Args:
            raw_response: Raw text response from LLM
            repo_url: Repository URL for context
            repo_name: Repository name for context
            
        Returns:
            ParsedCriteriaResponse object
        """
        warnings = []
        
        # Strategy 1: Try perfect JSON parsing
        parsed_data, method = self._try_json_parsing(raw_response)
        if parsed_data:
            logger.debug(f"Successfully parsed response using {method}")
            return self._create_response_from_json(parsed_data, repo_url, repo_name, method, warnings)
        
        # Strategy 2: Try cleaned JSON parsing
        parsed_data, method = self._try_cleaned_json_parsing(raw_response)
        if parsed_data:
            logger.debug(f"Successfully parsed response using {method}")
            warnings.append("Required JSON cleaning for parsing")
            return self._create_response_from_json(parsed_data, repo_url, repo_name, method, warnings)
        
        # Strategy 3: Try regex extraction
        parsed_data, method = self._try_regex_extraction(raw_response)
        if parsed_data:
            logger.debug(f"Successfully parsed response using {method}")
            warnings.append("Used regex extraction instead of JSON parsing")
            return self._create_response_from_extracted_data(parsed_data, repo_url, repo_name, method, warnings)
        
        # Strategy 4: Try line-by-line parsing
        parsed_data, method = self._try_line_parsing(raw_response)
        if parsed_data:
            logger.debug(f"Successfully parsed response using {method}")
            warnings.append("Used line-by-line parsing as fallback")
            return self._create_response_from_extracted_data(parsed_data, repo_url, repo_name, method, warnings)
        
        # Strategy 5: Fallback with defaults
        logger.warning(f"All parsing strategies failed for {repo_url}, using defaults")
        warnings.append("All parsing strategies failed, using default values")
        return self._create_fallback_response(repo_url, repo_name, warnings)
    
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
                                  method: str, warnings: List[str]) -> ParsedCriteriaResponse:
        """Create response object from successfully parsed JSON."""
        
        # Extract criteria assessments
        criteria_assessments = parsed_data.get("criteria_assessments", {})
        
        # Extract summary info
        summary = parsed_data.get("assessment_summary", {})
        target_score = summary.get("target_score")
        total_weight = summary.get("total_weight")
        overall_reasoning = summary.get("overall_reasoning")
        
        # Validate and fill missing criteria
        criteria_scores = self._validate_and_fill_criteria(criteria_assessments, warnings)
        
        return ParsedCriteriaResponse(
            repository_url=repo_url,
            repository_name=repo_name,
            criteria_scores=criteria_scores,
            target_score=target_score,
            total_weight=total_weight,
            overall_reasoning=overall_reasoning,
            parsing_method=method,
            parsing_success=True,
            parsing_warnings=warnings
        )
    
    def _create_response_from_extracted_data(self, extracted_data: Dict, repo_url: str, repo_name: str,
                                           method: str, warnings: List[str]) -> ParsedCriteriaResponse:
        """Create response object from extracted data."""
        
        criteria_assessments = extracted_data.get("criteria_assessments", {})
        summary = extracted_data.get("assessment_summary", {})
        
        # Validate and fill missing criteria
        criteria_scores = self._validate_and_fill_criteria(criteria_assessments, warnings)
        
        # Calculate target score if not provided
        target_score = summary.get("target_score")
        if target_score is None:
            target_score = self._calculate_target_score(criteria_scores)
            warnings.append("Calculated target score from individual criteria")
        
        return ParsedCriteriaResponse(
            repository_url=repo_url,
            repository_name=repo_name,
            criteria_scores=criteria_scores,
            target_score=target_score,
            total_weight=sum(c.get("weight", 0) for c in criteria_scores.values()),
            overall_reasoning="Extracted from partial parsing",
            parsing_method=method,
            parsing_success=True,
            parsing_warnings=warnings
        )
    
    def _create_fallback_response(self, repo_url: str, repo_name: str, 
                                warnings: List[str]) -> ParsedCriteriaResponse:
        """Create fallback response with default values."""
        
        # Use default scores (middle value of 5)
        criteria_scores = {}
        for criterion in self.default_criteria:
            criteria_scores[criterion] = {
                "score": 5,  # Middle value
                "weight": self.default_weights[criterion],
                "reasoning": "Default value due to parsing failure"
            }
        
        target_score = self._calculate_target_score(criteria_scores)
        
        return ParsedCriteriaResponse(
            repository_url=repo_url,
            repository_name=repo_name,
            criteria_scores=criteria_scores,
            target_score=target_score,
            total_weight=1.0,
            overall_reasoning="Default assessment due to parsing failure",
            parsing_method="fallback",
            parsing_success=False,
            parsing_warnings=warnings
        )
    
    def _validate_and_fill_criteria(self, criteria_assessments: Dict, warnings: List[str]) -> Dict[str, Dict]:
        """Validate criteria assessments and fill missing ones with defaults."""
        
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
                
                validated_criteria[criterion] = {
                    "score": int(score),
                    "weight": float(weight),
                    "reasoning": assessment.get("reasoning", "No reasoning provided")
                }
            else:
                # Missing criterion, use defaults
                validated_criteria[criterion] = {
                    "score": 5,
                    "weight": self.default_weights[criterion],
                    "reasoning": "Default value for missing criterion"
                }
                warnings.append(f"Missing assessment for {criterion}, using defaults")
        
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