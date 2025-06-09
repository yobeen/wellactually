# src/shared/model_answer_postprocessor.py
#!/usr/bin/env python3
"""
Model-specific answer postprocessing for handling tokenization artifacts.
Normalizes model responses to standard answer formats with family-specific rules.
"""

import logging
from typing import Dict, Any, Optional, List
import re

logger = logging.getLogger(__name__)

class ModelAnswerPostprocessor:
    """
    Handles model family-specific tokenization artifacts and answer normalization.
    """
    
    def __init__(self):
        """Initialize postprocessor with model family rules."""
        
        # Model family classification patterns
        self.family_patterns = {
            "llama": ["meta-llama", "llama", "alpaca"],
            "openai": ["openai", "gpt"],
            "google": ["google", "gemma", "palm"],
            "deepseek": ["deepseek"],
            "mistral": ["mistralai", "mixtral"],
            "xai": ["x-ai", "grok"],
        }
        
        # Valid answer sets by format
        self.valid_answers = {
            "choice": ["A", "B", "Equal"],
            "numeric": [str(i) for i in range(1, 11)],  # 1-10
            "auto": ["A", "B", "Equal"] + [str(i) for i in range(1, 11)]
        }
        
        logger.info("ModelAnswerPostprocessor initialized")
    
    def detect_model_family(self, model_id: str) -> str:
        """
        Detect model family from model identifier.
        
        Args:
            model_id: Model identifier (e.g., "meta-llama/llama-4-maverick")
            
        Returns:
            Model family name or "generic" if unknown
        """
        if not model_id:
            return "generic"
        
        model_lower = model_id.lower()
        
        for family, patterns in self.family_patterns.items():
            if any(pattern in model_lower for pattern in patterns):
                return family
        
        return "generic"
    
    def preprocess_token_by_family(self, raw_token: str, model_family: str) -> str:
        """
        Apply family-specific preprocessing to remove tokenization artifacts.
        
        Args:
            raw_token: Raw token text from API response
            model_family: Detected model family
            
        Returns:
            Preprocessed token text
        """
        if not raw_token:
            return raw_token
        
        try:
            # Universal space prefix removal for ALL families
            # Handle both unicode and different encodings of space characters
            cleaned = raw_token
            
            # Try different representations of space prefixes
            space_prefixes = ["Ġ", "\u0120", "▁", " "]
            for prefix in space_prefixes:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]
                    logger.debug(f"{model_family} preprocessing: '{raw_token}' -> '{cleaned}' (removed prefix: '{prefix}')")
                    break
            
            # Additional family-specific processing if needed
            if model_family == "google":
                # SentencePiece might have additional ▁ characters beyond the prefix
                cleaned = cleaned.replace("▁", "")
                if cleaned != raw_token:
                    logger.debug(f"Google additional cleanup: '{raw_token}' -> '{cleaned}'")
            
            return cleaned
                
        except Exception as e:
            logger.warning(f"Preprocessing failed for {model_family}: {e}")
            return raw_token  # Fallback to original
    
    def apply_family_rules(self, cleaned_token: str, model_family: str, 
                          expected_format: str = "choice") -> str:
        """
        Apply model family-specific answer normalization rules.
        
        Args:
            cleaned_token: Preprocessed token text
            model_family: Model family name
            expected_format: Expected answer format ("choice", "numeric", "auto")
            
        Returns:
            Normalized answer string
        """
        if not cleaned_token:
            return cleaned_token
        
        if model_family == "llama":
            return self._apply_llama_rules(cleaned_token, expected_format)
        elif model_family == "openai":
            return self._apply_openai_rules(cleaned_token, expected_format)
        elif model_family == "google":
            return self._apply_google_rules(cleaned_token, expected_format)
        else:
            return self._apply_generic_rules(cleaned_token, expected_format)
    
    def _apply_llama_rules(self, token: str, expected_format: str) -> str:
        """Apply Llama family-specific normalization rules."""
        token_lower = token.lower()
        
        # Handle Equal variations using user's insight
        # Key rule: if first 3 letters = "equ", interpret as "Equal"
        if len(token_lower) >= 3 and token_lower[:3] == "equ":
            return "Equal"
        
        # Standard A/B matching
        if token_lower == "a":
            return "A"
        elif token_lower == "b":
            return "B"
        
        # Handle case variations of Equal
        if token_lower in ["equal", "eq", "eql"]:
            return "Equal"
        
        # Numeric answers (1-10)
        if expected_format in ["numeric", "auto"] and token_lower in [str(i) for i in range(1, 11)]:
            return token_lower
        
        # Return original token if no specific rule applies
        return token
    
    def _apply_openai_rules(self, token: str, expected_format: str) -> str:
        """Apply OpenAI family-specific normalization rules."""
        token_lower = token.lower()
        
        # OpenAI typically has cleaner tokenization
        if token_lower == "a":
            return "A"
        elif token_lower == "b":
            return "B"
        elif token_lower in ["equal", "eq"]:
            return "Equal"
        
        # Numeric answers
        if expected_format in ["numeric", "auto"] and token_lower in [str(i) for i in range(1, 11)]:
            return token_lower
        
        return token
    
    def _apply_google_rules(self, token: str, expected_format: str) -> str:
        """Apply Google family-specific normalization rules."""
        token_lower = token.lower()
        
        # Similar to OpenAI but may have different patterns
        if token_lower == "a":
            return "A"
        elif token_lower == "b":
            return "B"
        elif token_lower in ["equal", "eq"]:
            return "Equal"
        
        # Numeric answers
        if expected_format in ["numeric", "auto"] and token_lower in [str(i) for i in range(1, 11)]:
            return token_lower
        
        return token
    
    def _apply_generic_rules(self, token: str, expected_format: str) -> str:
        """Apply generic normalization rules for unknown model families."""
        token_lower = token.lower()
        
        # Basic normalization
        if token_lower == "a":
            return "A"
        elif token_lower == "b":
            return "B"
        elif token_lower in ["equal", "eq", "equ", "eql"]:
            return "Equal"
        
        # Numeric answers
        if expected_format in ["numeric", "auto"] and token_lower in [str(i) for i in range(1, 11)]:
            return token_lower
        
        return token
    
    def normalize_answer_token(self, raw_token_obj: Dict[str, Any], 
                             model_id: str, expected_format: str = "choice") -> Dict[str, Any]:
        """
        Main normalization function that processes a token object.
        
        Args:
            raw_token_obj: Token object from API response with 'token' and 'logprob' fields
            model_id: Model identifier for family detection
            expected_format: Expected answer format
            
        Returns:
            Normalized token object with cleaned answer but original logprobs
        """
        if not isinstance(raw_token_obj, dict) or 'token' not in raw_token_obj:
            logger.warning(f"Invalid token object: {raw_token_obj}")
            return raw_token_obj
        
        try:
            # Step 1: Detect model family
            model_family = self.detect_model_family(model_id)
            
            # Step 2: Extract raw token text
            raw_token_text = raw_token_obj.get('token', '')
            
            # Step 3: Apply family-specific preprocessing (e.g., Ġ stripping)
            preprocessed_token = self.preprocess_token_by_family(raw_token_text, model_family)
            
            # Step 4: Apply normalization rules
            normalized_answer = self.apply_family_rules(preprocessed_token, model_family, expected_format)
            
            # Step 5: Create normalized token object (keep original logprobs)
            normalized_token_obj = raw_token_obj.copy()
            normalized_token_obj['token'] = normalized_answer
            normalized_token_obj['original_token'] = raw_token_text  # Keep for debugging
            normalized_token_obj['model_family'] = model_family
            
            logger.debug(f"Normalized '{raw_token_text}' → '{normalized_answer}' for {model_family}")
            
            return normalized_token_obj
            
        except Exception as e:
            logger.error(f"Error normalizing token for {model_id}: {e}")
            return raw_token_obj  # Return original on error
    
    def is_valid_answer(self, answer: str, expected_format: str = "auto") -> bool:
        """
        Check if normalized answer is valid for the expected format.
        
        Args:
            answer: Normalized answer string
            expected_format: Expected answer format
            
        Returns:
            True if answer is valid, False otherwise
        """
        valid_set = self.valid_answers.get(expected_format, self.valid_answers["auto"])
        return answer in valid_set
    
    def get_preprocessing_stats(self, token_objects: List[Dict], model_id: str) -> Dict[str, Any]:
        """
        Get statistics about preprocessing for debugging and validation.
        
        Args:
            token_objects: List of token objects to analyze
            model_id: Model identifier
            
        Returns:
            Dictionary with preprocessing statistics
        """
        model_family = self.detect_model_family(model_id)
        stats = {
            "model_family": model_family,
            "total_tokens": len(token_objects),
            "preprocessing_changes": 0,
            "normalization_changes": 0,
            "valid_answers_found": 0,
            "examples": []
        }
        
        for token_obj in token_objects:
            if not isinstance(token_obj, dict) or 'token' not in token_obj:
                continue
            
            raw_token = token_obj.get('token', '')
            preprocessed = self.preprocess_token_by_family(raw_token, model_family)
            normalized = self.apply_family_rules(preprocessed, model_family, "auto")
            
            if raw_token != preprocessed:
                stats["preprocessing_changes"] += 1
            
            if preprocessed != normalized:
                stats["normalization_changes"] += 1
            
            if self.is_valid_answer(normalized, "auto"):
                stats["valid_answers_found"] += 1
            
            # Collect examples for debugging
            if len(stats["examples"]) < 10 and (raw_token != normalized):
                stats["examples"].append({
                    "raw": raw_token,
                    "preprocessed": preprocessed,
                    "normalized": normalized,
                    "valid": self.is_valid_answer(normalized, "auto")
                })
        
        return stats


# Convenience function for standalone usage
def normalize_answer_token(raw_token_obj: Dict[str, Any], model_id: str, 
                          expected_format: str = "choice") -> Dict[str, Any]:
    """
    Convenience function to normalize a single answer token.
    
    Args:
        raw_token_obj: Token object from API response
        model_id: Model identifier
        expected_format: Expected answer format
        
    Returns:
        Normalized token object
    """
    postprocessor = ModelAnswerPostprocessor()
    return postprocessor.normalize_answer_token(raw_token_obj, model_id, expected_format)