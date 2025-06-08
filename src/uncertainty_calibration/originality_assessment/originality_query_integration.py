# src/uncertainty_calibration/originality_assessment/originality_query_integration.py
"""
Originality Query Integration

Integrates originality assessment with the existing LLM query engine to provide
originality assessments with the same return structure as criteria assessment.
"""

import logging
from typing import Dict, Any, Optional

from .originality_prompt_generator import OriginalityPromptGenerator
from .originality_response_parser import OriginalityResponseParser

logger = logging.getLogger(__name__)

class OriginalityQueryEngine:
    """
    Integrates originality assessment with LLM query engine.
    
    Returns originality assessments in the same format as criteria assessments:
    {
        "category": str,
        "score": float [0.1, 0.9], 
        "reasoning": str,
        "uncertainty": float [0, 1]
    }
    """
    
    def __init__(self, llm_engine):
        """
        Initialize the originality query engine.
        
        Args:
            llm_engine: MultiModelEngine instance for LLM queries
        """
        self.llm_engine = llm_engine
        self.prompt_generator = OriginalityPromptGenerator()
        self.response_parser = OriginalityResponseParser()
    
    def query_originality(self, repo_url: str, model_id: str = "openai/gpt-4o", 
                         temperature: float = 0.0, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query originality assessment for a repository.
        
        Args:
            repo_url: Repository URL to assess
            model_id: LLM model to use for assessment
            temperature: Sampling temperature for LLM
            parameters: Optional parameters (not used but kept for compatibility)
            
        Returns:
            Dictionary with originality assessment:
            {
                "category": str,           # Originality category (A-I)
                "score": float [0.1,0.9],  # Final originality score
                "reasoning": str,          # Overall reasoning for the score
                "uncertainty": float [0,1] # Confidence in assessment (1-confidence)
            }
        """
        try:
            # Get repository configuration
            repo_config = self.prompt_generator.get_repo_originality_config(repo_url)
            
            # Generate prompt
            prompt_messages = self.prompt_generator.create_originality_assessment_prompt(repo_url)
            
            # Query LLM
            response = self.llm_engine.query_model(
                model_id=model_id,
                messages=prompt_messages,
                temperature=temperature,
                max_tokens=4000
            )
            
            # Parse response
            parsed_response = self.response_parser.parse_response(
                response_text=response.content,
                expected_weights=repo_config['weights'],
                repo_url=repo_url,
                repo_name=repo_config['name'],
                originality_category=repo_config['category']
            )
            
            # Convert to standard format
            result = {
                "category": parsed_response.originality_category,
                "score": parsed_response.final_originality_score,
                "reasoning": parsed_response.overall_reasoning,
                "uncertainty": 1.0 - parsed_response.assessment_confidence  # Convert confidence to uncertainty
            }
            
            logger.debug(f"Originality query successful: {repo_url} -> {result['score']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Originality query failed for {repo_url}: {e}")
            
            # Return fallback result
            try:
                repo_config = self.prompt_generator.get_repo_originality_config(repo_url)
                category = repo_config.get('category', 'H')
            except:
                category = 'H'  # Default to Specialized Tools
            
            return {
                "category": category,
                "score": 0.5,  # Neutral score
                "reasoning": f"Assessment failed: {str(e)}",
                "uncertainty": 0.9  # High uncertainty due to failure
            }
    
    def get_repository_category(self, repo_url: str) -> str:
        """
        Get the originality category for a repository.
        
        Args:
            repo_url: Repository URL
            
        Returns:
            Originality category (A-I)
        """
        try:
            repo_config = self.prompt_generator.get_repo_originality_config(repo_url)
            return repo_config.get('category', 'H')
        except Exception:
            return 'H'  # Default category
    
    def get_category_weights(self, category: str) -> Dict[str, float]:
        """
        Get the weights for a specific originality category.
        
        Args:
            category: Originality category (A-I)
            
        Returns:
            Dictionary of criterion weights
        """
        try:
            framework_config = self.prompt_generator.framework_config
            category_config = framework_config['categories'].get(category, {})
            return category_config.get('weights', {})
        except Exception:
            # Return default weights
            return {
                'protocol_implementation': 0.125,
                'algorithmic_innovation': 0.125,
                'developer_experience': 0.125,
                'architectural_innovation': 0.125,
                'security_innovation': 0.125,
                'standards_leadership': 0.125,
                'cross_client_compatibility': 0.125,
                'domain_problem_solving': 0.125
            }


# Integration with existing query engine
def add_originality_query_method(query_engine_class):
    """
    Decorator to add originality query capability to existing query engine.
    
    This allows the existing query engine to handle originality assessments
    alongside comparison queries and other assessment types.
    """
    
    def query_originality(self, repo: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query originality assessment for a repository.
        
        Args:
            repo: Repository URL to assess
            parameters: Optional parameters
            
        Returns:
            Dictionary with originality assessment:
            {
                "category": str,
                "score": float [0.1,0.9], 
                "reasoning": str,
                "uncertainty": float [0,1]
            }
        """
        # Initialize originality engine if not exists
        if not hasattr(self, '_originality_engine'):
            self._originality_engine = OriginalityQueryEngine(self)
        
        return self._originality_engine.query_originality(repo, parameters=parameters)
    
    # Add method to class
    query_engine_class.query_originality = query_originality
    return query_engine_class


# Example usage with existing query engine
class EnhancedQueryEngine:
    """
    Example of how to integrate originality assessment with existing query engine.
    """
    
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine
        self.originality_engine = OriginalityQueryEngine(llm_engine)
    
    def query_comparison(self, repo_a: str, repo_b: str, parent: str) -> Dict[str, Any]:
        """Existing comparison query method."""
        # Implementation would go here
        pass
    
    def query_originality(self, repo: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        New originality query method with same interface as other query methods.
        
        Args:
            repo: Repository URL to assess
            parameters: Optional parameters
            
        Returns:
            Dictionary with originality assessment following standard format
        """
        return self.originality_engine.query_originality(repo, parameters=parameters)
    
    def generate_level2_assessments_from_repos(self, repos: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Generate Level 2 originality assessments from selected repositories.
        
        Args:
            repos: List of (repo, 'originality') tuples
            
        Returns:
            List of originality assessment dictionaries compatible with existing pipeline
        """
        logger.info(f"Generating {len(repos)} Level 2 originality assessments")
        
        assessments = []
        for repo, parent in repos:
            try:
                result = self.query_originality(repo)
                
                # Convert to pipeline format
                assessments.append({
                    'repo': repo,
                    'parent': parent,
                    'category': result['category'],
                    'weight': result['score'],  # Use score as weight for compatibility
                    'uncertainty': result['uncertainty'],
                    'explanation': result['reasoning'],
                    'data_source': 'synthetic'
                })
            except Exception as e:
                logger.warning(f"Failed to get originality assessment for {repo}: {e}")
                continue
        
        logger.info(f"Successfully generated {len(assessments)} Level 2 originality assessments")
        return assessments