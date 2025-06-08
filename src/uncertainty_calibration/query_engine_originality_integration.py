# src/uncertainty_calibration/query_engine_originality_integration.py
"""
Query Engine Originality Integration

Simple integration pattern for adding originality assessment to existing query engines.
This shows how to extend the existing query_engine.py with originality capabilities.
"""

from typing import Dict, Any, List, Tuple, Optional
from .originality_assessment import OriginalityQueryEngine

def add_originality_to_query_engine(query_engine_class):
    """
    Decorator to add originality assessment capability to existing query engine.
    
    Usage:
        @add_originality_to_query_engine
        class MyQueryEngine:
            # existing methods...
    """
    
    def __init_originality__(self):
        """Initialize originality engine lazily."""
        if not hasattr(self, '_originality_engine'):
            self._originality_engine = OriginalityQueryEngine(self)
    
    def query_originality(self, repo: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query originality assessment for a repository.
        
        Args:
            repo: Repository URL to assess
            parameters: Optional parameters
            
        Returns:
            Dictionary with originality assessment:
            {
                "category": str,           # Originality category (A-I)
                "score": float [0.1,0.9],  # Originality score
                "reasoning": str,          # Explanation
                "uncertainty": float [0,1] # Assessment uncertainty
            }
        """
        self.__init_originality__()
        return self._originality_engine.query_originality(repo, parameters=parameters)
    
    def generate_level2_assessments_from_repos(self, repos: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Generate Level 2 originality assessments from selected repositories.
        
        Args:
            repos: List of (repo, 'originality') tuples
            
        Returns:
            List of originality assessment dictionaries compatible with existing pipeline
        """
        self.__init_originality__()
        
        assessments = []
        for repo, parent in repos:
            try:
                result = self.query_originality(repo)
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
                # Log warning and continue
                continue
        
        return assessments
    
    def get_repository_originality_category(self, repo_url: str) -> str:
        """Get the originality category for a repository."""
        self.__init_originality__()
        return self._originality_engine.get_repository_category(repo_url)
    
    # Add methods to the class
    query_engine_class.__init_originality__ = __init_originality__
    query_engine_class.query_originality = query_originality
    query_engine_class.generate_level2_assessments_from_repos = generate_level2_assessments_from_repos
    query_engine_class.get_repository_originality_category = get_repository_originality_category
    
    return query_engine_class