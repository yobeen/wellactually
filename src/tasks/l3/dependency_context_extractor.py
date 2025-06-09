# src/uncertainty_calibration/dependency_context_extractor.py
"""
Dependency context extractor for Level 3 comparisons.
Extracts repository context from CSV files for dependency assessments.
"""

import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DependencyContextExtractor:
    """
    Extracts and manages dependency context from CSV files for Level 3 comparisons.
    """
    
    def __init__(self, parent_csv_path: str = "data/external/parent_repos.csv", 
                 dependencies_csv_path: str = "data/external/dependencies.csv"):
        """
        Initialize the dependency context extractor.
        
        Args:
            parent_csv_path: Path to parent repositories CSV
            dependencies_csv_path: Path to dependencies CSV
        """
        self.parent_csv_path = parent_csv_path
        self.dependencies_csv_path = dependencies_csv_path
        
        # Cache for loaded data
        self._parent_df = None
        self._dependencies_df = None
        
        logger.info("DependencyContextExtractor initialized")
    
    def extract_comparison_context(self, parent_url: str, dep_a_url: str, 
                                 dep_b_url: str) -> Dict[str, Any]:
        """
        Extract context for a dependency comparison.
        
        Args:
            parent_url: URL of the parent repository
            dep_a_url: URL of first dependency
            dep_b_url: URL of second dependency
            
        Returns:
            Dictionary with parent and dependency contexts
        """
        try:
            # Load CSV data if not cached
            self._ensure_data_loaded()
            
            # Extract parent context
            parent_context = self._extract_parent_context(parent_url)
            
            # Extract dependency contexts
            dep_a_context = self._extract_dependency_context(dep_a_url)
            dep_b_context = self._extract_dependency_context(dep_b_url)
            
            # Validate that dependencies are related to parent
            self._validate_dependency_relationship(parent_context, dep_a_context, dep_b_context)
            
            return {
                "parent": parent_context,
                "dependency_a": dep_a_context,
                "dependency_b": dep_b_context,
                "comparison_metadata": {
                    "parent_url": parent_url,
                    "dep_a_url": dep_a_url,
                    "dep_b_url": dep_b_url
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting comparison context: {e}")
            raise
    
    def _ensure_data_loaded(self):
        """Ensure CSV data is loaded and cached."""
        if self._parent_df is None:
            self._load_parent_data()
        
        if self._dependencies_df is None:
            self._load_dependencies_data()
    
    def _load_parent_data(self):
        """Load parent repositories CSV data."""
        try:
            if not Path(self.parent_csv_path).exists():
                raise FileNotFoundError(f"Parent CSV not found: {self.parent_csv_path}")
            
            self._parent_df = pd.read_csv(self.parent_csv_path)
            
            # Validate required columns
            required_columns = [
                'parent_url', 'name', 'description', 'primary_language', 
                'domain', 'architecture_type', 'key_functions'
            ]
            
            missing_columns = [col for col in required_columns if col not in self._parent_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in parent CSV: {missing_columns}")
            
            logger.info(f"Loaded {len(self._parent_df)} parent repositories")
            
        except Exception as e:
            logger.error(f"Error loading parent CSV: {e}")
            raise
    
    def _load_dependencies_data(self):
        """Load dependencies CSV data."""
        try:
            if not Path(self.dependencies_csv_path).exists():
                raise FileNotFoundError(f"Dependencies CSV not found: {self.dependencies_csv_path}")
            
            self._dependencies_df = pd.read_csv(self.dependencies_csv_path)
            
            # Validate required columns
            required_columns = [
                'dependency_url', 'name', 'description', 'category', 
                'primary_function', 'integration_patterns'
            ]
            
            missing_columns = [col for col in required_columns if col not in self._dependencies_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in dependencies CSV: {missing_columns}")
            
            logger.info(f"Loaded {len(self._dependencies_df)} dependencies")
            
        except Exception as e:
            logger.error(f"Error loading dependencies CSV: {e}")
            raise
    
    def _extract_parent_context(self, parent_url: str) -> Dict[str, Any]:
        """Extract context for a parent repository."""
        try:
            # Find parent in dataframe
            parent_row = self._parent_df[self._parent_df['parent_url'] == parent_url]
            
            if parent_row.empty:
                raise ValueError(f"Parent repository not found: {parent_url}")
            
            parent_data = parent_row.iloc[0]
            
            return {
                "url": parent_url,
                "name": parent_data.get('name', 'unknown'),
                "description": parent_data.get('description', ''),
                "primary_language": parent_data.get('primary_language', 'unknown'),
                "domain": parent_data.get('domain', 'unknown'),
                "architecture_type": parent_data.get('architecture_type', 'unknown'),
                "key_functions": parent_data.get('key_functions', ''),
                "dependency_management": parent_data.get('dependency_management_approach', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error extracting parent context for {parent_url}: {e}")
            raise
    
    def _extract_dependency_context(self, dependency_url: str) -> Dict[str, Any]:
        """Extract context for a dependency repository."""
        try:
            # Find dependency in dataframe
            dep_row = self._dependencies_df[self._dependencies_df['dependency_url'] == dependency_url]
            
            if dep_row.empty:
                raise ValueError(f"Dependency not found: {dependency_url}")
            
            dep_data = dep_row.iloc[0]
            
            return {
                "url": dependency_url,
                "name": dep_data.get('name', 'unknown'),
                "description": dep_data.get('description', ''),
                "category": dep_data.get('category', 'unknown'),
                "primary_function": dep_data.get('primary_function', ''),
                "integration_patterns": dep_data.get('integration_patterns', ''),
                "performance_characteristics": dep_data.get('performance_characteristics', ''),
                "parent_repos": dep_data.get('parent_repos', ''),
                "alternatives": dep_data.get('alternatives', '')
            }
            
        except Exception as e:
            logger.error(f"Error extracting dependency context for {dependency_url}: {e}")
            raise
    
    def _validate_dependency_relationship(self, parent_context: Dict, 
                                        dep_a_context: Dict, dep_b_context: Dict):
        """Validate that dependencies are related to the parent repository."""
        parent_name = parent_context.get('name', '').lower()
        
        # Check if parent is mentioned in dependency's parent_repos field
        for dep_context in [dep_a_context, dep_b_context]:
            parent_repos = dep_context.get('parent_repos', '').lower()
            dep_name = dep_context.get('name', '')
            
            # Simple validation - check if parent name appears in parent_repos
            if parent_name and parent_name not in parent_repos:
                logger.warning(f"Dependency {dep_name} may not be related to parent {parent_name}")
    
    def get_available_parents(self) -> pd.DataFrame:
        """Get list of available parent repositories."""
        self._ensure_data_loaded()
        return self._parent_df[['parent_url', 'name', 'description', 'domain']].copy()
    
    def get_dependencies_for_parent(self, parent_url: str) -> pd.DataFrame:
        """Get dependencies related to a specific parent."""
        self._ensure_data_loaded()
        
        # Find parent name
        parent_row = self._parent_df[self._parent_df['parent_url'] == parent_url]
        if parent_row.empty:
            raise ValueError(f"Parent repository not found: {parent_url}")
        
        parent_name = parent_row.iloc[0]['name'].lower()
        
        # Find related dependencies
        related_deps = self._dependencies_df[
            self._dependencies_df['parent_repos'].str.lower().str.contains(parent_name, na=False)
        ]
        
        return related_deps[['dependency_url', 'name', 'description', 'category']].copy()
    
    def validate_csv_schemas(self) -> Dict[str, Any]:
        """Validate that CSV files have the expected schema."""
        validation_results = {
            "parent_csv": {"valid": False, "errors": []},
            "dependencies_csv": {"valid": False, "errors": []}
        }
        
        try:
            # Validate parent CSV
            if Path(self.parent_csv_path).exists():
                parent_df = pd.read_csv(self.parent_csv_path)
                required_parent_cols = [
                    'parent_url', 'name', 'description', 'primary_language',
                    'domain', 'architecture_type', 'key_functions'
                ]
                
                missing_cols = [col for col in required_parent_cols if col not in parent_df.columns]
                if missing_cols:
                    validation_results["parent_csv"]["errors"].append(f"Missing columns: {missing_cols}")
                else:
                    validation_results["parent_csv"]["valid"] = True
                    validation_results["parent_csv"]["row_count"] = len(parent_df)
            else:
                validation_results["parent_csv"]["errors"].append("File not found")
            
            # Validate dependencies CSV
            if Path(self.dependencies_csv_path).exists():
                deps_df = pd.read_csv(self.dependencies_csv_path)
                required_deps_cols = [
                    'dependency_url', 'name', 'description', 'category',
                    'primary_function', 'integration_patterns'
                ]
                
                missing_cols = [col for col in required_deps_cols if col not in deps_df.columns]
                if missing_cols:
                    validation_results["dependencies_csv"]["errors"].append(f"Missing columns: {missing_cols}")
                else:
                    validation_results["dependencies_csv"]["valid"] = True
                    validation_results["dependencies_csv"]["row_count"] = len(deps_df)
            else:
                validation_results["dependencies_csv"]["errors"].append("File not found")
        
        except Exception as e:
            validation_results["error"] = str(e)
        
        return validation_results