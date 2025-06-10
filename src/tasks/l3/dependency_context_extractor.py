# src/uncertainty_calibration/dependency_context_extractor.py
"""
Dependency context extractor for Level 3 comparisons.
Extracts repository context from CSV files for dependency assessments.
"""

import pandas as pd
import json
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DependencyContextExtractor:
    """
    Extracts and manages dependency context from CSV files for Level 3 comparisons.
    """
    
    def __init__(self, parent_csv_path: str = "data/raw/DeepFunding Repos Enhanced via OpenQ - ENHANCED TEAMS.csv", 
                 dependencies_csv_path: str = "data/raw/DeepFunding Repos Enhanced via OpenQ - ENHANCED TEAMS.csv",
                 assessments_json_path: str = "data/processed/criteria_assessment/detailed_assessments.json"):
        """
        Initialize the dependency context extractor.
        
        Args:
            parent_csv_path: Path to parent repositories CSV
            dependencies_csv_path: Path to dependencies CSV
            assessments_json_path: Path to detailed assessments JSON for fallback data
        """
        self.parent_csv_path = parent_csv_path
        self.dependencies_csv_path = dependencies_csv_path
        self.assessments_json_path = assessments_json_path
        
        # Cache for loaded data
        self._parent_df = None
        self._dependencies_df = None
        self._assessments_data = None
        
        logger.info("DependencyContextExtractor initialized")
    
    def extract_comparison_context(self, parent_url: str, dep_a_url: str, 
                                 dep_b_url: str) -> Dict[str, Any]:
        """
        Extract context for a dependency comparison.
        Gracefully handles missing repositories by using fallback context.
        
        Args:
            parent_url: URL of the parent repository
            dep_a_url: URL of first dependency
            dep_b_url: URL of second dependency
            
        Returns:
            Dictionary with parent and dependency contexts
        """
        warnings = []
        
        try:
            # Load CSV data if not cached
            self._ensure_data_loaded()
            
            # Extract parent context (with fallback)
            parent_context, parent_warning = self._extract_parent_context_safe(parent_url)
            if parent_warning:
                warnings.append(parent_warning)
            
            # Extract dependency contexts (with fallbacks)
            dep_a_context, dep_a_warning = self._extract_dependency_context_safe(dep_a_url)
            if dep_a_warning:
                warnings.append(dep_a_warning)
                
            dep_b_context, dep_b_warning = self._extract_dependency_context_safe(dep_b_url)
            if dep_b_warning:
                warnings.append(dep_b_warning)
            
            # Log any warnings about missing repositories
            if warnings:
                logger.warning(f"Missing repository data: {'; '.join(warnings)}")
            
            return {
                "parent": parent_context,
                "dependency_a": dep_a_context,
                "dependency_b": dep_b_context,
                "comparison_metadata": {
                    "parent_url": parent_url,
                    "dep_a_url": dep_a_url,
                    "dep_b_url": dep_b_url,
                    "warnings": warnings,
                    "fallback_used": len(warnings) > 0
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
            
        if self._assessments_data is None:
            self._load_assessments_data()
    
    def _load_parent_data(self):
        """Load parent repositories CSV data."""
        try:
            if not Path(self.parent_csv_path).exists():
                raise FileNotFoundError(f"Parent CSV not found: {self.parent_csv_path}")
            
            self._parent_df = pd.read_csv(self.parent_csv_path)
            
            # Validate required columns (mapped to actual CSV structure)
            required_columns = [
                'githubLink', 'name', 'description', 'languages'
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
            
            # Validate required columns (mapped to actual CSV structure)
            required_columns = [
                'githubLink', 'name', 'description'
            ]
            
            missing_columns = [col for col in required_columns if col not in self._dependencies_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in dependencies CSV: {missing_columns}")
            
            logger.info(f"Loaded {len(self._dependencies_df)} dependencies")
            
        except Exception as e:
            logger.error(f"Error loading dependencies CSV: {e}")
            raise
    
    def _load_assessments_data(self):
        """Load detailed assessments JSON data for fallback context."""
        try:
            if not Path(self.assessments_json_path).exists():
                logger.warning(f"Assessments JSON not found: {self.assessments_json_path}")
                self._assessments_data = []
                return
            
            with open(self.assessments_json_path, 'r', encoding='utf-8') as f:
                self._assessments_data = json.load(f)
            
            logger.info(f"Loaded {len(self._assessments_data)} assessment records")
            
        except Exception as e:
            logger.warning(f"Error loading assessments JSON: {e}")
            self._assessments_data = []
    
    def _extract_parent_context(self, parent_url: str) -> Dict[str, Any]:
        """Extract context for a parent repository."""
        try:
            # Find parent in dataframe using flexible URL matching
            parent_row = self._find_repo_by_url(self._parent_df, parent_url)
            
            if parent_row.empty:
                raise ValueError(f"Parent repository not found: {parent_url}")
            
            parent_data = parent_row.iloc[0]
            
            return {
                "url": parent_url,
                "name": parent_data['name'] if 'name' in parent_data and pd.notna(parent_data['name']) else 'unknown',
                "description": parent_data['description'] if 'description' in parent_data and pd.notna(parent_data['description']) else '',
                "primary_language": parent_data['languages'].split(',')[0].strip() if 'languages' in parent_data and pd.notna(parent_data['languages']) else 'unknown',
                "domain": "blockchain",  # Default for this dataset
                "architecture_type": "unknown",  # Not available in current CSV
                "key_functions": parent_data['description'] if 'description' in parent_data and pd.notna(parent_data['description']) else '',
                "dependency_management": "unknown"  # Not available in current CSV
            }
            
        except Exception as e:
            logger.error(f"Error extracting parent context for {parent_url}: {e}")
            raise
    
    def _extract_dependency_context(self, dependency_url: str) -> Dict[str, Any]:
        """Extract context for a dependency repository."""
        try:
            # Find dependency in dataframe using flexible URL matching
            dep_row = self._find_repo_by_url(self._dependencies_df, dependency_url)
            
            if dep_row.empty:
                raise ValueError(f"Dependency not found: {dependency_url}")
            
            dep_data = dep_row.iloc[0]
            
            return {
                "url": dependency_url,
                "name": dep_data['name'] if 'name' in dep_data and pd.notna(dep_data['name']) else 'unknown',
                "description": dep_data['description'] if 'description' in dep_data and pd.notna(dep_data['description']) else '',
                "category": "dependency",  # Default category
                "primary_function": dep_data['description'] if 'description' in dep_data and pd.notna(dep_data['description']) else '',
                "integration_patterns": "unknown",  # Not available in current CSV
                "performance_characteristics": "unknown",  # Not available in current CSV
                "parent_repos": "unknown",  # Not available in current CSV
                "alternatives": "unknown"  # Not available in current CSV
            }
            
        except Exception as e:
            logger.error(f"Error extracting dependency context for {dependency_url}: {e}")
            raise
    
    def _extract_parent_context_safe(self, parent_url: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Extract parent context with fallback for missing repositories.
        
        Returns:
            Tuple of (context_dict, warning_message)
        """
        try:
            return self._extract_parent_context(parent_url), None
        except Exception:
            fallback_context = self._create_fallback_parent_context(parent_url)
            if fallback_context.get('assessment_fallback'):
                logger.info(f"Parent repository found in assessments data: {parent_url}")
                return fallback_context, None  # No warning if found in assessments
            else:
                logger.warning(f"Parent repository not found in any data source: {parent_url}")
                return fallback_context, f"Parent repo {parent_url} not found in CSV or assessments"
    
    def _extract_dependency_context_safe(self, dependency_url: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Extract dependency context with fallback for missing repositories.
        
        Returns:
            Tuple of (context_dict, warning_message)
        """
        try:
            return self._extract_dependency_context(dependency_url), None
        except Exception:
            fallback_context = self._create_fallback_dependency_context(dependency_url)
            if fallback_context.get('assessment_fallback'):
                logger.info(f"Dependency found in assessments data: {dependency_url}")
                return fallback_context, None  # No warning if found in assessments
            else:
                logger.warning(f"Dependency not found in any data source: {dependency_url}")
                return fallback_context, f"Dependency {dependency_url} not found in CSV or assessments"
    
    def _find_assessment_by_url(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """Find repository assessment data by URL."""
        if not self._assessments_data:
            return None
        
        # Normalize the search URL
        normalized_search = self._normalize_github_url(repo_url)
        
        for assessment in self._assessments_data:
            assessment_url = assessment.get('repository_url', '')
            if assessment_url:
                normalized_assessment = self._normalize_github_url(assessment_url)
                if normalized_search == normalized_assessment:
                    return assessment
        
        return None
    
    def _create_fallback_parent_context(self, parent_url: str) -> Dict[str, Any]:
        """Create fallback context for parent repository not found in CSV."""
        # Try to find in assessments data first
        assessment = self._find_assessment_by_url(parent_url)
        
        if assessment:
            logger.info(f"Found parent in assessments data: {parent_url}")
            return self._create_context_from_assessment(parent_url, assessment, is_parent=True)
        
        # Extract basic info from URL as final fallback
        name = self._extract_name_from_url(parent_url)
        
        return {
            "url": parent_url,
            "name": name,
            "description": f"Repository context not available in CSV data",
            "primary_language": "unknown",
            "domain": "unknown",
            "architecture_type": "unknown", 
            "key_functions": "unknown",
            "dependency_management": "unknown",
            "fallback_context": True
        }
    
    def _create_fallback_dependency_context(self, dependency_url: str) -> Dict[str, Any]:
        """Create fallback context for dependency repository not found in CSV."""
        # Try to find in assessments data first
        assessment = self._find_assessment_by_url(dependency_url)
        
        if assessment:
            logger.info(f"Found dependency in assessments data: {dependency_url}")
            return self._create_context_from_assessment(dependency_url, assessment, is_parent=False)
        
        # Extract basic info from URL as final fallback
        name = self._extract_name_from_url(dependency_url)
        
        return {
            "url": dependency_url,
            "name": name,
            "description": f"Dependency context not available in CSV data",
            "category": "unknown",
            "primary_function": "unknown",
            "integration_patterns": "unknown",
            "performance_characteristics": "unknown",
            "parent_repos": "unknown",
            "alternatives": "unknown",
            "fallback_context": True
        }
    
    def _create_context_from_assessment(self, repo_url: str, assessment: Dict[str, Any], 
                                       is_parent: bool = True) -> Dict[str, Any]:
        """Create context from assessment data."""
        name = assessment.get('repository_name', self._extract_name_from_url(repo_url))
        target_score = assessment.get('target_score', 0)
        
        # Extract key information from criteria scores
        criteria_scores = assessment.get('criteria_scores', {})
        core_protocol = criteria_scores.get('core_protocol', {})
        security = criteria_scores.get('security_infrastructure', {})
        
        # Create description from overall reasoning or core protocol reasoning
        description = assessment.get('overall_reasoning', '')
        if not description and core_protocol:
            description = core_protocol.get('reasoning', '')
        if not description:
            description = f"Repository with target score {target_score:.2f}"
        
        # Determine primary language and domain from criteria
        primary_language = "unknown"
        domain = "blockchain"  # Default for assessment data
        
        # Determine category from scores
        if core_protocol.get('score', 0) >= 7:
            category = "core_infrastructure"
        elif security.get('score', 0) >= 5:
            category = "security_tool"
        else:
            category = "ethereum_tool"
        
        if is_parent:
            return {
                "url": repo_url,
                "name": name,
                "description": description,
                "primary_language": primary_language,
                "domain": domain,
                "architecture_type": category,
                "key_functions": f"Target importance score: {target_score:.2f}",
                "dependency_management": "unknown",
                "assessment_fallback": True,
                "target_score": target_score
            }
        else:
            return {
                "url": repo_url,
                "name": name,
                "description": description,
                "category": category,
                "primary_function": f"Target importance score: {target_score:.2f}",
                "integration_patterns": "unknown",
                "performance_characteristics": f"Importance score: {target_score:.2f}",
                "parent_repos": "unknown",
                "alternatives": "unknown",
                "assessment_fallback": True,
                "target_score": target_score
            }
    
    def _find_repo_by_url(self, df: pd.DataFrame, search_url: str) -> pd.DataFrame:
        """
        Find repository in dataframe using flexible URL matching.
        Handles various URL formats like:
        - https://github.com/owner/repo
        - owner/repo
        - github.com/owner/repo
        """
        # First try exact match
        exact_match = df[df['githubLink'] == search_url]
        if not exact_match.empty:
            return exact_match
        
        # Normalize URLs for comparison
        normalized_search = self._normalize_github_url(search_url)
        
        # Try matching against normalized URLs
        for idx, row in df.iterrows():
            if pd.notna(row['githubLink']):
                normalized_csv_url = self._normalize_github_url(row['githubLink'])
                if normalized_search == normalized_csv_url:
                    return df.iloc[[idx]]
        
        # Return empty DataFrame if no match found
        return pd.DataFrame()
    
    def _normalize_github_url(self, url: str) -> str:
        """
        Normalize GitHub URL to owner/repo format for comparison.
        """
        try:
            # Remove common prefixes
            url = url.strip()
            url = url.replace('https://github.com/', '')
            url = url.replace('http://github.com/', '')
            url = url.replace('github.com/', '')
            url = url.replace('www.github.com/', '')
            
            # Remove trailing slash and .git
            url = url.rstrip('/')
            url = url.replace('.git', '')
            
            # Ensure it's in owner/repo format
            parts = url.split('/')
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            
            return url
        except Exception:
            return url

    def _extract_name_from_url(self, url: str) -> str:
        """Extract repository name from GitHub URL."""
        try:
            # Handle GitHub URLs like https://github.com/owner/repo
            if 'github.com' in url:
                parts = url.rstrip('/').split('/')
                if len(parts) >= 2:
                    return parts[-1]  # repo name
            return url.split('/')[-1] if '/' in url else url
        except Exception:
            return "unknown"
    
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