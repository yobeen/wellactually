# src/uncertainty_calibration/level1_prompts.py
"""
Level 1 Prompt Generator for LLM Data Augmentation

Generates prompts for seed repository comparisons (Level 1).
Focuses on ecosystem importance and contribution comparisons.
"""

import logging
import random
import yaml
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import itertools

logger = logging.getLogger(__name__)

class Level1PromptGenerator:
    """
    Generates prompts for Level 1 comparisons between seed repositories.
    
    Level 1 focuses on comparing seed repositories for their relative importance
    to the Ethereum ecosystem.
    """
    
    def __init__(self, config):
        """
        Initialize the Level 1 prompt generator.
        
        Args:
            config: LLM augmentation configuration
        """
        self.config = config
        self.seed_repos_cache = None
        self.categories_cache = None
        self.similarity_groups_cache = None
        
    def load_seed_repositories(self) -> List[Dict[str, Any]]:
        """
        Load seed repositories from configuration file.
        
        Returns:
            List of seed repository dictionaries
            
        Raises:
            FileNotFoundError: If seed repositories config file is missing
            ValueError: If config file is invalid or empty
            yaml.YAMLError: If YAML parsing fails
        """
        if self.seed_repos_cache is not None:
            return self.seed_repos_cache
        
        seed_file_path = self.config.repository_selection.level_1.seed_repositories_file
        
        if not os.path.exists(seed_file_path):
            raise FileNotFoundError(f"Seed repositories configuration file not found: {seed_file_path}")
        
        try:
            with open(seed_file_path, 'r') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file {seed_file_path}: {e}")
        except Exception as e:
            raise IOError(f"Failed to read seed repositories file {seed_file_path}: {e}")
        
        if not data or 'seed_repositories' not in data:
            raise ValueError(f"Invalid seed repositories config: missing 'seed_repositories' key in {seed_file_path}")
        
        seed_repos = data['seed_repositories']
        if not seed_repos or not isinstance(seed_repos, list):
            raise ValueError(f"Invalid seed repositories config: 'seed_repositories' must be a non-empty list in {seed_file_path}")
        
        # Validate required fields for each repository
        for i, repo in enumerate(seed_repos):
            if not isinstance(repo, dict):
                raise ValueError(f"Repository {i} in {seed_file_path} must be a dictionary")
            
            required_fields = ['url', 'name', 'category']
            for field in required_fields:
                if field not in repo:
                    raise ValueError(f"Repository {i} in {seed_file_path} missing required field: {field}")
        
        # Cache additional configuration data
        self.categories_cache = data.get('categories', {})
        self.similarity_groups_cache = data.get('similarity_groups', {})
        
        self.seed_repos_cache = seed_repos
        logger.info(f"Loaded {len(self.seed_repos_cache)} seed repositories from {seed_file_path}")
        
        return self.seed_repos_cache
    
    def get_categories(self) -> Dict[str, List[str]]:
        """
        Get category definitions from configuration.
        
        Returns:
            Dictionary mapping category groups to category lists
            
        Raises:
            ValueError: If categories not loaded or invalid
        """
        if self.categories_cache is None:
            # Ensure seed repositories are loaded first
            self.load_seed_repositories()
        
        if not self.categories_cache:
            raise ValueError("No categories defined in seed repositories configuration")
        
        return self.categories_cache
    
    def get_similarity_groups(self) -> Dict[str, List[str]]:
        """
        Get similarity groups from configuration.
        
        Returns:
            Dictionary mapping group names to repository lists
            
        Raises:
            ValueError: If similarity groups not loaded or invalid
        """
        if self.similarity_groups_cache is None:
            # Ensure seed repositories are loaded first
            self.load_seed_repositories()
        
        if not self.similarity_groups_cache:
            raise ValueError("No similarity groups defined in seed repositories configuration")
        
        return self.similarity_groups_cache
    
    def generate_repository_pairs(self, seed_repos: List[Dict[str, Any]], 
                                num_pairs: int, iteration: int) -> List[Tuple[Dict, Dict]]:
        """
        Generate repository pairs for comparison based on iteration strategy.
        
        Args:
            seed_repos: List of seed repositories
            num_pairs: Number of pairs to generate
            iteration: Current iteration number (affects strategy)
            
        Returns:
            List of (repo_a, repo_b) tuples
            
        Raises:
            ValueError: If insufficient repositories or invalid iteration
            RuntimeError: If pair generation fails
        """
        if not seed_repos:
            raise ValueError("Cannot generate pairs: empty seed repositories list")
        
        if len(seed_repos) < 2:
            raise ValueError(f"Cannot generate pairs: need at least 2 repositories, got {len(seed_repos)}")
        
        if num_pairs <= 0:
            raise ValueError(f"Invalid number of pairs requested: {num_pairs}")
        
        if iteration not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid iteration number: {iteration}. Must be 1, 2, 3, or 4")
        
        try:
            if iteration == 1:
                # High contrast pairs only (different categories)
                pairs = self._generate_high_contrast_pairs(seed_repos, num_pairs)
            elif iteration == 2:
                # Medium contrast pairs (some within categories)
                pairs = self._generate_medium_contrast_pairs(seed_repos, num_pairs)
            elif iteration == 3:
                # Low contrast pairs (similar functionality)
                pairs = self._generate_low_contrast_pairs(seed_repos, num_pairs)
            else:
                # Complete coverage (all remaining pairs)
                pairs = self._generate_complete_coverage_pairs(seed_repos, num_pairs)
            
            if not pairs:
                raise RuntimeError(f"Failed to generate any pairs for iteration {iteration}")
            
            logger.info(f"Generated {len(pairs)} Level 1 pairs for iteration {iteration}")
            return pairs
            
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            else:
                raise RuntimeError(f"Error generating repository pairs for iteration {iteration}: {e}")
    
    def _generate_high_contrast_pairs(self, repos: List[Dict], num_pairs: int) -> List[Tuple[Dict, Dict]]:
        """
        Generate pairs with high contrast (different categories).
        
        Raises:
            RuntimeError: If unable to generate sufficient high contrast pairs
        """
        pairs = []
        categories = {}
        
        # Group by category
        for repo in repos:
            category = repo.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(repo)
        
        if len(categories) < 2:
            raise RuntimeError("Cannot generate high contrast pairs: need at least 2 different categories")
        
        # Generate cross-category pairs
        category_list = list(categories.keys())
        for i, cat1 in enumerate(category_list):
            for cat2 in category_list[i+1:]:
                for repo1 in categories[cat1]:
                    for repo2 in categories[cat2]:
                        pairs.append((repo1, repo2))
        
        if not pairs:
            raise RuntimeError("Failed to generate any high contrast pairs")
        
        # Shuffle and limit
        random.shuffle(pairs)
        return pairs[:num_pairs]
    
    def _generate_medium_contrast_pairs(self, repos: List[Dict], num_pairs: int) -> List[Tuple[Dict, Dict]]:
        """
        Generate pairs with medium contrast (mix of within and across categories).
        
        Raises:
            RuntimeError: If unable to generate sufficient medium contrast pairs
        """
        pairs = []
        
        # 70% cross-category, 30% within-category
        cross_category_count = int(num_pairs * 0.7)
        within_category_count = num_pairs - cross_category_count
        
        # Generate cross-category pairs
        try:
            cross_pairs = self._generate_high_contrast_pairs(repos, cross_category_count)
            pairs.extend(cross_pairs)
        except RuntimeError as e:
            # If we can't generate enough cross-category pairs, adjust the split
            logger.warning(f"Could not generate {cross_category_count} cross-category pairs: {e}")
            cross_pairs = self._generate_high_contrast_pairs(repos, min(cross_category_count, len(repos) * (len(repos) - 1) // 4))
            pairs.extend(cross_pairs)
            within_category_count = num_pairs - len(cross_pairs)
        
        # Generate within-category pairs
        categories = {}
        for repo in repos:
            category = repo.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(repo)
        
        within_pairs = []
        for category, category_repos in categories.items():
            if len(category_repos) >= 2:
                for i, repo1 in enumerate(category_repos):
                    for repo2 in category_repos[i+1:]:
                        within_pairs.append((repo1, repo2))
        
        if within_category_count > 0 and not within_pairs:
            logger.warning("No within-category pairs available, using additional cross-category pairs")
            # Fall back to more cross-category pairs if no within-category pairs available
            additional_cross = self._generate_high_contrast_pairs(repos, within_category_count)
            pairs.extend(additional_cross[:within_category_count])
        else:
            random.shuffle(within_pairs)
            pairs.extend(within_pairs[:within_category_count])
        
        if not pairs:
            raise RuntimeError("Failed to generate any medium contrast pairs")
        
        return pairs
    
    def _generate_low_contrast_pairs(self, repos: List[Dict], num_pairs: int) -> List[Tuple[Dict, Dict]]:
        """
        Generate pairs with low contrast (similar functionality).
        
        Raises:
            RuntimeError: If unable to generate sufficient low contrast pairs
        """
        pairs = []
        
        # Use similarity groups from configuration
        try:
            similarity_groups = self.get_similarity_groups()
            
            # Generate pairs within similarity groups
            for group_name, repo_names in similarity_groups.items():
                # Find actual repo objects for this group
                group_repos = []
                for repo in repos:
                    repo_name = self._extract_repo_name(repo['url']).lower()
                    if repo_name in [name.lower() for name in repo_names]:
                        group_repos.append(repo)
                
                # Generate pairs within this similarity group
                if len(group_repos) >= 2:
                    for i, repo1 in enumerate(group_repos):
                        for repo2 in group_repos[i+1:]:
                            pairs.append((repo1, repo2))
            
        except ValueError:
            logger.warning("No similarity groups available, falling back to category-based similarity")
            # Fallback to within-category pairs
            categories = {}
            for repo in repos:
                category = repo.get('category', 'unknown')
                if category not in categories:
                    categories[category] = []
                categories[category].append(repo)
            
            for category, category_repos in categories.items():
                if len(category_repos) >= 2:
                    for i, repo1 in enumerate(category_repos):
                        for repo2 in category_repos[i+1:]:
                            pairs.append((repo1, repo2))
        
        # Add some cross-category pairs between similar categories from config
        try:
            categories_config = self.get_categories()
            
            # Find categories that might be similar (within same category group)
            for group_name, category_list in categories_config.items():
                if len(category_list) >= 2:
                    # Generate pairs between categories in the same group
                    for i, cat1 in enumerate(category_list):
                        for cat2 in category_list[i+1:]:
                            cat1_repos = [r for r in repos if r.get('category') == cat1]
                            cat2_repos = [r for r in repos if r.get('category') == cat2]
                            
                            for repo1 in cat1_repos:
                                for repo2 in cat2_repos:
                                    pairs.append((repo1, repo2))
                                    
        except ValueError:
            logger.warning("No category configuration available for cross-category similarity")
        
        if not pairs:
            raise RuntimeError("Failed to generate any low contrast pairs")
        
        random.shuffle(pairs)
        return pairs[:num_pairs]
    
    def _generate_complete_coverage_pairs(self, repos: List[Dict], num_pairs: int) -> List[Tuple[Dict, Dict]]:
        """
        Generate complete coverage of all possible pairs.
        
        Raises:
            RuntimeError: If unable to generate pairs
        """
        pairs = []
        
        # Generate all possible pairs
        for i, repo1 in enumerate(repos):
            for repo2 in repos[i+1:]:
                pairs.append((repo1, repo2))
        
        if not pairs:
            raise RuntimeError("Failed to generate any pairs for complete coverage")
        
        random.shuffle(pairs)
        return pairs[:num_pairs]
    
    def create_comparison_prompt(self, repo_a: Dict[str, Any], 
                               repo_b: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create a comparison prompt for two repositories.
        
        Args:
            repo_a: First repository dictionary
            repo_b: Second repository dictionary
            
        Returns:
            List of message dictionaries in OpenAI format
            
        Raises:
            ValueError: If repositories are missing required fields
        """
        # Validate repository data
        for repo, name in [(repo_a, 'repo_a'), (repo_b, 'repo_b')]:
            if not isinstance(repo, dict):
                raise ValueError(f"{name} must be a dictionary")
            if 'url' not in repo:
                raise ValueError(f"{name} missing required 'url' field")
        
        # Extract repository information
        url_a = repo_a['url']
        url_b = repo_b['url']
        name_a = repo_a.get('name', self._extract_repo_name(url_a))
        name_b = repo_b.get('name', self._extract_repo_name(url_b))
        
        # Create simple comparison prompt
        prompt = [
            {
                "role": "system",
                "content": "You are an expert evaluating the relative importance of open source repositories to the Ethereum ecosystem. Respond with only 'A' or 'B' or 'Equal' to indicate which repository's contribution is more important."
            },
            {
                "role": "user", 
                "content": f"""Which repository contributes more to the Ethereum ecosystem?

Repository A: {url_a}
Repository B: {url_b}

Choose: A or B or Equal"""
            }
        ]
        
        return prompt
    
    def _extract_repo_name(self, url: str) -> str:
        """
        Extract repository name from GitHub URL.
        
        Args:
            url: Repository URL
            
        Returns:
            Repository name
            
        Raises:
            ValueError: If URL is invalid or empty
        """
        if not url or not isinstance(url, str):
            raise ValueError("Repository URL must be a non-empty string")
        
        try:
            if 'github.com' in url:
                parts = url.rstrip('/').split('/')
                if len(parts) >= 2:
                    return parts[-1]  # Repository name
            return url.split('/')[-1] if '/' in url else url
        except Exception as e:
            raise ValueError(f"Failed to extract repository name from URL '{url}': {e}")