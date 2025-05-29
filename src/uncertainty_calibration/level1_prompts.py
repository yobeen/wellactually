# src/uncertainty_calibration/level1_prompts.py
"""
Level 1 Prompt Generator with structured Reasoning/Answer format.
Generates prompts for seed repository comparisons (Level 1).
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
    to the Ethereum ecosystem using structured reasoning format.
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
        """Load seed repositories from configuration file."""
        if self.seed_repos_cache is not None:
            return self.seed_repos_cache
        
        # For now, return a basic list since we don't have the seed file
        # In production, this would load from the actual configuration file
        basic_repos = [
            {"url": "https://github.com/ethereum/go-ethereum", "name": "go-ethereum", "category": "client"},
            {"url": "https://github.com/hyperledger/besu", "name": "besu", "category": "client"},
            {"url": "https://github.com/paradigmxyz/reth", "name": "reth", "category": "client"},
            {"url": "https://github.com/ethereum/solidity", "name": "solidity", "category": "language"},
            {"url": "https://github.com/vyperlang/vyper", "name": "vyper", "category": "language"},
            {"url": "https://github.com/ethers-io/ethers.js", "name": "ethers.js", "category": "library"},
            {"url": "https://github.com/web3/web3.js", "name": "web3.js", "category": "library"},
        ]
        
        self.seed_repos_cache = basic_repos
        logger.info(f"Loaded {len(self.seed_repos_cache)} basic seed repositories")
        
        return self.seed_repos_cache
    
    def generate_repository_pairs(self, seed_repos: List[Dict[str, Any]], 
                                num_pairs: int, iteration: int) -> List[Tuple[Dict, Dict]]:
        """Generate repository pairs for comparison based on iteration strategy."""
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
                pairs = self._generate_high_contrast_pairs(seed_repos, num_pairs)
            elif iteration == 2:
                pairs = self._generate_medium_contrast_pairs(seed_repos, num_pairs)
            elif iteration == 3:
                pairs = self._generate_low_contrast_pairs(seed_repos, num_pairs)
            else:
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
        """Generate pairs with high contrast (different categories)."""
        pairs = []
        categories = {}
        
        # Group by category
        for repo in repos:
            category = repo.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(repo)
        
        if len(categories) < 2:
            # Fallback to all pairs if not enough categories
            for i, repo1 in enumerate(repos):
                for repo2 in repos[i+1:]:
                    pairs.append((repo1, repo2))
        else:
            # Generate cross-category pairs
            category_list = list(categories.keys())
            for i, cat1 in enumerate(category_list):
                for cat2 in category_list[i+1:]:
                    for repo1 in categories[cat1]:
                        for repo2 in categories[cat2]:
                            pairs.append((repo1, repo2))
        
        # Shuffle and limit
        random.shuffle(pairs)
        return pairs[:num_pairs]
    
    def _generate_medium_contrast_pairs(self, repos: List[Dict], num_pairs: int) -> List[Tuple[Dict, Dict]]:
        """Generate pairs with medium contrast (mix of within and across categories)."""
        pairs = []
        
        # 70% cross-category, 30% within-category
        cross_category_count = int(num_pairs * 0.7)
        within_category_count = num_pairs - cross_category_count
        
        # Generate cross-category pairs
        try:
            cross_pairs = self._generate_high_contrast_pairs(repos, cross_category_count)
            pairs.extend(cross_pairs)
        except:
            pass
        
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
        
        random.shuffle(within_pairs)
        pairs.extend(within_pairs[:within_category_count])
        
        return pairs[:num_pairs] if pairs else self._generate_complete_coverage_pairs(repos, num_pairs)
    
    def _generate_low_contrast_pairs(self, repos: List[Dict], num_pairs: int) -> List[Tuple[Dict, Dict]]:
        """Generate pairs with low contrast (similar functionality)."""
        # For now, just use within-category pairs
        return self._generate_medium_contrast_pairs(repos, num_pairs)
    
    def _generate_complete_coverage_pairs(self, repos: List[Dict], num_pairs: int) -> List[Tuple[Dict, Dict]]:
        """Generate complete coverage of all possible pairs."""
        pairs = []
        
        # Generate all possible pairs
        for i, repo1 in enumerate(repos):
            for repo2 in repos[i+1:]:
                pairs.append((repo1, repo2))
        
        random.shuffle(pairs)
        return pairs[:num_pairs]
    
    def create_comparison_prompt(self, repo_a: Dict[str, Any], 
                               repo_b: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create a comparison prompt for two repositories with structured format.
        
        Args:
            repo_a: First repository dictionary
            repo_b: Second repository dictionary
            
        Returns:
            List of message dictionaries in OpenAI format
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
        
        # Create structured comparison prompt
        prompt = [
            {
                "role": "system",
                "content": "You are an expert evaluating the relative importance of open source repositories to the Ethereum ecosystem. Provide your reasoning and then give a clear answer."
            },
            {
                "role": "user", 
                "content": f"""Which repository contributes more to the Ethereum ecosystem?

Repository A: {url_a}
({name_a})

Repository B: {url_b}
({name_b})

Consider factors such as:
- Foundational importance to Ethereum infrastructure
- Community adoption and ecosystem integration
- Development activity and maintenance
- Impact on developers and end users
- Long-term strategic value

Please provide your analysis in this format:

Reasoning: [Explain your analysis comparing the two repositories, considering the factors above]

Answer: [A or B or Equal]"""
            }
        ]
        
        return prompt
            
    def _extract_repo_name(self, url: str) -> str:
        """Extract repository name from GitHub URL."""
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