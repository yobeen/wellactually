# src/uncertainty_calibration/level3_prompts.py
"""
Level 3 Prompt Generator with structured Reasoning/Answer format.
Generates prompts for dependency comparisons (Level 3).
"""

import logging
import random
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class Level3PromptGenerator:
    """
    Generates prompts for Level 3 dependency comparisons with structured format.
    
    Level 3 focuses on comparing relative importance of dependencies
    within each seed repository's ecosystem.
    """
    
    def __init__(self, config):
        """
        Initialize the Level 3 prompt generator.
        
        Args:
            config: LLM augmentation configuration
        """
        self.config = config
        
    def generate_dependency_pairs(self, graph: nx.DiGraph, repo_profiles: Dict[str, Any],
                                num_pairs: int, iteration: int) -> List[Tuple[str, str, str]]:
        """Generate dependency pairs for comparison using strategic sampling."""
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("Graph must be a NetworkX DiGraph")
        
        if num_pairs <= 0:
            raise ValueError(f"Number of pairs must be positive, got: {num_pairs}")
        
        if iteration not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid iteration number: {iteration}. Must be 1, 2, 3, or 4")
        
        # Get parent repositories (those that have dependencies)
        parent_repos = self._get_parent_repositories(graph)
        
        if not parent_repos:
            # Fallback to basic examples if no graph data
            return self._generate_fallback_pairs(num_pairs)
        
        # Apply iteration-specific sampling strategy
        if iteration == 1:
            pairs = self._generate_tier1_pairs(graph, parent_repos, num_pairs)
        elif iteration == 2:
            pairs = self._generate_tier1_and_tier2_pairs(graph, parent_repos, num_pairs)
        elif iteration == 3:
            pairs = self._generate_all_tiers_pairs(graph, parent_repos, num_pairs)
        else:
            pairs = self._generate_complete_strategic_coverage(graph, parent_repos, num_pairs)
        
        if not pairs:
            return self._generate_fallback_pairs(num_pairs)
        
        logger.info(f"Generated {len(pairs)} Level 3 dependency pairs for iteration {iteration}")
        return pairs
    
    def _get_parent_repositories(self, graph: nx.DiGraph) -> List[str]:
        """Get list of repositories that have dependencies."""
        parent_repos = set()
        
        for source, target in graph.edges():
            # Skip special nodes
            if target not in ['ethereum', 'originality'] and target.strip():
                parent_repos.add(target)
        
        return list(parent_repos)
    
    def _get_dependencies_for_parent(self, graph: nx.DiGraph, parent: str) -> List[str]:
        """Get list of dependencies for a parent repository."""
        dependencies = []
        
        for source, target in graph.edges():
            if target == parent and source.strip():
                dependencies.append(source)
        
        return dependencies
    
    def _generate_fallback_pairs(self, num_pairs: int) -> List[Tuple[str, str, str]]:
        """Generate fallback pairs when no graph data is available."""
        fallback_examples = [
            ("https://github.com/psf/requests", "https://github.com/pydantic/pydantic", "https://github.com/ethereum/web3.py"),
            ("https://github.com/readthedocs/sphinx_rtd_theme", "https://github.com/psf/requests", "https://github.com/vyperlang/titanoboa"),
            ("https://github.com/numpy/numpy", "https://github.com/pandas-dev/pandas", "https://github.com/ethereum/py-evm"),
        ]
        
        # Repeat examples to reach desired count
        pairs = []
        while len(pairs) < num_pairs and fallback_examples:
            pairs.extend(fallback_examples)
        
        return pairs[:num_pairs]
    
    def _generate_tier1_pairs(self, graph: nx.DiGraph, parent_repos: List[str], 
                            num_pairs: int) -> List[Tuple[str, str, str]]:
        """Generate Tier 1 pairs (obvious contrasts within each parent)."""
        pairs = []
        
        # For each parent, find dependencies and create high-contrast pairs
        for parent in parent_repos:
            dependencies = self._get_dependencies_for_parent(graph, parent)
            
            if len(dependencies) < 2:
                continue
            
            # Create pairs between dependencies (simple approach)
            for i, dep1 in enumerate(dependencies):
                for dep2 in dependencies[i+1:min(i+3, len(dependencies))]:
                    pairs.append((dep1, dep2, parent))
                    if len(pairs) >= num_pairs:
                        return pairs[:num_pairs]
        
        return pairs
    
    def _generate_tier1_and_tier2_pairs(self, graph: nx.DiGraph, parent_repos: List[str],
                                      num_pairs: int) -> List[Tuple[str, str, str]]:
        """Generate Tier 1 and Tier 2 pairs (clear + moderate contrasts)."""
        # 60% Tier 1, 40% Tier 2
        tier1_count = int(num_pairs * 0.6)
        tier2_count = num_pairs - tier1_count
        
        pairs = []
        
        # Generate Tier 1 pairs
        tier1_pairs = self._generate_tier1_pairs(graph, parent_repos, tier1_count)
        pairs.extend(tier1_pairs)
        
        # Generate additional pairs for Tier 2
        for parent in parent_repos:
            dependencies = self._get_dependencies_for_parent(graph, parent)
            
            if len(dependencies) >= 2:
                # Add more pairs from this parent
                for i, dep1 in enumerate(dependencies):
                    for dep2 in dependencies[i+1:]:
                        if (dep1, dep2, parent) not in pairs and (dep2, dep1, parent) not in pairs:
                            pairs.append((dep1, dep2, parent))
                            if len(pairs) >= num_pairs:
                                return pairs[:num_pairs]
        
        return pairs
    
    def _generate_all_tiers_pairs(self, graph: nx.DiGraph, parent_repos: List[str],
                                num_pairs: int) -> List[Tuple[str, str, str]]:
        """Generate all tiers of pairs (high, moderate, low contrast)."""
        return self._generate_tier1_and_tier2_pairs(graph, parent_repos, num_pairs)
    
    def _generate_complete_strategic_coverage(self, graph: nx.DiGraph, parent_repos: List[str],
                                            num_pairs: int) -> List[Tuple[str, str, str]]:
        """Generate complete strategic coverage of dependency space."""
        pairs = []
        
        # Generate all possible pairs for each parent
        for parent in parent_repos:
            dependencies = self._get_dependencies_for_parent(graph, parent)
            
            if len(dependencies) >= 2:
                for i, dep1 in enumerate(dependencies):
                    for dep2 in dependencies[i+1:]:
                        pairs.append((dep1, dep2, parent))
        
        # Shuffle and limit
        random.shuffle(pairs)
        return pairs[:num_pairs]
    
    def create_dependency_comparison_prompt(self, dep_a: str, dep_b: str, parent: str,
                                          repo_profiles: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Create a dependency comparison prompt with structured format.
        
        Args:
            dep_a: First dependency URL
            dep_b: Second dependency URL  
            parent: Parent repository URL
            repo_profiles: Repository profiles (not used in simplified version)
            
        Returns:
            List of message dictionaries in OpenAI format
        """
        # Validate inputs
        for param, name in [(dep_a, 'dep_a'), (dep_b, 'dep_b'), (parent, 'parent')]:
            if not param or not isinstance(param, str):
                raise ValueError(f"{name} must be a non-empty string")
        
        # Extract names for readability
        name_a = self._extract_repo_name(dep_a)
        name_b = self._extract_repo_name(dep_b)
        parent_name = self._extract_repo_name(parent)
        
        # Create structured dependency comparison prompt
        prompt = [
            {
                "role": "system",
                "content": "You are an expert evaluating the relative importance of dependencies to their parent repositories in the context of the Ethereum ecosystem. Consider functional criticality, replaceability, maintenance burden, and ecosystem impact. Provide your reasoning and then give a clear answer."
            },
            {
                "role": "user",
                "content": f"""Which dependency is more critical for the parent repository?

Dependency A: {dep_a}
({name_a})

Dependency B: {dep_b}
({name_b})

Parent Repository: {parent}
({parent_name})

Consider factors such as:
- Functional criticality: How essential is this dependency to core functionality?
- Replaceability: How difficult would it be to replace this dependency?
- Maintenance burden: What are the ongoing costs and risks?
- Ecosystem impact: How does this dependency affect the broader Ethereum ecosystem?
- Development workflow: How much does this dependency affect the development process?

Please provide your analysis in this format:

Reasoning: [Explain your analysis comparing the criticality of both dependencies for the parent repository, considering the factors above]

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