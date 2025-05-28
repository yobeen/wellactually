# src/uncertainty_calibration/level3_prompts.py
"""
Level 3 Prompt Generator for LLM Data Augmentation

Generates prompts for dependency comparisons (Level 3).
Uses strategic sampling to compare dependencies within each seed repository's ecosystem.
"""

import logging
import random
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class Level3PromptGenerator:
    """
    Generates prompts for Level 3 dependency comparisons.
    
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
        """
        Generate dependency pairs for comparison using strategic sampling.
        
        Args:
            graph: NetworkX graph
            repo_profiles: Repository profiles (not used in simplified version)
            num_pairs: Number of dependency pairs to generate
            iteration: Current iteration number
            
        Returns:
            List of (dependency_a, dependency_b, parent) tuples
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If unable to generate dependency pairs
        """
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("Graph must be a NetworkX DiGraph")
        
        if num_pairs <= 0:
            raise ValueError(f"Number of pairs must be positive, got: {num_pairs}")
        
        if iteration not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid iteration number: {iteration}. Must be 1, 2, 3, or 4")
        
        if graph.number_of_edges() == 0:
            raise RuntimeError("Cannot generate dependency pairs: graph has no edges")
        
        # Get parent repositories (those that have dependencies)
        parent_repos = self._get_parent_repositories(graph)
        
        if not parent_repos:
            raise RuntimeError("No parent repositories found with dependencies")
        
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
            raise RuntimeError(f"Failed to generate any dependency pairs for iteration {iteration}")
        
        logger.info(f"Generated {len(pairs)} Level 3 dependency pairs for iteration {iteration}")
        return pairs
    
    def _get_parent_repositories(self, graph: nx.DiGraph) -> List[str]:
        """
        Get list of repositories that have dependencies.
        
        Raises:
            RuntimeError: If no parent repositories found
        """
        parent_repos = set()
        
        for source, target in graph.edges():
            # Skip special nodes
            if target not in ['ethereum', 'originality'] and target.strip():
                parent_repos.add(target)
        
        parent_list = list(parent_repos)
        
        if not parent_list:
            raise RuntimeError("No parent repositories found in graph")
        
        return parent_list
    
    def _get_dependencies_for_parent(self, graph: nx.DiGraph, parent: str) -> List[str]:
        """
        Get list of dependencies for a parent repository.
        
        Args:
            graph: NetworkX graph
            parent: Parent repository URL
            
        Returns:
            List of dependency URLs
            
        Raises:
            ValueError: If parent is invalid
        """
        if not parent or not isinstance(parent, str):
            raise ValueError("Parent must be a non-empty string")
        
        dependencies = []
        
        for source, target in graph.edges():
            if target == parent and source.strip():
                dependencies.append(source)
        
        return dependencies
    
    def _generate_tier1_pairs(self, graph: nx.DiGraph, parent_repos: List[str], 
                            num_pairs: int) -> List[Tuple[str, str, str]]:
        """
        Generate Tier 1 pairs (obvious contrasts within each parent).
        
        Raises:
            RuntimeError: If unable to generate tier 1 pairs
        """
        pairs = []
        
        # For each parent, find dependencies and create high-contrast pairs
        for parent in parent_repos:
            dependencies = self._get_dependencies_for_parent(graph, parent)
            
            if len(dependencies) < 2:
                continue
            
            # Sort dependencies by some heuristic (URL length as proxy for complexity)
            dependencies_sorted = sorted(dependencies, key=lambda x: len(x))
            
            # Create pairs between "simple" and "complex" dependencies
            simple_deps = dependencies_sorted[:len(dependencies)//2]
            complex_deps = dependencies_sorted[len(dependencies)//2:]
            
            if not simple_deps or not complex_deps:
                # If we can't split into simple/complex, just take first few pairs
                for i, dep1 in enumerate(dependencies):
                    for dep2 in dependencies[i+1:min(i+3, len(dependencies))]:  # Limit to avoid too many pairs
                        pairs.append((dep1, dep2, parent))
                        if len(pairs) >= num_pairs:
                            return pairs[:num_pairs]
                continue
            
            # Pair simple with complex (high contrast)
            for simple in simple_deps:
                for complex in complex_deps:
                    pairs.append((simple, complex, parent))
                    if len(pairs) >= num_pairs:
                        return pairs[:num_pairs]
        
        if not pairs:
            raise RuntimeError("Failed to generate any tier 1 pairs")
        
        return pairs[:num_pairs]
    
    def _generate_tier1_and_tier2_pairs(self, graph: nx.DiGraph, parent_repos: List[str],
                                      num_pairs: int) -> List[Tuple[str, str, str]]:
        """
        Generate Tier 1 and Tier 2 pairs (clear + moderate contrasts).
        
        Raises:
            RuntimeError: If unable to generate tier 1 and tier 2 pairs
        """
        pairs = []
        
        # 60% Tier 1 (high contrast), 40% Tier 2 (moderate contrast)
        tier1_count = int(num_pairs * 0.6)
        tier2_count = num_pairs - tier1_count
        
        # Generate Tier 1 pairs
        try:
            tier1_pairs = self._generate_tier1_pairs(graph, parent_repos, tier1_count)
            pairs.extend(tier1_pairs)
        except RuntimeError as e:
            logger.warning(f"Could not generate all tier 1 pairs: {e}")
            # Generate what we can
            tier1_pairs = self._generate_tier1_pairs(graph, parent_repos, min(tier1_count, 100))
            pairs.extend(tier1_pairs)
            tier2_count = num_pairs - len(tier1_pairs)
        
        # Generate Tier 2 pairs (within similar dependency types)
        for parent in parent_repos:
            dependencies = self._get_dependencies_for_parent(graph, parent)
            
            if len(dependencies) < 2:
                continue
            
            # Group dependencies by similarity (simple heuristic: by URL pattern)
            dep_groups = self._group_dependencies_by_similarity(dependencies)
            
            # Create pairs within groups (moderate contrast)
            for group in dep_groups:
                if len(group) >= 2:
                    for i, dep1 in enumerate(group):
                        for dep2 in group[i+1:]:
                            pairs.append((dep1, dep2, parent))
                            if len(pairs) >= num_pairs:
                                return pairs[:num_pairs]
        
        if not pairs:
            raise RuntimeError("Failed to generate any tier 1 and tier 2 pairs")
        
        return pairs[:num_pairs]
    
    def _generate_all_tiers_pairs(self, graph: nx.DiGraph, parent_repos: List[str],
                                num_pairs: int) -> List[Tuple[str, str, str]]:
        """
        Generate all tiers of pairs (high, moderate, low contrast).
        
        Raises:
            RuntimeError: If unable to generate all tiers pairs
        """
        pairs = []
        
        # 40% Tier 1, 40% Tier 2, 20% Tier 3
        tier1_count = int(num_pairs * 0.4)
        tier2_count = int(num_pairs * 0.4)
        tier3_count = num_pairs - tier1_count - tier2_count
        
        # Generate previous tiers
        try:
            previous_pairs = self._generate_tier1_and_tier2_pairs(graph, parent_repos, 
                                                                tier1_count + tier2_count)
            pairs.extend(previous_pairs)
        except RuntimeError as e:
            logger.warning(f"Could not generate all previous tier pairs: {e}")
            # Generate what we can
            previous_pairs = self._generate_tier1_and_tier2_pairs(graph, parent_repos, 
                                                                min(tier1_count + tier2_count, 200))
            pairs.extend(previous_pairs)
            tier3_count = num_pairs - len(previous_pairs)
        
        # Generate Tier 3 pairs (low contrast - very similar dependencies)
        for parent in parent_repos:
            dependencies = self._get_dependencies_for_parent(graph, parent)
            
            if len(dependencies) < 2:
                continue
            
            # Create pairs between very similar dependencies
            similar_pairs = self._create_similar_dependency_pairs(dependencies)
            for dep1, dep2 in similar_pairs:
                pairs.append((dep1, dep2, parent))
                if len(pairs) >= num_pairs:
                    return pairs[:num_pairs]
        
        if not pairs:
            raise RuntimeError("Failed to generate any all tiers pairs")
        
        return pairs[:num_pairs]
    
    def _generate_complete_strategic_coverage(self, graph: nx.DiGraph, parent_repos: List[str],
                                            num_pairs: int) -> List[Tuple[str, str, str]]:
        """
        Generate complete strategic coverage of dependency space.
        
        Raises:
            RuntimeError: If unable to generate complete strategic coverage
        """
        pairs = []
        
        # Calculate total dependencies for proportional allocation
        total_dependencies = 0
        parent_dependency_counts = {}
        
        for parent in parent_repos:
            dependencies = self._get_dependencies_for_parent(graph, parent)
            parent_dependency_counts[parent] = len(dependencies)
            total_dependencies += len(dependencies)
        
        if total_dependencies == 0:
            raise RuntimeError("No dependencies found across all parent repositories")
        
        # Distribute pairs across all parents proportionally
        for parent in parent_repos:
            dependencies = self._get_dependencies_for_parent(graph, parent)
            
            if len(dependencies) < 2:
                continue
            
            # Calculate proportional allocation
            parent_dependency_count = parent_dependency_counts[parent]
            parent_pair_allocation = max(1, int((parent_dependency_count / total_dependencies) * num_pairs))
            
            # Generate all possible pairs for this parent
            parent_pairs = []
            for i, dep1 in enumerate(dependencies):
                for dep2 in dependencies[i+1:]:
                    parent_pairs.append((dep1, dep2, parent))
            
            # Sample from all possible pairs
            if len(parent_pairs) <= parent_pair_allocation:
                pairs.extend(parent_pairs)
            else:
                sampled_pairs = random.sample(parent_pairs, parent_pair_allocation)
                pairs.extend(sampled_pairs)
            
            if len(pairs) >= num_pairs:
                return pairs[:num_pairs]
        
        if not pairs:
            raise RuntimeError("Failed to generate any pairs for complete strategic coverage")
        
        return pairs[:num_pairs]
    
    def _group_dependencies_by_similarity(self, dependencies: List[str]) -> List[List[str]]:
        """
        Group dependencies by similarity (simple heuristic).
        
        Args:
            dependencies: List of dependency URLs
            
        Returns:
            List of dependency groups
            
        Raises:
            ValueError: If dependencies list is invalid
        """
        if not dependencies:
            raise ValueError("Dependencies list cannot be empty")
        
        groups = defaultdict(list)
        
        for dep in dependencies:
            if not dep or not isinstance(dep, str):
                continue
                
            # Simple grouping by domain or organization
            if 'github.com' in dep:
                try:
                    parts = dep.split('/')
                    if len(parts) >= 4:
                        # Group by organization
                        org = parts[3]
                        groups[org].append(dep)
                    else:
                        groups['other'].append(dep)
                except Exception:
                    groups['other'].append(dep)
            else:
                groups['other'].append(dep)
        
        result = [group for group in groups.values() if len(group) > 0]
        
        if not result:
            raise ValueError("No valid dependency groups could be created")
        
        return result
    
    def _create_similar_dependency_pairs(self, dependencies: List[str]) -> List[Tuple[str, str]]:
        """
        Create pairs between very similar dependencies.
        
        Args:
            dependencies: List of dependency URLs
            
        Returns:
            List of dependency pairs
            
        Raises:
            ValueError: If dependencies list is invalid
        """
        if not dependencies or len(dependencies) < 2:
            raise ValueError("Need at least 2 dependencies to create pairs")
        
        pairs = []
        
        # Group by organization first
        try:
            org_groups = self._group_dependencies_by_similarity(dependencies)
            
            # Create pairs within each organization
            for group in org_groups:
                if len(group) >= 2:
                    for i, dep1 in enumerate(group):
                        for dep2 in group[i+1:]:
                            pairs.append((dep1, dep2))
        except ValueError:
            # Fallback: create pairs from all dependencies
            for i, dep1 in enumerate(dependencies):
                for dep2 in dependencies[i+1:]:
                    pairs.append((dep1, dep2))
        
        return pairs
    
    def create_dependency_comparison_prompt(self, dep_a: str, dep_b: str, parent: str,
                                          repo_profiles: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Create a dependency comparison prompt.
        
        Args:
            dep_a: First dependency URL
            dep_b: Second dependency URL  
            parent: Parent repository URL
            repo_profiles: Repository profiles (not used in simplified version)
            
        Returns:
            List of message dictionaries in OpenAI format
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate inputs
        for param, name in [(dep_a, 'dep_a'), (dep_b, 'dep_b'), (parent, 'parent')]:
            if not param or not isinstance(param, str):
                raise ValueError(f"{name} must be a non-empty string")
        
        # Extract names for readability
        name_a = self._extract_repo_name(dep_a)
        name_b = self._extract_repo_name(dep_b)
        parent_name = self._extract_repo_name(parent)
        
        # Create dependency comparison prompt
        prompt = [
            {
                "role": "system",
                "content": "You are an expert evaluating the relative importance of dependencies to their parent repositories in the context of the Ethereum ecosystem. Consider functional criticality, replaceability, maintenance burden, and ecosystem impact. Respond with only 'A' or 'B' or 'Equal' to indicate which dependency is more critical."
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

Consider: functional criticality, replaceability, maintenance burden, ecosystem impact.

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
            ValueError: If URL is invalid
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