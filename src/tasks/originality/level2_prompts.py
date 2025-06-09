# src/uncertainty_calibration/level2_prompts.py
"""
Level 2 Prompt Generator with structured Reasoning/Answer format.
Generates prompts for originality assessments (Level 2).
"""

import logging
import random
import networkx as nx
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class Level2PromptGenerator:
    """
    Generates prompts for Level 2 originality assessments with structured format.
    
    Level 2 focuses on assessing how much of a repository's contribution
    comes from original work vs borrowed/adapted concepts and dependencies.
    """
    
    def __init__(self, config):
        """
        Initialize the Level 2 prompt generator.
        
        Args:
            config: LLM augmentation configuration
        """
        self.config = config
        
    def select_repositories_for_assessment(self, repo_profiles: Dict[str, Any], 
                                         graph: nx.DiGraph, num_samples: int, 
                                         iteration: int) -> List[Dict[str, Any]]:
        """Select repositories for originality assessment."""
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("Graph must be a NetworkX DiGraph")
        
        if num_samples <= 0:
            raise ValueError(f"Number of samples must be positive, got: {num_samples}")
        
        if iteration not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid iteration number: {iteration}. Must be 1, 2, 3, or 4")
        
        # Get all repositories that have dependencies (appear as parents in the graph)
        repositories_with_deps = set()
        
        # Find repositories that have incoming edges (dependencies point to them)
        for source, target in graph.edges():
            if target != 'ethereum' and target != 'originality':
                repositories_with_deps.add(target)
        
        if not repositories_with_deps:
            # Fallback to basic repository list
            repositories_with_deps = {
                "https://github.com/ethereum/go-ethereum",
                "https://github.com/ethereum/solidity", 
                "https://github.com/ethers-io/ethers.js",
                "https://github.com/hyperledger/besu",
                "https://github.com/paradigmxyz/reth"
            }
        
        # Convert to list and add basic info
        repo_list = []
        for repo_url in repositories_with_deps:
            if not isinstance(repo_url, str) or not repo_url.strip():
                continue
                
            repo_list.append({
                "url": repo_url,
                "name": self._extract_repo_name(repo_url),
                "in_degree": graph.in_degree(repo_url) if graph.has_node(repo_url) else 0,
                "out_degree": graph.out_degree(repo_url) if graph.has_node(repo_url) else 0
            })
        
        if not repo_list:
            raise RuntimeError("No valid repositories found for originality assessment")
        
        # Apply iteration-specific selection strategy
        if iteration == 1:
            selected = self._select_clear_cases(repo_list, num_samples)
        elif iteration == 2:
            selected = self._select_major_repositories(repo_list, num_samples)
        elif iteration == 3:
            selected = self._select_mid_tier_repositories(repo_list, num_samples)
        else:
            selected = self._select_remaining_repositories(repo_list, num_samples)
        
        logger.info(f"Selected {len(selected)} repositories for Level 2 assessment (iteration {iteration})")
        return selected
    
    def _select_clear_cases(self, repos: List[Dict], num_samples: int) -> List[Dict]:
        """Select clear high/low originality cases for iteration 1."""
        # Sort by complexity indicators (in_degree as proxy for dependencies)
        repos_sorted = sorted(repos, key=lambda x: x.get('in_degree', 0))
        
        # Take from both ends
        selected = []
        
        # Take low-dependency repos (likely high originality)
        low_dep_count = min(num_samples // 2, len(repos_sorted) // 2)
        selected.extend(repos_sorted[:low_dep_count])
        
        # Take high-dependency repos (likely low originality)
        high_dep_count = min(num_samples - low_dep_count, len(repos_sorted) - low_dep_count)
        if high_dep_count > 0:
            selected.extend(repos_sorted[-high_dep_count:])
        
        return selected
    
    def _select_major_repositories(self, repos: List[Dict], num_samples: int) -> List[Dict]:
        """Select major ecosystem repositories for iteration 2."""
        major_keywords = [
            'ethereum', 'openzeppelin', 'foundry', 'hardhat', 'solidity', 
            'vyper', 'prysm', 'lighthouse', 'geth', 'nethermind', 'web3',
            'ethers', 'remix', 'metamask', 'truffle', 'ganache'
        ]
        
        major_repos = []
        other_repos = []
        
        for repo in repos:
            repo_name = repo['name'].lower()
            repo_url = repo['url'].lower()
            
            is_major = any(keyword in repo_name or keyword in repo_url 
                          for keyword in major_keywords)
            
            if is_major:
                major_repos.append(repo)
            else:
                other_repos.append(repo)
        
        # Prefer major repos but fill with others if needed
        selected = major_repos[:num_samples]
        if len(selected) < num_samples and other_repos:
            selected.extend(other_repos[:num_samples - len(selected)])
        
        return selected
    
    def _select_mid_tier_repositories(self, repos: List[Dict], num_samples: int) -> List[Dict]:
        """Select mid-tier projects with moderate dependencies."""
        # Filter for repos with moderate dependency counts
        mid_tier = [repo for repo in repos 
                   if 5 <= repo.get('in_degree', 0) <= 20]
        
        if len(mid_tier) < num_samples:
            # Include all mid-tier and fill with others
            selected = mid_tier[:]
            remaining = [repo for repo in repos if repo not in mid_tier]
            if remaining:
                selected.extend(remaining[:num_samples - len(selected)])
        else:
            selected = random.sample(mid_tier, num_samples)
        
        return selected
    
    def _select_remaining_repositories(self, repos: List[Dict], num_samples: int) -> List[Dict]:
        """Select remaining repositories for complete coverage."""
        if len(repos) <= num_samples:
            return repos
        else:
            return random.sample(repos, num_samples)
    
    def create_originality_prompt(self, repo: Dict[str, Any], 
                                repo_profiles: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Create an originality assessment prompt for a repository with structured format.
        
        Args:
            repo: Repository dictionary
            repo_profiles: Repository profiles (not used in simplified version)
            
        Returns:
            List of message dictionaries in OpenAI format
        """
        if not isinstance(repo, dict):
            raise ValueError("Repository must be a dictionary")
        
        if 'url' not in repo:
            raise ValueError("Repository missing required 'url' field")
        
        url = repo['url']
        if not url or not isinstance(url, str):
            raise ValueError("Repository URL must be a non-empty string")
        
        name = repo.get('name', self._extract_repo_name(url))
        
        # Add context information if available
        context_info = []
        if 'in_degree' in repo:
            context_info.append(f"Dependencies: {repo['in_degree']}")
        if 'out_degree' in repo:
            context_info.append(f"Dependents: {repo['out_degree']}")
        
        context_str = f" ({', '.join(context_info)})" if context_info else ""
        
        # Create structured originality assessment prompt
        prompt = [
            {
                "role": "system",
                "content": "You are an expert evaluating the originality of repositories in the Ethereum ecosystem. Assess how much of the repository's value comes from original work versus borrowed/adapted concepts and dependencies. Provide your reasoning and then give a clear numerical rating."
            },
            {
                "role": "user",
                "content": f"""Assess the originality of this repository in the Ethereum ecosystem:

Repository: {url}{context_str}
({name})

Consider:
- How much value comes from original vs borrowed concepts?
- Degree of innovation vs adaptation of existing ideas?
- Dependency on external libraries vs internal contributions?
- Novel approaches or breakthrough innovations?
- Impact on defining new paradigms in the ecosystem?

Rate the originality on a scale where:
1 = Primarily wrapper/fork - minimal original contribution
2 = Heavy adaptation with some original features  
3 = Significant modification of existing concepts
4 = Substantial original work with external dependencies
5 = Mostly original with standard library dependencies
6 = Highly original with minimal external dependencies
7 = Novel approach with breakthrough innovations
8 = Pioneering work that defines new paradigms
9 = Foundational innovation with ecosystem-wide impact
10 = Revolutionary contribution that transforms the field

Please provide your analysis in this format:

Reasoning: [Explain your assessment of the repository's originality, considering the factors above]

Answer: [Number from 1-10]"""
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