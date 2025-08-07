# src/uncertainty_calibration/originality_assessment/originality_prompt_generator.py
"""
Originality Assessment Prompt Generator

Creates category-specific prompts for evaluating repository originality
based on the Ethereum Ecosystem Repository Originality Assessment Framework.
"""

import yaml
from typing import Dict, List, Any
from pathlib import Path

class OriginalityPromptGenerator:
    """
    Generates originality assessment prompts tailored to repository categories.
    """
    
    def __init__(self, config_path: str = "configs/seed_repositories.yaml"):
        """
        Initialize the prompt generator with originality framework configuration.
        
        Args:
            config_path: Path to the seed repositories configuration file
        """
        self.config_path = config_path
        self.framework_config = self._load_framework_config()
        
    def _load_framework_config(self) -> Dict[str, Any]:
        """Load originality framework configuration from YAML."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            framework_config = config.get('originality_framework', {})
            if not framework_config:
                raise ValueError("No originality_framework section found in config")
            return framework_config
        except Exception as e:
            raise ValueError(f"Failed to load originality framework config: {e}")
    
    def get_repo_originality_config(self, repo_url: str) -> Dict[str, Any]:
        """
        Get originality category and weights for a specific repository.
        
        Args:
            repo_url: Repository URL to look up
            
        Returns:
            Dictionary with category and weights
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            for repo in config.get('seed_repositories', []):
                if repo.get('url') == repo_url:
                    # Use originality_category if available, otherwise fall back to category
                    category = repo.get('originality_category') or repo.get('category')
                    weights = repo.get('originality_weights', {})
                    
                    return {
                        'category': category,
                        'weights': weights,
                        'name': repo.get('name'),
                        'description': repo.get('description', ''),
                        'primary_language': repo.get('primary_language', ''),
                        'domain': repo.get('domain', ''),
                        'key_functions': repo.get('key_functions', []),
                        'repo_url': repo_url
                    }
            
            raise ValueError(f"Repository not found in configuration: {repo_url}")
            
        except Exception as e:
            raise ValueError(f"Failed to get repository configuration: {e}")
    
    def create_originality_assessment_prompt(self, repo_url: str) -> List[Dict[str, str]]:
        """
        Create a comprehensive originality assessment prompt for a repository.
        
        Args:
            repo_url: Repository URL to assess
            
        Returns:
            List of message dictionaries in OpenAI format
        """
        repo_config = self.get_repo_originality_config(repo_url)
        category = repo_config['category']
        weights = repo_config['weights']
        
        if not category:
            raise ValueError(f"Missing originality category for repository: {repo_url}")
        
        if not weights:
            raise ValueError(f"Missing originality weights for repository: {repo_url}. Available keys: {list(repo_config.keys())}")
        
        # Get category-specific configuration
        category_config = self.framework_config.get('categories', {}).get(category, {})
        criteria_config = self.framework_config.get('criteria', {})
        
        # Build the prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(repo_config, category_config, criteria_config, weights)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with general instructions."""
        return """You are an expert evaluating the originality and innovation of open source repositories within the Ethereum ecosystem. You will assess repositories across specific criteria.

Your task is to evaluate how much original work and innovation each repository represents, considering both technical contributions and ecosystem impact. Focus on distinguishing between original innovations versus adaptations of existing solutions.

Key principles:
- Score on a 1-10 scale where 1 = minimal originality, 10 = groundbreaking innovation
- Consider the specific repository category and its unique challenges
- Evaluate innovation within the context of when the project was created
- Account for both technical originality and ecosystem influence
- Provide detailed reasoning for each assessment"""

    def _build_user_prompt(self, repo_config: Dict[str, Any], category_config: Dict[str, Any], 
                          criteria_config: Dict[str, Any], weights: Dict[str, float]) -> str:
        """Build the user prompt with repository-specific assessment request."""
        
        repo_url = repo_config.get('repo_url', '')
        repo_name = repo_config.get('name', 'Unknown')
        description = repo_config.get('description', '')
        category = repo_config.get('category', '')
        category_name = category_config.get('name', '')
        
        # Sort criteria by weight (descending) to emphasize important ones
        sorted_criteria = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        prompt = f"""# ORIGINALITY ASSESSMENT

## Repository Information
- **URL**: {repo_url}
- **Name**: {repo_name}
- **Description**: {description}
- **Category**: {category_name}

## Assessment Framework

You are evaluating this repository as a **{category_name}** project. Based on this category, certain criteria are weighted more heavily in the final assessment.

### Originality Criteria (in order of importance for this category):

"""
        
        # Add criteria in order of weight importance
        for criterion_key, weight in sorted_criteria:
            criterion_config = criteria_config.get(criterion_key, {})
            criterion_name = criterion_config.get('name', criterion_key)
            criterion_desc = criterion_config.get('description', '')
            examples = criterion_config.get('examples', {})
            
            emphasis = "🔥 **HIGH PRIORITY**" if weight >= 0.20 else "⭐ **MEDIUM PRIORITY**" if weight >= 0.15 else "📝 Standard"
            
            prompt += f"""
#### {weight:.0%} - {criterion_name} {emphasis}
**Description**: {criterion_desc}

**Scoring Guidelines** (1-10 scale):
- **1-3**: {self._get_scoring_guideline(criterion_key, 'low')}
- **4-6**: {self._get_scoring_guideline(criterion_key, 'medium')}
- **7-10**: {self._get_scoring_guideline(criterion_key, 'high')}

"""
            
            prompt += "\n"
        
        # Add category-specific evaluation focus
        focus_areas = category_config.get('focus_areas', [])
        evaluation_notes = category_config.get('evaluation_notes', '')
        
        if focus_areas:
            prompt += f"""
### Category-Specific Evaluation Focus for {category_name}:
- **Primary Focus Areas**: {', '.join(focus_areas)}
"""
        
        if evaluation_notes:
            prompt += f"- **Key Evaluation Points**: {evaluation_notes}\n"
        
        # Add response format
        prompt += f"""

## Required Response Format

Please provide your assessment in the following JSON format:

```json
{{
    "repository_name": "repo_name",
    "repository_url": "repo_url",
    "criteria_scores": {{
"""
        
        # Add each criterion to response format
        for criterion_key, weight in weights.items():
            criterion_name = criteria_config.get(criterion_key, {}).get('name', criterion_key)
            prompt += f"""        "{criterion_key}": {{
            "reasoning": "Detailed explanation of the score...",
            "score": [1-10 integer],
            "weight": [criteria weight],
        }},
"""
        
        prompt = prompt.rstrip(',\n') + """
    },
    "overall_reasoning": "Comprehensive explanation of the repository's originality and innovation...",
    "total_weight": "[sum of all weights - should be ~1.0]",
    "target_score": "[weighted sum: Σ(weight_i × score_i)]"
}
```

## Assessment Instructions

1. **Research the Repository**: Consider the repository's technical implementation, documentation, and ecosystem role
2. **Category Context**: Focus heavily on the high-priority criteria for this category type
3. **Historical Context**: Consider when this repository was created and what alternatives existed
4. **Innovation vs Adaptation**: Distinguish between original contributions and adaptations of existing solutions
5. **Ecosystem Impact**: Evaluate how the repository's innovations influenced the broader Ethereum ecosystem
6. **Detailed Reasoning**: Provide specific technical examples and concrete evidence for each score

Please begin your assessment now."""

        return prompt
    
    def _get_scoring_guideline(self, criterion_key: str, level: str) -> str:
        """Get scoring guidelines for different levels of originality."""
        guidelines = {
            'protocol_implementation': {
                'low': 'Thin wrapper around existing protocol implementations',
                'medium': 'Substantial protocol implementation with some novel optimizations',
                'high': 'Groundbreaking protocol implementation that sets new standards'
            },
            'algorithmic_innovation': {
                'low': 'Standard algorithms and data structures, no performance innovations',
                'medium': 'Significant performance optimizations or novel data structure usage',
                'high': 'Algorithmic innovations that become adopted across the ecosystem'
            },
            'developer_experience': {
                'low': 'Standard APIs or basic CLI tools',
                'medium': 'Innovative APIs or significantly improved developer workflows',
                'high': 'API/UX innovations that become industry standards'
            },
            'architectural_innovation': {
                'low': 'Monolithic design following standard patterns',
                'medium': 'Well-designed plugin system or innovative component architecture',
                'high': 'Architectural innovations that influence entire ecosystem design'
            },
            'security_innovation': {
                'low': 'Basic testing with standard security practices',
                'medium': 'Innovative testing approaches or security features',
                'high': 'Security innovations that become ecosystem standards'
            },
            'standards_leadership': {
                'low': 'Implements existing standards without contribution to specifications',
                'medium': 'Active participation in standards development, some EIP authorship',
                'high': 'Defining foundational standards that shape the ecosystem'
            },
            'cross_client_compatibility': {
                'low': 'Works in isolation, minimal interoperability features',
                'medium': 'Strong interoperability features and cross-client testing',
                'high': 'Foundational interoperability work that enables ecosystem growth'
            },
        }
        
        return guidelines.get(criterion_key, {}).get(level, 'Evaluate based on originality and innovation')