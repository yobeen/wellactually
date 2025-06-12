# src/uncertainty_calibration/level3_prompts.py
"""
Level 3 prompt generator for dependency comparisons.
Generates prompts using 4-dimension framework for assessing dependency importance.
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class Level3PromptGenerator:
    """
    Generates Level 3 dependency comparison prompts using 4-dimension framework.
    """
    
    def __init__(self):
        """Initialize the Level 3 prompt generator."""
        
        # Define the 4-dimension framework for dependency assessment
        self.dimension_framework = {
            "functional_necessity": {
                "name": "Functional Necessity",
                "weight": 0.4,
                "description": "How essential is this dependency for core parent functionality?",
                "indicators": [
                    "Core features that break without it",
                    "API surface area used by parent",
                    "Critical path involvement in main workflows",
                    "Frequency of usage in codebase",
                    "Impact on primary use cases"
                ],
                "scoring_guidance": "1=Optional enhancement, 3=Nice-to-have feature, 5=Important functionality, 8=Core feature dependency, 10=Parent breaks without it"
            },
            "performance_impact": {
                "name": "Performance Impact",
                "weight": 0.3,
                "description": "How much does this dependency affect parent performance and resource usage?",
                "indicators": [
                    "CPU and memory overhead",
                    "I/O operations and latency",
                    "Startup time contribution",
                    "Runtime performance bottlenecks",
                    "Resource consumption patterns"
                ],
                "scoring_guidance": "1=Negligible performance impact, 3=Minor overhead, 5=Moderate performance consideration, 8=Significant performance factor, 10=Performance critical component"
            },
            "replacement_difficulty": {
                "name": "Replacement Difficulty",
                "weight": 0.2,
                "description": "How difficult would it be to replace, remove, or substitute this dependency?",
                "indicators": [
                    "Availability of alternative libraries",
                    "API complexity and coupling", 
                    "Migration effort required",
                    "Vendor lock-in considerations",
                    "Community and ecosystem support"
                ],
                "scoring_guidance": "1=Easily replaceable with many alternatives, 3=Moderate replacement effort, 5=Limited alternatives available, 8=Difficult to replace, 10=Deeply integrated/irreplaceable"
            },
            "integration_depth": {
                "name": "Integration Depth",
                "weight": 0.1,
                "description": "How deeply integrated is this dependency into parent architecture and design?",
                "indicators": [
                    "Number of import/usage points",
                    "Architectural coupling level",
                    "Configuration and setup complexity",
                    "Data format dependencies",
                    "Build and deployment integration"
                ],
                "scoring_guidance": "1=Surface-level usage, 3=Moderate integration, 5=Well-integrated component, 8=Architectural dependency, 10=Foundational architectural component"
            }
        }

        self.simplified_framework = """
###
        
Functional Necessity - How essential is a dependency for core parent functionality?
Indicators: Core features that break without it, API surface area used by parent, Critical path involvement in main workflows, Frequency of usage in codebase, Impact on primary use cases.
Scoring: Low - Optional enhancement, Less than average - Nice-to-have feature, Average =Important functionality, High - Core feature dependency, Critical - Parent breaks without it.

###

Performance Impact - How much does this dependency affect parent performance and resource usage?
Indicators: CPU and memory overhead, I/O operations and latency, Startup time contribution, Runtime performance bottlenecks, Resource consumption patterns.
Scoring: Low - Negligible performance impact, Less than average - Minor overhead, Average - Moderate performance consideration, High - Significant performance factor, Critical - Performance critical component.

###

Replacement Difficulty - How difficult would it be to replace, remove, or substitute this dependency?
Indicators: Availability of alternative libraries, API complexity and coupling, Migration effort required, Vendor lock-in considerations, Community and ecosystem support.
Scoring: Low - Easily replaceable with many alternatives, Less than average - Moderate replacement effort, Average - Limited alternatives available, High - Difficult to replace, Critical - Deeply integrated/irreplaceable.

###

Integration Depth - How deeply integrated is this dependency into parent architecture and design?
Indicators: Number of import/usage points, Architectural coupling level, Configuration and setup complexity, Data format dependencies, Build and deployment integration.
Scoring: Low - Surface-level usage, Less than average - Moderate integration, Average - Well-integrated component, High - Architectural dependency, Critical - Foundational architectural component.

###
"""
    
    def create_dependency_comparison_prompt(self, parent_context: Dict[str, Any], 
                                          dep_a_context: Dict[str, Any], 
                                          dep_b_context: Dict[str, Any],
                                          simplified: bool = False) -> List[Dict[str, str]]:
        """
        Create a dependency comparison prompt using the 4-dimension framework.
        
        Args:
            parent_context: Context about the parent repository
            dep_a_context: Context about first dependency
            dep_b_context: Context about second dependency
            simplified: If True, returns only overall assessment choice without detailed reasoning/dimensions
            
        Returns:
            List of message dictionaries in OpenAI format
        """
        try:
            # Build framework section
            framework_section = self._build_framework_section()
            
            # Build context sections
            parent_section = self._build_parent_context_section(parent_context)
            dependencies_section = self._build_dependencies_section(dep_a_context, dep_b_context)
            
            # Build response format
            response_format = self._build_response_format(dep_a_context, dep_b_context, simplified)
            
            if simplified:
                # Simplified system message
                system_content = f"""You are an expert software architect evaluating dependency relationships within software projects. You will assess which of two dependencies is more important for a specific parent repository.

Your goal is to determine which dependency is more critical for the parent repository's success and functionality based on your expert judgment.

You may use this framework for assessment:
{self.simplified_framework}

Provide your response in the exact JSON format specified with only the overall choice."""
                
                # Simplified user message
                user_content = f"""Please assess which dependency is more important for the parent repository:

{parent_section}

{dependencies_section}

{response_format}

Based on your expert analysis, determine which dependency is more critical overall and provide only the choice."""
            else:
                # Full system message with framework
                system_content = f"""You are an expert software architect evaluating dependency relationships within software projects. You will assess which of two dependencies is more important for a specific parent repository using a structured 4-dimension framework.

DEPENDENCY IMPORTANCE ASSESSMENT FRAMEWORK:

{framework_section}

For each dimension, provide:
1. A score from 1-10 for BOTH dependencies (where 1 = minimal importance, 10 = maximum importance)
2. A weight (you may adjust from defaults if justified, but they should sum to approximately 1.0)
3. Brief reasoning comparing both dependencies on this dimension

Your goal is to determine which dependency is more critical for the parent repository's success and functionality.

Provide your response in the exact JSON format specified."""
                
                # Full user message with framework analysis
                user_content = f"""Please assess which dependency is more important for the parent repository using the 4-dimension framework:

{parent_section}

{dependencies_section}

{response_format}

Analyze each dimension carefully, score both dependencies, provide reasoning, and determine which dependency is more critical overall."""
            
            prompt = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user", 
                    "content": user_content
                }
            ]
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error creating dependency comparison prompt: {e}")
            raise
    
    def _build_framework_section(self) -> str:
        """Build the 4-dimension framework description."""
        sections = []
        
        for dimension_id, dimension in self.dimension_framework.items():
            section = f"""
**{dimension['name']} (Weight: {dimension['weight']:.0%})**
- **Description**: {dimension['description']}
- **Key Indicators**: {', '.join(dimension['indicators'])}
- **Scoring Guidance**: {dimension['scoring_guidance']}"""
            sections.append(section)
        
        return "\n".join(sections)
    
    def _build_parent_context_section(self, parent_context: Dict[str, Any]) -> str:
        """Build the parent repository context section."""
        parent_fallback = parent_context.get('fallback_context', False)
        parent_note = " (Note: Limited context)" if parent_fallback else ""
        
        return f"""
PARENT REPOSITORY CONTEXT:{parent_note}
Repository: {parent_context.get('url', 'unknown')}
Name: {parent_context.get('name', 'unknown')}
Description: {parent_context.get('description', 'No description available')}
Primary Language: {parent_context.get('primary_language', 'unknown')}
Domain: {parent_context.get('domain', 'unknown')}
Architecture Type: {parent_context.get('architecture_type', 'unknown')}
Key Functions: {parent_context.get('key_functions', 'unknown')}
Dependency Management: {parent_context.get('dependency_management', 'unknown')}"""
    
    def _build_dependencies_section(self, dep_a_context: Dict[str, Any], 
                                  dep_b_context: Dict[str, Any]) -> str:
        """Build the dependencies context section."""
        
        # Check if fallback context is being used
        dep_a_fallback = dep_a_context.get('fallback_context', False)
        dep_b_fallback = dep_b_context.get('fallback_context', False)
        
        dep_a_note = " (Note: Limited context)" if dep_a_fallback else ""
        dep_b_note = " (Note: Limited context)" if dep_b_fallback else ""
        
        dep_a_section = f"""
DEPENDENCY A:{dep_a_note}
Repository: {dep_a_context.get('url', 'unknown')}
Name: {dep_a_context.get('name', 'unknown')}
Description: {dep_a_context.get('description', 'No description available')}
Category: {dep_a_context.get('category', 'unknown')}
Integration Patterns: {dep_a_context.get('integration_patterns', 'unknown')}
"""

        dep_b_section = f"""
DEPENDENCY B:{dep_b_note}
Repository: {dep_b_context.get('url', 'unknown')}
Name: {dep_b_context.get('name', 'unknown')}
Description: {dep_b_context.get('description', 'No description available')}
Category: {dep_b_context.get('category', 'unknown')}
Integration Patterns: {dep_b_context.get('integration_patterns', 'unknown')}
"""
        
        return dep_a_section + "\n" + dep_b_section
    
    def _build_response_format(self, dep_a_context: Dict[str, Any], 
                             dep_b_context: Dict[str, Any], simplified: bool = False) -> str:
        """Build the expected response format section."""
        
        dep_a_name = dep_a_context.get('name', 'dependency_a')
        dep_b_name = dep_b_context.get('name', 'dependency_b')
        
        if simplified:
            # Simplified format with only overall assessment choice
            response_format = f"""
Please respond in this exact JSON format:

{{
  {{
    "choice": "[A|B|Equal]"
  }}
}}"""
        else:
            # Full format with dimensions and detailed reasoning
            example_dimensions = []
            for dimension_id, dimension in self.dimension_framework.items():
                example_dimensions.append(f'''    "{dimension_id}": {{
      "name": "{dimension['name']}",
      "score_a": [1-10],
      "score_b": [1-10],
      "weight": {dimension['weight']:.3f},
      "reasoning": "[Compare both dependencies on this dimension]"
    }}''')
            
            dimensions_json = ',\n'.join(example_dimensions)
            
            response_format = f"""
Please respond in this exact JSON format:

{{
  "parent_url": "{dep_a_context.get('url', 'parent_url')}",
  "parent_name": "parent_repository_name",
  "dependency_a_url": "{dep_a_context.get('url', 'dep_a_url')}",
  "dependency_a_name": "{dep_a_name}",
  "dependency_b_url": "{dep_b_context.get('url', 'dep_b_url')}",
  "dependency_b_name": "{dep_b_name}",
  "dimension_assessments": {{
{dimensions_json}
  }},
  "overall_assessment": {{
    "choice": "[A|B|Equal]",
    "weighted_score_a": "[calculated weighted sum for dependency A]",
    "weighted_score_b": "[calculated weighted sum for dependency B]", 
    "reasoning": "[Overall comparison explaining which dependency is more important and why]"
  }}
}}"""
        
        return response_format
    
    def get_default_weights(self) -> Dict[str, float]:
        """Get the default weights for all dimensions."""
        return {dimension_id: dimension['weight'] 
                for dimension_id, dimension in self.dimension_framework.items()}
    
    def get_dimension_names(self) -> Dict[str, str]:
        """Get mapping of dimension IDs to human-readable names."""
        return {dimension_id: dimension['name'] 
                for dimension_id, dimension in self.dimension_framework.items()}
    
    def validate_weights_sum(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate that weights sum to approximately 1.0.
        
        Args:
            weights: Dictionary of dimension weights
            
        Returns:
            Validation result dictionary
        """
        total_weight = sum(weights.values())
        deviation = abs(total_weight - 1.0)
        
        return {
            'total_weight': total_weight,
            'deviation': deviation,
            'is_valid': deviation <= 0.1,  # Allow 10% deviation
            'needs_normalization': deviation > 0.05,  # Normalize if >5% deviation
            'warning': deviation > 0.1
        }