# src/uncertainty_calibration/criteria_assessment/criteria_prompt_generator.py
"""
Criteria-based prompt generator for repository assessment.
Generates comprehensive prompts with all 11 importance criteria matching the detailed template.
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class CriteriaPromptGenerator:
    """
    Generates criteria-based assessment prompts for repositories.
    """
    
    def __init__(self):
        """Initialize the criteria prompt generator with detailed template."""
        # Define the 11 criteria with their weights, descriptions, and scoring guidance
        self.criteria_template = {
            "core_protocol": {
                "name": "Core Protocol Implementation",
                "weight": 0.25,
                "description": "Measures how critical the repository is to Ethereum's core functioning and protocol implementation",
                "indicators": [
                    "Official Ethereum Foundation repositories",
                    "Execution clients", 
                    "Consensus clients",
                    "Core language implementations",
                    "Layer 2 protocol implementations"
                ],
                "scoring_guidance": "Official implementations score highest (10x multiplier), major client implementations (3-5x), alternative implementations (1-2x)"
            },
            "market_adoption": {
                "name": "Market Adoption & Network Effects", 
                "weight": 0.20,
                "description": "Evaluates real-world usage, developer adoption, and network share",
                "indicators": [
                    "Number of active users/developers",
                    "Market share for clients (% of validators/nodes)",
                    "Total Value Locked (TVL) for DeFi protocols",
                    "Transaction volume and fee generation",
                    "Integration in major projects"
                ],
                "scoring_guidance": "Dominant market position (5-10x multiplier), significant adoption (2-5x), emerging adoption (1-2x)"
            },
            "developer_ecosystem": {
                "name": "Developer Ecosystem Impact",
                "weight": 0.15,
                "description": "Measures how much the tool enables and accelerates Ethereum development",
                "indicators": [
                    "Development frameworks",
                    "Core libraries",
                    "Smart contract libraries", 
                    "Testing, debugging, and deployment tools",
                    "Documentation and educational resources"
                ],
                "scoring_guidance": "Essential developer tools (3-5x multiplier), widely-used libraries (2-3x), specialized tools (1-2x)"
            },
            "general_purpose_tools": {
                "name": "General Purpose Tool Dependency",
                "weight": 0.10,
                "description": "Recognizes non-blockchain tools that are essential for ecosystem operations",
                "indicators": [
                    "Data analysis tools",
                    "Programming languages and runtimes",
                    "Database systems",
                    "Web frameworks and libraries",
                    "DevOps tools"
                ],
                "scoring_guidance": "Universal dependencies (5-10x multiplier), widely-used tools (2-5x), specialized tools (1-2x)"
            },
            "security_infrastructure": {
                "name": "Security & Infrastructure Criticality",
                "weight": 0.10,
                "description": "Assesses the security implications and infrastructure dependencies",
                "indicators": [
                    "Smart contract security tools and audited libraries",
                    "Account abstraction and wallet infrastructure",
                    "Key management and cryptographic libraries",
                    "Audit firms' tools and frameworks",
                    "Bug bounty platforms"
                ],
                "scoring_guidance": "Security-critical infrastructure (5-10x multiplier), important security tools (2-5x), auxiliary security projects (1-2x)"
            },
            "defi_infrastructure": {
                "name": "DeFi & Financial Infrastructure",
                "weight": 0.05,
                "description": "Evaluates importance to decentralized finance ecosystem",
                "indicators": [
                    "Core DeFi protocols",
                    "Oracle networks",
                    "Stablecoin infrastructure",
                    "Liquidity and market-making tools",
                    "Cross-chain bridges"
                ],
                "scoring_guidance": "Systemic DeFi infrastructure (3-5x multiplier), major protocols (2-3x), niche protocols (1-2x)"
            },
            "data_analytics": {
                "name": "Data & Analytics Infrastructure",
                "weight": 0.05,
                "description": "Measures contribution to blockchain data accessibility and analysis",
                "indicators": [
                    "Blockchain explorers",
                    "Data indexing protocols",
                    "Analytics platforms",
                    "MEV infrastructure",
                    "Data availability solutions"
                ],
                "scoring_guidance": "Essential data infrastructure (3-5x multiplier), major analytics tools (2-3x), specialized tools (1-2x)"
            },
            "innovation_research": {
                "name": "Innovation & Research Impact",
                "weight": 0.03,
                "description": "Recognizes technical innovation and research contributions",
                "indicators": [
                    "Novel consensus mechanisms or cryptographic implementations",
                    "Zero-knowledge proof systems",
                    "Academic research tools and implementations",
                    "Experimental protocols and proof-of-concepts"
                ],
                "scoring_guidance": "Breakthrough innovations (2-3x multiplier), significant improvements (1.5-2x), incremental advances (1-1.5x)"
            },
            "ecosystem_coordination": {
                "name": "Ecosystem Coordination & Standards",
                "weight": 0.03,
                "description": "Evaluates contribution to ecosystem coordination and standardization",
                "indicators": [
                    "EIP (Ethereum Improvement Proposal) repositories",
                    "Token standards and registries",
                    "Chain and token lists",
                    "Governance tools and frameworks",
                    "DAO infrastructure"
                ],
                "scoring_guidance": "Core standards (2-3x multiplier), widely-adopted standards (1.5-2x), niche standards (1x)"
            },
            "community_trust": {
                "name": "Community Trust & Project Maturity",
                "weight": 0.02,
                "description": "Evaluates project stability, maintenance, and community confidence",
                "indicators": [
                    "Years in production",
                    "Corporate/foundation backing",
                    "Maintenance activity and responsiveness",
                    "Community size and engagement"
                ],
                "scoring_guidance": "Battle-tested projects (2x multiplier), established projects (1.5x), newer projects (1x)"
            },
            "user_applications": {
                "name": "User-Facing Applications",
                "weight": 0.02,
                "description": "Recognizes importance of end-user applications and interfaces",
                "indicators": [
                    "Major wallets",
                    "NFT platforms and marketplaces",
                    "Gaming infrastructure",
                    "Social protocols and identity systems"
                ],
                "scoring_guidance": "Dominant user applications (2-3x multiplier), popular applications (1.5-2x), niche applications (1x)"
            }
        }
    
    def create_criteria_assessment_prompt(self, repo_info: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Create a comprehensive criteria assessment prompt for a repository.
        
        Args:
            repo_info: Repository information dictionary with 'url', 'name', etc.
            
        Returns:
            List of message dictionaries in OpenAI format
        """
        repo_url = repo_info.get('url', '')
        repo_name = repo_info.get('name', 'unknown')
        repo_full_name = repo_info.get('full_name', 'unknown/unknown')
        
        if not repo_url:
            raise ValueError("Repository URL is required")
        
        # Build criteria section for prompt
        criteria_section = self._build_criteria_section()
        
        # Build response format section
        response_format = self._build_response_format()
        
        prompt = [
            {
                "role": "system",
                "content": f"""You are an expert evaluating the importance and contribution of open source repositories to the Ethereum ecosystem. You will assess repositories across 11 specific criteria with predefined weights.

IMPORTANCE CRITERIA FOR ETHEREUM ECOSYSTEM REPOSITORIES:

{criteria_section}

For each criterion, provide:
1. A score from 1-10 (where 1 = minimal contribution, 10 = maximum contribution)
2. A weight (you may adjust from defaults if justified, but they should sum to approximately 1.0)
3. Brief reasoning for your assessment

Provide your response in the exact JSON format specified."""
            },
            {
                "role": "user",
                "content": f"""Please assess this repository according to the 11 importance criteria:

Repository: {repo_url}
Name: {repo_name} ({repo_full_name})

{response_format}

Assess each criterion carefully and provide scores, weights, and reasoning. Ensure weights sum to approximately 1.0."""
            }
        ]
        
        return prompt
    
    def _build_criteria_section(self) -> str:
        """Build the criteria description section for the prompt."""
        sections = []
        
        for criterion_id, criterion in self.criteria_template.items():
            section = f"""
**{criterion['name']} (Weight: {criterion['weight']:.0%})**
- **Description**: {criterion['description']}
- **Indicators**: {', '.join(criterion['indicators'])}
- **Scoring guidance**: {criterion['scoring_guidance']}"""
            sections.append(section)
        
        return "\n".join(sections)
    
    def _build_response_format(self) -> str:
        """Build the expected response format section."""
        
        example_criteria = []
        for criterion_id, criterion in self.criteria_template.items():
            example_criteria.append(f'''    "{criterion_id}": {{
      "name": "{criterion['name']}",
      "score": [1-10],
      "weight": {criterion['weight']:.3f},
      "reasoning": "[Brief explanation for the score]"
    }}''')
        
        response_format = f"""
Please respond in this exact JSON format:

{{
  "repository_url": "{'{repo_url}'}",
  "repository_name": "{'{repo_name}'}",
  "criteria_assessments": {{
{',\n'.join(example_criteria)}
  }},
  "assessment_summary": {{
    "total_weight": "[sum of all weights - should be ~1.0]",
    "target_score": "[weighted sum: Σ(weight_i × score_i)]",
    "overall_reasoning": "[Brief overall assessment]"
  }}
}}"""
        
        return response_format
    
    def get_default_weights(self) -> Dict[str, float]:
        """Get the default weights for all criteria."""
        return {criterion_id: criterion['weight'] 
                for criterion_id, criterion in self.criteria_template.items()}
    
    def get_criteria_names(self) -> Dict[str, str]:
        """Get mapping of criterion IDs to human-readable names."""
        return {criterion_id: criterion['name'] 
                for criterion_id, criterion in self.criteria_template.items()}
    
    def validate_weights_sum(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate that weights sum to approximately 1.0.
        
        Args:
            weights: Dictionary of criterion weights
            
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