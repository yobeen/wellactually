# src/uncertainty_calibration/originality_assessment/originality_assessment_pipeline.py
"""
Originality Assessment Pipeline

Main orchestrator for the originality assessment process, following the same pattern
as the criteria assessment but focused on evaluating repository originality.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import existing infrastructure
from src.uncertainty_calibration.multi_model_engine import MultiModelEngine
from src.uncertainty_calibration.model_metadata import get_model_metadata

# Import originality assessment components
from .originality_prompt_generator import OriginalityPromptGenerator
from .originality_response_parser import OriginalityResponseParser, ParsedOriginalityResponse

logger = logging.getLogger(__name__)

class OriginalityAssessmentPipeline:
    """
    Main pipeline for originality-based repository assessment.
    """
    
    def __init__(self, config):
        """
        Initialize the originality assessment pipeline.
        
        Args:
            config: Configuration object (OmegaConf format)
        """
        self.config = config
        
        # Initialize components
        self.prompt_generator = OriginalityPromptGenerator()
        self.response_parser = OriginalityResponseParser()
        
        # Initialize LLM engine
        self.llm_engine = MultiModelEngine(config)
        
        # Results storage
        self.assessment_results = {}
        self.originality_scores = {}
        
    def run_full_assessment(self, model_id: str = "openai/gpt-4o", 
                          temperature: float = 0.0,
                          repositories: Optional[List[str]] = None,
                          output_dir: str = None) -> Dict[str, Any]:
        """
        Run the complete originality assessment pipeline.
        
        Args:
            model_id: LLM model to use for assessment
            temperature: Sampling temperature for LLM
            repositories: List of repository URLs to assess (None = all seed repos)
            output_dir: Directory to save results
            
        Returns:
            Dictionary with assessment results and summary statistics
        """
        logger.info(f"Starting originality assessment pipeline with model {model_id}")
        
        # Create timestamped output directory if not specified
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/originality_assessment_{timestamp}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load repositories to assess
        if repositories is None:
            repositories = self._load_seed_repositories()
        
        logger.info(f"Assessing {len(repositories)} repositories for originality")
        
        # Run assessments
        assessment_results = self._run_assessments(repositories, model_id, temperature)
        
        # Calculate originality scores and save detailed results
        originality_scores = self._calculate_originality_scores(assessment_results, output_path)
        
        logger.info(f"Originality assessment completed. Results saved to {output_path}")
        
        return {
            'assessment_results': assessment_results,
            'originality_scores': originality_scores,
            'output_path': str(output_path),
            'total_repositories': len(repositories),
            'successful_assessments': len(assessment_results),
            'failed_assessments': len(repositories) - len(assessment_results)
        }
    
    def assess_single_repository(self, repo_url: str, model_id: str = "openai/gpt-4o", 
                               temperature: float = 0.0) -> ParsedOriginalityResponse:
        """
        Assess a single repository for originality.
        
        Args:
            repo_url: Repository URL to assess
            model_id: LLM model to use
            temperature: Sampling temperature
            
        Returns:
            ParsedOriginalityResponse with assessment results
        """
        logger.info(f"Assessing repository: {repo_url}")
        
        try:
            # Get repository configuration
            repo_config = self.prompt_generator.get_repo_originality_config(repo_url)
            
            # Generate prompt
            prompt_messages = self.prompt_generator.create_originality_assessment_prompt(repo_url)
            
            # Query LLM
            response = self.llm_engine.query_single_model_with_temperature(
                model_id=model_id,
                prompt=prompt_messages,  # FIXED: use 'prompt' parameter name
                temperature=temperature
            )
            
            # Check if query was successful
            if not response.success:
                raise ValueError(f"LLM query failed: {response.error}")
            
            # Parse response - FIXED parameter mapping
            parsed_response = self.response_parser.parse_response(
                response_text=response.content,
                expected_weights=repo_config['weights'],  # FIXED: use 'weights' key
                repo_url=repo_url,
                repo_name=repo_config['name'],
                originality_category=repo_config['category']  # FIXED: use 'category' key
            )
            
            logger.info(f"Successfully assessed {repo_url} - Score: {parsed_response.final_originality_score:.3f}")
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error assessing repository {repo_url}: {e}")
            raise
    
    def _load_seed_repositories(self) -> List[str]:
        """Load seed repository URLs from configuration."""
        try:
            with open("configs/seed_repositories.yaml", 'r') as f:
                config = yaml.safe_load(f)
            
            repositories = []
            for repo in config.get('seed_repositories', []):
                if repo.get('url') and repo.get('originality_category'):
                    repositories.append(repo['url'])
            
            logger.info(f"Loaded {len(repositories)} seed repositories")
            return repositories
            
        except Exception as e:
            logger.error(f"Failed to load seed repositories: {e}")
            raise
    
    def _run_assessments(self, repositories: List[str], model_id: str, 
                        temperature: float) -> Dict[str, ParsedOriginalityResponse]:
        """Run originality assessments for all repositories."""
        assessment_results = {}
        
        for i, repo_url in enumerate(repositories, 1):
            try:
                logger.info(f"[{i}/{len(repositories)}] Assessing: {repo_url}")
                
                result = self.assess_single_repository(
                    repo_url=repo_url,
                    model_id=model_id,
                    temperature=temperature
                )
                
                assessment_results[repo_url] = result
                logger.info(f"✓ Completed {repo_url} - Score: {result.final_originality_score:.3f}")
                
            except Exception as e:
                logger.error(f"✗ Failed to assess {repo_url}: {e}")
                # Continue with next repository
                continue
        
        logger.info(f"Assessment completed: {len(assessment_results)}/{len(repositories)} successful")
        return assessment_results
    
    def _calculate_originality_scores(self, assessment_results: Dict[str, ParsedOriginalityResponse],
                                    output_path: Path) -> Dict[str, float]:
        """
        Calculate originality scores and save detailed JSON results (like criteria assessment).
        
        Args:
            assessment_results: Dictionary of assessment results
            output_path: Path to save detailed results
            
        Returns:
            Dictionary mapping repo URLs to originality scores
        """
        logger.info("Calculating originality scores and saving detailed results")
        
        # Extract scores
        originality_scores = {}
        for repo_url, assessment in assessment_results.items():
            originality_scores[repo_url] = assessment.final_originality_score
        
        # Save detailed assessment results (following criteria assessment pattern)
        detailed_results = []
        for repo_url, assessment in assessment_results.items():
            result_record = {
                "repository_url": repo_url,
                "repository_name": assessment.repository_name,
                "originality_category": assessment.originality_category,
                "final_originality_score": assessment.final_originality_score,
                "assessment_confidence": assessment.assessment_confidence,
                "parsing_method": assessment.parsing_method,
                "parsing_success": assessment.parsing_success,
                "parsing_warnings": assessment.parsing_warnings,
                "criteria_scores": assessment.criteria_scores,
                "overall_reasoning": assessment.overall_reasoning
            }
            detailed_results.append(result_record)
        
        # Save as JSON (detailed LLM results)
        with open(output_path / "detailed_originality_assessments.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Save originality scores summary
        originality_scores_summary = [
            {"repository_url": url, "originality_score": score}
            for url, score in originality_scores.items()
        ]
        
        with open(output_path / "originality_scores.json", 'w') as f:
            json.dump(originality_scores_summary, f, indent=2)
        
        # Save CSV for easy analysis
        import csv
        with open(output_path / "originality_scores.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['repository_url', 'repository_name', 'originality_category', 'originality_score', 'confidence'])
            
            for repo_url, assessment in assessment_results.items():
                writer.writerow([
                    repo_url,
                    assessment.repository_name,
                    assessment.originality_category,
                    assessment.final_originality_score,
                    assessment.assessment_confidence
                ])
        
        logger.info(f"Saved detailed results to {output_path}")
        return originality_scores
