# src/tasks/criteria/criteria_assessment_pipeline.py
"""
Main pipeline for criteria-based repository assessment.
Contains business logic for orchestrating the complete assessment workflow.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Import existing infrastructure  
from src.shared.multi_model_engine import MultiModelEngine
from src.shared.model_metadata import get_model_metadata

# Import criteria assessment components
from .repo_extractor import RepositoryExtractor
from .criteria_prompt_generator import CriteriaPromptGenerator
from .fuzzy_response_parser import FuzzyCriteriaResponseParser, ParsedCriteriaResponse
from .weight_validator import WeightValidator
from .target_score_calculator import TargetScoreCalculator
from .ratio_comparator import RatioComparator

logger = logging.getLogger(__name__)

class CriteriaAssessmentPipeline:
    """
    Main pipeline for criteria-based repository assessment.
    """
    
    def __init__(self, config):
        """
        Initialize the criteria assessment pipeline.
        
        Args:
            config: Configuration object (OmegaConf format)
        """
        self.config = config
        
        # Generate timestamp for this pipeline run
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Initialize components
        self.repo_extractor = RepositoryExtractor()
        self.prompt_generator = CriteriaPromptGenerator()
        self.response_parser = FuzzyCriteriaResponseParser()
        self.weight_validator = WeightValidator()
        self.score_calculator = TargetScoreCalculator()
        self.ratio_comparator = RatioComparator()
        
        # Initialize LLM engine
        self.llm_engine = MultiModelEngine(config)
        
        # Results storage
        self.assessment_results = {}
        self.target_scores = {}
        
    def run_full_assessment(self, model_id: str = "openai/gpt-4o", 
                          temperature: float = 0.0,
                          train_csv_path: str = "data/raw/train.csv",
                          output_dir: str = None,
                          specific_repos: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete criteria assessment pipeline.
        
        Args:
            model_id: LLM model to use for assessment
            temperature: Sampling temperature
            train_csv_path: Path to training CSV file
            output_dir: Directory to save results (if None, uses timestamped default)
            specific_repos: List of specific repository URLs to assess (if None, assesses all repos from CSV)
            
        Returns:
            Dictionary with pipeline results and summary
        """
        logger.info("Starting criteria assessment pipeline")
        
        try:
            # Create timestamped output directory if not specified
            if output_dir is None:
                output_dir = f"results/{self.timestamp}/criteria_assessment"
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Results will be saved to: {output_path}")
            
            # Step 1: Extract unique repositories
            logger.info("Step 1: Extracting unique repositories")
            if specific_repos:
                repo_urls = specific_repos
                logger.info(f"Using {len(repo_urls)} specific repositories provided")
            else:
                repo_urls = self.repo_extractor.extract_unique_repos(train_csv_path)
                logger.info(f"Found {len(repo_urls)} unique repositories")
            
            # Step 2: Assess each repository
            logger.info("Step 2: Assessing repositories with criteria")
            assessment_results = self._assess_repositories(repo_urls, model_id, temperature)
            
            # Step 3: Calculate target scores
            logger.info("Step 3: Calculating target scores")
            target_scores = self._calculate_target_scores(assessment_results)
            
            # Step 4: Compare with training data
            logger.info("Step 4: Comparing with training data")
            comparison_summary = self.ratio_comparator.compare_with_training_data(
                target_scores, train_csv_path
            )
            
            # Step 5: Save results
            logger.info("Step 5: Saving results")
            self._save_results(assessment_results, target_scores, comparison_summary, output_path)
            
            # Generate summary
            pipeline_summary = {
                "pipeline_metadata": {
                    "model_used": model_id,
                    "temperature": temperature,
                    "timestamp": self.timestamp,
                    "run_datetime": datetime.now().isoformat(),
                    "repositories_assessed": len(repo_urls),
                    "successful_assessments": len(assessment_results),
                    "train_csv_path": train_csv_path,
                    "output_directory": str(output_path)
                },
                "assessment_summary": {
                    "total_repositories": len(repo_urls),
                    "successful_assessments": len(assessment_results),
                    "failed_assessments": len(repo_urls) - len(assessment_results),
                    "average_target_score": sum(target_scores.values()) / len(target_scores) if target_scores else 0.0
                },
                "comparison_summary": {
                    "total_comparisons": comparison_summary.total_comparisons,
                    "directional_accuracy": comparison_summary.directional_accuracy,
                    "mean_ratio_error": comparison_summary.mean_ratio_error,
                    "correlation_coefficient": comparison_summary.correlation_coefficient
                },
                "model_costs": {
                    "total_cost_usd": self.llm_engine.get_total_cost()
                }
            }
            
            logger.info("Criteria assessment pipeline completed successfully")
            return pipeline_summary
            
        except Exception as e:
            logger.error(f"Error in criteria assessment pipeline: {e}")
            raise
    
    def _assess_repositories(self, repo_urls: List[str], model_id: str, 
                           temperature: float) -> Dict[str, ParsedCriteriaResponse]:
        """Assess each repository using the criteria framework."""
        
        assessment_results = {}
        
        for i, repo_url in enumerate(repo_urls, 1):
            try:
                logger.info(f"Assessing repository {i}/{len(repo_urls)}: {repo_url}")
                
                # Get repository info
                repo_info = self.repo_extractor.get_repo_info(repo_url)
                
                # Generate criteria assessment prompt
                prompt = self.prompt_generator.create_criteria_assessment_prompt(repo_info)
                
                # Query LLM
                response = self.llm_engine.query_single_model_with_temperature(
                    model_id, prompt, temperature
                )
                
                if response.success:
                    # Parse response
                    parsed_response = self.response_parser.parse_response(
                        response.content, repo_url, repo_info.get('name', 'unknown')
                    )
                    
                    assessment_results[repo_url] = parsed_response
                    
                    if parsed_response.parsing_success:
                        logger.debug(f"Successfully assessed {repo_url}")
                    else:
                        logger.warning(f"Assessment parsing issues for {repo_url}: {parsed_response.parsing_warnings}")
                else:
                    logger.error(f"LLM query failed for {repo_url}: {response.error}")
                    
            except Exception as e:
                logger.error(f"Error assessing {repo_url}: {e}")
                continue
        
        logger.info(f"Successfully assessed {len(assessment_results)}/{len(repo_urls)} repositories")
        return assessment_results
    
    def _calculate_target_scores(self, assessment_results: Dict[str, ParsedCriteriaResponse]) -> Dict[str, float]:
        """Calculate target scores from assessment results."""
        
        target_scores = {}
        
        for repo_url, assessment in assessment_results.items():
            try:
                # Validate and normalize weights
                weights = {criterion: data.get("weight", 0.0) 
                          for criterion, data in assessment.criteria_scores.items()}
                
                weight_validation = self.weight_validator.validate_and_normalize_weights(weights)
                
                # Update assessment with normalized weights
                for criterion in assessment.criteria_scores:
                    if criterion in weight_validation.normalized_weights:
                        assessment.criteria_scores[criterion]["weight"] = weight_validation.normalized_weights[criterion]
                
                # Calculate target score
                score_result = self.score_calculator.calculate_target_score(assessment.criteria_scores)
                
                target_scores[repo_url] = score_result.target_score
                
                # Log warnings if any
                if weight_validation.validation_warnings:
                    logger.debug(f"Weight validation warnings for {repo_url}: {weight_validation.validation_warnings}")
                
                if score_result.calculation_warnings:
                    logger.debug(f"Score calculation warnings for {repo_url}: {score_result.calculation_warnings}")
                    
            except Exception as e:
                logger.error(f"Error calculating target score for {repo_url}: {e}")
                # Use fallback score
                target_scores[repo_url] = 5.0
        
        logger.info(f"Calculated target scores for {len(target_scores)} repositories")
        return target_scores
    
    def _save_results(self, assessment_results: Dict[str, ParsedCriteriaResponse],
                     target_scores: Dict[str, float],
                     comparison_summary,
                     output_path: Path):
        """Save all results to files."""
        
        # Save detailed assessment results
        detailed_results = []
        for repo_url, assessment in assessment_results.items():
            result_record = {
                "repository_url": repo_url,
                "repository_name": assessment.repository_name,
                "target_score": target_scores.get(repo_url, 0.0),
                "total_weight": assessment.total_weight,
                "parsing_method": assessment.parsing_method,
                "parsing_success": assessment.parsing_success,
                "parsing_warnings": assessment.parsing_warnings,
                "criteria_scores": assessment.criteria_scores,
                "overall_reasoning": assessment.overall_reasoning
            }
            detailed_results.append(result_record)
        
        # Save as JSON
        with open(output_path / "detailed_assessments.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Save target scores summary
        target_scores_summary = [
            {"repository_url": url, "target_score": score}
            for url, score in target_scores.items()
        ]
        
        with open(output_path / "target_scores.json", 'w') as f:
            json.dump(target_scores_summary, f, indent=2)
        
        # Save comparison results
        self.ratio_comparator.save_comparison_results(
            comparison_summary, 
            output_path / "comparison_results.csv"
        )
        
        # Save comparison summary
        comparison_summary_dict = {
            "total_comparisons": comparison_summary.total_comparisons,
            "directional_accuracy": comparison_summary.directional_accuracy,
            "mean_ratio_error": comparison_summary.mean_ratio_error,
            "median_ratio_error": comparison_summary.median_ratio_error,
            "correlation_coefficient": comparison_summary.correlation_coefficient,
            "agreement_by_preference": comparison_summary.agreement_by_preference,
            "analysis_warnings": comparison_summary.analysis_warnings
        }
        
        with open(output_path / "comparison_summary.json", 'w') as f:
            json.dump(comparison_summary_dict, f, indent=2, default=str)
        
        logger.info(f"Saved all results to {output_path}")