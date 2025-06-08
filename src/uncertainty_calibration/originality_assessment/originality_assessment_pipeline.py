# src/uncertainty_calibration/originality_assessment/originality_assessment_pipeline.py
"""
Enhanced Originality Assessment Pipeline with Uncertainty Calibration

Main orchestrator for the originality assessment process with perplexity-based uncertainty
calculation for both individual criteria and overall reasoning assessments.
"""

import os
import json
import yaml
import logging
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics

# Import existing infrastructure
from src.uncertainty_calibration.multi_model_engine import MultiModelEngine
from src.uncertainty_calibration.model_metadata import get_model_metadata

# Import enhanced originality assessment components
from .originality_prompt_generator import OriginalityPromptGenerator
from .originality_response_parser import OriginalityResponseParser, ParsedOriginalityResponse

logger = logging.getLogger(__name__)

class OriginalityAssessmentPipeline:
    """
    Enhanced pipeline for originality-based repository assessment with uncertainty calculation.
    """
    
    def __init__(self, config):
        """
        Initialize the enhanced originality assessment pipeline.
        
        Args:
            config: Configuration object (OmegaConf format)
        """
        self.config = config
        
        # Generate timestamp for this pipeline run
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Initialize components
        self.prompt_generator = OriginalityPromptGenerator()
        self.response_parser = OriginalityResponseParser()
        
        # Initialize LLM engine with logprobs enabled
        self.llm_engine = MultiModelEngine(config)
        
        # Results storage
        self.assessment_results = {}
        self.originality_scores = {}
        self.uncertainty_metrics = {}
        
    def run_full_assessment(self, model_id: str = "openai/gpt-4o", 
                          temperature: float = 0.0,
                          repositories: Optional[List[str]] = None,
                          output_dir: str = None) -> Dict[str, Any]:
        """
        Run the complete originality assessment pipeline with uncertainty calculation.
        
        Args:
            model_id: Model identifier for LLM
            temperature: Generation temperature
            repositories: List of repository URLs to assess (if None, uses seed repositories)
            output_dir: Output directory for results
            
        Returns:
            Dictionary with assessment results, scores, and uncertainty metrics
        """
        try:
            logger.info("Starting enhanced originality assessment pipeline")
            
            # Determine repositories to assess
            if repositories is None:
                repositories = self._get_seed_repositories()
            
            logger.info(f"Assessing {len(repositories)} repositories")
            
            # Create output directory
            if output_dir is None:
                output_dir = f"results/originality_assessment/{self.timestamp}"
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Assess each repository
            assessment_results = self._assess_repositories_with_uncertainty(
                repositories, model_id, temperature
            )
            
            # Calculate originality scores
            originality_scores = self._calculate_originality_scores(assessment_results)
            
            # Calculate uncertainty metrics
            uncertainty_metrics = self._calculate_uncertainty_metrics(assessment_results)
            
            # Save enhanced results with uncertainty
            self._save_enhanced_results(
                assessment_results, originality_scores, uncertainty_metrics, output_path
            )
            
            # Generate summary
            pipeline_summary = {
                "pipeline_metadata": {
                    "timestamp": self.timestamp,
                    "model_id": model_id,
                    "temperature": temperature,
                    "output_directory": str(output_path)
                },
                "assessment_summary": {
                    "total_repositories": len(repositories),
                    "successful_assessments": len(assessment_results),
                    "failed_assessments": len(repositories) - len(assessment_results),
                    "average_originality_score": statistics.mean(originality_scores.values()) if originality_scores else 0.0
                },
                "uncertainty_summary": uncertainty_metrics.get("summary", {}),
                "originality_scores": originality_scores,
                "model_costs": {
                    "total_cost_usd": self.llm_engine.get_total_cost()
                }
            }
            
            logger.info("Enhanced originality assessment pipeline completed successfully")
            return pipeline_summary
            
        except Exception as e:
            logger.error(f"Error in enhanced originality assessment pipeline: {e}")
            raise
    
    def _assess_repositories_with_uncertainty(self, repo_urls: List[str], model_id: str, 
                                            temperature: float) -> Dict[str, ParsedOriginalityResponse]:
        """
        Assess each repository with uncertainty calculation.
        
        Args:
            repo_urls: List of repository URLs to assess
            model_id: Model identifier
            temperature: Generation temperature
            
        Returns:
            Dictionary mapping repo URLs to parsed assessment responses with uncertainty
        """
        assessment_results = {}
        
        for i, repo_url in enumerate(repo_urls, 1):
            try:
                logger.info(f"Assessing repository {i}/{len(repo_urls)}: {repo_url}")
                
                # Get repository configuration
                repo_config = self.prompt_generator.get_repo_originality_config(repo_url)
                expected_weights = repo_config.get('weights', {})
                category = repo_config.get('category', 'Unknown')
                repo_name = repo_config.get('name', repo_url.split('/')[-1])
                
                # Generate assessment prompt
                messages = self.prompt_generator.create_originality_assessment_prompt(repo_url)
                
                # Get LLM response with logprobs enabled (using custom method for longer responses)
                response = self._query_originality_assessment(
                    model_id=model_id,
                    messages=messages,
                    temperature=temperature
                )
                
                # Check if response was successful
                if not response.success:
                    logger.error(f"LLM query failed for {repo_url}: {response.error}")
                    continue
                
                # Extract response components
                response_text = response.content
                logprobs_data = response.full_content_logprobs or []
                
                # Parse response with uncertainty calculation
                parsed_response = self.response_parser.parse_response(
                    response_text=response_text,
                    expected_weights=expected_weights,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    originality_category=category,
                    logprobs_data=logprobs_data  # Pass logprobs for perplexity calculation
                )
                
                assessment_results[repo_url] = parsed_response
                
                # Log uncertainty information
                logger.info(f"Assessment completed for {repo_name}")
                logger.debug(f"  Originality score: {parsed_response.final_originality_score:.3f}")
                logger.debug(f"  Overall uncertainty: {parsed_response.overall_reasoning_uncertainty:.3f}")
                logger.debug(f"  Aggregate uncertainty: {parsed_response.aggregate_uncertainty:.3f}")
                
            except Exception as e:
                logger.error(f"Error assessing repository {repo_url}: {e}")
                continue
        
        logger.info(f"Successfully assessed {len(assessment_results)} repositories")
        return assessment_results
    
    def _query_originality_assessment(self, model_id: str, messages: List[Dict[str, str]], 
                                    temperature: float) -> 'ModelResponse':
        """
        Custom query method for originality assessments that need longer responses.
        Works around the MultiModelEngine's hardcoded max_tokens=10 limitation.
        
        Args:
            model_id: Model identifier
            messages: Message list in OpenAI format
            temperature: Sampling temperature
            
        Returns:
            ModelResponse with full originality assessment
        """
        try:
            # Create custom payload for originality assessment with higher max_tokens
            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": 4000,  # Much higher for detailed originality assessments
                "temperature": temperature,
                "logprobs": True,
                "top_logprobs": 5
            }
            
            # Add provider filtering if configured (copied from MultiModelEngine)
            if hasattr(self.llm_engine.api_config.openrouter, 'providers') and self.llm_engine.api_config.openrouter.providers:
                providers = list(self.llm_engine.api_config.openrouter.providers)
                filtered_providers = self.llm_engine._filter_providers_for_model(model_id, providers)
                if filtered_providers:
                    payload["provider"] = {
                        "only": filtered_providers
                    }
            
            # Apply rate limiting
            self.llm_engine._wait_if_needed()
            
            # Make API request with retry logic
            api_response = None
            for attempt in range(self.llm_engine.max_retries + 1):
                try:
                    response = requests.post(
                        self.llm_engine.base_url,
                        headers=self.llm_engine.headers,
                        json=payload,
                        timeout=self.llm_engine.timeout
                    )
                    
                    if response.status_code == 200:
                        api_response = response.json()
                        break
                    elif response.status_code == 429:  # Rate limited
                        if attempt < self.llm_engine.max_retries:
                            wait_time = (self.llm_engine.backoff_factor ** attempt) * 60
                            logger.warning(f"Rate limited for {model_id}, waiting {wait_time:.1f}s")
                            time.sleep(wait_time)
                            continue
                        else:
                            api_response = self.llm_engine._create_api_error_response(
                                f"Rate limit exceeded after {self.llm_engine.max_retries} retries"
                            )
                            break
                    else:  # Other HTTP error
                        error_msg = f"HTTP {response.status_code}: {response.text}"
                        if attempt < self.llm_engine.max_retries:
                            logger.warning(f"HTTP error for {model_id}: {error_msg}, retrying...")
                            time.sleep(self.llm_engine.backoff_factor ** attempt)
                            continue
                        else:
                            api_response = self.llm_engine._create_api_error_response(error_msg)
                            break
                            
                except Exception as e:
                    error_msg = f"Request exception: {str(e)}"
                    if attempt < self.llm_engine.max_retries:
                        logger.warning(f"Exception for {model_id}: {error_msg}, retrying...")
                        time.sleep(self.llm_engine.backoff_factor ** attempt)
                        continue
                    else:
                        api_response = self.llm_engine._create_api_error_response(error_msg)
                        break
            
            if api_response is None:
                api_response = self.llm_engine._create_api_error_response("Unexpected error in retry loop")
            
            # Parse response using the engine's response parser
            parsed_response = self.llm_engine.response_parser.parse_response(
                model_id, api_response, temperature
            )
            
            # Update cost tracking
            self.llm_engine.total_cost += parsed_response.cost_usd
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error in custom originality query: {e}")
            # Return error response
            from src.uncertainty_calibration.response_parser import ModelResponse
            return ModelResponse(
                model_id=model_id,
                success=False,
                content="",
                logprobs=None,
                uncertainty=1.0,
                raw_choice="",
                cost_usd=0.0,
                tokens_used=0,
                temperature=temperature,
                error=str(e),
                timestamp="",
                answer_token_info={"extraction_success": False, "method": "error"},
                full_content_logprobs=None
            )
    
    def _calculate_originality_scores(self, assessment_results: Dict[str, ParsedOriginalityResponse]) -> Dict[str, float]:
        """Extract originality scores from assessment results."""
        originality_scores = {}
        
        for repo_url, assessment in assessment_results.items():
            originality_scores[repo_url] = assessment.final_originality_score
        
        return originality_scores
    
    def _calculate_uncertainty_metrics(self, assessment_results: Dict[str, ParsedOriginalityResponse]) -> Dict[str, Any]:
        """
        Calculate comprehensive uncertainty metrics from assessment results.
        
        Args:
            assessment_results: Dictionary of assessment responses with uncertainty
            
        Returns:
            Dictionary with detailed uncertainty analysis
        """
        if not assessment_results:
            return {"summary": {}, "per_repository": {}, "per_criterion": {}}
        
        # Collect uncertainty data
        overall_uncertainties = []
        aggregate_uncertainties = []
        criteria_uncertainties_by_criterion = {criterion: [] for criterion in self.response_parser.default_criteria}
        per_repo_metrics = {}
        
        for repo_url, assessment in assessment_results.items():
            # Overall reasoning uncertainty
            overall_uncertainties.append(assessment.overall_reasoning_uncertainty)
            
            # Aggregate uncertainty
            aggregate_uncertainties.append(assessment.aggregate_uncertainty)
            
            # Per-criterion uncertainties
            for criterion, uncertainty in assessment.criteria_uncertainties.items():
                if criterion in criteria_uncertainties_by_criterion:
                    criteria_uncertainties_by_criterion[criterion].append(uncertainty)
            
            # Per-repository metrics
            repo_name = repo_url.split('/')[-1]
            per_repo_metrics[repo_name] = {
                "overall_reasoning_uncertainty": assessment.overall_reasoning_uncertainty,
                "aggregate_uncertainty": assessment.aggregate_uncertainty,
                "criteria_uncertainties": assessment.criteria_uncertainties,
                "originality_score": assessment.final_originality_score,
                "parsing_method": assessment.parsing_method,
                "parsing_success": assessment.parsing_success
            }
        
        # Calculate summary statistics
        uncertainty_summary = {
            "overall_reasoning": {
                "mean": statistics.mean(overall_uncertainties),
                "median": statistics.median(overall_uncertainties),
                "std": statistics.stdev(overall_uncertainties) if len(overall_uncertainties) > 1 else 0.0,
                "min": min(overall_uncertainties),
                "max": max(overall_uncertainties)
            },
            "aggregate_criteria": {
                "mean": statistics.mean(aggregate_uncertainties),
                "median": statistics.median(aggregate_uncertainties),
                "std": statistics.stdev(aggregate_uncertainties) if len(aggregate_uncertainties) > 1 else 0.0,
                "min": min(aggregate_uncertainties),
                "max": max(aggregate_uncertainties)
            },
            "parsing_success_rate": sum(1 for a in assessment_results.values() if a.parsing_success) / len(assessment_results)
        }
        
        # Calculate per-criterion uncertainty statistics
        per_criterion_stats = {}
        for criterion, uncertainties in criteria_uncertainties_by_criterion.items():
            if uncertainties:
                per_criterion_stats[criterion] = {
                    "mean": statistics.mean(uncertainties),
                    "median": statistics.median(uncertainties),
                    "std": statistics.stdev(uncertainties) if len(uncertainties) > 1 else 0.0,
                    "count": len(uncertainties)
                }
        
        return {
            "summary": uncertainty_summary,
            "per_repository": per_repo_metrics,
            "per_criterion": per_criterion_stats
        }
    
    def _save_enhanced_results(self, assessment_results: Dict[str, ParsedOriginalityResponse],
                             originality_scores: Dict[str, float],
                             uncertainty_metrics: Dict[str, Any],
                             output_path: Path):
        """
        Save enhanced results with uncertainty information to files.
        
        Args:
            assessment_results: Assessment results with uncertainty
            originality_scores: Calculated originality scores
            uncertainty_metrics: Calculated uncertainty metrics
            output_path: Output directory path
        """
        # Save detailed assessments with uncertainty
        detailed_assessments = []
        for repo_url, assessment in assessment_results.items():
            assessment_dict = {
                "repository_url": assessment.repository_url,
                "repository_name": assessment.repository_name,
                "originality_category": assessment.originality_category,
                "final_originality_score": assessment.final_originality_score,
                "assessment_confidence": assessment.assessment_confidence,
                
                # Uncertainty metrics
                "overall_reasoning_uncertainty": assessment.overall_reasoning_uncertainty,
                "aggregate_uncertainty": assessment.aggregate_uncertainty,
                "criteria_uncertainties": assessment.criteria_uncertainties,
                
                # Detailed criteria
                "criteria_scores": assessment.criteria_scores,
                "overall_reasoning": assessment.overall_reasoning,
                
                # Parsing metadata
                "parsing_method": assessment.parsing_method,
                "parsing_success": assessment.parsing_success,
                "parsing_warnings": assessment.parsing_warnings
            }
            detailed_assessments.append(assessment_dict)
        
        # Save files
        with open(output_path / "detailed_originality_assessments_with_uncertainty.json", 'w') as f:
            json.dump(detailed_assessments, f, indent=2)
        
        with open(output_path / "originality_scores.json", 'w') as f:
            json.dump(originality_scores, f, indent=2)
        
        with open(output_path / "uncertainty_metrics.json", 'w') as f:
            json.dump(uncertainty_metrics, f, indent=2)
        
        # Create CSV with scores and uncertainties
        import pandas as pd
        
        csv_data = []
        for repo_url, assessment in assessment_results.items():
            csv_data.append({
                "repository_url": repo_url,
                "repository_name": assessment.repository_name,
                "originality_category": assessment.originality_category,
                "originality_score": assessment.final_originality_score,
                "overall_reasoning_uncertainty": assessment.overall_reasoning_uncertainty,
                "aggregate_uncertainty": assessment.aggregate_uncertainty,
                "assessment_confidence": assessment.assessment_confidence,
                "parsing_success": assessment.parsing_success,
                "parsing_method": assessment.parsing_method
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path / "originality_scores_with_uncertainty.csv", index=False)
        
        logger.info(f"Enhanced results saved to {output_path}")
        logger.info(f"Created files:")
        logger.info(f"  - detailed_originality_assessments_with_uncertainty.json")
        logger.info(f"  - originality_scores.json") 
        logger.info(f"  - uncertainty_metrics.json")
        logger.info(f"  - originality_scores_with_uncertainty.csv")
    
    def _get_seed_repositories(self) -> List[str]:
        """Get list of seed repositories from configuration."""
        try:
            with open(self.prompt_generator.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            repositories = []
            for repo in config.get('seed_repositories', []):
                if repo.get('url'):
                    repositories.append(repo['url'])
            
            return repositories
            
        except Exception as e:
            logger.error(f"Error loading seed repositories: {e}")
            return []