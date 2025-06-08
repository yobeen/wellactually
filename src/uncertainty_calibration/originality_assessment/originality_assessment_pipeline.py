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
            from datetime import datetime
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
        
        # Calculate originality scores
        originality_scores = self._calculate_originality_scores(assessment_results)
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(assessment_results, originality_scores)
        
        # Save results
        self._save_results(assessment_results, originality_scores, summary_stats, output_path)
        
        logger.info(f"Originality assessment completed. Results saved to {output_path}")
        
        return {
            'assessment_results': assessment_results,
            'originality_scores': originality_scores,
            'summary_statistics': summary_stats,
            'output_path': str(output_path)
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
            response = self.llm_engine.query_model(
                model_id=model_id,
                messages=prompt_messages,
                temperature=temperature,
                max_tokens=4000
            )
            
            # Parse response
            parsed_response = self.response_parser.parse_response(
                response_text=response.content,
                expected_weights=repo_config['weights'],
                repo_url=repo_url,
                repo_name=repo_config['name'],
                originality_category=repo_config['category']
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
            logger.info(f"Processing repository {i}/{len(repositories)}: {repo_url}")
            
            try:
                assessment = self.assess_single_repository(repo_url, model_id, temperature)
                assessment_results[repo_url] = assessment
                
            except Exception as e:
                logger.error(f"Failed to assess {repo_url}: {e}")
                # Continue with other repositories
                continue
        
        logger.info(f"Successfully assessed {len(assessment_results)}/{len(repositories)} repositories")
        return assessment_results
    
    def _calculate_originality_scores(self, assessment_results: Dict[str, ParsedOriginalityResponse]) -> Dict[str, float]:
        """Calculate final originality scores for all repositories."""
        originality_scores = {}
        
        for repo_url, assessment in assessment_results.items():
            try:
                # Score is already calculated in the assessment
                originality_scores[repo_url] = assessment.final_originality_score
                
            except Exception as e:
                logger.error(f"Error calculating originality score for {repo_url}: {e}")
                # Use fallback score
                originality_scores[repo_url] = 0.5
        
        logger.info(f"Calculated originality scores for {len(originality_scores)} repositories")
        return originality_scores
    
    def _generate_summary_statistics(self, assessment_results: Dict[str, ParsedOriginalityResponse],
                                   originality_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate summary statistics for the assessment results."""
        if not originality_scores:
            return {}
        
        scores = list(originality_scores.values())
        
        # Basic statistics
        summary = {
            'total_repositories': len(assessment_results),
            'successful_assessments': len([r for r in assessment_results.values() if r.parsing_success]),
            'failed_assessments': len([r for r in assessment_results.values() if not r.parsing_success]),
            'score_statistics': {
                'mean': sum(scores) / len(scores),
                'median': sorted(scores)[len(scores) // 2],
                'min': min(scores),
                'max': max(scores),
                'std': self._calculate_std(scores)
            }
        }
        
        # Category breakdown
        category_stats = {}
        for repo_url, assessment in assessment_results.items():
            category = assessment.originality_category
            if category not in category_stats:
                category_stats[category] = {'count': 0, 'scores': []}
            
            category_stats[category]['count'] += 1
            category_stats[category]['scores'].append(originality_scores.get(repo_url, 0.5))
        
        # Calculate category averages
        for category, stats in category_stats.items():
            if stats['scores']:
                stats['average_score'] = sum(stats['scores']) / len(stats['scores'])
            else:
                stats['average_score'] = 0.5
        
        summary['category_breakdown'] = category_stats
        
        # Parsing method statistics
        parsing_methods = {}
        for assessment in assessment_results.values():
            method = assessment.parsing_method
            if method not in parsing_methods:
                parsing_methods[method] = 0
            parsing_methods[method] += 1
        
        summary['parsing_methods'] = parsing_methods
        
        # Top and bottom repositories by score
        sorted_repos = sorted(originality_scores.items(), key=lambda x: x[1], reverse=True)
        summary['top_repositories'] = sorted_repos[:5]
        summary['bottom_repositories'] = sorted_repos[-5:]
        
        return summary
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _save_results(self, assessment_results: Dict[str, ParsedOriginalityResponse],
                     originality_scores: Dict[str, float],
                     summary_stats: Dict[str, Any],
                     output_path: Path):
        """Save all results to files."""
        
        # Save detailed assessment results
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
        
        # Save as JSON
        with open(output_path / "detailed_originality_assessments.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Save originality scores summary
        originality_scores_summary = [
            {"repository_url": url, "originality_score": score}
            for url, score in originality_scores.items()
        ]
        
        with open(output_path / "originality_scores.json", 'w') as f:
            json.dump(originality_scores_summary, f, indent=2)
        
        # Save summary statistics
        with open(output_path / "originality_summary.json", 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
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
        
        logger.info(f"Saved all results to {output_path}")

def main():
    """Main entry point for originality assessment."""
    import argparse
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser(description='Run originality assessment pipeline')
    parser.add_argument('--config', default='configs/config.yaml', help='Configuration file path')
    parser.add_argument('--model', default='openai/gpt-4o', help='Model ID to use')
    parser.add_argument('--output', default='results/originality_assessment', help='Output directory')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    pipeline = OriginalityAssessmentPipeline(config)
    results = pipeline.run_full_assessment(
        model_id=args.model,
        temperature=args.temperature,
        output_dir=args.output
    )
    
    print(f"\nOriginality Assessment Complete!")
    print(f"Results saved to: {results['output_path']}")
    print(f"Repositories assessed: {results['summary_statistics']['total_repositories']}")
    print(f"Mean originality score: {results['summary_statistics']['score_statistics']['mean']:.3f}")

if __name__ == "__main__":
    main()