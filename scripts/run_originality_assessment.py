# scripts/run_originality_assessment.py
"""
Main execution script for originality-based repository assessment.
Orchestrates the complete pipeline from seed repositories to final assessment results.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.tasks.originality.originality_assessment_pipeline import OriginalityAssessmentPipeline

logger = logging.getLogger(__name__)

def main():
    """Main entry point for originality assessment."""
    import argparse
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser(description='Run originality assessment pipeline')
    parser.add_argument('--config', default='configs/uncertainty_calibration/llm.yaml', 
                        help='Configuration file path')
    parser.add_argument('--model', default='openai/gpt-4o', 
                        help='Model ID to use (e.g., openai/gpt-4o, anthropic/claude-3-sonnet)')
    parser.add_argument('--output', default=None, 
                        help='Output directory (default: timestamped results/originality_assessment_YYYYMMDD_HHMMSS)')
    parser.add_argument('--temperature', type=float, default=0.0, 
                        help='Sampling temperature for LLM (0.0 = deterministic)')
    parser.add_argument('--repositories', nargs='+', default=None,
                        help='Specific repository URLs to assess (default: all seed repositories)')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key:")
        print("export OPENROUTER_API_KEY='your-api-key-here'")
        return 1
    
    # Load configuration
    try:
        config = OmegaConf.load(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Run assessment
    try:
        logger.info("="*60)
        logger.info("STARTING ORIGINALITY ASSESSMENT PIPELINE")
        logger.info("="*60)
        
        pipeline = OriginalityAssessmentPipeline(config)
        
        # Run with specified parameters
        results = pipeline.run_full_assessment(
            model_id=args.model,
            temperature=args.temperature,
            repositories=args.repositories,
            output_dir=args.output
        )
        
        # Print summary
        print("\n" + "="*60)
        print("ORIGINALITY ASSESSMENT RESULTS")
        print("="*60)
        print(f"Model used: {args.model}")
        print(f"Temperature: {args.temperature}")
        print(f"Repositories assessed: {results['successful_assessments']}/{results['total_repositories']}")
        print(f"Failed assessments: {results['failed_assessments']}")
        
        if results['originality_scores']:
            scores = list(results['originality_scores'].values())
            print(f"Mean originality score: {sum(scores)/len(scores):.3f}")
            print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
            print(f"Standard deviation: {(sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5:.3f}")
        
        print(f"Total cost: ${pipeline.llm_engine.get_total_cost():.2f}")
        print(f"Results saved to: {results['output_path']}")
        
        # Show top and bottom repositories by originality score
        if results['originality_scores']:
            print("\n" + "-"*40)
            print("TOP 5 MOST ORIGINAL REPOSITORIES:")
            print("-"*40)
            sorted_scores = sorted(results['originality_scores'].items(), key=lambda x: x[1], reverse=True)
            for i, (repo_url, score) in enumerate(sorted_scores[:5], 1):
                repo_name = repo_url.split('/')[-1]
                print(f"{i}. {repo_name}: {score:.3f}")
            
            print("\n" + "-"*40)
            print("BOTTOM 5 LEAST ORIGINAL REPOSITORIES:")
            print("-"*40)
            for i, (repo_url, score) in enumerate(sorted_scores[-5:], 1):
                repo_name = repo_url.split('/')[-1]
                print(f"{i}. {repo_name}: {score:.3f}")
        
        print("\n" + "="*60)
        print("Files saved:")
        print(f"  • detailed_originality_assessments.json - Full LLM responses and analysis")
        print(f"  • originality_scores.json - Summary scores")
        print(f"  • originality_scores.csv - Spreadsheet format")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nAssessment interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Error running originality assessment: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())