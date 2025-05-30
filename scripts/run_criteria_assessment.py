#!/usr/bin/env python3
# scripts/run_criteria_assessment.py
"""
Standalone script to run criteria-based repository assessment.
Evaluates repositories against 11 importance criteria and compares with human preferences.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.uncertainty_calibration.criteria_assessment import CriteriaAssessmentPipeline

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_config(config_path: str = "configs/uncertainty_calibration/llm.yaml"):
    """Load configuration from YAML file."""
    try:
        config = OmegaConf.load(config_path)
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)

def validate_environment():
    """Validate that required environment variables are set."""
    required_vars = ["OPENROUTER_API_KEY"]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables and try again.")
        print("Example: export OPENROUTER_API_KEY=your_api_key_here")
        sys.exit(1)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run criteria-based repository assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic assessment with default settings (timestamped output)
  python scripts/run_criteria_assessment.py
  
  # Use different model and temperature
  python scripts/run_criteria_assessment.py --model "meta-llama/llama-4-maverick" --temperature 0.2
  
  # Custom input/output paths
  python scripts/run_criteria_assessment.py --train-csv "data/raw/train.csv" --output-dir "my_results"
  
  # Verbose logging
  python scripts/run_criteria_assessment.py --verbose
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model", 
        type=str, 
        default="openai/gpt-4o",
        help="LLM model to use for assessment (default: openai/gpt-4o)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    
    # Data paths
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/raw/train.csv",
        help="Path to training CSV file (default: data/raw/train.csv)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/YYYY-MM-DD_HH-MM-SS/criteria_assessment)"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/uncertainty_calibration/llm.yaml",
        help="Path to configuration file (default: configs/uncertainty_calibration/llm.yaml)"
    )
    
    # Options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without running assessment"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate environment
    logger.info("Validating environment...")
    validate_environment()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Validate input files
    if not Path(args.train_csv).exists():
        print(f"Error: Training CSV file not found: {args.train_csv}")
        sys.exit(1)
    
    # Initialize pipeline to get timestamp for display
    pipeline = CriteriaAssessmentPipeline(config)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"results/{pipeline.timestamp}/criteria_assessment"
    
    # Print configuration
    print("="*60)
    print("CRITERIA ASSESSMENT CONFIGURATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Training CSV: {args.train_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Configuration: {args.config}")
    print(f"Verbose logging: {args.verbose}")
    print(f"Pipeline timestamp: {pipeline.timestamp}")
    print("="*60)
    
    if args.dry_run:
        print("Dry run completed successfully. All configurations are valid.")
        return 0
    
    try:
        # Run assessment
        logger.info("Starting criteria assessment...")
        results = pipeline.run_full_assessment(
            model_id=args.model,
            temperature=args.temperature,
            train_csv_path=args.train_csv,
            output_dir=args.output_dir  # Will use pipeline's timestamped default if None
        )
        
        # Print results summary
        print("\n" + "="*60)
        print("CRITERIA ASSESSMENT RESULTS")
        print("="*60)
        
        # Pipeline metadata
        metadata = results['pipeline_metadata']
        print(f"Model used: {metadata['model_used']}")
        print(f"Temperature: {metadata['temperature']}")
        print(f"Pipeline timestamp: {metadata['timestamp']}")
        print(f"Run completed: {metadata['run_datetime']}")
        print(f"Output directory: {metadata['output_directory']}")
        print()
        
        # Assessment summary
        assessment = results['assessment_summary']
        print(f"Repositories processed: {assessment['total_repositories']}")
        print(f"Successful assessments: {assessment['successful_assessments']}")
        print(f"Failed assessments: {assessment['failed_assessments']}")
        print(f"Average target score: {assessment['average_target_score']:.3f}")
        print()
        
        # Comparison summary
        comparison = results['comparison_summary']
        print(f"Training comparisons analyzed: {comparison['total_comparisons']}")
        print(f"Directional accuracy: {comparison['directional_accuracy']:.1%}")
        print(f"Mean ratio error: {comparison['mean_ratio_error']:.3f}")
        print(f"Correlation with human preferences: {comparison['correlation_coefficient']:.3f}")
        print()
        
        # Cost summary
        costs = results['model_costs']
        print(f"Total API cost: ${costs['total_cost_usd']:.2f}")
        print()
        
        # File outputs
        output_path = Path(metadata['output_directory'])
        print("Generated files:")
        print(f"  - {output_path / 'detailed_assessments.json'}")
        print(f"  - {output_path / 'target_scores.json'}")
        print(f"  - {output_path / 'comparison_results.csv'}")
        print(f"  - {output_path / 'comparison_summary.json'}")
        
        print("\n" + "="*60)
        print("Assessment completed successfully!")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nAssessment interrupted by user.")
        return 130
        
    except Exception as e:
        logger.error(f"Error during criteria assessment: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())