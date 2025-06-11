#!/usr/bin/env python3
"""
Script to generate all L1 comparison pairs and their responses.

This script:
1. Extracts all repository URLs from criteria assessment data
2. Generates all possible pairs (~450 combinations)
3. Calculates comparison responses using the same logic as the API
4. Logs missing assessments for repos not in criteria data
5. Saves results to JSON file for fast extraction

Usage:
    python scripts/generate_l1_comparison_pairs.py
"""

import json
import itertools
import logging
import time
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import argparse
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CriteriaAssessment:
    """Container for repository criteria assessment data."""
    repository_url: str
    repository_name: str
    target_score: float
    total_weight: float
    criteria_scores: Dict[str, Dict[str, Any]]
    overall_reasoning: str
    parsing_method: str
    parsing_success: bool
    parsing_warnings: List[str]

@dataclass
class ComparisonResult:
    """Container for comparison results."""
    choice: str  # "A" or "B" or "Equal"
    multiplier: float
    raw_uncertainty: float
    calibrated_uncertainty: float
    explanation: str
    model_used: str
    temperature: float
    processing_time_ms: float
    method: str
    comparison_level: str
    # Additional metadata
    score_a: float
    score_b: float
    ratio: float
    repo_a: str
    repo_b: str

class L1ComparisonGenerator:
    """Generate and collect all L1 comparison pairs using criteria assessment logic."""
    
    def __init__(self, 
                 assessments_file: str = "data/processed/criteria_assessment/detailed_assessments.json",
                 level1_list_file: str = "data/processed/criteria_assessment/level1list.csv",
                 output_file: str = "data/processed/l1_comparison_results.json"):
        self.assessments_file = assessments_file
        self.level1_list_file = level1_list_file
        self.output_file = output_file
        self.assessments: Dict[str, CriteriaAssessment] = {}
        self.repositories = []
        self.results = []
        self.missing_repos = []
        
        # Comparison parameters
        self.min_multiplier = 1.0
        self.max_multiplier = 999.0
        
    def load_assessments(self) -> Dict[str, CriteriaAssessment]:
        """Load repository assessments from JSON file."""
        try:
            with open(self.assessments_file, 'r') as f:
                data = json.load(f)
            
            assessments = {}
            for item in data:
                try:
                    assessment = CriteriaAssessment(
                        repository_url=item.get("repository_url", ""),
                        repository_name=item.get("repository_name", ""),
                        target_score=float(item.get("target_score", 0.0)),
                        total_weight=float(item.get("total_weight", 1.0)),
                        criteria_scores=item.get("criteria_scores", {}),
                        overall_reasoning=item.get("overall_reasoning", ""),
                        parsing_method=item.get("parsing_method", "unknown"),
                        parsing_success=item.get("parsing_success", False),
                        parsing_warnings=item.get("parsing_warnings", [])
                    )
                    
                    if assessment.repository_url:
                        assessments[assessment.repository_url] = assessment
                    else:
                        logger.warning(f"Skipping assessment with missing repository_url")
                        
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"Error parsing assessment item: {e}")
                    continue
            
            logger.info(f"Loaded {len(assessments)} criteria assessments from {self.assessments_file}")
            self.assessments = assessments
            return assessments
            
        except Exception as e:
            logger.error(f"Error loading assessments: {e}")
            raise
    
    def load_repositories(self) -> List[str]:
        """Load repository URLs from level1list.csv, filtered by assessment availability."""
        try:
            # Load assessments first
            if not self.assessments:
                self.load_assessments()
            
            with open(self.level1_list_file, 'r') as f:
                all_repos = [line.strip() for line in f if line.strip()]
            
            # Filter to only repos that have assessment data
            repos_with_assessments = []
            missing_repos = []
            
            for repo in all_repos:
                norm_repo = self._normalize_url(repo)
                if norm_repo in self.assessments:
                    repos_with_assessments.append(repo)
                else:
                    missing_repos.append(repo)
            
            repos_with_assessments.sort()  # Sort for consistent ordering
            
            logger.info(f"Loaded {len(all_repos)} repositories from {self.level1_list_file}")
            logger.info(f"Found {len(repos_with_assessments)} repositories with assessment data")
            
            if missing_repos:
                logger.warning(f"Excluded {len(missing_repos)} repositories missing assessment data:")
                for repo in missing_repos:
                    logger.warning(f"  - {self._extract_repo_name(repo)}")
            
            self.repositories = repos_with_assessments
            return repos_with_assessments
            
        except Exception as e:
            logger.error(f"Error loading repository list: {e}")
            raise
    
    def generate_pairs(self) -> List[Tuple[str, str]]:
        """Generate all possible repository pairs for L1 comparison."""
        if not self.repositories:
            raise ValueError("No repositories loaded. Call load_repositories() first.")
        
        # Generate all combinations (n choose 2)
        pairs = list(itertools.combinations(self.repositories, 2))
        logger.info(f"Generated {len(pairs)} repository pairs")
        
        return pairs
    
    def _normalize_url(self, url: str) -> str:
        """Normalize repository URL to match assessment data."""
        # Handle known URL mismatches between level1list and assessments
        url_mappings = {
            'https://github.com/ledgerwatch/erigon': 'https://github.com/erigontech/erigon',
            'https://github.com/anza-xyz/grandine': 'https://github.com/grandinetech/grandine',
            'https://github.com/ApeWorX/ape': 'https://github.com/apeworx/ape',
            'https://github.com/Nethereum/Nethereum': 'https://github.com/nethereum/nethereum',
        }
        
        return url_mappings.get(url, url)
    
    def compare_repositories(self, repo_a: str, repo_b: str) -> ComparisonResult:
        """
        Compare two repositories using criteria assessment data.
        
        Args:
            repo_a: First repository URL
            repo_b: Second repository URL
            
        Returns:
            ComparisonResult with choice, multiplier, uncertainties, and reasoning
        """
        start_time = time.time()
        
        # Normalize URLs and get assessments
        norm_repo_a = self._normalize_url(repo_a)
        norm_repo_b = self._normalize_url(repo_b)
        
        assessment_a = self.assessments.get(norm_repo_a)
        assessment_b = self.assessments.get(norm_repo_b)
        
        if not assessment_a or not assessment_b:
            missing = []
            if not assessment_a:
                missing.append(repo_a)
                self.missing_repos.append(repo_a)
            if not assessment_b:
                missing.append(repo_b)
                self.missing_repos.append(repo_b)
            
            logger.warning(f"Missing assessments for: {missing}")
            
            # Return fallback result
            return ComparisonResult(
                choice="Equal",
                multiplier=1.0,
                raw_uncertainty=0.5,
                calibrated_uncertainty=0.5,
                explanation=f"Assessment data not available for: {', '.join([self._extract_repo_name(r) for r in missing])}",
                model_used="criteria_based",
                temperature=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                method="special_case_criteria",
                comparison_level="L1",
                score_a=0.0,
                score_b=0.0,
                ratio=1.0,
                repo_a=repo_a,
                repo_b=repo_b
            )
        
        logger.debug(f"Comparing {assessment_a.repository_name} vs {assessment_b.repository_name}")
        
        # Extract target scores
        score_a = assessment_a.target_score
        score_b = assessment_b.target_score
        
        logger.debug(f"Scores: {assessment_a.repository_name}={score_a:.3f}, "
                    f"{assessment_b.repository_name}={score_b:.3f}")
        
        # Calculate comparison metrics
        choice, multiplier, ratio = self._calculate_comparison_metrics(score_a, score_b)
        
        # Calculate uncertainties
        raw_uncertainty, calibrated_uncertainty = self._calculate_uncertainties(
            assessment_a, assessment_b, abs(score_a - score_b)
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            assessment_a, assessment_b, choice, ratio, score_a, score_b
        )
        
        # Map choice to string format
        choice_str = "A" if choice == 1 else "B"
        
        result = ComparisonResult(
            choice=choice_str,
            multiplier=multiplier,
            raw_uncertainty=raw_uncertainty,
            calibrated_uncertainty=calibrated_uncertainty,
            explanation=explanation,
            model_used="criteria_based",
            temperature=0.0,
            processing_time_ms=(time.time() - start_time) * 1000,
            method="special_case_criteria",
            comparison_level="L1",
            score_a=score_a,
            score_b=score_b,
            ratio=ratio,
            repo_a=repo_a,
            repo_b=repo_b
        )
        
        logger.info(f"✓ Compared {assessment_a.repository_name} vs {assessment_b.repository_name}: "
                   f"{choice_str} (multiplier={multiplier:.2f})")
        
        return result
    
    def _calculate_comparison_metrics(self, score_a: float, score_b: float) -> Tuple[int, float, float]:
        """
        Calculate choice, multiplier, and ratio from target scores.
        
        Args:
            score_a: Target score for repository A
            score_b: Target score for repository B
            
        Returns:
            (choice, multiplier, ratio) tuple
        """
        # Handle edge cases
        if score_a <= 0 or score_b <= 0:
            logger.warning(f"Invalid scores: A={score_a}, B={score_b}")
            return random.choice([1, 2]), 1.0, 1.0
        
        # Calculate ratio and determine choice
        if score_a > score_b:
            choice = 1
            ratio = score_a / score_b
        elif score_b > score_a:
            choice = 2
            ratio = score_b / score_a
        else:
            # Equal scores
            choice = random.choice([1, 2])
            ratio = 1.0
        
        # Calculate multiplier using exponential transformation
        try:
            multiplier = math.exp(2 * ratio)
            
            # Clamp to reasonable range
            multiplier = max(self.min_multiplier, min(self.max_multiplier, multiplier))
            
        except (OverflowError, ValueError):
            logger.warning(f"Multiplier calculation overflow for ratio {ratio}")
            multiplier = self.max_multiplier
        
        return choice, multiplier, ratio
    
    def _calculate_uncertainties(self, assessment_a: CriteriaAssessment, 
                               assessment_b: CriteriaAssessment, 
                               score_difference: float) -> Tuple[float, float]:
        """
        Calculate uncertainty measures with special case detection.
        
        Args:
            assessment_a: First repository assessment
            assessment_b: Second repository assessment
            score_difference: Absolute difference between scores
            
        Returns:
            (raw_uncertainty, calibrated_uncertainty) tuple
        """
        # Check for parsing failure indicator (all uncertainties = 0.5)
        def has_parsing_failure(assessment: CriteriaAssessment) -> bool:
            if not assessment.criteria_scores:
                return True
            
            uncertainties = [
                criterion.get("raw_uncertainty", 0.5)
                for criterion in assessment.criteria_scores.values()
            ]
            
            # Check if all uncertainties are approximately 0.5 (within floating point tolerance)
            return len(uncertainties) > 0 and all(abs(u - 0.5) < 1e-10 for u in uncertainties)
        
        # Detect parsing failures
        parsing_failure_a = has_parsing_failure(assessment_a)
        parsing_failure_b = has_parsing_failure(assessment_b)
        
        if parsing_failure_a or parsing_failure_b:
            logger.warning(f"Parsing failure detected - using fallback uncertainties")
            logger.debug(f"Failure flags: A={parsing_failure_a}, B={parsing_failure_b}")
            
            # Use specified fallback values
            return 0.18, 0.15
        
        # Normal uncertainty calculation based on score differences
        # Lower uncertainty when scores differ more significantly
        base_uncertainty = 1.0 / (1.0 + score_difference * 2.0)
        
        raw_uncertainty = max(0.05, min(0.95, base_uncertainty))
        calibrated_uncertainty = max(0.05, min(0.90, base_uncertainty * 0.9))
        
        return raw_uncertainty, calibrated_uncertainty
    
    def _generate_explanation(self, assessment_a: CriteriaAssessment, 
                            assessment_b: CriteriaAssessment, 
                            choice: int, ratio: float, 
                            score_a: float, score_b: float) -> str:
        """
        Generate comparison reasoning based on assessment data.
        
        Args:
            assessment_a: First repository assessment
            assessment_b: Second repository assessment
            choice: Chosen repository (1 or 2)
            ratio: Score ratio
            score_a: Target score for repository A
            score_b: Target score for repository B
            
        Returns:
            Formatted reasoning string
        """
        try:
            # Determine which repository was chosen
            chosen_repo = assessment_a if choice == 1 else assessment_b
            other_repo = assessment_b if choice == 1 else assessment_a
            chosen_score = score_a if choice == 1 else score_b
            other_score = score_b if choice == 1 else score_a
            
            # Start with overall comparison
            reasoning_parts = []
            
            # Overall score comparison
            score_diff = abs(chosen_score - other_score)
            if score_diff < 0.1:
                reasoning_parts.append(
                    f"{chosen_repo.repository_name} and {other_repo.repository_name} have very similar "
                    f"target scores ({chosen_score:.2f} vs {other_score:.2f}), making this a close comparison."
                )
            else:
                reasoning_parts.append(
                    f"{chosen_repo.repository_name} (score: {chosen_score:.2f}) outperforms "
                    f"{other_repo.repository_name} (score: {other_score:.2f}) "
                    f"with a {ratio:.2f}x advantage in overall assessment."
                )
            
            # Add simple score-based reasoning
            if score_diff >= 1.0:
                reasoning_parts.append(
                    f"The chosen repository demonstrates superior performance across "
                    f"multiple evaluation criteria, resulting in a significant scoring advantage."
                )
            
            # Add overall reasoning if available
            if chosen_repo.overall_reasoning:
                reasoning_parts.append(f"Assessment summary: {chosen_repo.overall_reasoning}")
            
            return " ".join(reasoning_parts)
            
        except Exception as e:
            logger.warning(f"Error generating reasoning: {e}")
            
            # Fallback reasoning
            chosen_name = assessment_a.repository_name if choice == 1 else assessment_b.repository_name
            return (f"{chosen_name} was selected based on criteria assessment scores "
                   f"({score_a:.2f} vs {score_b:.2f}).")
    
    def generate_all_comparisons(self) -> List[ComparisonResult]:
        """Generate all L1 comparison results."""
        pairs = self.generate_pairs()
        
        logger.info(f"Starting to generate {len(pairs)} comparisons...")
        
        results = []
        for i, (repo_a, repo_b) in enumerate(pairs):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(pairs)} comparisons completed")
            
            result = self.compare_repositories(repo_a, repo_b)
            results.append(result)
        
        logger.info(f"Generated {len(results)} comparison results")
        self.results = results
        return results
    
    def save_results(self) -> None:
        """Save results to JSON file."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Create output directory if it doesn't exist
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to dictionaries
        results_dict = [asdict(result) for result in self.results]
        
        # Calculate statistics
        successful = len([r for r in results_dict if 'error' not in r])
        failed = len(results_dict) - successful
        
        choices = [r.get('choice') for r in results_dict if 'choice' in r]
        choice_counts = {choice: choices.count(choice) for choice in set(choices) if choice}
        
        multipliers = [r.get('multiplier') for r in results_dict if 'multiplier' in r and r.get('multiplier', 0) > 0]
        avg_multiplier = sum(multipliers) / len(multipliers) if multipliers else 0
        
        # Remove duplicates from missing repos
        unique_missing = list(set(self.missing_repos))
        
        # Add metadata
        output_data = {
            'metadata': {
                'total_repositories': len(self.repositories),
                'total_pairs': len(self.results),
                'successful_comparisons': successful,
                'failed_comparisons': failed,
                'missing_assessments': len(unique_missing),
                'generation_timestamp': time.time(),
                'assessments_source': self.assessments_file,
                'statistics': {
                    'choice_distribution': choice_counts,
                    'average_multiplier': round(avg_multiplier, 2),
                    'multiplier_range': [round(min(multipliers), 2), round(max(multipliers), 2)] if multipliers else [0, 0]
                }
            },
            'repositories': self.repositories,
            'missing_assessments': unique_missing,
            'comparisons': results_dict
        }
        
        try:
            with open(self.output_file, 'w') as f:
                json.dump(output_data, f, indent=2, sort_keys=True)
            
            logger.info(f"Saved results to {self.output_file}")
            
            # Print summary statistics
            logger.info(f"Summary: {successful} successful, {failed} failed comparisons")
            logger.info(f"Choice distribution: {choice_counts}")
            logger.info(f"Average multiplier: {avg_multiplier:.2f}")
            
            if unique_missing:
                logger.warning(f"Missing assessments for {len(unique_missing)} repositories:")
                for repo in unique_missing:
                    logger.warning(f"  - {self._extract_repo_name(repo)}")
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def _extract_repo_name(self, url: str) -> str:
        """Extract repository name from GitHub URL."""
        if 'github.com' in url:
            return url.rstrip('/').split('/')[-1]
        return url.split('/')[-1] if '/' in url else url

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate L1 comparison pairs and collect responses")
    parser.add_argument('--assessments-file', 
                       default='data/processed/criteria_assessment/detailed_assessments.json',
                       help='Path to criteria assessments JSON file')
    parser.add_argument('--level1-list-file',
                       default='data/processed/criteria_assessment/level1list.csv',
                       help='Path to level1 repository list CSV file')
    parser.add_argument('--output-file', 
                       default='data/processed/l1_comparison_results.json',
                       help='Output file for comparison results')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only generate pairs without making comparisons')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = L1ComparisonGenerator(
        assessments_file=args.assessments_file,
        level1_list_file=args.level1_list_file,
        output_file=args.output_file
    )
    
    try:
        # Load repositories
        repos = generator.load_repositories()
        logger.info(f"Loaded repositories: {[generator._extract_repo_name(r) for r in repos[:5]]}...")
        
        # Generate pairs
        pairs = generator.generate_pairs()
        logger.info(f"Generated {len(pairs)} total pairs")
        
        if args.dry_run:
            logger.info("Dry run mode - showing first 5 pairs:")
            for i, (repo_a, repo_b) in enumerate(pairs[:5]):
                logger.info(f"  {i+1}. {generator._extract_repo_name(repo_a)} vs {generator._extract_repo_name(repo_b)}")
            logger.info(f"Total pairs that would be processed: {len(pairs)}")
            return
        
        # Generate all comparisons
        logger.info("Starting comparison generation process...")
        start_time = time.time()
        
        results = generator.generate_all_comparisons()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generation completed in {elapsed_time:.2f} seconds")
        
        # Save results
        generator.save_results()
        
        logger.info("L1 comparison pairs generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()