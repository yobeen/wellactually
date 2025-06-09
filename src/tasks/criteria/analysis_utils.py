# src/uncertainty_calibration/criteria_assessment/analysis_utils.py
"""
Analysis utilities for criteria assessment results.
Provides functions to analyze and visualize criteria assessment outcomes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class CriteriaAnalysisUtils:
    """
    Utility functions for analyzing criteria assessment results.
    """
    
    def __init__(self):
        """Initialize the analysis utilities."""
        self.criteria_names = {
            "core_protocol": "Core Protocol Implementation",
            "market_adoption": "Market Adoption & Network Effects",
            "developer_ecosystem": "Developer Ecosystem Impact",
            "general_purpose_tools": "General Purpose Tool Dependency",
            "security_infrastructure": "Security & Infrastructure Criticality",
            "defi_infrastructure": "DeFi & Financial Infrastructure",
            "data_analytics": "Data & Analytics Infrastructure",
            "innovation_research": "Innovation & Research Impact",
            "ecosystem_coordination": "Ecosystem Coordination & Standards",
            "community_trust": "Community Trust & Project Maturity",
            "user_applications": "User-Facing Applications"
        }
    
    def load_assessment_results(self, results_dir: str) -> Dict[str, Any]:
        """
        Load criteria assessment results from directory.
        
        Args:
            results_dir: Directory containing assessment results
            
        Returns:
            Dictionary with loaded results
        """
        results_path = Path(results_dir)
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        results = {}
        
        # Load detailed assessments
        detailed_path = results_path / "detailed_assessments.json"
        if detailed_path.exists():
            with open(detailed_path, 'r') as f:
                results['detailed_assessments'] = json.load(f)
        
        # Load target scores
        scores_path = results_path / "target_scores.json"
        if scores_path.exists():
            with open(scores_path, 'r') as f:
                results['target_scores'] = json.load(f)
        
        # Load comparison results
        comparison_path = results_path / "comparison_results.csv"
        if comparison_path.exists():
            results['comparison_results'] = pd.read_csv(comparison_path)
        
        # Load comparison summary
        summary_path = results_path / "comparison_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                results['comparison_summary'] = json.load(f)
        
        logger.info(f"Loaded assessment results from {results_dir}")
        return results
    
    def analyze_score_distribution(self, detailed_assessments: List[Dict]) -> Dict[str, Any]:
        """
        Analyze the distribution of scores across criteria and repositories.
        
        Args:
            detailed_assessments: List of detailed assessment dictionaries
            
        Returns:
            Dictionary with distribution analysis
        """
        # Extract all scores by criterion
        scores_by_criterion = defaultdict(list)
        target_scores = []
        
        for assessment in detailed_assessments:
            target_scores.append(assessment.get('target_score', 0))
            
            criteria_scores = assessment.get('criteria_scores', {})
            for criterion, data in criteria_scores.items():
                score = data.get('score', 0)
                scores_by_criterion[criterion].append(score)
        
        # Calculate statistics for each criterion
        criterion_stats = {}
        for criterion, scores in scores_by_criterion.items():
            if scores:
                criterion_stats[criterion] = {
                    'name': self.criteria_names.get(criterion, criterion),
                    'mean': np.mean(scores),
                    'median': np.median(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'count': len(scores)
                }
        
        # Overall target score statistics
        target_score_stats = {
            'mean': np.mean(target_scores),
            'median': np.median(target_scores),
            'std': np.std(target_scores),
            'min': np.min(target_scores),
            'max': np.max(target_scores),
            'count': len(target_scores)
        }
        
        return {
            'criterion_statistics': criterion_stats,
            'target_score_statistics': target_score_stats,
            'raw_scores_by_criterion': dict(scores_by_criterion),
            'raw_target_scores': target_scores
        }
    
    def analyze_weight_patterns(self, detailed_assessments: List[Dict]) -> Dict[str, Any]:
        """
        Analyze how weights were assigned across assessments.
        
        Args:
            detailed_assessments: List of detailed assessment dictionaries
            
        Returns:
            Dictionary with weight pattern analysis
        """
        # Template weights for comparison
        template_weights = {
            "core_protocol": 0.25,
            "market_adoption": 0.20,
            "developer_ecosystem": 0.15,
            "general_purpose_tools": 0.10,
            "security_infrastructure": 0.10,
            "defi_infrastructure": 0.05,
            "data_analytics": 0.05,
            "innovation_research": 0.03,
            "ecosystem_coordination": 0.03,
            "community_trust": 0.02,
            "user_applications": 0.02
        }
        
        # Extract weights by criterion
        weights_by_criterion = defaultdict(list)
        
        for assessment in detailed_assessments:
            criteria_scores = assessment.get('criteria_scores', {})
            for criterion, data in criteria_scores.items():
                weight = data.get('weight', 0)
                weights_by_criterion[criterion].append(weight)
        
        # Calculate weight statistics
        weight_stats = {}
        deviations_from_template = {}
        
        for criterion, weights in weights_by_criterion.items():
            if weights:
                template_weight = template_weights.get(criterion, 0)
                
                weight_stats[criterion] = {
                    'name': self.criteria_names.get(criterion, criterion),
                    'template_weight': template_weight,
                    'mean_assigned': np.mean(weights),
                    'median_assigned': np.median(weights),
                    'std_assigned': np.std(weights),
                    'min_assigned': np.min(weights),
                    'max_assigned': np.max(weights)
                }
                
                # Calculate deviations from template
                deviations = [abs(w - template_weight) for w in weights]
                deviations_from_template[criterion] = {
                    'mean_deviation': np.mean(deviations),
                    'max_deviation': np.max(deviations),
                    'fraction_changed': sum(1 for d in deviations if d > 0.01) / len(deviations)
                }
        
        return {
            'template_weights': template_weights,
            'weight_statistics': weight_stats,
            'deviations_from_template': deviations_from_template,
            'raw_weights_by_criterion': dict(weights_by_criterion)
        }
    
    def analyze_comparison_accuracy(self, comparison_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze accuracy of criteria-based predictions vs human preferences.
        
        Args:
            comparison_results: DataFrame with comparison results
            
        Returns:
            Dictionary with accuracy analysis
        """
        if comparison_results.empty:
            return {"error": "No comparison results available"}
        
        # Overall accuracy
        total_comparisons = len(comparison_results)
        correct_predictions = comparison_results['directional_agreement'].sum()
        overall_accuracy = correct_predictions / total_comparisons if total_comparisons > 0 else 0
        
        # Accuracy by preference type
        accuracy_by_preference = {}
        for pref_type in ['A', 'B', 'Equal']:
            subset = comparison_results[comparison_results['human_preference'] == pref_type]
            if len(subset) > 0:
                accuracy = subset['directional_agreement'].sum() / len(subset)
                accuracy_by_preference[pref_type] = {
                    'count': len(subset),
                    'correct': int(subset['directional_agreement'].sum()),
                    'accuracy': accuracy
                }
            else:
                accuracy_by_preference[pref_type] = {
                    'count': 0,
                    'correct': 0,
                    'accuracy': 0.0
                }
        
        # Ratio error analysis
        finite_errors = comparison_results[
            np.isfinite(comparison_results['ratio_error'])
        ]['ratio_error']
        
        ratio_error_stats = {
            'mean': np.mean(finite_errors) if len(finite_errors) > 0 else float('inf'),
            'median': np.median(finite_errors) if len(finite_errors) > 0 else float('inf'),
            'std': np.std(finite_errors) if len(finite_errors) > 0 else float('inf'),
            'count_finite': len(finite_errors),
            'count_infinite': len(comparison_results) - len(finite_errors)
        }
        
        # Correlation analysis
        try:
            # Convert human preferences to ratios
            human_ratios = []
            predicted_ratios = []
            
            for _, row in comparison_results.iterrows():
                if np.isfinite(row['predicted_ratio']):
                    predicted_ratios.append(row['predicted_ratio'])
                    
                    if row['human_choice'] == 1.0:
                        human_ratio = row['human_multiplier']
                    elif row['human_choice'] == 2.0:
                        human_ratio = 1.0 / row['human_multiplier']
                    else:
                        human_ratio = 1.0
                    
                    human_ratios.append(human_ratio)
            
            correlation = np.corrcoef(predicted_ratios, human_ratios)[0, 1] if len(predicted_ratios) >= 2 else 0
            if np.isnan(correlation):
                correlation = 0.0
                
        except Exception as e:
            correlation = 0.0
            logger.warning(f"Error calculating correlation: {e}")
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_comparisons': total_comparisons,
            'correct_predictions': int(correct_predictions),
            'accuracy_by_preference': accuracy_by_preference,
            'ratio_error_statistics': ratio_error_stats,
            'correlation_coefficient': correlation
        }
    
    def create_score_distribution_plots(self, distribution_analysis: Dict, 
                                      save_path: Optional[str] = None):
        """
        Create plots showing score distributions across criteria.
        
        Args:
            distribution_analysis: Results from analyze_score_distribution
            save_path: Optional path to save plots
        """
        criterion_stats = distribution_analysis['criterion_statistics']
        
        if not criterion_stats:
            logger.warning("No criterion statistics available for plotting")
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Mean scores by criterion
        criteria = list(criterion_stats.keys())
        mean_scores = [criterion_stats[c]['mean'] for c in criteria]
        criterion_labels = [self.criteria_names.get(c, c)[:20] for c in criteria]
        
        bars = ax1.bar(range(len(criteria)), mean_scores, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_xlabel('Criteria', fontweight='bold')
        ax1.set_ylabel('Mean Score', fontweight='bold')
        ax1.set_title('Mean Scores by Criterion', fontweight='bold', fontsize=14)
        ax1.set_xticks(range(len(criteria)))
        ax1.set_xticklabels(criterion_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, mean_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Score ranges (min-max) by criterion
        min_scores = [criterion_stats[c]['min'] for c in criteria]
        max_scores = [criterion_stats[c]['max'] for c in criteria]
        ranges = [max_scores[i] - min_scores[i] for i in range(len(criteria))]
        
        bars2 = ax2.bar(range(len(criteria)), ranges, color='lightcoral', edgecolor='darkred', alpha=0.7)
        ax2.set_xlabel('Criteria', fontweight='bold')
        ax2.set_ylabel('Score Range (Max - Min)', fontweight='bold')
        ax2.set_title('Score Ranges by Criterion', fontweight='bold', fontsize=14)
        ax2.set_xticks(range(len(criteria)))
        ax2.set_xticklabels(criterion_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Target score distribution
        target_scores = distribution_analysis['raw_target_scores']
        ax3.hist(target_scores, bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        ax3.set_xlabel('Target Score', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Distribution of Target Scores', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Add statistics text
        stats = distribution_analysis['target_score_statistics']
        stats_text = f"Mean: {stats['mean']:.2f}\nMedian: {stats['median']:.2f}\nStd: {stats['std']:.2f}"
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
        
        # 4. Score variance by criterion
        std_scores = [criterion_stats[c]['std'] for c in criteria]
        bars4 = ax4.bar(range(len(criteria)), std_scores, color='gold', edgecolor='orange', alpha=0.7)
        ax4.set_xlabel('Criteria', fontweight='bold')
        ax4.set_ylabel('Standard Deviation', fontweight='bold')
        ax4.set_title('Score Variability by Criterion', fontweight='bold', fontsize=14)
        ax4.set_xticks(range(len(criteria)))
        ax4.set_xticklabels(criterion_labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved score distribution plots to {save_path}")
        
        plt.show()
    
    def create_accuracy_analysis_plots(self, accuracy_analysis: Dict,
                                     save_path: Optional[str] = None):
        """
        Create plots showing prediction accuracy analysis.
        
        Args:
            accuracy_analysis: Results from analyze_comparison_accuracy
            save_path: Optional path to save plots
        """
        if 'error' in accuracy_analysis:
            logger.warning(f"Cannot create accuracy plots: {accuracy_analysis['error']}")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall accuracy
        accuracy = accuracy_analysis['overall_accuracy']
        total = accuracy_analysis['total_comparisons']
        correct = accuracy_analysis['correct_predictions']
        incorrect = total - correct
        
        labels = ['Correct', 'Incorrect']
        sizes = [correct, incorrect]
        colors = ['lightgreen', 'lightcoral']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
               textprops={'fontweight': 'bold'})
        ax1.set_title(f'Overall Prediction Accuracy\n({correct}/{total} correct)', 
                     fontweight='bold', fontsize=14)
        
        # 2. Accuracy by preference type
        pref_data = accuracy_analysis['accuracy_by_preference']
        preferences = list(pref_data.keys())
        accuracies = [pref_data[p]['accuracy'] for p in preferences]
        counts = [pref_data[p]['count'] for p in preferences]
        
        bars = ax2.bar(preferences, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'],
                      edgecolor='navy', alpha=0.7)
        ax2.set_xlabel('Human Preference', fontweight='bold')
        ax2.set_ylabel('Prediction Accuracy', fontweight='bold')
        ax2.set_title('Accuracy by Preference Type', fontweight='bold', fontsize=14)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar, acc, count in zip(bars, accuracies, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.1%}\n(n={count})', ha='center', va='bottom', fontweight='bold')
        
        # 3. Ratio error distribution (if available)
        ratio_stats = accuracy_analysis['ratio_error_statistics']
        if ratio_stats['count_finite'] > 0:
            ax3.text(0.5, 0.6, f"Ratio Error Statistics", ha='center', va='center',
                    transform=ax3.transAxes, fontsize=16, fontweight='bold')
            
            stats_text = f"""Mean Error: {ratio_stats['mean']:.3f}
Median Error: {ratio_stats['median']:.3f}
Std Dev: {ratio_stats['std']:.3f}
Finite Errors: {ratio_stats['count_finite']}
Infinite Errors: {ratio_stats['count_infinite']}"""
            
            ax3.text(0.5, 0.3, stats_text, ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, "No finite ratio errors\navailable for analysis", 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        
        ax3.set_title('Ratio Error Analysis', fontweight='bold', fontsize=14)
        ax3.axis('off')
        
        # 4. Correlation information
        correlation = accuracy_analysis['correlation_coefficient']
        
        ax4.text(0.5, 0.6, f"Correlation Analysis", ha='center', va='center',
                transform=ax4.transAxes, fontsize=16, fontweight='bold')
        
        corr_text = f"""Correlation Coefficient: {correlation:.3f}

Interpretation:
{self._interpret_correlation(correlation)}"""
        
        ax4.text(0.5, 0.3, corr_text, ha='center', va='center',
                transform=ax4.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax4.set_title('Predicted vs Human Ratios', fontweight='bold', fontsize=14)
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved accuracy analysis plots to {save_path}")
        
        plt.show()
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient."""
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.8:
            strength = "Very Strong"
        elif abs_corr >= 0.6:
            strength = "Strong"
        elif abs_corr >= 0.4:
            strength = "Moderate"
        elif abs_corr >= 0.2:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        direction = "Positive" if correlation >= 0 else "Negative"
        
        return f"{strength} {direction} correlation"
    
    def generate_summary_report(self, results_dir: str, output_path: str = None) -> str:
        """
        Generate a comprehensive summary report of the criteria assessment.
        
        Args:
            results_dir: Directory containing assessment results
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        try:
            results = self.load_assessment_results(results_dir)
            
            # Analyze results
            distribution_analysis = self.analyze_score_distribution(
                results.get('detailed_assessments', [])
            )
            
            weight_analysis = self.analyze_weight_patterns(
                results.get('detailed_assessments', [])
            )
            
            comparison_df = results.get('comparison_results', pd.DataFrame())
            accuracy_analysis = self.analyze_comparison_accuracy(comparison_df)
            
            # Generate report
            report_lines = [
                "=" * 80,
                "CRITERIA ASSESSMENT SUMMARY REPORT",
                "=" * 80,
                "",
                f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Results directory: {results_dir}",
                "",
                "ASSESSMENT OVERVIEW",
                "-" * 40
            ]
            
            # Add assessment statistics
            detailed_assessments = results.get('detailed_assessments', [])
            report_lines.extend([
                f"Total repositories assessed: {len(detailed_assessments)}",
                f"Successful assessments: {sum(1 for a in detailed_assessments if a.get('parsing_success', False))}",
                "",
                "TARGET SCORE STATISTICS",
                "-" * 40
            ])
            
            target_stats = distribution_analysis['target_score_statistics']
            report_lines.extend([
                f"Mean target score: {target_stats['mean']:.3f}",
                f"Median target score: {target_stats['median']:.3f}",
                f"Standard deviation: {target_stats['std']:.3f}",
                f"Range: {target_stats['min']:.3f} - {target_stats['max']:.3f}",
                "",
                "CRITERIA SCORE ANALYSIS",
                "-" * 40
            ])
            
            # Add criteria statistics
            criterion_stats = distribution_analysis['criterion_statistics']
            for criterion, stats in criterion_stats.items():
                name = self.criteria_names.get(criterion, criterion)
                report_lines.append(f"{name}:")
                report_lines.append(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, Range: {stats['min']}-{stats['max']}")
            
            report_lines.extend([
                "",
                "WEIGHT ANALYSIS",
                "-" * 40
            ])
            
            # Add weight analysis
            weight_stats = weight_analysis['weight_statistics']
            significant_deviations = []
            
            for criterion, stats in weight_stats.items():
                template_weight = stats['template_weight']
                mean_assigned = stats['mean_assigned']
                deviation = abs(mean_assigned - template_weight)
                
                if deviation > 0.02:  # 2% deviation threshold
                    significant_deviations.append((criterion, deviation, template_weight, mean_assigned))
            
            if significant_deviations:
                report_lines.append("Significant weight deviations from template:")
                for criterion, deviation, template, assigned in significant_deviations:
                    name = self.criteria_names.get(criterion, criterion)
                    report_lines.append(f"  {name}: {template:.3f} → {assigned:.3f} (Δ{deviation:.3f})")
            else:
                report_lines.append("No significant weight deviations from template")
            
            report_lines.extend([
                "",
                "PREDICTION ACCURACY",
                "-" * 40
            ])
            
            # Add accuracy analysis
            if 'error' not in accuracy_analysis:
                overall_acc = accuracy_analysis['overall_accuracy']
                total_comp = accuracy_analysis['total_comparisons']
                correlation = accuracy_analysis['correlation_coefficient']
                
                report_lines.extend([
                    f"Total comparisons: {total_comp}",
                    f"Overall directional accuracy: {overall_acc:.1%}",
                    f"Correlation with human preferences: {correlation:.3f}",
                    ""
                ])
                
                # Accuracy by preference type
                pref_acc = accuracy_analysis['accuracy_by_preference']
                for pref_type, stats in pref_acc.items():
                    count = stats['count']
                    accuracy = stats['accuracy']
                    report_lines.append(f"  {pref_type} preferences: {accuracy:.1%} ({count} comparisons)")
            else:
                report_lines.append("No comparison data available")
            
            report_lines.extend([
                "",
                "=" * 80,
                "END OF REPORT",
                "=" * 80
            ])
            
            report_text = "\n".join(report_lines)
            
            # Save report if path provided
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Saved summary report to {output_path}")
            
            return report_text
            
        except Exception as e:
            error_report = f"Error generating summary report: {e}"
            logger.error(error_report)
            return error_report