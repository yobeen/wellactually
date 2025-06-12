#!/usr/bin/env python3
"""
Validation script for L1 comparison results against human-labeled data.

This script validates the predicted choices from criteria assessment scores 
against human-labeled comparisons in train.csv where parent == 'ethereum'.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_human_labels(csv_path: str) -> pd.DataFrame:
    """Load human-labeled L1 comparisons from train.csv."""
    df = pd.read_csv(csv_path)
    # Filter for L1 comparisons (parent == 'ethereum')
    l1_df = df[df['parent'] == 'ethereum'].copy()
    
    # Normalize choice values
    l1_df['choice_normalized'] = l1_df['choice'].apply(normalize_choice)
    
    print(f"Loaded {len(l1_df)} human-labeled L1 comparisons")
    return l1_df


def load_predicted_results(json_path: str) -> List[Dict]:
    """Load predicted L1 comparison results from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    comparisons = data.get('comparisons', [])
    print(f"Loaded {len(comparisons)} predicted L1 comparisons")
    return comparisons


def get_fresh_logprobs_data(predicted_results: List[Dict]) -> Dict:
    """Make fresh API calls to get logprobs data for perplexity calculation."""
    # Import here to avoid circular imports
    import sys
    sys.path.append('/home/ubuntu/wellactually/src')
    
    from shared.multi_model_engine import MultiModelEngine
    from tasks.l1.level1_prompts import Level1PromptGenerator
    
    # Initialize components
    prompt_generator = Level1PromptGenerator()
    engine = MultiModelEngine()
    
    logprobs_data = {}
    
    print(f"Making fresh API calls for {len(predicted_results)} comparisons...")
    
    for i, result in enumerate(predicted_results):
        repo_a = result['repo_a']
        repo_b = result['repo_b']
        
        print(f"Processing {i+1}/{len(predicted_results)}: {repo_a} vs {repo_b}")
        
        # Generate prompt for this comparison
        prompt = prompt_generator.generate_comparison_prompt(repo_a, repo_b)
        
        # Make API call with logprobs
        try:
            response = engine.query_with_logprobs(prompt, temperature=0.0, max_tokens=50)
            
            if response and 'full_content_logprobs' in response:
                comparison_key = create_cache_key(repo_a, repo_b)
                logprobs_data[comparison_key] = response['full_content_logprobs']
                
        except Exception as e:
            print(f"Error getting logprobs for {repo_a} vs {repo_b}: {e}")
            continue
    
    print(f"Successfully retrieved logprobs for {len(logprobs_data)} comparisons")
    return logprobs_data


def calculate_answer_token_perplexity(logprobs_dict: Dict[str, float]) -> float:
    """
    Calculate perplexity using only answer token probabilities (A, B, Equal).
    
    Args:
        logprobs_dict: Dictionary mapping tokens to their log probabilities
        
    Returns:
        Answer token perplexity as uncertainty measure [0, 1]
    """
    # Extract answer token probabilities
    answer_tokens = ['A', 'B', 'Equal']
    answer_probs = []
    
    for token in answer_tokens:
        if token in logprobs_dict:
            # Convert from linear probability to log probability if needed
            prob = logprobs_dict[token]
            if prob > 1.0:  # Assume it's already linear probability
                answer_probs.append(prob)
            else:  # Assume it's log probability
                answer_probs.append(np.exp(prob))
    
    if not answer_probs:
        return 1.0  # Maximum uncertainty if no answer tokens found
    
    # Calculate total probability mass for answer tokens
    total_prob = sum(answer_probs)
    
    if total_prob <= 0:
        return 1.0  # Maximum uncertainty for invalid probabilities
    
    # Answer token perplexity = 1 / total_answer_probability
    perplexity = 1.0 / total_prob
    
    # Normalize to [0, 1] uncertainty scale
    # Higher perplexity = higher uncertainty
    # Use log scale with reasonable bounds
    max_perplexity = 100.0  # Reasonable upper bound
    uncertainty = np.log(min(perplexity, max_perplexity)) / np.log(max_perplexity)
    
    return max(0.0, min(1.0, uncertainty))


def create_cache_key(repo_a: str, repo_b: str) -> str:
    """Create cache key pattern for matching logprobs entries."""
    # Clean repo URLs
    clean_a = repo_a.strip()
    clean_b = repo_b.strip()
    return f"{clean_a}|{clean_b}"


def normalize_choice(choice) -> str:
    """Normalize choice values to standard format."""
    if isinstance(choice, (int, float)):
        if choice == 1.0:
            return 'A'
        elif choice == 2.0:
            return 'B'
        elif choice == 0.0:
            return 'Equal'
    elif isinstance(choice, str):
        choice_upper = choice.upper()
        if choice_upper in ['A', '1', '1.0']:
            return 'A'
        elif choice_upper in ['B', '2', '2.0']:
            return 'B'
        elif choice_upper in ['EQUAL', 'TIE', '0', '0.0']:
            return 'Equal'
    
    return str(choice)


def create_comparison_key(repo_a: str, repo_b: str) -> str:
    """Create a standardized key for repo comparisons."""
    # Sort repos to handle A vs B and B vs A as the same comparison
    repos = sorted([repo_a.strip(), repo_b.strip()])
    return f"{repos[0]}|{repos[1]}"


def flip_choice(choice: str) -> str:
    """Flip choice when repo order is reversed."""
    if choice == 'A':
        return 'B'
    elif choice == 'B':
        return 'A'
    else:
        return choice


def match_comparisons(human_df: pd.DataFrame, predicted_results: List[Dict]) -> Tuple[List[str], List[str], List[Dict]]:
    """Match human labels with predicted results."""
    # Create lookup for human labels
    human_lookup = {}
    for _, row in human_df.iterrows():
        key = create_comparison_key(row['repo_a'], row['repo_b'])
        if key not in human_lookup:
            human_lookup[key] = []
        human_lookup[key].append({
            'choice': row['choice_normalized'],
            'multiplier': row['multiplier'],
            'juror': row['juror'],
            'repo_a': row['repo_a'],
            'repo_b': row['repo_b']
        })
    
    matched_human = []
    matched_predicted = []
    match_details = []
    
    for pred in predicted_results:
        pred_key = create_comparison_key(pred['repo_a'], pred['repo_b'])
        pred_choice = normalize_choice(pred['choice'])
        
        if pred_key in human_lookup:
            for human_entry in human_lookup[pred_key]:
                # Check if repo order matches
                repos_match_order = (pred['repo_a'].strip() == human_entry['repo_a'].strip() and 
                                   pred['repo_b'].strip() == human_entry['repo_b'].strip())
                
                human_choice = human_entry['choice']
                if not repos_match_order:
                    # Flip the human choice if repo order is reversed
                    human_choice = flip_choice(human_choice)
                
                matched_human.append(human_choice)
                matched_predicted.append(pred_choice)
                match_details.append({
                    'human_choice': human_choice,
                    'predicted_choice': pred_choice,
                    'human_multiplier': human_entry['multiplier'],
                    'predicted_multiplier': pred.get('multiplier', None),
                    'juror': human_entry['juror'],
                    'repo_a': pred['repo_a'],
                    'repo_b': pred['repo_b'],
                    'repos_match_order': repos_match_order,
                    'predicted_score_a': pred.get('score_a', None),
                    'predicted_score_b': pred.get('score_b', None),
                    'predicted_ratio': pred.get('ratio', None)
                })
    
    return matched_human, matched_predicted, match_details


def match_with_logprobs(match_details: List[Dict], logprobs_cache: Dict) -> List[Dict]:
    """
    Enhance match details with logprobs data and answer token perplexity.
    
    Args:
        match_details: List of matched comparison details
        logprobs_cache: Dictionary of cached logprobs data
        
    Returns:
        Enhanced match details with perplexity metrics
    """
    enhanced_details = []
    
    for detail in match_details:
        enhanced_detail = detail.copy()
        
        # Try to find matching cache entries
        cache_key = create_cache_key(detail['repo_a'], detail['repo_b'])
        reverse_cache_key = create_cache_key(detail['repo_b'], detail['repo_a'])
        
        # Look for cache entries with different model prefixes
        matching_entries = []
        for full_key, cache_entry in logprobs_cache.items():
            if cache_key in full_key or reverse_cache_key in full_key:
                matching_entries.append(cache_entry)
        
        if matching_entries:
            # Calculate perplexity metrics from cache entries
            perplexities = []
            uncertainties = []
            
            for entry in matching_entries:
                if 'logprobs' in entry:
                    perplexity = calculate_answer_token_perplexity(entry['logprobs'])
                    perplexities.append(perplexity)
                
                if 'uncertainty' in entry:
                    uncertainties.append(entry['uncertainty'])
            
            # Add aggregated perplexity metrics
            if perplexities:
                enhanced_detail['answer_token_perplexity_mean'] = np.mean(perplexities)
                enhanced_detail['answer_token_perplexity_std'] = np.std(perplexities)
                enhanced_detail['answer_token_perplexity_min'] = np.min(perplexities)
                enhanced_detail['answer_token_perplexity_max'] = np.max(perplexities)
                enhanced_detail['num_models_with_logprobs'] = len(perplexities)
            
            if uncertainties:
                enhanced_detail['cached_uncertainty_mean'] = np.mean(uncertainties)
                enhanced_detail['cached_uncertainty_std'] = np.std(uncertainties)
        
        enhanced_details.append(enhanced_detail)
    
    return enhanced_details


def calculate_metrics(human_choices: List[str], predicted_choices: List[str]) -> Dict:
    """Calculate validation metrics."""
    accuracy = accuracy_score(human_choices, predicted_choices)
    
    # Get unique labels
    labels = sorted(list(set(human_choices + predicted_choices)))
    
    # Confusion matrix
    cm = confusion_matrix(human_choices, predicted_choices, labels=labels)
    
    # Classification report
    report = classification_report(human_choices, predicted_choices, labels=labels, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'labels': labels,
        'classification_report': report,
        'total_comparisons': len(human_choices)
    }


def analyze_perplexity_patterns(enhanced_match_details: List[Dict]) -> Dict:
    """
    Analyze patterns between answer token perplexity and prediction accuracy.
    
    Args:
        enhanced_match_details: Match details with perplexity metrics
        
    Returns:
        Dictionary containing perplexity analysis results
    """
    analysis = {
        'total_with_perplexity': 0,
        'perplexity_vs_accuracy': {},
        'perplexity_stats': {},
        'high_perplexity_disagreements': [],
        'low_perplexity_disagreements': []
    }
    
    # Extract data for analysis
    perplexities = []
    is_correct = []
    disagreements_high_perp = []
    disagreements_low_perp = []
    
    for detail in enhanced_match_details:
        if 'answer_token_perplexity_mean' in detail:
            perp = detail['answer_token_perplexity_mean']
            correct = detail['human_choice'] == detail['predicted_choice']
            
            perplexities.append(perp)
            is_correct.append(correct)
            
            # Track disagreements by perplexity level
            if not correct:
                if perp > 0.5:  # High perplexity threshold
                    disagreements_high_perp.append(detail)
                else:  # Low perplexity threshold
                    disagreements_low_perp.append(detail)
    
    analysis['total_with_perplexity'] = len(perplexities)
    
    if perplexities:
        # Basic perplexity statistics
        analysis['perplexity_stats'] = {
            'mean': np.mean(perplexities),
            'std': np.std(perplexities),
            'min': np.min(perplexities),
            'max': np.max(perplexities),
            'median': np.median(perplexities)
        }
        
        # Accuracy by perplexity bins
        perp_array = np.array(perplexities)
        correct_array = np.array(is_correct)
        
        # Create perplexity bins
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (perp_array >= low) & (perp_array < high)
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(correct_array[mask])
                bin_count = np.sum(mask)
                analysis['perplexity_vs_accuracy'][bin_labels[i]] = {
                    'accuracy': bin_accuracy,
                    'count': int(bin_count),
                    'avg_perplexity': np.mean(perp_array[mask])
                }
        
        # Store disagreement samples
        analysis['high_perplexity_disagreements'] = disagreements_high_perp[:5]
        analysis['low_perplexity_disagreements'] = disagreements_low_perp[:5]
    
    return analysis


def analyze_disagreements(match_details: List[Dict]) -> Dict:
    """Analyze cases where human and predicted choices disagree."""
    disagreements = [m for m in match_details if m['human_choice'] != m['predicted_choice']]
    
    analysis = {
        'total_disagreements': len(disagreements),
        'disagreement_rate': len(disagreements) / len(match_details) if match_details else 0,
        'disagreement_patterns': Counter([
            f"{d['human_choice']} -> {d['predicted_choice']}" for d in disagreements
        ]),
        'disagreements_by_juror': Counter([d['juror'] for d in disagreements]),
        'sample_disagreements': disagreements[:10]  # First 10 for inspection
    }
    
    return analysis


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], save_path: str = None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('L1 Comparison Validation - Confusion Matrix')
    plt.xlabel('Predicted Choice')
    plt.ylabel('Human Choice')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_report(metrics: Dict, disagreement_analysis: Dict, match_details: List[Dict], 
                   perplexity_analysis: Dict = None) -> str:
    """Generate a comprehensive validation report."""
    report = []
    report.append("="*60)
    report.append("L1 COMPARISON VALIDATION REPORT")
    report.append("="*60)
    report.append("")
    
    # Overall metrics
    report.append(f"Total matched comparisons: {metrics['total_comparisons']}")
    report.append(f"Overall accuracy: {metrics['accuracy']:.3f}")
    report.append("")
    
    # Confusion matrix summary
    report.append("CONFUSION MATRIX:")
    cm = metrics['confusion_matrix']
    labels = metrics['labels']
    
    # Create a formatted table
    header_label = "Actual\\Pred"
    header = f"{header_label:<12}" + "".join([f"{label:>8}" for label in labels])
    report.append(header)
    report.append("-" * len(header))
    
    for i, label in enumerate(labels):
        row = f"{label:<12}" + "".join([f"{cm[i,j]:>8}" for j in range(len(labels))])
        report.append(row)
    report.append("")
    
    # Per-class metrics
    report.append("PER-CLASS METRICS:")
    class_report = metrics['classification_report']
    for label in labels:
        if label in class_report:
            precision = class_report[label]['precision']
            recall = class_report[label]['recall']
            f1 = class_report[label]['f1-score']
            support = class_report[label]['support']
            report.append(f"{label:>6}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Support={support}")
    report.append("")
    
    # Disagreement analysis
    report.append("DISAGREEMENT ANALYSIS:")
    report.append(f"Total disagreements: {disagreement_analysis['total_disagreements']}")
    report.append(f"Disagreement rate: {disagreement_analysis['disagreement_rate']:.3f}")
    report.append("")
    
    report.append("Common disagreement patterns:")
    for pattern, count in disagreement_analysis['disagreement_patterns'].most_common(5):
        report.append(f"  {pattern}: {count} cases")
    report.append("")
    
    report.append("Disagreements by juror:")
    for juror, count in disagreement_analysis['disagreements_by_juror'].most_common(5):
        report.append(f"  {juror}: {count} disagreements")
    report.append("")
    
    # Sample disagreements
    report.append("SAMPLE DISAGREEMENTS:")
    for i, disagreement in enumerate(disagreement_analysis['sample_disagreements'][:5]):
        report.append(f"{i+1}. Human: {disagreement['human_choice']}, Predicted: {disagreement['predicted_choice']}")
        report.append(f"   Repos: {disagreement['repo_a']} vs {disagreement['repo_b']}")
        report.append(f"   Juror: {disagreement['juror']}")
        if disagreement['predicted_score_a'] and disagreement['predicted_score_b']:
            report.append(f"   Scores: A={disagreement['predicted_score_a']:.2f}, B={disagreement['predicted_score_b']:.2f}")
        report.append("")
    
    # Add perplexity analysis if available
    if perplexity_analysis and perplexity_analysis['total_with_perplexity'] > 0:
        report.append("="*60)
        report.append("ANSWER TOKEN PERPLEXITY ANALYSIS")
        report.append("="*60)
        report.append("")
        
        report.append(f"Comparisons with perplexity data: {perplexity_analysis['total_with_perplexity']}")
        
        # Perplexity statistics
        stats = perplexity_analysis['perplexity_stats']
        report.append("PERPLEXITY STATISTICS:")
        report.append(f"  Mean: {stats['mean']:.4f}")
        report.append(f"  Std:  {stats['std']:.4f}")
        report.append(f"  Min:  {stats['min']:.4f}")
        report.append(f"  Max:  {stats['max']:.4f}")
        report.append(f"  Median: {stats['median']:.4f}")
        report.append("")
        
        # Accuracy by perplexity bins
        report.append("ACCURACY BY PERPLEXITY LEVEL:")
        for bin_label, bin_data in perplexity_analysis['perplexity_vs_accuracy'].items():
            accuracy = bin_data['accuracy']
            count = bin_data['count']
            avg_perp = bin_data['avg_perplexity']
            report.append(f"  {bin_label}: Accuracy={accuracy:.3f}, Count={count}, Avg Perplexity={avg_perp:.4f}")
        report.append("")
        
        # High perplexity disagreements
        if perplexity_analysis['high_perplexity_disagreements']:
            report.append("HIGH PERPLEXITY DISAGREEMENTS (model uncertain but wrong):")
            for i, disagreement in enumerate(perplexity_analysis['high_perplexity_disagreements'][:3]):
                perp = disagreement.get('answer_token_perplexity_mean', 'N/A')
                report.append(f"{i+1}. Human: {disagreement['human_choice']}, Predicted: {disagreement['predicted_choice']}")
                report.append(f"   Perplexity: {perp:.4f}")
                report.append(f"   Repos: {disagreement['repo_a']} vs {disagreement['repo_b']}")
            report.append("")
        
        # Low perplexity disagreements
        if perplexity_analysis['low_perplexity_disagreements']:
            report.append("LOW PERPLEXITY DISAGREEMENTS (model confident but wrong):")
            for i, disagreement in enumerate(perplexity_analysis['low_perplexity_disagreements'][:3]):
                perp = disagreement.get('answer_token_perplexity_mean', 'N/A')
                report.append(f"{i+1}. Human: {disagreement['human_choice']}, Predicted: {disagreement['predicted_choice']}")
                report.append(f"   Perplexity: {perp:.4f}")
                report.append(f"   Repos: {disagreement['repo_a']} vs {disagreement['repo_b']}")
            report.append("")
    
    return "\n".join(report)


def main():
    """Main validation function."""
    # Paths
    train_csv_path = "/home/ubuntu/wellactually/data/raw/train.csv"
    results_json_path = "/home/ubuntu/wellactually/data/processed/l1_comparison_results.json"
    cache_path = "/home/ubuntu/wellactually/correlation_cache.json"
    
    print("Loading data...")
    
    # Load data
    human_df = load_human_labels(train_csv_path)
    predicted_results = load_predicted_results(results_json_path)
    logprobs_cache = load_logprobs_cache(cache_path)
    
    print("Matching comparisons...")
    
    # Match comparisons
    human_choices, predicted_choices, match_details = match_comparisons(human_df, predicted_results)
    
    if not human_choices:
        print("ERROR: No matching comparisons found!")
        return
    
    print(f"Found {len(human_choices)} matching comparisons")
    
    # Enhance with logprobs data
    print("Enhancing with logprobs data...")
    enhanced_match_details = match_with_logprobs(match_details, logprobs_cache)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(human_choices, predicted_choices)
    
    # Analyze disagreements
    print("Analyzing disagreements...")
    disagreement_analysis = analyze_disagreements(enhanced_match_details)
    
    # Analyze perplexity patterns
    print("Analyzing perplexity patterns...")
    perplexity_analysis = analyze_perplexity_patterns(enhanced_match_details)
    
    # Generate report
    report = generate_report(metrics, disagreement_analysis, enhanced_match_details, perplexity_analysis)
    print(report)
    
    # Save report
    report_path = "/home/ubuntu/wellactually/results/l1_validation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        metrics['labels'],
        "/home/ubuntu/wellactually/plots/l1_validation_confusion_matrix.png"
    )
    
    # Save detailed match results with perplexity data
    match_df = pd.DataFrame(enhanced_match_details)
    match_df.to_csv("/home/ubuntu/wellactually/results/l1_validation_matches.csv", index=False)
    print(f"Detailed matches saved to: /home/ubuntu/wellactually/results/l1_validation_matches.csv")


if __name__ == "__main__":
    main()