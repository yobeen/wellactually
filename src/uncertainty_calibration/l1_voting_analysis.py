# src/uncertainty_calibration/l1_voting_analysis.py
"""
Voting analysis with per-model rejection for L1 validation.
Implements majority voting with uncertainty-based tie-breaking after independent model rejection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging

from .l1_utils import (
    group_data_by_model,
    sanitize_model_name,
    generate_timestamp,
    create_results_directory,
    save_voting_metadata,
    convert_calibration_points_to_dataframe,
    validate_calibration_data
)
from .l1_visualization import (
    create_voting_accuracy_curves,
    create_voting_detailed_analysis
)
from .l1_core_analysis import analyze_accuracy, analyze_precision_rejection

logger = logging.getLogger(__name__)

def run_voting_analysis(calibration_data_points, rejection_rates: Optional[List[float]] = None,
                       base_save_dir: Optional[str] = None) -> Path:
    """
    Main entry point for voting analysis with per-model rejection.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        rejection_rates: List of rejection rates to test (0-100)
        base_save_dir: Optional base directory override
        
    Returns:
        Path to results directory
    """
    print("\n" + "="*60)
    print("L1 VOTING ANALYSIS WITH PER-MODEL REJECTION")
    print("="*60)
    
    # Validate input data
    validation_result = validate_calibration_data(calibration_data_points)
    if not validation_result['valid']:
        print(f"Data validation failed: {validation_result['error']}")
        return None
    
    # Check if we have multiple models
    models_data = group_data_by_model(calibration_data_points)
    if len(models_data) < 2:
        print("Voting analysis requires at least 2 models!")
        return None
    
    print(f"Analyzing voting with {len(models_data)} models: {list(models_data.keys())}")
    
    # Default rejection rates if not provided
    if rejection_rates is None:
        rejection_rates = list(range(0, 95, 5))  # 0%, 5%, 10%, ..., 90%
    
    # Create directory structure
    timestamp = generate_timestamp()
    model_names = list(models_data.keys()) + ["voting"]
    base_dir = create_results_directory(timestamp, model_names, base_save_dir)
    voting_dir = base_dir / "voting"
    
    # Run voting analysis
    voting_results = analyze_voting_with_per_model_rejection(
        calibration_data_points, rejection_rates, voting_dir
    )
    
    # Get individual model results for comparison
    individual_results = {}
    for model_id, model_data_points in models_data.items():
        model_dir = base_dir / sanitize_model_name(model_id)
        overall_acc, class_acc, results_df = analyze_accuracy(model_data_points, model_dir)
        precision_data = analyze_precision_rejection(model_data_points, model_dir)
        
        individual_results[model_id] = {
            'overall_accuracy': overall_acc,
            'class_accuracy': class_acc,
            'results_df': results_df,
            'precision_data': precision_data,
            'data_points': model_data_points
        }
    
    # Create voting visualizations
    create_voting_accuracy_curves(voting_results, voting_dir, individual_results)
    create_voting_detailed_analysis(voting_results, voting_dir)
    
    # Save metadata
    save_voting_metadata(base_dir, models_data, timestamp, voting_results, rejection_rates)
    
    # Print summary
    print_voting_summary(voting_results, individual_results)
    
    print(f"\nVoting analysis complete!")
    print(f"Results saved to: {base_dir}")
    print(f"Voting-specific results in: {voting_dir}")
    
    return base_dir

def analyze_voting_with_per_model_rejection(calibration_data_points: List, 
                                          rejection_rates: List[float],
                                          save_dir: Path) -> List[Dict]:
    """
    Analyze voting performance with per-model rejection at different rates.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        rejection_rates: List of rejection rates to test
        save_dir: Directory to save results
        
    Returns:
        List of voting results at each rejection rate
    """
    # Group data by model
    models_data = group_data_by_model(calibration_data_points)
    
    # Get all unique questions
    all_questions = set()
    for model_data in models_data.values():
        for dp in model_data:
            all_questions.add(dp.question_id)
    
    # Create human labels mapping
    human_labels = {}
    for dp in calibration_data_points:
        if dp.human_multiplier <= 1.2:
            human_label = "Equal"
        elif dp.human_choice == 1.0:
            human_label = "A"
        elif dp.human_choice == 2.0:
            human_label = "B"
        else:
            human_label = "Equal"
        human_labels[dp.question_id] = human_label
    
    results = []
    
    for rejection_rate in rejection_rates:
        print(f"Processing rejection rate: {rejection_rate}%")
        
        # Step 1: Apply per-model rejection
        filtered_predictions = apply_per_model_rejection(models_data, rejection_rate)
        
        # Step 2: Regroup by question
        question_votes = regroup_by_question(filtered_predictions)
        
        # Step 3: Apply voting with dual-mode evaluation
        voting_result = voting_with_dual_mode(question_votes, human_labels, all_questions)
        
        # Add rejection rate to result
        voting_result['rejection_rate'] = rejection_rate
        voting_result['total_questions'] = len(all_questions)
        
        results.append(voting_result)
    
    # Save detailed results
    results_df = pd.DataFrame([
        {
            'rejection_rate': r['rejection_rate'],
            'mode1_accuracy': r['mode1_accuracy'],
            'mode2_accuracy': r['mode2_accuracy'],
            'samples_remaining': r['samples_remaining'],
            'questions_without_votes': len(r['questions_without_votes']),
            'total_questions': r['total_questions']
        }
        for r in results
    ])
    
    results_df.to_csv(save_dir / "voting_results_summary.csv", index=False)
    
    return results

def apply_per_model_rejection(models_data: Dict, rejection_rate: float) -> Dict:
    """
    Apply rejection threshold independently to each model.
    
    Args:
        models_data: Dictionary mapping model_id to list of CalibrationDataPoint objects
        rejection_rate: Rejection rate (0-100)
        
    Returns:
        Dictionary mapping model_id to filtered predictions
    """
    filtered_predictions = {}
    
    for model_id, model_data_points in models_data.items():
        if not model_data_points:
            filtered_predictions[model_id] = []
            continue
        
        # Calculate rejection threshold for this model
        uncertainties = [dp.raw_uncertainty for dp in model_data_points]
        threshold = calculate_rejection_threshold(uncertainties, rejection_rate)
        
        # Filter predictions based on threshold
        filtered = []
        for dp in model_data_points:
            if dp.raw_uncertainty <= threshold:
                filtered.append({
                    'question_id': dp.question_id,
                    'prediction': dp.model_prediction,
                    'uncertainty': dp.raw_uncertainty,
                    'model_id': model_id
                })
        
        filtered_predictions[model_id] = filtered
        
        logger.debug(f"Model {model_id}: {len(filtered)}/{len(model_data_points)} predictions kept "
                    f"(threshold={threshold:.3f})")
    
    return filtered_predictions

def calculate_rejection_threshold(uncertainties: List[float], rejection_rate: float) -> float:
    """
    Calculate rejection threshold for given rejection rate.
    
    Args:
        uncertainties: List of uncertainty values
        rejection_rate: Percentage of samples to reject (0-100)
        
    Returns:
        Threshold value
    """
    if rejection_rate <= 0:
        return 1.0  # Keep all
    
    if rejection_rate >= 100:
        return -1.0  # Reject all
    
    # Sort uncertainties in descending order (highest uncertainty first)
    sorted_uncertainties = sorted(uncertainties, reverse=True)
    
    # Calculate number of samples to reject
    n_reject = int(len(uncertainties) * rejection_rate / 100)
    
    if n_reject >= len(uncertainties):
        return -1.0  # Reject all
    elif n_reject == 0:
        return 1.0  # Keep all
    else:
        # Threshold is the uncertainty of the last rejected sample
        return sorted_uncertainties[n_reject - 1]

def regroup_by_question(filtered_predictions: Dict) -> Dict:
    """
    Regroup filtered predictions by question ID.
    
    Args:
        filtered_predictions: Dictionary mapping model_id to filtered predictions
        
    Returns:
        Dictionary mapping question_id to list of predictions
    """
    question_votes = defaultdict(list)
    
    for model_id, predictions in filtered_predictions.items():
        for pred in predictions:
            question_votes[pred['question_id']].append({
                'model_id': model_id,
                'prediction': pred['prediction'],
                'uncertainty': pred['uncertainty']
            })
    
    return dict(question_votes)

def voting_with_dual_mode(question_votes: Dict, human_labels: Dict, 
                         all_questions: set) -> Dict:
    """
    Apply majority voting with dual-mode evaluation.
    
    Args:
        question_votes: Dictionary mapping question_id to list of votes
        human_labels: Dictionary mapping question_id to human labels
        all_questions: Set of all question IDs
        
    Returns:
        Dictionary with voting results and dual-mode accuracies
    """
    questions_with_votes = []
    questions_without_votes = []
    
    for question_id in all_questions:
        if question_id in question_votes and question_votes[question_id]:
            # Apply majority voting
            voted_answer = majority_vote_with_uncertainty_tiebreak(question_votes[question_id])
            questions_with_votes.append((question_id, voted_answer))
        else:
            questions_without_votes.append(question_id)
    
    # Calculate accuracies
    correct_votes = 0
    for question_id, voted_answer in questions_with_votes:
        if question_id in human_labels and voted_answer == human_labels[question_id]:
            correct_votes += 1
    
    # Mode 1: Count missing as wrong
    mode1_accuracy = correct_votes / len(all_questions) if all_questions else 0.0
    
    # Mode 2: Exclude missing from calculation
    mode2_accuracy = correct_votes / len(questions_with_votes) if questions_with_votes else 0.0
    
    return {
        'questions_with_votes': questions_with_votes,
        'questions_without_votes': questions_without_votes,
        'mode1_accuracy': mode1_accuracy,
        'mode2_accuracy': mode2_accuracy,
        'samples_remaining': len(questions_with_votes),
        'correct_votes': correct_votes
    }

def majority_vote_with_uncertainty_tiebreak(votes: List[Dict]) -> str:
    """
    Apply majority voting with uncertainty-based tie-breaking.
    
    Args:
        votes: List of vote dictionaries with prediction and uncertainty
        
    Returns:
        Winning answer
    """
    if not votes:
        return "Equal"  # Default fallback
    
    # Count votes for each answer
    vote_counts = Counter(vote['prediction'] for vote in votes)
    max_votes = max(vote_counts.values())
    
    # Get answers with maximum votes
    tied_answers = [answer for answer, count in vote_counts.items() if count == max_votes]
    
    if len(tied_answers) == 1:
        return tied_answers[0]
    
    # Tie-breaking: select answer with minimum uncertainty sum
    uncertainty_sums = {}
    for answer in tied_answers:
        uncertainty_sum = sum(
            vote['uncertainty'] for vote in votes 
            if vote['prediction'] == answer
        )
        uncertainty_sums[answer] = uncertainty_sum
    
    # Return answer with minimum uncertainty sum
    winner = min(uncertainty_sums.keys(), key=lambda x: uncertainty_sums[x])
    
    logger.debug(f"Tie-break: {tied_answers} -> {winner} "
                f"(uncertainty sums: {uncertainty_sums})")
    
    return winner

def print_voting_summary(voting_results: List[Dict], individual_results: Dict):
    """Print summary of voting analysis results."""
    
    print("\nVOTING ANALYSIS SUMMARY")
    print("="*50)
    
    # Baseline performance (0% rejection)
    baseline = voting_results[0]
    print(f"Baseline Performance (0% rejection):")
    print(f"  Mode 1 Accuracy (penalize missing): {baseline['mode1_accuracy']:.3f}")
    print(f"  Mode 2 Accuracy (exclude missing): {baseline['mode2_accuracy']:.3f}")
    print(f"  Questions with votes: {baseline['samples_remaining']}/{baseline['total_questions']}")
    
    # Best performance for each mode
    best_mode1 = max(voting_results, key=lambda x: x['mode1_accuracy'])
    best_mode2 = max(voting_results, key=lambda x: x['mode2_accuracy'])
    
    print(f"\nBest Mode 1 Performance:")
    print(f"  Accuracy: {best_mode1['mode1_accuracy']:.3f} at {best_mode1['rejection_rate']:.0f}% rejection")
    print(f"  Improvement: +{best_mode1['mode1_accuracy'] - baseline['mode1_accuracy']:.3f}")
    print(f"  Questions remaining: {best_mode1['samples_remaining']}")
    
    print(f"\nBest Mode 2 Performance:")
    print(f"  Accuracy: {best_mode2['mode2_accuracy']:.3f} at {best_mode2['rejection_rate']:.0f}% rejection")
    print(f"  Improvement: +{best_mode2['mode2_accuracy'] - baseline['mode2_accuracy']:.3f}")
    print(f"  Questions remaining: {best_mode2['samples_remaining']}")
    
    # Individual model comparison
    print(f"\nIndividual Model Baselines:")
    best_individual = max(individual_results.items(), key=lambda x: x[1]['overall_accuracy'])
    for model_id, results in individual_results.items():
        marker = " ⭐" if model_id == best_individual[0] else ""
        print(f"  {model_id}: {results['overall_accuracy']:.3f}{marker}")
    
    # Voting advantage
    voting_baseline = baseline['mode2_accuracy']
    best_individual_acc = best_individual[1]['overall_accuracy']
    voting_advantage = voting_baseline - best_individual_acc
    
    print(f"\nVoting vs Best Individual Model:")
    print(f"  Best individual: {best_individual_acc:.3f} ({best_individual[0]})")
    print(f"  Voting baseline: {voting_baseline:.3f}")
    if voting_advantage > 0:
        print(f"  Voting advantage: +{voting_advantage:.3f}")
    else:
        print(f"  Individual advantage: +{-voting_advantage:.3f}")

def analyze_voting_model_contributions(question_votes: Dict, save_dir: Path) -> Dict:
    """
    Analyze how much each model contributes to voting decisions.
    
    Args:
        question_votes: Dictionary mapping question_id to list of votes
        save_dir: Directory to save analysis
        
    Returns:
        Dictionary with contribution statistics
    """
    model_stats = defaultdict(lambda: {
        'total_votes': 0,
        'winning_votes': 0,
        'decisive_votes': 0,  # Votes that broke ties
        'questions_participated': set()
    })
    
    for question_id, votes in question_votes.items():
        if not votes:
            continue
        
        # Count votes per model for this question
        for vote in votes:
            model_id = vote['model_id']
            model_stats[model_id]['total_votes'] += 1
            model_stats[model_id]['questions_participated'].add(question_id)
        
        # Determine winner and contributions
        winner = majority_vote_with_uncertainty_tiebreak(votes)
        vote_counts = Counter(vote['prediction'] for vote in votes)
        
        # Mark winning votes
        for vote in votes:
            if vote['prediction'] == winner:
                model_stats[vote['model_id']]['winning_votes'] += 1
        
        # Check if this was decided by tie-breaking
        max_votes = max(vote_counts.values())
        tied_count = sum(1 for count in vote_counts.values() if count == max_votes)
        
        if tied_count > 1:  # There was a tie
            # Models that voted for winner in tie situation
            for vote in votes:
                if vote['prediction'] == winner:
                    model_stats[vote['model_id']]['decisive_votes'] += 1
    
    # Convert to regular dict and calculate percentages
    contribution_stats = {}
    for model_id, stats in model_stats.items():
        total_votes = stats['total_votes']
        contribution_stats[model_id] = {
            'total_votes': total_votes,
            'winning_votes': stats['winning_votes'],
            'winning_percentage': stats['winning_votes'] / total_votes if total_votes > 0 else 0,
            'decisive_votes': stats['decisive_votes'],
            'questions_participated': len(stats['questions_participated']),
            'participation_rate': len(stats['questions_participated'])
        }
    
    # Save contribution analysis
    contrib_df = pd.DataFrame.from_dict(contribution_stats, orient='index')
    contrib_df.to_csv(save_dir / "model_contributions.csv")
    
    return contribution_stats