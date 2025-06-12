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
    
    # Run hierarchical voting analysis
    print("\n" + "-"*60)
    print("HIERARCHICAL VOTING ANALYSIS")
    print("-"*60)
    hierarchical_result = analyze_hierarchical_voting(calibration_data_points, voting_dir)
    
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
    
    # Save metadata (include hierarchical result)
    save_voting_metadata(base_dir, models_data, timestamp, voting_results, rejection_rates, hierarchical_result)
    
    # Print summary
    print_voting_summary(voting_results, individual_results)
    print_hierarchical_summary(hierarchical_result)
    
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

def analyze_hierarchical_voting(calibration_data_points: List, save_dir: Path) -> Dict:
    """
    Analyze hierarchical voting with chain-like rejection: GPT-4 -> Llama -> Gemma -> Mistral.
    
    Process:
    1. GPT-4 50% rejection on full dataset - take decisions for all questions
    2. Remaining: Llama 50% rejection on full dataset - take decisions for remaining questions  
    3. Remaining: Gemma 50% rejection on full dataset - take decisions for remaining questions
    4. Remaining: Mistral 50% rejection on full dataset - take decisions for remaining questions
    5. Final remaining: majority vote
    
    Each model performs rejection on the complete dataset but only makes decisions 
    for questions not yet decided by previous models in the hierarchy.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        save_dir: Directory to save results
        
    Returns:
        Dictionary with hierarchical voting results
    """
    print(f"Processing hierarchical voting...")
    
    # Group data by model
    models_data = group_data_by_model(calibration_data_points)
    
    # Define model hierarchy (order matters)
    model_hierarchy = [
            "google/gemma-3-27b-it",
            "openai/gpt-4o",
            "meta-llama/llama-4-maverick",
    ]
    
    # Filter to only include models we have data for
    available_models = [m for m in model_hierarchy if m in models_data]
    
    if len(available_models) < 2:
        print(f"Hierarchical voting requires at least 2 models. Available: {available_models}")
        return {}
    
    print(f"Using hierarchical order: {available_models}")
    
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
    
    # Apply hierarchical voting
    hierarchical_result = apply_hierarchical_voting(
        models_data, available_models, all_questions, human_labels
    )
    
    # Save detailed results
    hierarchy_df = pd.DataFrame([
        {
            'stage': stage,
            'model': data['model'],
            'questions_decided': data['questions_decided'],
            'questions_remaining': data['questions_remaining'],
            'stage_accuracy': data['stage_accuracy'],
            'cumulative_accuracy': data['cumulative_accuracy']
        }
        for stage, data in hierarchical_result['stage_results'].items()
    ])
    
    hierarchy_df.to_csv(save_dir / "hierarchical_voting_results.csv", index=False)
    
    # Save final decisions
    final_decisions = []
    for question_id, decision_info in hierarchical_result['final_decisions'].items():
        final_decisions.append({
            'question_id': question_id,
            'decision': decision_info['decision'],
            'decided_by': decision_info['decided_by'],
            'stage': decision_info['stage'],
            'human_label': human_labels.get(question_id, 'Unknown'),
            'correct': decision_info['decision'] == human_labels.get(question_id, 'Unknown')
        })
    
    decisions_df = pd.DataFrame(final_decisions)
    decisions_df.to_csv(save_dir / "hierarchical_decisions.csv", index=False)
    
    return hierarchical_result

def apply_hierarchical_voting(models_data: Dict, model_hierarchy: List[str], 
                            all_questions: set, human_labels: Dict) -> Dict:
    """
    Apply hierarchical voting with chain-like rejection.
    
    Args:
        models_data: Dictionary mapping model_id to list of CalibrationDataPoint objects
        model_hierarchy: List of model IDs in hierarchical order
        all_questions: Set of all question IDs
        human_labels: Dictionary mapping question_id to human labels
        
    Returns:
        Dictionary with hierarchical voting results
    """
    remaining_questions = set(all_questions)
    final_decisions = {}
    stage_results = {}
    rejection_rate = 50.0  # 50% rejection at each stage
    
    for stage_idx, model_id in enumerate(model_hierarchy):
        if not remaining_questions:
            break
            
        stage_name = f"stage_{stage_idx + 1}"
        print(f"Stage {stage_idx + 1}: {model_id} - rejection on full dataset, deciding {len(remaining_questions)} remaining questions")
        
        # Get ALL model data for full dataset rejection
        model_data_points = models_data.get(model_id, [])
        
        if not model_data_points:
            stage_results[stage_name] = {
                'model': model_id,
                'questions_decided': 0,
                'questions_remaining': len(remaining_questions),
                'stage_accuracy': 0.0,
                'cumulative_accuracy': 0.0
            }
            continue
        
        # Apply 50% rejection to ALL model predictions (full dataset)
        uncertainties = [dp.raw_uncertainty for dp in model_data_points]
        threshold = calculate_rejection_threshold(uncertainties, rejection_rate)
        
        # Select confident predictions (below threshold) from full dataset
        all_confident_predictions = []
        for dp in model_data_points:
            if dp.raw_uncertainty <= threshold:
                all_confident_predictions.append(dp)
        
        # Filter to only use confident predictions for remaining questions
        confident_predictions = [
            dp for dp in all_confident_predictions 
            if dp.question_id in remaining_questions
        ]
        
        # Make decisions for confident predictions
        questions_decided_this_stage = set()
        stage_correct = 0
        
        for dp in confident_predictions:
            question_id = dp.question_id
            decision = dp.model_prediction
            
            final_decisions[question_id] = {
                'decision': decision,
                'decided_by': model_id,
                'stage': stage_idx + 1,
                'uncertainty': dp.raw_uncertainty
            }
            
            questions_decided_this_stage.add(question_id)
            
            # Check if correct
            if question_id in human_labels and decision == human_labels[question_id]:
                stage_correct += 1
        
        # Update remaining questions
        remaining_questions -= questions_decided_this_stage
        
        # Calculate accuracies
        stage_accuracy = stage_correct / len(questions_decided_this_stage) if questions_decided_this_stage else 0.0
        
        total_decided = len(final_decisions)
        total_correct = sum(
            1 for q_id, decision_info in final_decisions.items()
            if q_id in human_labels and decision_info['decision'] == human_labels[q_id]
        )
        cumulative_accuracy = total_correct / total_decided if total_decided > 0 else 0.0
        
        stage_results[stage_name] = {
            'model': model_id,
            'questions_decided': len(questions_decided_this_stage),
            'questions_remaining': len(remaining_questions),
            'stage_accuracy': stage_accuracy,
            'cumulative_accuracy': cumulative_accuracy
        }
        
        print(f"  Decided {len(questions_decided_this_stage)} questions with {stage_accuracy:.3f} accuracy")
        print(f"  {len(remaining_questions)} questions remaining")
    
    # Handle remaining questions with majority vote
    if remaining_questions:
        print(f"Final stage: Majority vote for {len(remaining_questions)} remaining questions")
        
        # Collect all available predictions for remaining questions
        remaining_votes = {}
        for question_id in remaining_questions:
            votes = []
            for model_id, model_data_points in models_data.items():
                for dp in model_data_points:
                    if dp.question_id == question_id:
                        votes.append({
                            'model_id': model_id,
                            'prediction': dp.model_prediction,
                            'uncertainty': dp.raw_uncertainty
                        })
            if votes:
                remaining_votes[question_id] = votes
        
        # Apply majority voting to remaining questions
        majority_correct = 0
        for question_id, votes in remaining_votes.items():
            if votes:
                decision = majority_vote_with_uncertainty_tiebreak(votes)
                final_decisions[question_id] = {
                    'decision': decision,
                    'decided_by': 'majority_vote',
                    'stage': len(model_hierarchy) + 1,
                    'uncertainty': np.mean([v['uncertainty'] for v in votes])
                }
                
                if question_id in human_labels and decision == human_labels[question_id]:
                    majority_correct += 1
        
        majority_accuracy = majority_correct / len(remaining_votes) if remaining_votes else 0.0
        
        # Add majority vote stage results
        total_decided = len(final_decisions)
        total_correct = sum(
            1 for q_id, decision_info in final_decisions.items()
            if q_id in human_labels and decision_info['decision'] == human_labels[q_id]
        )
        final_cumulative_accuracy = total_correct / total_decided if total_decided > 0 else 0.0
        
        stage_results[f'stage_{len(model_hierarchy) + 1}'] = {
            'model': 'majority_vote',
            'questions_decided': len(remaining_votes),
            'questions_remaining': 0,
            'stage_accuracy': majority_accuracy,
            'cumulative_accuracy': final_cumulative_accuracy
        }
        
        print(f"  Majority vote decided {len(remaining_votes)} questions with {majority_accuracy:.3f} accuracy")
    
    # Calculate overall performance
    total_correct = sum(
        1 for q_id, decision_info in final_decisions.items()
        if q_id in human_labels and decision_info['decision'] == human_labels[q_id]
    )
    overall_accuracy = total_correct / len(all_questions) if all_questions else 0.0
    
    return {
        'final_decisions': final_decisions,
        'stage_results': stage_results,
        'overall_accuracy': overall_accuracy,
        'questions_decided': len(final_decisions),
        'total_questions': len(all_questions),
        'model_hierarchy': model_hierarchy
    }

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

def print_hierarchical_summary(hierarchical_result: Dict):
    """Print summary of hierarchical voting analysis results."""
    
    if not hierarchical_result:
        print("\nNo hierarchical voting results to display.")
        return
    
    print("\nHIERARCHICAL VOTING SUMMARY")
    print("="*50)
    
    print(f"Overall Accuracy: {hierarchical_result['overall_accuracy']:.3f}")
    print(f"Questions Decided: {hierarchical_result['questions_decided']}/{hierarchical_result['total_questions']}")
    print(f"Model Hierarchy: {' → '.join(hierarchical_result['model_hierarchy'])}")
    
    print("\nStage-by-Stage Results:")
    for stage, data in hierarchical_result['stage_results'].items():
        stage_num = stage.replace('stage_', '')
        print(f"  Stage {stage_num} ({data['model']}):")
        print(f"    Questions decided: {data['questions_decided']}")
        print(f"    Stage accuracy: {data['stage_accuracy']:.3f}")
        print(f"    Cumulative accuracy: {data['cumulative_accuracy']:.3f}")
        print(f"    Questions remaining: {data['questions_remaining']}")
    
    # Analyze decision distribution
    decision_by_stage = {}
    for q_id, decision_info in hierarchical_result['final_decisions'].items():
        stage = decision_info['stage']
        if stage not in decision_by_stage:
            decision_by_stage[stage] = 0
        decision_by_stage[stage] += 1
    
    print(f"\nDecision Distribution:")
    total_decisions = sum(decision_by_stage.values())
    for stage in sorted(decision_by_stage.keys()):
        count = decision_by_stage[stage]
        percentage = (count / total_decisions) * 100 if total_decisions > 0 else 0
        stage_name = hierarchical_result['stage_results'].get(f'stage_{stage}', {}).get('model', f'stage_{stage}')
        print(f"  Stage {stage} ({stage_name}): {count} questions ({percentage:.1f}%)")

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