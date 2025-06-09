# src/uncertainty_calibration/l1_multi_model.py
"""
Multi-model analysis functions for L1 validation.
Handles cross-model comparisons, statistical analysis, and model performance evaluation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any
from scipy import stats
from .l1_visualization import (
    create_model_accuracy_comparison,
    create_uncertainty_distribution_comparison, 
    create_precision_rejection_comparison
)
import warnings
warnings.filterwarnings('ignore')

def generate_cross_model_analysis(individual_results: Dict, comparison_dir: Path) -> Dict:
    """
    Generate cross-model comparison analysis and plots.
    
    Args:
        individual_results: Dictionary of model results
        comparison_dir: Directory to save comparison plots
        
    Returns:
        Dictionary with comparison statistics
    """
    print("\nGenerating cross-model comparisons...")
    
    # Extract comparison data
    model_names = list(individual_results.keys())
    comparison_stats = {}
    
    # 1. Model Accuracy Comparison
    create_model_accuracy_comparison(individual_results, comparison_dir)
    
    # 2. Uncertainty Distribution Comparison
    create_uncertainty_distribution_comparison(individual_results, comparison_dir)
    
    # 3. Precision-Rejection Comparison
    create_precision_rejection_comparison(individual_results, comparison_dir)
    
    # 4. Summary Statistics Table
    summary_stats = create_cross_model_summary(individual_results, comparison_dir)
    
    # 5. Statistical Significance Testing
    significance_results = perform_statistical_tests(individual_results, comparison_dir)
    
    comparison_stats = {
        'summary_statistics': summary_stats,
        'statistical_significance': significance_results,
        'best_model_overall': max(individual_results.keys(), 
                                key=lambda x: individual_results[x]['overall_accuracy']),
        'model_count': len(model_names),
        'total_comparisons': len(model_names) * (len(model_names) - 1) // 2
    }
    
    return comparison_stats

def create_cross_model_summary(individual_results: Dict, save_dir: Path) -> Dict:
    """Create cross-model summary statistics."""
    summary_data = []
    
    for model, results in individual_results.items():
        df = results['results_df']
        precision_data = results['precision_data']
        
        # Calculate additional metrics
        mean_uncertainty = df['raw_uncertainty'].mean()
        uncertainty_std = df['raw_uncertainty'].std()
        
        # Best precision-rejection performance
        max_improvement = 0
        if len(precision_data) > 0:
            baseline = results['overall_accuracy']
            max_acc = precision_data['accuracy'].max()
            max_improvement = max_acc - baseline
        
        summary_data.append({
            'model': model,
            'overall_accuracy': results['overall_accuracy'],
            'class_A_accuracy': results['class_accuracy']['A'],
            'class_B_accuracy': results['class_accuracy']['B'],
            'class_Equal_accuracy': results['class_accuracy']['Equal'],
            'mean_uncertainty': mean_uncertainty,
            'uncertainty_std': uncertainty_std,
            'max_precision_improvement': max_improvement,
            'total_predictions': len(df)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.round(4)
    
    # Save summary table
    summary_df.to_csv(save_dir / "cross_model_summary.csv", index=False)
    
    # Create summary dictionary
    summary_dict = summary_df.to_dict('records')
    
    return summary_dict

def perform_statistical_tests(individual_results: Dict, save_dir: Path) -> Dict:
    """Perform statistical significance tests between models."""
    
    model_names = list(individual_results.keys())
    significance_results = {}
    
    # Pairwise accuracy comparisons
    pairwise_tests = {}
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            # Get accuracy arrays for both models
            df1 = individual_results[model1]['results_df']
            df2 = individual_results[model2]['results_df']
            
            acc1 = df1['is_correct'].values
            acc2 = df2['is_correct'].values
            
            # Perform chi-square test for independence
            try:
                # Create contingency table
                correct1, total1 = acc1.sum(), len(acc1)
                correct2, total2 = acc2.sum(), len(acc2)
                
                contingency = [[correct1, total1 - correct1],
                             [correct2, total2 - correct2]]
                
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                
                pairwise_tests[f"{model1} vs {model2}"] = {
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'significant_at_0.05': p_value < 0.05,
                    'accuracy_difference': (correct1/total1) - (correct2/total2)
                }
            except Exception as e:
                pairwise_tests[f"{model1} vs {model2}"] = {
                    'error': str(e)
                }
    
    significance_results['pairwise_accuracy_tests'] = pairwise_tests
    
    # Additional statistical analyses
    significance_results['model_rankings'] = create_model_rankings(individual_results)
    significance_results['uncertainty_correlations'] = analyze_uncertainty_correlations(individual_results)
    
    # Save statistical results
    with open(save_dir / "statistical_comparison.json", 'w') as f:
        json.dump(significance_results, f, indent=2, default=str)
    
    return significance_results

def create_model_rankings(individual_results: Dict) -> Dict:
    """Create model rankings based on different metrics."""
    models_data = []
    
    for model, results in individual_results.items():
        df = results['results_df']
        precision_data = results['precision_data']
        
        # Calculate various metrics
        overall_acc = results['overall_accuracy']
        mean_uncertainty = df['raw_uncertainty'].mean()
        uncertainty_calibration = calculate_uncertainty_calibration_score(df)
        
        # Best precision-rejection improvement
        max_improvement = 0
        if len(precision_data) > 0:
            max_acc = precision_data['accuracy'].max()
            max_improvement = max_acc - overall_acc
        
        models_data.append({
            'model': model,
            'overall_accuracy': overall_acc,
            'mean_uncertainty': mean_uncertainty,
            'uncertainty_calibration': uncertainty_calibration,
            'max_precision_improvement': max_improvement
        })
    
    rankings = {}
    
    # Rank by different metrics
    metrics = ['overall_accuracy', 'uncertainty_calibration', 'max_precision_improvement']
    for metric in metrics:
        sorted_models = sorted(models_data, key=lambda x: x[metric], reverse=True)
        rankings[f'rank_by_{metric}'] = [model['model'] for model in sorted_models]
    
    # Combined ranking (average of ranks)
    combined_scores = {}
    for model_data in models_data:
        model = model_data['model']
        total_rank = 0
        for metric in metrics:
            rank = rankings[f'rank_by_{metric}'].index(model) + 1
            total_rank += rank
        combined_scores[model] = total_rank / len(metrics)
    
    rankings['rank_combined'] = sorted(combined_scores.keys(), key=lambda x: combined_scores[x])
    
    return rankings

def analyze_uncertainty_correlations(individual_results: Dict) -> Dict:
    """Analyze correlations between model uncertainties."""
    correlations = {}
    
    # Get uncertainty data for each model on same questions
    model_uncertainties = {}
    common_questions = None
    
    for model, results in individual_results.items():
        df = results['results_df']
        if 'question_id' in df.columns:
            model_questions = set(df['question_id'])
            if common_questions is None:
                common_questions = model_questions
            else:
                common_questions = common_questions.intersection(model_questions)
            
            model_uncertainties[model] = df.set_index('question_id')['raw_uncertainty']
    
    if common_questions and len(common_questions) > 1:
        # Calculate pairwise correlations on common questions
        model_names = list(model_uncertainties.keys())
        correlation_matrix = {}
        
        for i, model1 in enumerate(model_names):
            correlation_matrix[model1] = {}
            for model2 in model_names:
                if model1 == model2:
                    correlation_matrix[model1][model2] = 1.0
                else:
                    # Get uncertainties for common questions
                    unc1 = [model_uncertainties[model1][q] for q in common_questions 
                           if q in model_uncertainties[model1].index and q in model_uncertainties[model2].index]
                    unc2 = [model_uncertainties[model2][q] for q in common_questions 
                           if q in model_uncertainties[model1].index and q in model_uncertainties[model2].index]
                    
                    if len(unc1) > 1:
                        corr, p_value = stats.pearsonr(unc1, unc2)
                        correlation_matrix[model1][model2] = {
                            'correlation': corr,
                            'p_value': p_value,
                            'n_samples': len(unc1)
                        }
                    else:
                        correlation_matrix[model1][model2] = None
        
        correlations['uncertainty_correlations'] = correlation_matrix
        correlations['common_questions_count'] = len(common_questions)
    else:
        correlations['error'] = "No common questions found or insufficient data"
    
    return correlations

def calculate_uncertainty_calibration_score(df: pd.DataFrame) -> float:
    """
    Calculate a simple uncertainty calibration score.
    Higher score means better calibration (uncertainty correlates with incorrectness).
    """
    try:
        # Calculate correlation between uncertainty and incorrectness
        incorrectness = (~df['is_correct']).astype(int)
        uncertainty = df['raw_uncertainty']
        
        if len(uncertainty.unique()) > 1:
            correlation, _ = stats.pearsonr(uncertainty, incorrectness)
            # Return correlation (higher = better calibration)
            return correlation if not np.isnan(correlation) else 0.0
        else:
            return 0.0
    except:
        return 0.0

def calculate_model_agreement_matrix(individual_results: Dict) -> Dict:
    """
    Calculate agreement matrix between models on same questions.
    
    Args:
        individual_results: Dictionary of model results
        
    Returns:
        Dictionary with agreement statistics
    """
    agreement_data = {}
    
    # Get predictions for each model on same questions
    model_predictions = {}
    common_questions = None
    
    for model, results in individual_results.items():
        df = results['results_df']
        if 'question_id' in df.columns:
            model_questions = set(df['question_id'])
            if common_questions is None:
                common_questions = model_questions
            else:
                common_questions = common_questions.intersection(model_questions)
            
            model_predictions[model] = df.set_index('question_id')['model_prediction']
    
    if common_questions and len(common_questions) > 1:
        model_names = list(model_predictions.keys())
        agreement_matrix = {}
        
        for i, model1 in enumerate(model_names):
            agreement_matrix[model1] = {}
            for model2 in model_names:
                if model1 == model2:
                    agreement_matrix[model1][model2] = 1.0
                else:
                    # Calculate agreement on common questions
                    agreements = []
                    for q in common_questions:
                        if q in model_predictions[model1].index and q in model_predictions[model2].index:
                            pred1 = model_predictions[model1][q]
                            pred2 = model_predictions[model2][q]
                            agreements.append(1 if pred1 == pred2 else 0)
                    
                    if agreements:
                        agreement_rate = np.mean(agreements)
                        agreement_matrix[model1][model2] = {
                            'agreement_rate': agreement_rate,
                            'n_comparisons': len(agreements)
                        }
                    else:
                        agreement_matrix[model1][model2] = None
        
        agreement_data['prediction_agreements'] = agreement_matrix
        agreement_data['common_questions_count'] = len(common_questions)
        
        # Calculate overall agreement statistics
        all_agreements = []
        for model1 in model_names:
            for model2 in model_names:
                if model1 != model2 and agreement_matrix[model1][model2] is not None:
                    all_agreements.append(agreement_matrix[model1][model2]['agreement_rate'])
        
        if all_agreements:
            agreement_data['overall_statistics'] = {
                'mean_agreement': np.mean(all_agreements),
                'std_agreement': np.std(all_agreements),
                'min_agreement': np.min(all_agreements),
                'max_agreement': np.max(all_agreements)
            }
    else:
        agreement_data['error'] = "No common questions found"
    
    return agreement_data