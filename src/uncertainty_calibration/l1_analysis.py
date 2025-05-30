# src/uncertainty_calibration/l1_analysis.py
"""
Enhanced analysis functions for L1 validation results with multi-model support.
Supports both single and multiple model analysis with structured result saving.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def run_analysis(calibration_data_points, base_save_dir: Optional[str] = None):
    """
    Main analysis function with automatic single/multi-model detection.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        base_save_dir: Optional base directory override
        
    Returns:
        Path to results directory
    """
    if not calibration_data_points:
        print("No calibration data points provided!")
        return None
    
    # Group data by model
    models_data = group_data_by_model(calibration_data_points)
    
    if len(models_data) == 1:
        return run_single_model_analysis(calibration_data_points, base_save_dir)
    else:
        return run_multi_model_analysis(calibration_data_points, base_save_dir)

def group_data_by_model(calibration_data_points) -> Dict[str, List]:
    """
    Group calibration data points by model ID.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        
    Returns:
        Dictionary mapping model_id to list of data points
    """
    models_data = defaultdict(list)
    
    for dp in calibration_data_points:
        model_id = getattr(dp, 'model_id', 'unknown_model')
        models_data[model_id].append(dp)
    
    return dict(models_data)

def sanitize_model_name(model_id: str) -> str:
    """
    Convert model ID to filesystem-safe name.
    
    Args:
        model_id: Original model identifier
        
    Returns:
        Filesystem-safe model name
    """
    return model_id.replace('/', '-').replace(':', '-').replace('@', '-')

def generate_timestamp() -> str:
    """Generate filesystem-safe timestamp."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def create_results_directory(timestamp: str, model_names: List[str], 
                           base_save_dir: Optional[str] = None) -> Path:
    """
    Create structured results directory.
    
    Args:
        timestamp: Timestamp string
        model_names: List of model names for subdirectories
        base_save_dir: Optional base directory override
        
    Returns:
        Path to base results directory
    """
    if base_save_dir:
        base_dir = Path(base_save_dir) / timestamp
    else:
        base_dir = Path("results") / timestamp
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model subdirectories
    for model_name in model_names:
        clean_name = sanitize_model_name(model_name)
        model_dir = base_dir / clean_name
        model_dir.mkdir(exist_ok=True)
    
    # Create comparison directory if multiple models
    if len(model_names) > 1:
        comparison_dir = base_dir / "comparison"
        comparison_dir.mkdir(exist_ok=True)
    
    return base_dir

def run_single_model_analysis(calibration_data_points, base_save_dir: Optional[str] = None) -> Path:
    """
    Run analysis for a single model.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        base_save_dir: Optional base directory override
        
    Returns:
        Path to results directory
    """
    print("\n" + "="*50)
    print("L1 VALIDATION ANALYSIS RESULTS (SINGLE MODEL)")
    print("="*50)
    
    # Extract model info
    models_data = group_data_by_model(calibration_data_points)
    model_id = list(models_data.keys())[0]
    
    # Create directory structure
    timestamp = generate_timestamp()
    base_dir = create_results_directory(timestamp, [model_id], base_save_dir)
    model_dir = base_dir / sanitize_model_name(model_id)
    
    # Run analysis
    overall_acc, class_acc, results_df = analyze_accuracy(calibration_data_points, model_dir)
    precision_data = analyze_precision_rejection(calibration_data_points, model_dir)
    create_single_model_plots(results_df, precision_data, overall_acc, model_dir, model_id)
    
    # Save metadata
    save_single_model_metadata(base_dir, model_id, calibration_data_points, timestamp)
    
    print(f"\nSingle model analysis complete!")
    print(f"Results saved to: {base_dir}")
    
    return base_dir

def run_multi_model_analysis(calibration_data_points, base_save_dir: Optional[str] = None) -> Path:
    """
    Run analysis for multiple models with cross-model comparisons.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        base_save_dir: Optional base directory override
        
    Returns:
        Path to results directory
    """
    print("\n" + "="*60)
    print("L1 VALIDATION ANALYSIS RESULTS (MULTI-MODEL)")
    print("="*60)
    
    # Group data by model
    models_data = group_data_by_model(calibration_data_points)
    model_ids = list(models_data.keys())
    
    print(f"Analyzing {len(model_ids)} models: {', '.join(model_ids)}")
    
    # Create directory structure
    timestamp = generate_timestamp()
    base_dir = create_results_directory(timestamp, model_ids, base_save_dir)
    
    # Individual model analysis
    individual_results = {}
    
    for model_id, model_data_points in models_data.items():
        print(f"\nAnalyzing model: {model_id}")
        
        model_dir = base_dir / sanitize_model_name(model_id)
        
        # Run individual analysis
        overall_acc, class_acc, results_df = analyze_accuracy(model_data_points, model_dir)
        precision_data = analyze_precision_rejection(model_data_points, model_dir)
        create_single_model_plots(results_df, precision_data, overall_acc, model_dir, model_id)
        
        # Store results for comparison
        individual_results[model_id] = {
            'overall_accuracy': overall_acc,
            'class_accuracy': class_acc,
            'results_df': results_df,
            'precision_data': precision_data,
            'data_points': model_data_points
        }
        
        print(f"  Overall accuracy: {overall_acc:.3f}")
    
    # Cross-model comparison analysis
    comparison_dir = base_dir / "comparison"
    comparison_results = generate_cross_model_analysis(individual_results, comparison_dir)
    
    # Save metadata
    save_multi_model_metadata(base_dir, models_data, timestamp, comparison_results)
    
    print(f"\nMulti-model analysis complete!")
    print(f"Results saved to: {base_dir}")
    print(f"Cross-model comparisons saved to: {comparison_dir}")
    
    return base_dir

def analyze_accuracy(calibration_data_points, save_dir: Path):
    """
    Calculate overall and per-class accuracy from CalibrationDataPoint objects.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        save_dir: Directory to save results
        
    Returns:
        tuple: (overall_accuracy, class_accuracies_dict, results_dataframe)
    """
    if not calibration_data_points:
        return 0.0, {}, pd.DataFrame()
    
    # Convert to DataFrame for easier analysis
    data = []
    for dp in calibration_data_points:
        # Convert human choice back to A/B/Equal format
        if dp.human_multiplier <= 1.2:
            human_label = "Equal"
        elif dp.human_choice == 1.0:
            human_label = "A"
        elif dp.human_choice == 2.0:
            human_label = "B"
        else:
            human_label = "Equal"
            
        data.append({
            'human_label': human_label,
            'model_prediction': dp.model_prediction,
            'is_correct': dp.is_correct,
            'raw_uncertainty': dp.raw_uncertainty,
            'human_multiplier': dp.human_multiplier
        })
    
    df = pd.DataFrame(data)
    
    # Overall accuracy
    overall_acc = df['is_correct'].mean()
    
    # Per-class accuracy
    class_acc = {}
    for class_name in ['A', 'B', 'Equal']:
        class_data = df[df['human_label'] == class_name]
        if len(class_data) > 0:
            class_acc[class_name] = class_data['is_correct'].mean()
        else:
            class_acc[class_name] = 0.0
    
    # Save detailed results
    df.to_csv(save_dir / "l1_validation_results.csv", index=False)
    
    print(f"Accuracy Analysis:")
    print(f"  Overall Accuracy: {overall_acc:.3f}")
    print("  Per-class Accuracy:")
    for class_name, acc in class_acc.items():
        count = len(df[df['human_label'] == class_name])
        print(f"    {class_name}: {acc:.3f} (n={count})")
    
    return overall_acc, class_acc, df

def analyze_precision_rejection(calibration_data_points, save_dir: Path):
    """
    Generate precision-rejection curve data.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        save_dir: Directory to save results
        
    Returns:
        DataFrame with rejection_rate, accuracy, samples_remaining columns
    """
    if not calibration_data_points:
        return pd.DataFrame()
    
    # Convert to DataFrame and sort by uncertainty descending
    data = []
    for dp in calibration_data_points:
        data.append({
            'raw_uncertainty': dp.raw_uncertainty,
            'is_correct': dp.is_correct
        })
    
    df = pd.DataFrame(data)
    sorted_df = df.sort_values('raw_uncertainty', ascending=False)
    
    rejection_rates = np.arange(0, 95, 5)  # 0%, 5%, 10%, ..., 90%
    precision_data = []
    
    for rate in rejection_rates:
        # Calculate number of samples to remove
        n_total = len(sorted_df)
        n_remove = int(n_total * rate / 100)
        
        # Keep remaining samples
        remaining_df = sorted_df.iloc[n_remove:]
        
        if len(remaining_df) > 0:
            accuracy = remaining_df['is_correct'].mean()
        else:
            accuracy = 0.0
        
        precision_data.append({
            'rejection_rate': rate,
            'accuracy': accuracy,
            'samples_remaining': len(remaining_df)
        })
    
    precision_df = pd.DataFrame(precision_data)
    
    # Save results
    precision_df.to_csv(save_dir / "l1_precision_rejection.csv", index=False)
    
    return precision_df

def create_single_model_plots(results_df: pd.DataFrame, precision_data: pd.DataFrame, 
                            overall_acc: float, save_dir: Path, model_name: str):
    """
    Create and save analysis plots for a single model.
    
    Args:
        results_df: DataFrame with prediction results
        precision_data: DataFrame with precision-rejection data
        overall_acc: Overall accuracy value
        save_dir: Directory to save plots
        model_name: Model name for plot titles
    """
    # Set style with high contrast colors
    plt.style.use('default')
    
    # Define high-contrast color palette
    primary_blue = '#1f77b4'
    vibrant_red = '#d62728'
    vibrant_green = '#2ca02c'
    vibrant_orange = '#ff7f0e'
    deep_purple = '#9467bd'
    
    # 1. Accuracy Analysis Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Overall accuracy
    ax1.bar(['Overall'], [overall_acc], color=primary_blue, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title(f'Overall Accuracy - {model_name}', fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.text(0, overall_acc + 0.02, f'{overall_acc:.3f}', ha='center', fontweight='bold', fontsize=11)
    
    # Per-class accuracy
    class_acc = {}
    for class_name in ['A', 'B', 'Equal']:
        class_data = results_df[results_df['human_label'] == class_name]
        if len(class_data) > 0:
            class_acc[class_name] = class_data['is_correct'].mean()
        else:
            class_acc[class_name] = 0.0
    
    classes = list(class_acc.keys())
    accuracies = list(class_acc.values())
    class_colors = [vibrant_red, vibrant_green, vibrant_orange]
    bars = ax2.bar(classes, accuracies, color=class_colors, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title(f'Per-Class Accuracy - {model_name}', fontweight='bold')
    ax2.set_ylim(0, 1)
    
    # Add count labels
    for i, (class_name, bar) in enumerate(zip(classes, bars)):
        count = len(results_df[results_df['human_label'] == class_name])
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f'{height:.3f}\n(n={count})', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / "l1_accuracy_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Rejection Curve
    plt.figure(figsize=(10, 6))
    plt.plot(precision_data['rejection_rate'], precision_data['accuracy'], 
             marker='o', linewidth=3, markersize=8, color=primary_blue, markerfacecolor='white',
             markeredgecolor=primary_blue, markeredgewidth=2)
    plt.axhline(y=overall_acc, color=vibrant_red, linestyle='--', linewidth=2, alpha=0.8, 
                label=f'Baseline (no rejection): {overall_acc:.3f}')
    plt.xlabel('Rejection Rate (%)', fontweight='bold')
    plt.ylabel('Accuracy on Remaining Samples', fontweight='bold')
    plt.title(f'Precision-Rejection Curve - {model_name}\n(Higher uncertainty samples rejected first)', 
              fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    plt.xlim(0, 90)
    plt.ylim(0, 1)
    
    # Add annotation for improvement
    if len(precision_data) > 0:
        final_acc = precision_data.iloc[-1]['accuracy']
        improvement = final_acc - overall_acc
        plt.text(45, 0.1, f'Max improvement: +{improvement:.3f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='gold', alpha=0.9, edgecolor='black'),
                fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_dir / "l1_precision_rejection.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Uncertainty Distribution by Correctness
    plt.figure(figsize=(10, 6))
    correct_uncertainty = results_df[results_df['is_correct']]['raw_uncertainty']
    incorrect_uncertainty = results_df[~results_df['is_correct']]['raw_uncertainty']
    
    plt.hist(correct_uncertainty, bins=20, alpha=0.75, label='Correct Predictions', 
             color=vibrant_green, density=True, edgecolor='black', linewidth=0.5)
    plt.hist(incorrect_uncertainty, bins=20, alpha=0.75, label='Incorrect Predictions', 
             color=vibrant_red, density=True, edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Raw Uncertainty', fontweight='bold')
    plt.ylabel('Density', fontweight='bold')
    plt.title(f'Uncertainty Distribution by Prediction Correctness - {model_name}', fontweight='bold')
    plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "l1_uncertainty_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

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

def create_model_accuracy_comparison(individual_results: Dict, save_dir: Path):
    """Create model accuracy comparison plots."""
    model_names = list(individual_results.keys())
    overall_accs = [individual_results[model]['overall_accuracy'] for model in model_names]
    
    # High-contrast color palette for multiple models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Overall accuracy comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall accuracy bar plot
    bars = ax1.bar(range(len(model_names)), overall_accs, 
                   color=colors[:len(model_names)], edgecolor='black', linewidth=1)
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('Overall Accuracy', fontweight='bold')
    ax1.set_title('Overall Accuracy Comparison', fontweight='bold')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels([name.replace('/', '\n') for name in model_names], rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, overall_accs)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Per-class accuracy comparison
    class_names = ['A', 'B', 'Equal']
    x = np.arange(len(class_names))
    width = 0.8 / len(model_names)  # Adjust width based on number of models
    
    for i, model in enumerate(model_names):
        class_acc = individual_results[model]['class_accuracy']
        accuracies = [class_acc[cls] for cls in class_names]
        bars = ax2.bar(x + i * width, accuracies, width, 
                      label=model.replace('/', '-'), color=colors[i], 
                      edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            if acc > 0.05:  # Only show label if bar is tall enough
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Class', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Per-Class Accuracy Comparison', fontweight='bold')
    ax2.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax2.set_xticklabels(class_names)
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_dir / "model_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_uncertainty_distribution_comparison(individual_results: Dict, save_dir: Path):
    """Create uncertainty distribution comparison plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # High-contrast colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Correct predictions uncertainty distribution
    for i, (model, results) in enumerate(individual_results.items()):
        df = results['results_df']
        correct_uncertainty = df[df['is_correct']]['raw_uncertainty']
        ax1.hist(correct_uncertainty, bins=15, alpha=0.7, label=model.replace('/', '-'),
                color=colors[i % len(colors)], density=True, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Raw Uncertainty', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Uncertainty Distribution - Correct Predictions', fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Incorrect predictions uncertainty distribution
    for i, (model, results) in enumerate(individual_results.items()):
        df = results['results_df']
        incorrect_uncertainty = df[~df['is_correct']]['raw_uncertainty']
        ax2.hist(incorrect_uncertainty, bins=15, alpha=0.7, label=model.replace('/', '-'),
                color=colors[i % len(colors)], density=True, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Raw Uncertainty', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('Uncertainty Distribution - Incorrect Predictions', fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "uncertainty_distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_precision_rejection_comparison(individual_results: Dict, save_dir: Path):
    """Create precision-rejection curve comparison."""
    plt.figure(figsize=(12, 8))
    
    # High-contrast colors and line styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, (model, results) in enumerate(individual_results.items()):
        precision_data = results['precision_data']
        plt.plot(precision_data['rejection_rate'], precision_data['accuracy'],
                marker=markers[i % len(markers)], linewidth=3, markersize=8,
                label=model.replace('/', '-'), color=colors[i % len(colors)],
                linestyle=line_styles[i % len(line_styles)], markerfacecolor='white',
                markeredgecolor=colors[i % len(colors)], markeredgewidth=2)
        
        # Add baseline for this model
        baseline = results['overall_accuracy']
        plt.axhline(y=baseline, color=colors[i % len(colors)], 
                   linestyle=':', alpha=0.6, linewidth=1)
    
    plt.xlabel('Rejection Rate (%)', fontweight='bold')
    plt.ylabel('Accuracy on Remaining Samples', fontweight='bold')
    plt.title('Precision-Rejection Curves Comparison\n(Higher uncertainty samples rejected first)', 
              fontweight='bold')
    plt.legend(frameon=True, fancybox=True, shadow=True, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 90)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_dir / "precision_rejection_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

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
    from scipy import stats
    
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
    
    # Save statistical results
    with open(save_dir / "statistical_comparison.json", 'w') as f:
        json.dump(significance_results, f, indent=2, default=str)
    
    return significance_results

def save_single_model_metadata(base_dir: Path, model_id: str, 
                             calibration_data_points: List, timestamp: str):
    """Save metadata for single model analysis."""
    metadata = {
        "timestamp": timestamp,
        "analysis_type": "single_model",
        "model_analyzed": model_id,
        "total_data_points": len(calibration_data_points),
        "temperature_used": getattr(calibration_data_points[0], 'temperature', 'unknown') if calibration_data_points else 'unknown'
    }
    
    with open(base_dir / "run_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

def save_multi_model_metadata(base_dir: Path, models_data: Dict, 
                            timestamp: str, comparison_results: Dict):
    """Save metadata for multi-model analysis."""
    metadata = {
        "timestamp": timestamp,
        "analysis_type": "multi_model",
        "models_analyzed": list(models_data.keys()),
        "total_data_points": sum(len(data_points) for data_points in models_data.values()),
        "data_points_per_model": {model: len(data_points) for model, data_points in models_data.items()},
        "best_model_by_accuracy": comparison_results.get('best_model_overall', 'unknown'),
        "model_count": len(models_data),
        "temperature_used": getattr(list(models_data.values())[0][0], 'temperature', 'unknown') if models_data else 'unknown'
    }
    
    # Save run metadata
    with open(base_dir / "run_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save analysis summary
    analysis_summary = {
        "overall_summary": {
            "best_model_by_accuracy": comparison_results.get('best_model_overall', 'unknown'),
            "total_models_compared": comparison_results.get('model_count', 0),
            "total_statistical_comparisons": comparison_results.get('total_comparisons', 0)
        },
        "comparison_results": comparison_results
    }
    
    with open(base_dir / "analysis_summary.json", 'w') as f:
        json.dump(analysis_summary, f, indent=2, default=str)

# Backward compatibility function
def run_analysis_legacy(calibration_data_points):
    """
    Legacy function for backward compatibility.
    Uses the original behavior but with new structured output.
    """
    return run_analysis(calibration_data_points)