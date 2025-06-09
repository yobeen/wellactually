# src/uncertainty_calibration/l1_visualization.py
"""
Visualization functions for L1 validation analysis.
Handles all plotting and chart generation for single-model, multi-model, and voting analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style with high contrast colors
plt.style.use('default')

# Define high-contrast color palette
PRIMARY_BLUE = '#1f77b4'
VIBRANT_RED = '#d62728'
VIBRANT_GREEN = '#2ca02c'
VIBRANT_ORANGE = '#ff7f0e'
DEEP_PURPLE = '#9467bd'

# High-contrast colors for multiple models
MULTI_MODEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

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
    
    # 1. Accuracy Analysis Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Overall accuracy
    ax1.bar(['Overall'], [overall_acc], color=PRIMARY_BLUE, edgecolor='black', linewidth=1)
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
    class_colors = [VIBRANT_RED, VIBRANT_GREEN, VIBRANT_ORANGE]
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
             marker='o', linewidth=3, markersize=8, color=PRIMARY_BLUE, markerfacecolor='white',
             markeredgecolor=PRIMARY_BLUE, markeredgewidth=2)
    plt.axhline(y=overall_acc, color=VIBRANT_RED, linestyle='--', linewidth=2, alpha=0.8, 
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
             color=VIBRANT_GREEN, density=True, edgecolor='black', linewidth=0.5)
    plt.hist(incorrect_uncertainty, bins=20, alpha=0.75, label='Incorrect Predictions', 
             color=VIBRANT_RED, density=True, edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Raw Uncertainty', fontweight='bold')
    plt.ylabel('Density', fontweight='bold')
    plt.title(f'Uncertainty Distribution by Prediction Correctness - {model_name}', fontweight='bold')
    plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "l1_uncertainty_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_model_accuracy_comparison(individual_results: Dict, save_dir: Path):
    """Create model accuracy comparison plots."""
    model_names = list(individual_results.keys())
    overall_accs = [individual_results[model]['overall_accuracy'] for model in model_names]
    
    # Overall accuracy comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall accuracy bar plot
    bars = ax1.bar(range(len(model_names)), overall_accs, 
                   color=MULTI_MODEL_COLORS[:len(model_names)], edgecolor='black', linewidth=1)
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
                      label=model.replace('/', '-'), color=MULTI_MODEL_COLORS[i], 
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
    
    # Correct predictions uncertainty distribution
    for i, (model, results) in enumerate(individual_results.items()):
        df = results['results_df']
        correct_uncertainty = df[df['is_correct']]['raw_uncertainty']
        ax1.hist(correct_uncertainty, bins=15, alpha=0.7, label=model.replace('/', '-'),
                color=MULTI_MODEL_COLORS[i % len(MULTI_MODEL_COLORS)], density=True, 
                edgecolor='black', linewidth=0.5)
    
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
                color=MULTI_MODEL_COLORS[i % len(MULTI_MODEL_COLORS)], density=True, 
                edgecolor='black', linewidth=0.5)
    
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
    
    # Line styles and markers for variety
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, (model, results) in enumerate(individual_results.items()):
        precision_data = results['precision_data']
        plt.plot(precision_data['rejection_rate'], precision_data['accuracy'],
                marker=markers[i % len(markers)], linewidth=3, markersize=8,
                label=model.replace('/', '-'), color=MULTI_MODEL_COLORS[i % len(MULTI_MODEL_COLORS)],
                linestyle=line_styles[i % len(line_styles)], markerfacecolor='white',
                markeredgecolor=MULTI_MODEL_COLORS[i % len(MULTI_MODEL_COLORS)], markeredgewidth=2)
        
        # Add baseline for this model
        baseline = results['overall_accuracy']
        plt.axhline(y=baseline, color=MULTI_MODEL_COLORS[i % len(MULTI_MODEL_COLORS)], 
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

def create_voting_accuracy_curves(voting_results: List[Dict], save_dir: Path, 
                                 individual_results: Optional[Dict] = None):
    """
    Create voting accuracy curves for both evaluation modes.
    
    Args:
        voting_results: List of voting results at different rejection rates
        save_dir: Directory to save plots
        individual_results: Optional individual model results for comparison
    """
    rejection_rates = [r['rejection_rate'] for r in voting_results]
    mode1_accuracies = [r['mode1_accuracy'] for r in voting_results]
    mode2_accuracies = [r['mode2_accuracy'] for r in voting_results]
    samples_remaining = [r['samples_remaining'] for r in voting_results]
    total_questions = voting_results[0]['total_questions']
    
    # 1. Dual-mode accuracy comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mode comparison
    ax1.plot(rejection_rates, mode1_accuracies, marker='o', linewidth=3, markersize=8,
             color=VIBRANT_RED, label='Mode 1: Penalize Missing', markerfacecolor='white',
             markeredgecolor=VIBRANT_RED, markeredgewidth=2)
    ax1.plot(rejection_rates, mode2_accuracies, marker='s', linewidth=3, markersize=8,
             color=PRIMARY_BLUE, label='Mode 2: Exclude Missing', markerfacecolor='white',
             markeredgecolor=PRIMARY_BLUE, markeredgewidth=2)
    
    # Add individual model baselines if provided
    if individual_results:
        for i, (model, results) in enumerate(individual_results.items()):
            baseline = results['overall_accuracy']
            ax1.axhline(y=baseline, color=MULTI_MODEL_COLORS[i % len(MULTI_MODEL_COLORS)], 
                       linestyle=':', alpha=0.7, linewidth=1, 
                       label=f'{model.replace("/", "-")} baseline')
    
    ax1.set_xlabel('Rejection Rate (%)', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Voting Accuracy by Evaluation Mode', fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(rejection_rates))
    ax1.set_ylim(0, 1)
    
    # Sample retention
    retention_percentages = [s / total_questions * 100 for s in samples_remaining]
    ax2.plot(rejection_rates, retention_percentages, marker='D', linewidth=3, markersize=8,
             color=VIBRANT_GREEN, markerfacecolor='white', markeredgecolor=VIBRANT_GREEN, 
             markeredgewidth=2)
    ax2.set_xlabel('Rejection Rate (%)', fontweight='bold')
    ax2.set_ylabel('Questions with Votes (%)', fontweight='bold')
    ax2.set_title('Sample Retention in Voting', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(rejection_rates))
    ax2.set_ylim(0, 100)
    
    # Add annotations
    if len(voting_results) > 1:
        best_mode1 = max(voting_results, key=lambda x: x['mode1_accuracy'])
        best_mode2 = max(voting_results, key=lambda x: x['mode2_accuracy'])
        
        ax1.annotate(f'Best Mode 1: {best_mode1["mode1_accuracy"]:.3f}\n@{best_mode1["rejection_rate"]:.0f}%',
                    xy=(best_mode1['rejection_rate'], best_mode1['mode1_accuracy']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontweight='bold', fontsize=9)
        
        ax1.annotate(f'Best Mode 2: {best_mode2["mode2_accuracy"]:.3f}\n@{best_mode2["rejection_rate"]:.0f}%',
                    xy=(best_mode2['rejection_rate'], best_mode2['mode2_accuracy']),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                    fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_dir / "voting_accuracy_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_voting_detailed_analysis(voting_results: List[Dict], save_dir: Path):
    """
    Create detailed voting analysis plots.
    
    Args:
        voting_results: List of voting results at different rejection rates
        save_dir: Directory to save plots
    """
    rejection_rates = [r['rejection_rate'] for r in voting_results]
    mode1_accuracies = [r['mode1_accuracy'] for r in voting_results]
    mode2_accuracies = [r['mode2_accuracy'] for r in voting_results]
    samples_remaining = [r['samples_remaining'] for r in voting_results]
    
    # Calculate accuracy differences and improvements
    baseline_mode1 = mode1_accuracies[0]
    baseline_mode2 = mode2_accuracies[0]
    mode1_improvements = [acc - baseline_mode1 for acc in mode1_accuracies]
    mode2_improvements = [acc - baseline_mode2 for acc in mode2_accuracies]
    mode_differences = [m2 - m1 for m1, m2 in zip(mode1_accuracies, mode2_accuracies)]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy improvements over baseline
    ax1.plot(rejection_rates, mode1_improvements, marker='o', linewidth=2, 
             color=VIBRANT_RED, label='Mode 1 Improvement')
    ax1.plot(rejection_rates, mode2_improvements, marker='s', linewidth=2, 
             color=PRIMARY_BLUE, label='Mode 2 Improvement')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Rejection Rate (%)', fontweight='bold')
    ax1.set_ylabel('Accuracy Improvement', fontweight='bold')
    ax1.set_title('Accuracy Improvement Over Baseline', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Mode difference (Mode 2 - Mode 1)
    ax2.plot(rejection_rates, mode_differences, marker='D', linewidth=2, 
             color=DEEP_PURPLE, markerfacecolor='white', markeredgewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Rejection Rate (%)', fontweight='bold')
    ax2.set_ylabel('Mode 2 - Mode 1 Accuracy', fontweight='bold')
    ax2.set_title('Evaluation Mode Difference', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Efficiency plot (accuracy vs samples lost)
    samples_lost = [voting_results[0]['samples_remaining'] - s for s in samples_remaining]
    ax3.scatter(samples_lost, mode1_accuracies, c=rejection_rates, cmap='viridis', 
               s=100, alpha=0.7, edgecolors='black')
    ax3.set_xlabel('Questions Lost to Rejection', fontweight='bold')
    ax3.set_ylabel('Mode 1 Accuracy', fontweight='bold')
    ax3.set_title('Accuracy vs Questions Lost', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar for rejection rates
    scatter = ax3.scatter(samples_lost, mode1_accuracies, c=rejection_rates, cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black')
    plt.colorbar(scatter, ax=ax3, label='Rejection Rate (%)')
    
    # 4. Sample retention curve with thresholds
    retention_percentages = [s / voting_results[0]['samples_remaining'] * 100 for s in samples_remaining]
    ax4.plot(rejection_rates, retention_percentages, marker='o', linewidth=3, 
             color=VIBRANT_GREEN, markerfacecolor='white', markeredgewidth=2)
    
    # Add threshold lines
    for threshold in [90, 75, 50, 25]:
        if any(r <= threshold for r in retention_percentages):
            ax4.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
            ax4.text(max(rejection_rates) * 0.8, threshold + 2, f'{threshold}%', 
                    fontweight='bold', fontsize=9)
    
    ax4.set_xlabel('Rejection Rate (%)', fontweight='bold')
    ax4.set_ylabel('Questions Retained (%)', fontweight='bold')
    ax4.set_title('Question Retention Rate', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_dir / "voting_detailed_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()