# src/uncertainty_calibration/l1_analysis.py
"""
Analysis functions for L1 validation results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List

def analyze_accuracy(calibration_data_points):
    """
    Calculate overall and per-class accuracy from CalibrationDataPoint objects.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        
    Returns:
        tuple: (overall_accuracy, class_accuracies_dict)
    """
    if not calibration_data_points:
        return 0.0, {}
    
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
            'raw_uncertainty': dp.raw_uncertainty
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
    
    print(f"\nAccuracy Analysis:")
    print(f"Overall Accuracy: {overall_acc:.3f}")
    print("Per-class Accuracy:")
    for class_name, acc in class_acc.items():
        count = len(df[df['human_label'] == class_name])
        print(f"  {class_name}: {acc:.3f} (n={count})")
    
    return overall_acc, class_acc, df

def analyze_precision_rejection(calibration_data_points):
    """
    Generate precision-rejection curve data.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
        
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
    
    return pd.DataFrame(precision_data)

def create_plots(results_df, precision_data, overall_acc):
    """
    Create and save analysis plots.
    
    Args:
        results_df: DataFrame with prediction results
        precision_data: DataFrame with precision-rejection data
        overall_acc: Overall accuracy value
    """
    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Accuracy Analysis Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Overall accuracy
    ax1.bar(['Overall'], [overall_acc], color='skyblue')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Overall Accuracy')
    ax1.set_ylim(0, 1)
    ax1.text(0, overall_acc + 0.02, f'{overall_acc:.3f}', ha='center')
    
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
    bars = ax2.bar(classes, accuracies, color=['lightcoral', 'lightgreen', 'lightsalmon'])
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Per-Class Accuracy')
    ax2.set_ylim(0, 1)
    
    # Add count labels
    for i, (class_name, bar) in enumerate(zip(classes, bars)):
        count = len(results_df[results_df['human_label'] == class_name])
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f'{height:.3f}\n(n={count})', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "l1_accuracy_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Precision-Rejection Curve
    plt.figure(figsize=(10, 6))
    plt.plot(precision_data['rejection_rate'], precision_data['accuracy'], 
             marker='o', linewidth=2, markersize=6)
    plt.axhline(y=overall_acc, color='red', linestyle='--', alpha=0.7, 
                label=f'Baseline (no rejection): {overall_acc:.3f}')
    plt.xlabel('Rejection Rate (%)')
    plt.ylabel('Accuracy on Remaining Samples')
    plt.title('Precision-Rejection Curve\n(Higher uncertainty samples rejected first)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 90)
    plt.ylim(0, 1)
    
    # Add annotation for improvement
    final_acc = precision_data.iloc[-1]['accuracy']
    improvement = final_acc - overall_acc
    plt.text(45, 0.1, f'Max improvement: +{improvement:.3f}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(plots_dir / "l1_precision_rejection.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Uncertainty Distribution by Correctness
    plt.figure(figsize=(10, 6))
    correct_uncertainty = results_df[results_df['is_correct']]['raw_uncertainty']
    incorrect_uncertainty = results_df[~results_df['is_correct']]['raw_uncertainty']
    
    plt.hist(correct_uncertainty, bins=20, alpha=0.7, label='Correct Predictions', 
             color='green', density=True)
    plt.hist(incorrect_uncertainty, bins=20, alpha=0.7, label='Incorrect Predictions', 
             color='red', density=True)
    
    plt.xlabel('Raw Uncertainty')
    plt.ylabel('Density')
    plt.title('Uncertainty Distribution by Prediction Correctness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "l1_uncertainty_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()

def run_analysis(calibration_data_points):
    """
    Run complete analysis on calibration data points.
    
    Args:
        calibration_data_points: List of CalibrationDataPoint objects
    """
    print("\n" + "="*50)
    print("L1 VALIDATION ANALYSIS RESULTS")
    print("="*50)
    
    # Analyze accuracy
    overall_acc, class_acc, results_df = analyze_accuracy(calibration_data_points)
    
    # Analyze precision-rejection
    print("\nGenerating precision-rejection curve...")
    precision_data = analyze_precision_rejection(calibration_data_points)
    
    # Create plots
    print("\nCreating plots...")
    create_plots(results_df, precision_data, overall_acc)
    
    # Save detailed results
    results_df.to_csv("data/l1_validation_results.csv", index=False)
    precision_data.to_csv("data/l1_precision_rejection.csv", index=False)
    
    print(f"\nAnalysis complete!")
    print(f"Total predictions analyzed: {len(calibration_data_points)}")
    print(f"Charts saved to plots/ directory")
    print(f"Detailed results saved to data/l1_validation_results.csv")