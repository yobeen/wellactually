#!/usr/bin/env python3
"""
Quick validation script to test correlation between LLM log-probabilities and human multipliers.
Tests core assumption for ensemble calibration project.
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr, ttest_ind
from pathlib import Path

def load_data():
    """Load human preferences and LLM cache data."""
    # Load human preferences
    train_df = pd.read_csv('train.csv')
    
    # Load LLM responses
    with open('correlation_cache.json', 'r') as f:
        llm_cache = json.load(f)
    
    return train_df, llm_cache

def preprocess_human_data(train_df):
    """Preprocess human data: map choices and apply Equal rule."""
    processed = train_df.copy()
    
    # Map choice values: 1.0 -> A, 2.0 -> B
    processed['human_choice'] = processed['choice'].map({1.0: 'A', 2.0: 'B'})
    
    # Apply Equal rule: multiplier <= 1.2 -> Equal
    processed.loc[processed['multiplier'] <= 1.2, 'human_choice'] = 'Equal'
    
    # Create matching key
    processed['match_key'] = processed['repo_a'] + '|' + processed['repo_b']
    
    return processed

def extract_logprob_features(logprobs, human_choice):
    """Extract correlation features from logprobs."""
    # Get probabilities for A, B, Equal (with small epsilon for numerical stability)
    eps = 1e-10
    prob_a = logprobs.get('A', eps)
    prob_b = logprobs.get('B', eps)
    prob_equal = logprobs.get('Equal', eps)
    
    # Normalize probabilities
    total_prob = prob_a + prob_b + prob_equal
    prob_a_norm = prob_a / total_prob
    prob_b_norm = prob_b / total_prob
    prob_equal_norm = prob_equal / total_prob
    
    features = {}
    
    # 1. Probability of human-chosen option
    choice_probs = {'A': prob_a_norm, 'B': prob_b_norm, 'Equal': prob_equal_norm}
    features['prob_chosen'] = choice_probs.get(human_choice, eps)
    
    # 2. Probability ratio (A vs B) - skip for Equal
    if human_choice != 'Equal':
        features['prob_ratio'] = prob_a_norm / (prob_b_norm + eps)
        if human_choice == 'B':
            features['prob_ratio'] = 1.0 / features['prob_ratio']  # Flip for B choices
    else:
        features['prob_ratio'] = 1.0  # Neutral for Equal
    
    # 3. Confidence (max probability)
    features['confidence'] = max(prob_a_norm, prob_b_norm, prob_equal_norm)
    
    # 4. Entropy (uncertainty measure)
    probs = [prob_a_norm, prob_b_norm, prob_equal_norm]
    features['entropy'] = -sum(p * np.log(p + eps) for p in probs)
    
    # 5. Probability gap |P(A) - P(B)|
    features['prob_gap'] = abs(prob_a_norm - prob_b_norm)
    
    return features

def match_human_llm_data(human_df, llm_cache):
    """Match human preferences with LLM responses."""
    matched_data = []
    
    for _, row in human_df.iterrows():
        repo_a = row['repo_a'].strip()
        repo_b = row['repo_b'].strip()
        human_choice = row['human_choice']
        multiplier = row['multiplier']
        
        # Try different key formats to match cache entries
        possible_keys = [
            f"openai/gpt-4o|{repo_a}|{repo_b}",
            f"openai/gpt-4o|{repo_b}|{repo_a}",  # Reverse order
        ]
        
        # Find matching LLM response
        llm_entry = None
        matched_key = None
        for key in possible_keys:
            if key in llm_cache:
                llm_entry = llm_cache[key]
                matched_key = key
                break
        
        if llm_entry and 'logprobs' in llm_entry:
            # Extract features
            features = extract_logprob_features(llm_entry['logprobs'], human_choice)
            
            # Add to matched data
            entry = {
                'matched_key': matched_key,
                'human_choice': human_choice,
                'multiplier': multiplier,
                'llm_choice': llm_entry.get('model_choice', ''),
                'llm_content': llm_entry.get('content', ''),
                **features
            }
            matched_data.append(entry)
    
    return pd.DataFrame(matched_data)

def calculate_correlations(matched_df):
    """Calculate correlations between logprob features and multipliers."""
    feature_cols = ['prob_chosen', 'prob_ratio', 'confidence', 'entropy', 'prob_gap']
    
    correlations = {}
    
    for feature in feature_cols:
        if feature in matched_df.columns:
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(matched_df[feature], matched_df['multiplier'])
            
            # Spearman correlation
            spearman_r, spearman_p = spearmanr(matched_df[feature], matched_df['multiplier'])
            
            correlations[feature] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            }
    
    return correlations

def analyze_choice_agreement(matched_df):
    """Analyze agreement between model choices and human choices."""
    # Create agreement indicator
    matched_df['choice_agreement'] = (matched_df['llm_choice'] == matched_df['human_choice']).astype(int)
    
    # Calculate basic accuracy
    accuracy = matched_df['choice_agreement'].mean()
    
    # Agreement by human choice type
    agreement_by_choice = matched_df.groupby('human_choice')['choice_agreement'].agg(['mean', 'count'])
    
    # Analyze multiplier distribution for correct vs incorrect predictions
    correct_mask = matched_df['choice_agreement'] == 1
    incorrect_mask = matched_df['choice_agreement'] == 0
    
    correct_multipliers = matched_df.loc[correct_mask, 'multiplier']
    incorrect_multipliers = matched_df.loc[incorrect_mask, 'multiplier']
    
    # Correlation between model confidence and choice correctness
    confidence_accuracy_corr = None
    if 'confidence' in matched_df.columns:
        conf_acc_r, conf_acc_p = pearsonr(matched_df['confidence'], matched_df['choice_agreement'])
        confidence_accuracy_corr = {'r': conf_acc_r, 'p': conf_acc_p}
    
    return {
        'overall_accuracy': accuracy,
        'agreement_by_choice': agreement_by_choice,
        'correct_multipliers': correct_multipliers,
        'incorrect_multipliers': incorrect_multipliers,
        'confidence_accuracy_correlation': confidence_accuracy_corr
    }

def create_comprehensive_visualizations(matched_df, correlations, choice_analysis, save_plots=True, show_plots=False):
    """Create comprehensive visualizations for correlation analysis."""
    feature_cols = ['prob_chosen', 'prob_ratio', 'confidence', 'entropy', 'prob_gap']
    valid_features = [f for f in feature_cols if f in matched_df.columns]
    
    if not valid_features:
        print("No valid features found for visualization")
        return
    
    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. MODEL vs HUMAN CHOICE ANALYSIS
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Overall accuracy
    accuracy = choice_analysis['overall_accuracy']
    axes[0, 0].bar(['Model Accuracy'], [accuracy], color='skyblue', alpha=0.8)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title(f'Model Choice Accuracy: {accuracy:.1%}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add accuracy text
    axes[0, 0].text(0, accuracy + 0.05, f'{accuracy:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Agreement by choice type
    agreement_by_choice = choice_analysis['agreement_by_choice']
    choices = agreement_by_choice.index
    accuracies = agreement_by_choice['mean']
    counts = agreement_by_choice['count']
    
    bars = axes[0, 1].bar(choices, accuracies, alpha=0.8)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy by Human Choice Type')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, acc, count in zip(bars, accuracies, counts):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, acc + 0.02, 
                       f'{acc:.1%}\n(n={count})', ha='center', va='bottom')
    
    # Confusion matrix
    confusion_df = matched_df.groupby(['human_choice', 'llm_choice']).size().unstack(fill_value=0)
    sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
    axes[0, 2].set_title('Confusion Matrix: LLM vs Human')
    axes[0, 2].set_xlabel('LLM Choice')
    axes[0, 2].set_ylabel('Human Choice')
    
    # Multiplier distribution for correct vs incorrect predictions
    correct_mult = choice_analysis['correct_multipliers']
    incorrect_mult = choice_analysis['incorrect_multipliers']
    
    axes[1, 0].hist([correct_mult, incorrect_mult], bins=15, alpha=0.7, 
                   label=['Correct Predictions', 'Incorrect Predictions'], 
                   color=['green', 'red'])
    axes[1, 0].set_xlabel('Human Multiplier')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Multiplier Distribution by Prediction Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confidence vs Accuracy
    if choice_analysis['confidence_accuracy_correlation']:
        conf_corr = choice_analysis['confidence_accuracy_correlation']
        scatter = axes[1, 1].scatter(matched_df['confidence'], matched_df['choice_agreement'], 
                                   alpha=0.6, c=matched_df['multiplier'], cmap='viridis')
        axes[1, 1].set_xlabel('Model Confidence')
        axes[1, 1].set_ylabel('Choice Agreement (1=Correct, 0=Wrong)')
        axes[1, 1].set_title(f'Confidence vs Accuracy\nr={conf_corr["r"]:.3f}, p={conf_corr["p"]:.3f}')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='Human Multiplier')
        
        # Add trend line
        z = np.polyfit(matched_df['confidence'], matched_df['choice_agreement'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(matched_df['confidence'].min(), matched_df['confidence'].max(), 100)
        axes[1, 1].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    # Choice agreement vs multiplier
    axes[1, 2].boxplot([correct_mult, incorrect_mult], tick_labels=['Correct', 'Incorrect'])
    axes[1, 2].set_ylabel('Human Multiplier')
    axes[1, 2].set_title('Multiplier by Prediction Accuracy')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add statistical test
    if len(correct_mult) > 0 and len(incorrect_mult) > 0:
        t_stat, t_p = ttest_ind(correct_mult, incorrect_mult)
        axes[1, 2].text(0.5, 0.95, f't-test p={t_p:.3f}', transform=axes[1, 2].transAxes, 
                       ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(plots_dir / "01_model_human_choice_analysis.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plots_dir / '01_model_human_choice_analysis.png'}")
    if show_plots:
        plt.show()
    plt.close()
    
    # 2. CORRELATION HEATMAP
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create correlation matrix
    corr_data = []
    feature_names = []
    for feature in valid_features:
        if feature in correlations:
            corr_data.append([
                correlations[feature]['pearson_r'],
                correlations[feature]['spearman_r']
            ])
            feature_names.append(feature.replace('_', ' ').title())
    
    corr_matrix = np.array(corr_data)
    
    # Heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pearson r', 'Spearman r'])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    
    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(2):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                          ha="center", va="center", color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
    
    ax.set_title('Correlation with Human Multipliers', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    plt.tight_layout()
    if save_plots:
        plt.savefig(plots_dir / "02_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plots_dir / '02_correlation_heatmap.png'}")
    if show_plots:
        plt.show()
    plt.close()
    
    # 3. SCATTER PLOTS WITH REGRESSION LINES
    n_features = len(valid_features)
    cols = min(3, n_features)
    rows = (n_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = [axes] if n_features == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(valid_features):
        ax = axes[i]
        
        # Color by choice agreement (correct vs incorrect)
        colors = matched_df['choice_agreement'].map({1: 'green', 0: 'red'})
        scatter = ax.scatter(matched_df[feature], matched_df['multiplier'], 
                           c=colors, alpha=0.7, s=50)
        
        # Add regression line
        if len(matched_df) > 1:
            z = np.polyfit(matched_df[feature], matched_df['multiplier'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(matched_df[feature].min(), matched_df[feature].max(), 100)
            ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2)
        
        # Correlation info
        corr_info = correlations.get(feature, {})
        pearson_r = corr_info.get('pearson_r', 0)
        pearson_p = corr_info.get('pearson_p', 1)
        
        # Format p-value
        if pearson_p < 0.001:
            p_str = "p<0.001"
        elif pearson_p < 0.01:
            p_str = f"p<0.01"
        elif pearson_p < 0.05:
            p_str = f"p<0.05"
        else:
            p_str = f"p={pearson_p:.3f}"
        
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Human Multiplier')
        ax.set_title(f'{feature.replace("_", " ").title()}\nr={pearson_r:.3f}, {p_str}')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [Patch(facecolor='green', label='Correct Choice'),
                          Patch(facecolor='red', label='Incorrect Choice')]
        ax.legend(handles=legend_elements)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(plots_dir / "03_feature_correlations_scatter.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plots_dir / '03_feature_correlations_scatter.png'}")
    if show_plots:
        plt.show()
    plt.close()
    
    # 4. DISTRIBUTION ANALYSIS
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Multiplier distribution
    axes[0, 0].hist(matched_df['multiplier'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(matched_df['multiplier'].mean(), color='red', linestyle='--', label=f'Mean: {matched_df["multiplier"].mean():.2f}')
    axes[0, 0].set_xlabel('Human Multiplier')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Human Multipliers')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Choice agreement pie chart
    agreement_counts = matched_df['choice_agreement'].value_counts()
    agreement_labels = ['Incorrect', 'Correct']
    axes[0, 1].pie(agreement_counts.values, labels=[agreement_labels[i] for i in agreement_counts.index], 
                   autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
    axes[0, 1].set_title(f'Model Choice Accuracy\n({accuracy:.1%} correct)')
    
    # Model choice distribution
    llm_choice_counts = matched_df['llm_choice'].value_counts()
    axes[0, 2].pie(llm_choice_counts.values, labels=llm_choice_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 2].set_title('LLM Choice Distribution')
    
    # Feature distributions
    for i, feature in enumerate(valid_features[:3]):
        if i < 3:
            ax = axes[1, i]
            ax.hist(matched_df[feature], bins=15, alpha=0.7, edgecolor='black')
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {feature.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(plots_dir / "04_distribution_analysis.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plots_dir / '04_distribution_analysis.png'}")
    if show_plots:
        plt.show()
    plt.close()
    
    # 5. CORRELATION SUMMARY PLOT
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Prepare data for plotting
    features = []
    pearson_rs = []
    spearman_rs = []
    p_values = []
    
    for feature in valid_features:
        if feature in correlations:
            features.append(feature.replace('_', ' ').title())
            pearson_rs.append(correlations[feature]['pearson_r'])
            spearman_rs.append(correlations[feature]['spearman_r'])
            p_values.append(correlations[feature]['pearson_p'])
    
    x = np.arange(len(features))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, pearson_rs, width, label='Pearson r', alpha=0.8)
    bars2 = ax.bar(x + width/2, spearman_rs, width, label='Spearman r', alpha=0.8)
    
    # Color bars by significance
    for i, (bar1, bar2, p_val) in enumerate(zip(bars1, bars2, p_values)):
        color = 'green' if p_val < 0.05 else 'orange' if p_val < 0.1 else 'red'
        bar1.set_edgecolor(color)
        bar2.set_edgecolor(color)
        bar1.set_linewidth(3)
        bar2.set_linewidth(3)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Feature Correlations with Human Multipliers\n(Green edge: p<0.05, Orange: p<0.1, Red: p≥0.1)')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Weak threshold')
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Strong threshold')
    ax.axhline(y=-0.3, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.5, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(plots_dir / "05_correlation_summary.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plots_dir / '05_correlation_summary.png'}")
    if show_plots:
        plt.show()
    plt.close()
    
    # 6. FEATURE INTERCORRELATION MATRIX
    if len(valid_features) >= 2:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Feature correlation matrix
        feature_data = matched_df[valid_features + ['multiplier', 'choice_agreement']]
        corr_matrix = feature_data.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Feature Intercorrelations')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(plots_dir / "06_feature_intercorrelations.png", dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {plots_dir / '06_feature_intercorrelations.png'}")
        if show_plots:
            plt.show()
        plt.close()
    
    if save_plots:
        print(f"\n✅ All visualizations saved to '{plots_dir}' directory")
        print("   Files created:")
        print("   - 01_model_human_choice_analysis.png")
        print("   - 02_correlation_heatmap.png") 
        print("   - 03_feature_correlations_scatter.png")
        print("   - 04_distribution_analysis.png")
        print("   - 05_correlation_summary.png")
        print("   - 06_feature_intercorrelations.png")

def print_matched_data_summary(matched_df):
    """Print summary of matched model vs human data."""
    # Create choice agreement column for this summary
    matched_df_copy = matched_df.copy()
    matched_df_copy['choice_agreement'] = (matched_df_copy['llm_choice'] == matched_df_copy['human_choice']).astype(int)
    
    print("\n=== MATCHED DATA SUMMARY ===")
    print(f"{'#':<3} {'Model Choice':<12} {'Model Conf':<11} {'Human Choice':<12} {'Human Conf':<11} {'Match':<6}")
    print("-" * 65)
    
    for i, row in matched_df_copy.iterrows():
        model_choice = row['llm_choice']
        model_conf = row.get('prob_chosen', 0.0)  # Probability of chosen option
        human_choice = row['human_choice']
        human_conf = row['multiplier']
        match = "✓" if row['choice_agreement'] == 1 else "✗"
        
        print(f"{i+1:<3} {model_choice:<12} {model_conf:<11.3f} {human_choice:<12} {human_conf:<11.2f} {match:<6}")
    
    print("-" * 65)
    print(f"Total pairs: {len(matched_df_copy)}")
    print(f"Agreement rate: {matched_df_copy['choice_agreement'].mean():.1%}")
    print(f"Mean human confidence: {matched_df_copy['multiplier'].mean():.2f}")
    
    # Handle prob_chosen column safely
    if 'prob_chosen' in matched_df_copy.columns:
        mean_model_conf = matched_df_copy['prob_chosen'].mean()
    else:
        mean_model_conf = 0.0
    print(f"Mean model confidence: {mean_model_conf:.3f}")

def print_results(matched_df, correlations, choice_analysis):
    """Print summary of correlation analysis."""
    print("=== CORRELATION VALIDATION RESULTS ===\n")
    
    print(f"Matched {len(matched_df)} human-LLM pairs")
    print(f"Human multiplier range: {matched_df['multiplier'].min():.2f} - {matched_df['multiplier'].max():.2f}")
    print(f"Mean multiplier: {matched_df['multiplier'].mean():.2f}\n")
    
    # Choice Agreement Analysis
    print("MODEL vs HUMAN CHOICE AGREEMENT:")
    print("-" * 50)
    accuracy = choice_analysis['overall_accuracy']
    print(f"Overall Model Accuracy: {accuracy:.1%}")
    
    print("\nAccuracy by Human Choice Type:")
    agreement_by_choice = choice_analysis['agreement_by_choice']
    for choice_type in agreement_by_choice.index:
        acc = agreement_by_choice.loc[choice_type, 'mean']
        count = agreement_by_choice.loc[choice_type, 'count']
        print(f"  {choice_type}: {acc:.1%} ({count} samples)")
    
    # Confidence-Accuracy Correlation
    if choice_analysis['confidence_accuracy_correlation']:
        conf_corr = choice_analysis['confidence_accuracy_correlation']
        sig = "***" if conf_corr['p'] < 0.001 else "**" if conf_corr['p'] < 0.01 else "*" if conf_corr['p'] < 0.05 else ""
        print(f"\nModel Confidence vs Choice Accuracy: r={conf_corr['r']:.3f}{sig}")
        
    # Multiplier analysis for correct vs incorrect
    correct_mult = choice_analysis['correct_multipliers']
    incorrect_mult = choice_analysis['incorrect_multipliers']
    if len(correct_mult) > 0 and len(incorrect_mult) > 0:
        print(f"\nMultiplier Analysis:")
        print(f"  Correct predictions - Mean: {correct_mult.mean():.2f}, Std: {correct_mult.std():.2f}")
        print(f"  Incorrect predictions - Mean: {incorrect_mult.mean():.2f}, Std: {incorrect_mult.std():.2f}")
    
    print(f"\nCORRELATIONS WITH HUMAN MULTIPLIER:")
    print("-" * 50)
    
    for feature, stats in correlations.items():
        pearson_r = stats['pearson_r']
        pearson_p = stats['pearson_p']
        spearman_r = stats['spearman_r']
        spearman_p = stats['spearman_p']
        
        # Significance indicators
        sig_p = "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else ""
        sig_s = "***" if spearman_p < 0.001 else "**" if spearman_p < 0.01 else "*" if spearman_p < 0.05 else ""
        
        print(f"{feature:15} | Pearson: {pearson_r:6.3f}{sig_p:3} | Spearman: {spearman_r:6.3f}{sig_s:3}")
    
    print("\n*** p<0.001, ** p<0.01, * p<0.05")
    
    # Interpretation
    print("\n=== INTERPRETATION ===")
    
    # Choice Agreement Assessment
    if accuracy >= 0.7:
        print(f"🟢 GOOD model-human agreement! ({accuracy:.1%} accuracy)")
        print("   → LLMs are reasonably aligned with human preferences")
    elif accuracy >= 0.5:
        print(f"🟡 MODERATE model-human agreement ({accuracy:.1%} accuracy)")
        print("   → Some alignment, but significant disagreement exists")
    else:
        print(f"🔴 POOR model-human agreement ({accuracy:.1%} accuracy)")
        print("   → LLMs frequently disagree with human judgments")
    
    # Correlation Assessment
    best_feature = max(correlations.keys(), key=lambda x: abs(correlations[x]['pearson_r']))
    best_corr = correlations[best_feature]['pearson_r']
    
    if abs(best_corr) > 0.5:
        print(f"🟢 STRONG correlation found! {best_feature} (r={best_corr:.3f})")
        print("   → Log-probability approach looks viable!")
    elif abs(best_corr) > 0.3:
        print(f"🟡 MODERATE correlation found: {best_feature} (r={best_corr:.3f})")
        print("   → Log-probability approach might work with refinement")
    else:
        print(f"🔴 WEAK correlations found (best: {best_feature}, r={best_corr:.3f})")
        print("   → Consider bucket-based multiplier approach instead")
    
    # Combined Assessment
    print(f"\n=== OVERALL RECOMMENDATION ===")
    if accuracy >= 0.6 and abs(best_corr) >= 0.4:
        print("✅ PROCEED with ensemble approach - both choice agreement and multiplier correlation show promise")
    elif accuracy >= 0.5 or abs(best_corr) >= 0.3:
        print("⚠️  PROCEED WITH CAUTION - consider simpler ensemble methods first")
    else:
        print("❌ RECONSIDER APPROACH - weak evidence for both choice agreement and multiplier correlation")
        print("   → Try bucket-based multiplier estimation or simple voting instead")

    """Print summary of correlation analysis."""
    print("=== CORRELATION VALIDATION RESULTS ===\n")
    
    print(f"Matched {len(matched_df)} human-LLM pairs")
    print(f"Human multiplier range: {matched_df['multiplier'].min():.2f} - {matched_df['multiplier'].max():.2f}")
    print(f"Mean multiplier: {matched_df['multiplier'].mean():.2f}\n")
    
    # Choice Agreement Analysis
    print("MODEL vs HUMAN CHOICE AGREEMENT:")
    print("-" * 50)
    accuracy = choice_analysis['overall_accuracy']
    print(f"Overall Model Accuracy: {accuracy:.1%}")
    
    print("\nAccuracy by Human Choice Type:")
    agreement_by_choice = choice_analysis['agreement_by_choice']
    for choice_type in agreement_by_choice.index:
        acc = agreement_by_choice.loc[choice_type, 'mean']
        count = agreement_by_choice.loc[choice_type, 'count']
        print(f"  {choice_type}: {acc:.1%} ({count} samples)")
    
    # Confidence-Accuracy Correlation
    if choice_analysis['confidence_accuracy_correlation']:
        conf_corr = choice_analysis['confidence_accuracy_correlation']
        sig = "***" if conf_corr['p'] < 0.001 else "**" if conf_corr['p'] < 0.01 else "*" if conf_corr['p'] < 0.05 else ""
        print(f"\nModel Confidence vs Choice Accuracy: r={conf_corr['r']:.3f}{sig}")
        
    # Multiplier analysis for correct vs incorrect
    correct_mult = choice_analysis['correct_multipliers']
    incorrect_mult = choice_analysis['incorrect_multipliers']
    if len(correct_mult) > 0 and len(incorrect_mult) > 0:
        print(f"\nMultiplier Analysis:")
        print(f"  Correct predictions - Mean: {correct_mult.mean():.2f}, Std: {correct_mult.std():.2f}")
        print(f"  Incorrect predictions - Mean: {incorrect_mult.mean():.2f}, Std: {incorrect_mult.std():.2f}")
    
    print(f"\nCORRELATIONS WITH HUMAN MULTIPLIER:")
    print("-" * 50)
    
    for feature, stats in correlations.items():
        pearson_r = stats['pearson_r']
        pearson_p = stats['pearson_p']
        spearman_r = stats['spearman_r']
        spearman_p = stats['spearman_p']
        
        # Significance indicators
        sig_p = "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else ""
        sig_s = "***" if spearman_p < 0.001 else "**" if spearman_p < 0.01 else "*" if spearman_p < 0.05 else ""
        
        print(f"{feature:15} | Pearson: {pearson_r:6.3f}{sig_p:3} | Spearman: {spearman_r:6.3f}{sig_s:3}")
    
    print("\n*** p<0.001, ** p<0.01, * p<0.05")
    
    # Interpretation
    print("\n=== INTERPRETATION ===")
    
    # Choice Agreement Assessment
    if accuracy >= 0.7:
        print(f"🟢 GOOD model-human agreement! ({accuracy:.1%} accuracy)")
        print("   → LLMs are reasonably aligned with human preferences")
    elif accuracy >= 0.5:
        print(f"🟡 MODERATE model-human agreement ({accuracy:.1%} accuracy)")
        print("   → Some alignment, but significant disagreement exists")
    else:
        print(f"🔴 POOR model-human agreement ({accuracy:.1%} accuracy)")
        print("   → LLMs frequently disagree with human judgments")
    
    # Correlation Assessment
    best_feature = max(correlations.keys(), key=lambda x: abs(correlations[x]['pearson_r']))
    best_corr = correlations[best_feature]['pearson_r']
    
    if abs(best_corr) > 0.5:
        print(f"🟢 STRONG correlation found! {best_feature} (r={best_corr:.3f})")
        print("   → Log-probability approach looks viable!")
    elif abs(best_corr) > 0.3:
        print(f"🟡 MODERATE correlation found: {best_feature} (r={best_corr:.3f})")
        print("   → Log-probability approach might work with refinement")
    else:
        print(f"🔴 WEAK correlations found (best: {best_feature}, r={best_corr:.3f})")
        print("   → Consider bucket-based multiplier approach instead")
    
    # Combined Assessment
    print(f"\n=== OVERALL RECOMMENDATION ===")
    if accuracy >= 0.6 and abs(best_corr) >= 0.4:
        print("✅ PROCEED with ensemble approach - both choice agreement and multiplier correlation show promise")
    elif accuracy >= 0.5 or abs(best_corr) >= 0.3:
        print("⚠️  PROCEED WITH CAUTION - consider simpler ensemble methods first")
    else:
        print("❌ RECONSIDER APPROACH - weak evidence for both choice agreement and multiplier correlation")
        print("   → Try bucket-based multiplier estimation or simple voting instead")

def main(save_plots=True, show_plots=False):
    """Main execution function."""
    try:
        # Load data
        print("Loading data...")
        train_df, llm_cache = load_data()
        
        # Preprocess human data
        print("Preprocessing human preferences...")
        human_df = preprocess_human_data(train_df)
        
        # Match data
        print("Matching human preferences with LLM responses...")
        matched_df = match_human_llm_data(human_df, llm_cache)
        
        if len(matched_df) == 0:
            print("❌ No matches found between human data and LLM cache!")
            print("Check that repo URLs match between train.csv and correlation_cache.json")
            return
        
        # Print matched data summary
        print_matched_data_summary(matched_df)
        
        # Calculate correlations
        print("Calculating correlations...")
        correlations = calculate_correlations(matched_df)
        
        # Analyze choice agreement
        print("Analyzing model vs human choice agreement...")
        choice_analysis = analyze_choice_agreement(matched_df)
        
        # Print results
        print_results(matched_df, correlations, choice_analysis)
        
        # Visualize
        print("\nGenerating comprehensive visualizations...")
        create_comprehensive_visualizations(matched_df, correlations, choice_analysis, 
                                           save_plots=save_plots, show_plots=show_plots)
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure train.csv and correlation_cache.json are in the current directory")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Default: save plots to files, don't show them interactively
    # To show plots instead: main(save_plots=False, show_plots=True)
    # To do both: main(save_plots=True, show_plots=True)
    main(save_plots=True, show_plots=False)