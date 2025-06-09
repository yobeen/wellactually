#!/usr/bin/env python3
"""
Correlation Analysis: Log Probabilities vs Human Multipliers

Validates whether LLM log probabilities correlate with human multiplier judgments
for repository importance comparisons.

Usage:
    # First run (will query LLM and cache results)
    python logprob_multiplier_correlation.py
    
    # Subsequent runs (will use cached data)
    python logprob_multiplier_correlation.py
    
    # Force new queries even if cache exists
    python logprob_multiplier_correlation.py --force-query
    
    # Clear cache and start fresh
    python logprob_multiplier_correlation.py --clear-cache
    
    # Use specific model
    python logprob_multiplier_correlation.py --model "openai/gpt-4o"
    
    # Debug logprob extraction issues
    python logprob_multiplier_correlation.py --debug-logprobs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix
import logging
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Import existing components
import sys
sys.path.append('src')
from src.shared.multi_model_engine import MultiModelEngine
from src.tasks.l1.level1_prompts import Level1PromptGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogprobMultiplierAnalyzer:
    """Analyzes correlation between LLM log probabilities and human multipliers."""
    
    def __init__(self, config, cache_file: str = "correlation_cache.json"):
        """
        Initialize the analyzer.
        
        Args:
            config: Configuration object with API settings
            cache_file: File to cache LLM responses to avoid re-querying
        """
        self.config = config
        self.cache_file = cache_file
        self.engine = MultiModelEngine(config)
        
        # Initialize prompt generator with minimal config
        # For correlation analysis, we'll create prompts directly without seed repos
        try:
            self.prompt_gen = Level1PromptGenerator(config)
        except Exception as e:
            logger.warning(f"Could not fully initialize Level1PromptGenerator: {e}")
            logger.info("Will create prompts directly for correlation analysis")
            self.prompt_gen = None
            
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """Load cached responses if available."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save responses to cache."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _try_load_from_cache(self, human_df: pd.DataFrame, model_id: str) -> Optional[List[Dict]]:
        """
        Try to load complete analysis from cache.
        
        Args:
            human_df: Human judgment DataFrame
            model_id: Model identifier
            
        Returns:
            List of result dictionaries if cache is complete, None otherwise
        """
        if not self.cache:
            logger.info("Cache is empty")
            return None
        
        logger.info(f"Checking cache for {len(human_df)} samples with model {model_id}")
        
        results = []
        missing_samples = []
        
        for idx, row in human_df.iterrows():
            cache_key = f"{model_id}|{row['repo_a']}|{row['repo_b']}"
            
            if cache_key in self.cache:
                model_response = self.cache[cache_key]
                
                # Extract features from cached response
                features = self.extract_logprob_features(
                    model_response['logprobs'],
                    row['choice'],
                    model_response['model_choice']
                )
                
                # Combine all data
                result = {
                    'sample_id': idx,
                    'repo_a': row['repo_a'],
                    'repo_b': row['repo_b'],
                    'human_choice': row['choice'],
                    'human_multiplier': row['multiplier'],
                    'model_choice': model_response['model_choice'],
                    'model_content': model_response['content'],
                    'cost_usd': model_response['cost_usd'],
                    **features
                }
                
                results.append(result)
            else:
                missing_samples.append((idx, row['repo_a'], row['repo_b']))
        
        if missing_samples:
            logger.info(f"Cache incomplete: missing {len(missing_samples)} samples")
            logger.info(f"Missing samples: {missing_samples[:5]}...")  # Show first 5
            return None
        else:
            logger.info(f"Cache complete: found all {len(results)} samples")
            return results
    
    def _confirm_expensive_operation(self) -> bool:
        """
        Ask user to confirm expensive LLM queries.
        
        Returns:
            True if user confirms, False otherwise
        """
        try:
            # Check if we have cost estimates
            estimated_cost = 0.0
            try:
                if hasattr(self.config, 'cost_management'):
                    cost_per_1k = self.config.cost_management.cost_per_1k_tokens
                    # Rough estimate: ~50 tokens per query
                    estimated_cost = (50 / 1000) * cost_per_1k * 140  # ~140 samples
            except:
                estimated_cost = 1.0  # Fallback estimate
            
            print(f"\n{'='*50}")
            print("COST WARNING")
            print(f"{'='*50}")
            print(f"About to make ~140 LLM API calls")
            print(f"Estimated cost: ~${estimated_cost:.2f}")
            print("This will take several minutes to complete.")
            print(f"{'='*50}")
            
            response = input("Continue with LLM queries? (y/N): ").strip().lower()
            return response in ['y', 'yes']
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return False
        except:
            # In non-interactive environments, assume yes
            return True
    
    def load_human_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and filter human judgment data for L1 samples.
        
        Args:
            csv_path: Path to train.csv file
            
        Returns:
            DataFrame with L1 human judgments
        """
        logger.info(f"Loading human data from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} total samples")
            
            # Filter for L1 samples
            l1_mask = df['juror'].str.contains('L1', na=False)
            l1_df = df[l1_mask].copy()
            
            logger.info(f"Found {len(l1_df)} L1 samples")
            
            # Validate required columns
            required_cols = ['repo_a', 'repo_b', 'choice', 'multiplier']
            missing_cols = [col for col in required_cols if col not in l1_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean data
            l1_df = l1_df.dropna(subset=required_cols)
            logger.info(f"After cleaning: {len(l1_df)} samples")
            
            return l1_df
            
        except Exception as e:
            logger.error(f"Failed to load human data: {e}")
            raise
    
    def extract_logprob_features(self, logprobs: Dict[str, float], 
                                human_choice: str, model_choice: str) -> Dict:
        """
        Extract correlation features from log probabilities.
        
        Args:
            logprobs: Dictionary of token probabilities
            human_choice: Human's choice (A/B/Equal)
            model_choice: Model's choice
            
        Returns:
            Dictionary of extracted features
        """
        # Get probabilities for each option (handle different token formats)
        prob_A = 0.0
        prob_B = 0.0
        prob_Equal = 0.0
        
        # Look for different possible token representations
        for token, prob in logprobs.items():
            token_clean = token.strip().upper()
            if token_clean in ['A']:
                prob_A = max(prob_A, prob)
            elif token_clean in ['B']:
                prob_B = max(prob_B, prob)
            elif token_clean in ['EQUAL', 'E']:
                prob_Equal = max(prob_Equal, prob)
        
        # Normalize probabilities
        total_prob = prob_A + prob_B + prob_Equal
        if total_prob > 0:
            prob_A /= total_prob
            prob_B /= total_prob
            prob_Equal /= total_prob
        else:
            # Fallback if no matching tokens found
            logger.warning(f"No matching tokens found in logprobs: {list(logprobs.keys())}")
            return {
                'prob_ratio': 1.0,
                'confidence': 0.5,
                'probability_gap': 0.0,
                'entropy': 1.0,
                'prob_chosen': 0.5,
                'prob_A': 0.33,
                'prob_B': 0.33,
                'prob_Equal': 0.33,
                'model_agrees': False
            }
        
        # Calculate features
        # 1. Probability ratio (chosen vs not chosen)
        if human_choice == 'A' and prob_B > 0:
            prob_ratio = prob_A / prob_B
        elif human_choice == 'B' and prob_A > 0:
            prob_ratio = prob_B / prob_A
        elif human_choice == 'Equal':
            prob_ratio = prob_Equal / max(prob_A, prob_B, 1e-10)
        else:
            prob_ratio = 1.0
        
        # 2. Confidence (max probability)
        confidence = max(prob_A, prob_B, prob_Equal)
        
        # 3. Probability gap (difference between top two)
        probs_sorted = sorted([prob_A, prob_B, prob_Equal], reverse=True)
        probability_gap = probs_sorted[0] - probs_sorted[1]
        
        # 4. Entropy (uncertainty measure)
        entropy = -sum(p * np.log2(p + 1e-10) for p in [prob_A, prob_B, prob_Equal] if p > 0)
        
        # 5. Probability of chosen option
        if human_choice == 'A':
            prob_chosen = prob_A
        elif human_choice == 'B':
            prob_chosen = prob_B
        else:
            prob_chosen = prob_Equal
        
        # 6. Model agreement
        model_agrees = (human_choice == model_choice)
        
        return {
            'prob_ratio': prob_ratio,
            'confidence': confidence,
            'probability_gap': probability_gap,
            'entropy': entropy,
            'prob_chosen': prob_chosen,
            'prob_A': prob_A,
            'prob_B': prob_B,
            'prob_Equal': prob_Equal,
            'model_agrees': model_agrees
        }
    
    def query_model_for_sample(self, repo_a: str, repo_b: str, 
                              model_id: str = None) -> Optional[Dict]:
        """
        Query LLM for a single repository comparison.
        
        Args:
            repo_a: First repository URL/name
            repo_b: Second repository URL/name
            model_id: Model to query (defaults to GPT-4o from config)
            
        Returns:
            Response dictionary or None if failed
        """
        # Use model from config if not specified
        if model_id is None:
            try:
                model_id = self.config.models.primary_models.gpt_4o
            except:
                model_id = "openai/gpt-4o"  # Fallback
        
        # Create cache key
        cache_key = f"{model_id}|{repo_a}|{repo_b}"
        
        # Check cache first
        if cache_key in self.cache:
            logger.debug(f"Using cached response for {repo_a} vs {repo_b}")
            return self.cache[cache_key]
        
        try:
            # Create comparison prompt
            if self.prompt_gen is not None:
                prompt = self.prompt_gen.create_comparison_prompt(
                    {'url': repo_a}, {'url': repo_b}
                )
            else:
                # Create prompt directly for correlation analysis
                prompt = [
                    {
                        "role": "system",
                        "content": "You are an expert evaluating the relative importance of open source repositories to the Ethereum ecosystem. Respond with only 'A' or 'B' or 'Equal' to indicate which repository's contribution is more important."
                    },
                    {
                        "role": "user", 
                        "content": f"""Which repository contributes more to the Ethereum ecosystem?

Repository A: {repo_a}
Repository B: {repo_b}

Choose: A or B or Equal"""
                    }
                ]
            
            # Query model
            response = self.engine.query_single_model_with_temperature(
                model_id=model_id,
                prompt=prompt,
                temperature=0.0
            )
            
            if response.success:
                result = {
                    'model_choice': response.raw_choice,
                    'content': response.content,
                    'logprobs': response.logprobs,
                    'uncertainty': response.uncertainty,
                    'cost_usd': response.cost_usd
                }
                
                # Cache the result
                self.cache[cache_key] = result
                return result
            else:
                logger.error(f"Model query failed: {response.error}")
                return None
                
        except Exception as e:
            logger.error(f"Error querying model for {repo_a} vs {repo_b}: {e}")
            return None
    
    def analyze_correlations(self, csv_path: str, 
                           model_id: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Run full correlation analysis.
        
        Args:
            csv_path: Path to train.csv
            model_id: Model to test (defaults to GPT-4o from config)
            
        Returns:
            Tuple of (results_df, correlation_stats)
        """
        logger.info("Starting correlation analysis...")
        
        # Use model from config if not specified
        if model_id is None:
            try:
                model_id = self.config.models.primary_models.gpt_4o
                logger.info(f"Using model from config: {model_id}")
            except:
                model_id = "openai/gpt-4o"  # Fallback
                logger.info(f"Using fallback model: {model_id}")
        
        # Load human data
        human_df = self.load_human_data(csv_path)
        
        # Check if we can use cached data
        cached_results = self._try_load_from_cache(human_df, model_id)
        
        if cached_results is not None:
            logger.info(f"✓ Using cached data for {len(cached_results)} samples")
            print(f"Cache hit: Using cached responses for all {len(cached_results)} samples")
            print(f"Model used for cached data: {model_id}")
            print("Skipping LLM queries - proceeding directly to analysis...")
            results_df = pd.DataFrame(cached_results)
            # Calculate correlations
            correlation_stats = self.calculate_correlation_stats(results_df)
            return results_df, correlation_stats
        
        # If cache is incomplete, proceed with querying
        logger.info("Cache incomplete or empty, proceeding with LLM queries...")
        
        results = []
        total_samples = len(human_df)
        
        for idx, row in human_df.iterrows():
            logger.info(f"Processing sample {idx + 1}/{total_samples}: {row['repo_a']} vs {row['repo_b']}")
            
            # Query model
            model_response = self.query_model_for_sample(
                row['repo_a'], row['repo_b'], model_id
            )
            
            if model_response is None:
                logger.warning(f"Skipping sample {idx + 1} due to query failure")
                continue
            
            # Extract features
            features = self.extract_logprob_features(
                model_response['logprobs'],
                row['choice'],
                model_response['model_choice']
            )
            
            # Combine all data
            result = {
                'sample_id': idx,
                'repo_a': row['repo_a'],
                'repo_b': row['repo_b'],
                'human_choice': row['choice'],
                'human_multiplier': row['multiplier'],
                'model_choice': model_response['model_choice'],
                'model_content': model_response['content'],
                'cost_usd': model_response['cost_usd'],
                **features
            }
            
            results.append(result)
            
            # Save cache periodically
            if len(results) % 10 == 0:
                self._save_cache()
                logger.info(f"Processed {len(results)} samples, saved cache")
            
            # Small delay to respect rate limits
            time.sleep(0.1)
        
        # Final cache save
        self._save_cache()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        logger.info(f"Completed analysis on {len(results_df)} samples")
        
        # Calculate correlations
        correlation_stats = self.calculate_correlation_stats(results_df)
        
        return results_df, correlation_stats
    
    def calculate_correlation_stats(self, results_df: pd.DataFrame) -> Dict:
        """Calculate correlation statistics with robust error handling."""
        feature_cols = [
            'prob_ratio', 'confidence', 'probability_gap', 
            'entropy', 'prob_chosen', 'prob_A', 'prob_B', 'prob_Equal'
        ]
        
        correlations = {}
        
        # Check for basic data validity
        if len(results_df) < 3:
            logger.warning(f"Insufficient data for correlation analysis: {len(results_df)} samples")
            return correlations
        
        # Check multiplier variance
        multiplier_var = results_df['human_multiplier'].var()
        if multiplier_var == 0:
            logger.warning("Human multipliers have zero variance - cannot calculate correlations")
            return correlations
        
        logger.info(f"Calculating correlations for {len(results_df)} samples")
        
        for feature in feature_cols:
            if feature in results_df.columns:
                feature_data = results_df[feature].dropna()
                multiplier_data = results_df.loc[feature_data.index, 'human_multiplier']
                
                # Check if we have enough valid data
                if len(feature_data) < 3:
                    logger.warning(f"Insufficient valid data for {feature}: {len(feature_data)} samples")
                    continue
                
                # Check for constant values
                feature_var = feature_data.var()
                if feature_var == 0 or np.isnan(feature_var):
                    logger.warning(f"Feature {feature} has zero variance (constant values: {feature_data.iloc[0]:.4f})")
                    correlations[feature] = {
                        'pearson_r': np.nan,
                        'pearson_p': np.nan,
                        'spearman_r': np.nan,
                        'spearman_p': np.nan,
                        'n_samples': len(feature_data),
                        'feature_var': feature_var,
                        'constant_value': feature_data.iloc[0]
                    }
                    continue
                
                try:
                    # Pearson correlation
                    pearson_r, pearson_p = stats.pearsonr(feature_data, multiplier_data)
                    
                    # Spearman correlation  
                    spearman_r, spearman_p = stats.spearmanr(feature_data, multiplier_data)
                    
                    correlations[feature] = {
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'n_samples': len(feature_data),
                        'feature_var': feature_var,
                        'feature_mean': feature_data.mean(),
                        'feature_std': feature_data.std()
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate correlation for {feature}: {e}")
                    correlations[feature] = {
                        'pearson_r': np.nan,
                        'pearson_p': np.nan,
                        'spearman_r': np.nan,
                        'spearman_p': np.nan,
                        'n_samples': len(feature_data),
                        'error': str(e)
                    }
        
        return correlations
    
    def create_visualizations(self, results_df: pd.DataFrame, 
                            correlations: Dict, output_dir: str = "plots"):
        """Create comprehensive correlation visualizations with robust error handling."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter out features with valid correlations for visualization
        valid_features = []
        for feature, stats in correlations.items():
            if not np.isnan(stats.get('pearson_r', np.nan)):
                valid_features.append(feature)
        
        if not valid_features:
            logger.warning("No valid features for correlation visualization")
            # Create a simple message plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No valid correlations found\nAll features may be constant or have insufficient data", 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title("Correlation Analysis Results")
            plt.savefig(f"{output_dir}/correlation_scatter_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        feature_cols = valid_features[:5]  # Limit to top 5 valid features
        
        # 1. Main correlation scatter plots
        n_plots = min(6, len(feature_cols) + 1)  # +1 for potential empty plot
        cols = 3
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, feature in enumerate(feature_cols):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            try:
                # Check if feature has variance
                if results_df[feature].var() == 0:
                    ax.text(0.5, 0.5, f"{feature}\n(Constant value)", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{feature.replace('_', ' ').title()}\n(No variance)")
                    continue
                
                # Scatter plot with color coding by human choice
                choices = results_df['human_choice'].dropna().unique()
                colors = ['red', 'blue', 'green', 'orange', 'purple']
                
                for choice_idx, choice in enumerate(choices):
                    if choice_idx >= len(colors):
                        break
                    mask = results_df['human_choice'] == choice
                    if mask.any():
                        ax.scatter(results_df[mask][feature], results_df[mask]['human_multiplier'], 
                                  alpha=0.7, s=40, c=colors[choice_idx], label=f'Choice: {choice}')
                
                # Add trend line with error handling
                try:
                    valid_mask = results_df[feature].notna() & results_df['human_multiplier'].notna()
                    if valid_mask.sum() >= 2:  # Need at least 2 points for trend line
                        x_vals = results_df[valid_mask][feature]
                        y_vals = results_df[valid_mask]['human_multiplier']
                        
                        if x_vals.var() > 0:  # Only fit if there's variance
                            z = np.polyfit(x_vals, y_vals, 1)
                            p = np.poly1d(z)
                            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                            ax.plot(x_line, p(x_line), "black", linestyle="--", alpha=0.8, linewidth=2)
                except (np.RankWarning, np.linalg.LinAlgError, ValueError) as e:
                    logger.debug(f"Could not fit trend line for {feature}: {e}")
                
                # Labels and correlation info
                if feature in correlations:
                    corr_info = correlations[feature]
                    title = f"{feature.replace('_', ' ').title()}\n"
                    
                    pearson_r = corr_info.get('pearson_r', np.nan)
                    pearson_p = corr_info.get('pearson_p', np.nan)
                    spearman_r = corr_info.get('spearman_r', np.nan)
                    
                    if not np.isnan(pearson_r):
                        title += f"Pearson r={pearson_r:.3f} (p={pearson_p:.3f})\n"
                        title += f"Spearman ρ={spearman_r:.3f}"
                    else:
                        title += "No valid correlation"
                    
                    ax.set_title(title, fontsize=11)
                
                ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=10)
                ax.set_ylabel('Human Multiplier', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                if len(choices) > 0:
                    ax.legend(fontsize=8)
                    
            except Exception as e:
                logger.warning(f"Error creating scatter plot for {feature}: {e}")
                ax.text(0.5, 0.5, f"Error plotting {feature}", 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Remove empty subplots
        for idx in range(len(feature_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Log Probability Features vs Human Multipliers', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_scatter_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distribution analysis
        self._create_distribution_plots(results_df, output_dir)
        
        # 3. Correlation heatmap
        self._create_correlation_heatmap(correlations, output_dir)
        
        # 4. Feature importance ranking
        self._create_feature_importance_plot(correlations, output_dir)
        
        # 5. Model choice vs human choice confusion matrix
        self._create_confusion_matrix_plot(results_df, output_dir)
        
        logger.info(f"All visualizations saved to {output_dir}/")
    
    def _create_distribution_plots(self, results_df: pd.DataFrame, output_dir: str):
        """Create distribution analysis plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Multiplier distribution
            axes[0,0].hist(results_df['human_multiplier'].dropna(), bins=20, alpha=0.7, 
                          color='skyblue', edgecolor='black')
            axes[0,0].set_title('Distribution of Human Multipliers')
            axes[0,0].set_xlabel('Multiplier Value')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].grid(True, alpha=0.3)
            
            # Model agreement vs multiplier
            if 'model_agrees' in results_df.columns:
                agreement_stats = results_df.groupby('model_agrees')['human_multiplier'].agg(['mean', 'std', 'count'])
                if len(agreement_stats) > 0:
                    x_pos = list(range(len(agreement_stats)))
                    means = agreement_stats['mean'].values
                    stds = agreement_stats['std'].fillna(0).values
                    labels = [str(x) for x in agreement_stats.index]
                    
                    axes[0,1].bar(x_pos, means, yerr=stds, alpha=0.7, 
                                 color=['lightcoral', 'lightgreen'][:len(x_pos)], 
                                 capsize=5, edgecolor='black')
                    axes[0,1].set_title('Multiplier by Model Agreement')
                    axes[0,1].set_xlabel('Model Agrees with Human')
                    axes[0,1].set_ylabel('Mean Human Multiplier')
                    axes[0,1].set_xticks(x_pos)
                    axes[0,1].set_xticklabels(labels)
                    axes[0,1].grid(True, alpha=0.3)
            
            # Choice distribution
            if 'human_choice' in results_df.columns:
                choice_counts = results_df['human_choice'].value_counts()
                if len(choice_counts) > 0:
                    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'][:len(choice_counts)]
                    axes[1,0].pie(choice_counts.values, labels=choice_counts.index, autopct='%1.1f%%', colors=colors)
                    axes[1,0].set_title('Distribution of Human Choices')
            
            # Probability features distribution
            prob_features = ['prob_A', 'prob_B', 'prob_Equal']
            for i, feature in enumerate(prob_features):
                if feature in results_df.columns and results_df[feature].var() > 0:
                    axes[1,1].hist(results_df[feature].dropna(), bins=15, alpha=0.6, 
                                  label=feature, density=True)
            axes[1,1].set_title('Distribution of Model Probabilities')
            axes[1,1].set_xlabel('Probability')
            axes[1,1].set_ylabel('Density')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/distribution_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error creating distribution plots: {e}")
    
    def _create_correlation_heatmap(self, correlations: Dict, output_dir: str):
        """Create correlation heatmap."""
        try:
            valid_correlations = {k: v for k, v in correlations.items() 
                                if not np.isnan(v.get('pearson_r', np.nan))}
            
            if not valid_correlations:
                logger.warning("No valid correlations for heatmap")
                return
            
            pearson_corrs = [corr['pearson_r'] for corr in valid_correlations.values()]
            spearman_corrs = [corr['spearman_r'] for corr in valid_correlations.values()]
            feature_names = [f.replace('_', ' ').title() for f in valid_correlations.keys()]
            
            correlation_matrix = pd.DataFrame({
                'Pearson': pearson_corrs,
                'Spearman': spearman_corrs
            }, index=feature_names)
            
            plt.figure(figsize=(10, max(8, len(feature_names) * 0.5)))
            sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                       fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'},
                       square=True, linewidths=0.5)
            plt.title('Feature Correlations with Human Multipliers\n(Pearson vs Spearman)', fontsize=14)
            plt.ylabel('Features')
            plt.xlabel('Correlation Type')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error creating correlation heatmap: {e}")
    
    def _create_feature_importance_plot(self, correlations: Dict, output_dir: str):
        """Create feature importance ranking plot."""
        try:
            valid_correlations = {k: v for k, v in correlations.items() 
                                if not np.isnan(v.get('pearson_r', np.nan))}
            
            if not valid_correlations:
                logger.warning("No valid correlations for feature importance plot")
                return
            
            feature_importance = [(feature.replace('_', ' ').title(), abs(corr['pearson_r'])) 
                                 for feature, corr in valid_correlations.items()]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            if not feature_importance:
                return
            
            features, importances = zip(*feature_importance)
            
            plt.figure(figsize=(10, max(6, len(features) * 0.4)))
            bars = plt.barh(range(len(features)), importances, color='steelblue', alpha=0.7)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Absolute Correlation with Human Multiplier')
            plt.title('Feature Importance Ranking\n(Based on Absolute Pearson Correlation)')
            plt.grid(True, alpha=0.3, axis='x')
            
            # Add correlation values on bars
            for i, (bar, importance) in enumerate(zip(bars, importances)):
                plt.text(importance + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.3f}', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error creating feature importance plot: {e}")
    
    def _create_confusion_matrix_plot(self, results_df: pd.DataFrame, output_dir: str):
        """Create model vs human choice confusion matrix."""
        try:
            # Get actual unique labels present in the data
            human_choices = results_df['human_choice'].dropna()
            model_choices = results_df['model_choice'].dropna()
            
            # Find common indices (where both human and model choices exist)
            common_idx = human_choices.index.intersection(model_choices.index)
            if len(common_idx) == 0:
                logger.warning("No common indices for confusion matrix")
                return
            
            human_choices_aligned = human_choices.loc[common_idx]
            model_choices_aligned = model_choices.loc[common_idx]
            
            # Get unique labels from actual data
            all_labels = sorted(set(human_choices_aligned.unique()) | set(model_choices_aligned.unique()))
            
            if len(all_labels) == 0:
                logger.warning("No valid labels for confusion matrix")
                return
            
            # Create confusion matrix with actual labels
            cm = confusion_matrix(human_choices_aligned, model_choices_aligned, labels=all_labels)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=all_labels, yticklabels=all_labels)
            plt.title('Model vs Human Choice Agreement')
            plt.xlabel('Model Choice')
            plt.ylabel('Human Choice')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/choice_confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error creating confusion matrix: {e}")
    
    def print_summary(self, correlations: Dict, results_df: pd.DataFrame):
        """Print analysis summary with enhanced debugging information."""
        print("\n" + "="*60)
        print("LOGPROB-MULTIPLIER CORRELATION ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nDataset: {len(results_df)} samples analyzed")
        if 'model_agrees' in results_df.columns:
            print(f"Model agreement rate: {results_df['model_agrees'].mean():.3f}")
        print(f"Total API cost: ${results_df['cost_usd'].sum():.4f}")
        
        print(f"\nMultiplier distribution:")
        multipliers = results_df['human_multiplier'].dropna()
        print(f"  Mean: {multipliers.mean():.2f}")
        print(f"  Std:  {multipliers.std():.2f}")
        print(f"  Range: {multipliers.min():.1f} - {multipliers.max():.1f}")
        print(f"  Unique values: {multipliers.nunique()}")
        
        # Feature variance analysis
        print(f"\nFeature Quality Analysis:")
        print("-" * 40)
        feature_cols = ['prob_ratio', 'confidence', 'probability_gap', 'entropy', 'prob_chosen', 
                       'prob_A', 'prob_B', 'prob_Equal']
        
        for feature in feature_cols:
            if feature in results_df.columns:
                feature_data = results_df[feature].dropna()
                if len(feature_data) > 0:
                    variance = feature_data.var()
                    mean_val = feature_data.mean()
                    std_val = feature_data.std()
                    unique_vals = feature_data.nunique()
                    
                    status = "CONSTANT" if variance == 0 else "VALID"
                    print(f"  {feature:<20}: {status:<8} | var={variance:.6f} | mean={mean_val:.4f} | unique={unique_vals}")
                    
                    if variance == 0:
                        print(f"    └─ All values: {feature_data.iloc[0]:.6f}")
        
        # Choice distribution analysis
        print(f"\nChoice Distribution:")
        if 'human_choice' in results_df.columns:
            human_dist = results_df['human_choice'].value_counts()
            print(f"  Human choices: {dict(human_dist)}")
        
        if 'model_choice' in results_df.columns:
            model_dist = results_df['model_choice'].value_counts()
            print(f"  Model choices: {dict(model_dist)}")
        
        # Correlation results
        print(f"\nCorrelation Results (Pearson):")
        print("-" * 40)
        
        valid_correlations = {k: v for k, v in correlations.items() 
                            if not np.isnan(v.get('pearson_r', np.nan))}
        
        if not valid_correlations:
            print("  No valid correlations found!")
            print("  All features may be constant or have insufficient variance.")
        else:
            sorted_features = sorted(valid_correlations.items(), 
                                   key=lambda x: abs(x[1]['pearson_r']), reverse=True)
            
            for feature, stats in sorted_features:
                r = stats['pearson_r']
                p = stats['pearson_p']
                n = stats.get('n_samples', 0)
                significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {feature:<20}: r={r:6.3f}  p={p:.4f}  n={n:3d}  {significance}")
        
        # Interpretation
        print(f"\nInterpretation:")
        if valid_correlations:
            best_feature, best_stats = sorted(valid_correlations.items(), 
                                            key=lambda x: abs(x[1]['pearson_r']), reverse=True)[0]
            best_r = abs(best_stats['pearson_r'])
            
            if best_r > 0.7:
                interpretation = "STRONG correlation - logprobs are excellent predictors"
            elif best_r > 0.5:
                interpretation = "MODERATE correlation - logprobs show promise"
            elif best_r > 0.3:
                interpretation = "WEAK correlation - logprobs have limited utility"
            else:
                interpretation = "MINIMAL correlation - logprobs are poor predictors"
            
            print(f"  Best feature: {best_feature} (r={best_stats['pearson_r']:.3f})")
            print(f"  Assessment: {interpretation}")
        else:
            print("  Assessment: NO CORRELATIONS FOUND")
            print("  Recommendation: Check logprob extraction and feature engineering")
        
        # Data quality warnings
        constant_features = [f for f, stats in correlations.items() 
                           if stats.get('feature_var', 1) == 0]
        
        if constant_features:
            print(f"\n⚠️  DATA QUALITY WARNINGS:")
            print(f"  Constant features found: {constant_features}")
            print(f"  This suggests issues with logprob extraction or model responses")
            print(f"  Consider debugging with: --debug-logprobs flag")
        
        print("\n" + "="*60)
    
    def debug_logprob_extraction(self, results_df: pd.DataFrame, n_samples: int = 5):
        """Debug logprob extraction by showing sample data."""
        print(f"\n{'='*60}")
        print("LOGPROB EXTRACTION DEBUGGING")
        print(f"{'='*60}")
        
        print(f"Showing first {n_samples} samples for debugging:")
        
        for idx in range(min(n_samples, len(results_df))):
            row = results_df.iloc[idx]
            print(f"\nSample {idx + 1}:")
            print(f"  Repos: {row.get('repo_a', 'N/A')} vs {row.get('repo_b', 'N/A')}")
            print(f"  Human choice: {row.get('human_choice', 'N/A')} (multiplier: {row.get('human_multiplier', 'N/A')})")
            print(f"  Model choice: {row.get('model_choice', 'N/A')}")
            print(f"  Raw probs: A={row.get('prob_A', 'N/A'):.4f}, B={row.get('prob_B', 'N/A'):.4f}, Equal={row.get('prob_Equal', 'N/A'):.4f}")
            print(f"  Features: ratio={row.get('prob_ratio', 'N/A'):.4f}, conf={row.get('confidence', 'N/A'):.4f}, ent={row.get('entropy', 'N/A'):.4f}")
        
        print(f"\n{'='*60}")


def main():
    """Main execution function."""
    import sys
    import yaml
    from omegaconf import OmegaConf
    import argparse
    
    # Add command line arguments
    parser = argparse.ArgumentParser(description='LLM Log Probability vs Human Multiplier Correlation Analysis')
    parser.add_argument('--force-query', action='store_true', 
                       help='Force new LLM queries even if cache exists')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear existing cache before starting')
    parser.add_argument('--model', type=str, default=None,
                       help='Model ID to use (overrides config)')
    parser.add_argument('--debug-logprobs', action='store_true',
                       help='Show detailed logprob extraction debugging')
    args = parser.parse_args()
    
    # Load LLM configuration (primary config for API settings)
    llm_config_path = 'configs/uncertainty_calibration/llm.yaml'
    try:
        with open(llm_config_path, 'r') as f:
            config = OmegaConf.load(f)
        logger.info(f"Loaded LLM config from {llm_config_path}")
    except FileNotFoundError:
        logger.error(f"LLM configuration file not found: {llm_config_path}")
        sys.exit(1)
    
    # Load main project config for paths
    main_config_path = 'configs/config.yaml'
    try:
        with open(main_config_path, 'r') as f:
            main_config = OmegaConf.load(f)
        logger.info(f"Loaded main config from {main_config_path}")
        
        # Use data path from main config
        csv_path = os.path.join(main_config.data.raw_dir, "train.csv")
        
    except FileNotFoundError:
        logger.warning(f"Main configuration file not found: {main_config_path}")
        # Fallback to hardcoded path
        csv_path = "data/raw/train.csv"
    
    # Verify data file exists
    if not os.path.exists(csv_path):
        logger.error(f"Training data file not found: {csv_path}")
        sys.exit(1)
    
    # Handle cache clearing
    cache_file = "correlation_cache.json"
    if args.clear_cache and os.path.exists(cache_file):
        os.remove(cache_file)
        logger.info("Cleared existing cache")
    
    # Initialize analyzer with LLM config (contains API settings)
    analyzer = LogprobMultiplierAnalyzer(config, cache_file)
    
    # Show cache status
    cache_size = len(analyzer.cache)
    logger.info(f"Cache status: {cache_size} cached responses")
    
    # Handle force query option
    if args.force_query:
        logger.info("Force query mode: will clear cache for this analysis")
        analyzer.cache = {}
    
    # Show available models from config
    try:
        primary_models = config.models.primary_models
        logger.info("Available models from config:")
        for model_name, model_id in primary_models.items():
            logger.info(f"  {model_name}: {model_id}")
    except:
        logger.warning("Could not load model configuration")
    
    # Determine model to use
    model_to_use = args.model
    if model_to_use:
        logger.info(f"Using model from command line: {model_to_use}")
    
    try:
        # Run analysis
        results_df, correlations = analyzer.analyze_correlations(csv_path, model_to_use)
        
        # Handle case where user cancelled expensive operation
        if results_df is None:
            logger.info("Analysis was cancelled or no data available")
            sys.exit(0)
        
        # Check if we have any valid correlations
        valid_correlations = {k: v for k, v in correlations.items() 
                            if not np.isnan(v.get('pearson_r', np.nan))}
        
        if not valid_correlations:
            logger.warning("No valid correlations found - all features may be constant")
            print("\n⚠️  WARNING: No valid correlations could be calculated!")
            print("This usually means:")
            print("  1. All logprob features have constant values")
            print("  2. Logprob extraction is failing") 
            print("  3. Model is giving identical responses")
            print("\nRecommendations:")
            print("  - Run with --debug-logprobs to investigate")
            print("  - Check if model responses vary")
            print("  - Verify logprob extraction logic")
            
            # Still create basic visualizations
            try:
                output_dir = main_config.outputs.plots_dir if 'main_config' in locals() else "plots"
                os.makedirs(output_dir, exist_ok=True)
                analyzer.create_visualizations(results_df, correlations, output_dir)
            except Exception as e:
                logger.warning(f"Could not create visualizations: {e}")
        else:
            # Create output directory from main config if available
            try:
                output_dir = main_config.outputs.plots_dir
                os.makedirs(output_dir, exist_ok=True)
            except:
                output_dir = "plots"
            
            # Create comprehensive visualizations
            analyzer.create_visualizations(results_df, correlations, output_dir)
        
        # Print summary (always do this)
        analyzer.print_summary(correlations, results_df)
        
        # Debug logprobs if requested
        if args.debug_logprobs:
            analyzer.debug_logprob_extraction(results_df)
        
        # Save results with proper paths (always do this)
        try:
            results_dir = main_config.outputs.results_dir
            os.makedirs(results_dir, exist_ok=True)
            results_path = os.path.join(results_dir, "correlation_analysis_results.csv")
            stats_path = os.path.join(results_dir, "correlation_stats.json")
        except:
            results_path = "correlation_analysis_results.csv"
            stats_path = "correlation_stats.json"
        
        results_df.to_csv(results_path, index=False)
        
        with open(stats_path, 'w') as f:
            json.dump(correlations, f, indent=2)
        
        logger.info(f"Analysis complete! Results saved to {results_path}")
        
        # Print visualization summary
        output_dir = getattr(main_config.outputs, 'plots_dir', 'plots') if 'main_config' in locals() else 'plots'
        print(f"\nGenerated visualizations in {output_dir}/:")
        print(f"  - correlation_scatter_plots.png: Main correlation analysis")
        print(f"  - distribution_analysis.png: Data distribution insights")
        print(f"  - correlation_heatmap.png: Correlation comparison")
        print(f"  - feature_importance.png: Feature ranking")
        print(f"  - choice_confusion_matrix.png: Model vs human agreement")
        
        # Print cost summary from config
        if hasattr(config, 'cost_management') and config.cost_management.track_costs:
            total_cost = results_df['cost_usd'].sum()
            max_cost = config.cost_management.max_cost_per_experiment
            print(f"\nCost Summary:")
            print(f"  Total cost: ${total_cost:.4f}")
            print(f"  Budget limit: ${max_cost:.2f}")
            print(f"  Budget used: {(total_cost/max_cost)*100:.1f}%")
        
        # Print cache summary
        final_cache_size = len(analyzer.cache)
        print(f"\nCache Summary:")
        print(f"  Cache file: {cache_file}")
        print(f"  Cached responses: {final_cache_size}")
        print(f"  Cache added this run: {final_cache_size - cache_size}")
        print(f"  Next run will use cached data (unless --force-query or --clear-cache)")
        print(f"  To rerun analysis: python {sys.argv[0]}")
        print(f"  To force new queries: python {sys.argv[0]} --force-query")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()