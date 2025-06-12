#!/usr/bin/env python3
"""
Cross-validation test for optimal weighted sum approach.

This script performs proper cross-validation on the weighted sum optimization
to provide a fair comparison with ML models.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')


class WeightedSumCVTester:
    def __init__(self, assessments_path: str, train_csv_path: str):
        self.assessments_path = assessments_path
        self.train_csv_path = train_csv_path
        self.criteria_names = []
        self.repo_scores = {}
        self.comparison_data = []
        self.current_weights = {}
        
    def load_data(self):
        """Load assessments and comparison data."""
        print("Loading data...")
        
        # Load assessments
        with open(self.assessments_path, 'r') as f:
            assessments = json.load(f)
        
        # Extract criteria names and build repo scores lookup
        if assessments:
            self.criteria_names = list(assessments[0]['criteria_scores'].keys())
        
        for assessment in assessments:
            repo_url = assessment['repository_url']
            scores = {}
            weights = {}
            
            for criterion, data in assessment['criteria_scores'].items():
                scores[criterion] = data['score']
                weights[criterion] = data['weight']
            
            self.repo_scores[repo_url] = {
                'scores': scores,
                'target_score': assessment['target_score']
            }
            
            if not self.current_weights:
                self.current_weights = weights
        
        # Load human comparisons
        df = pd.read_csv(self.train_csv_path)
        l1_df = df[df['parent'] == 'ethereum'].copy()
        
        for _, row in l1_df.iterrows():
            repo_a = row['repo_a']
            repo_b = row['repo_b']
            choice = row['choice']
            
            if repo_a in self.repo_scores and repo_b in self.repo_scores:
                if choice == 1.0:  # A wins
                    target = 1
                elif choice == 2.0:  # B wins
                    target = 0
                else:
                    continue  # Skip ties
                
                self.comparison_data.append({
                    'repo_a': repo_a,
                    'repo_b': repo_b,
                    'target': target,
                    'juror': row['juror'],
                    'multiplier': row['multiplier']
                })
        
        print(f"Loaded {len(self.repo_scores)} repositories")
        print(f"Created {len(self.comparison_data)} comparison examples")
        return self
        
    def weighted_sum_accuracy(self, weights: np.ndarray, comparison_indices: List[int]) -> float:
        """Calculate accuracy using weighted sum for given comparison subset."""
        correct = 0
        total = 0
        
        for idx in comparison_indices:
            comparison = self.comparison_data[idx]
            repo_a = comparison['repo_a']
            repo_b = comparison['repo_b']
            target = comparison['target']
            
            scores_a = self.repo_scores[repo_a]['scores']
            scores_b = self.repo_scores[repo_b]['scores']
            
            # Calculate weighted scores
            weighted_a = sum(scores_a[criterion] * weights[i] 
                           for i, criterion in enumerate(self.criteria_names)
                           if criterion in scores_a)
            weighted_b = sum(scores_b[criterion] * weights[i]
                           for i, criterion in enumerate(self.criteria_names)
                           if criterion in scores_b)
            
            # Predict: 1 if A > B, 0 if B > A
            predicted = 1 if weighted_a > weighted_b else 0
            
            if predicted == target:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0
    
    def optimize_weights_on_fold(self, train_indices: List[int]) -> np.ndarray:
        """Optimize weights using only training fold data."""
        def objective(weights):
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            return -self.weighted_sum_accuracy(weights, train_indices)
        
        # Optimize using differential evolution
        bounds = [(0, 1) for _ in self.criteria_names]
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50,  # Reduced for speed in CV
            popsize=10   # Reduced for speed in CV
        )
        
        optimal_weights = result.x / np.sum(result.x)  # Normalize
        return optimal_weights
    
    def cross_validate_weighted_sum(self, n_folds: int = 5) -> Dict:
        """Perform cross-validation on weighted sum optimization."""
        print(f"Running {n_folds}-fold cross-validation for weighted sum...")
        
        # Convert to arrays for sklearn compatibility
        X = np.arange(len(self.comparison_data))  # Just indices
        y = np.array([comp['target'] for comp in self.comparison_data])
        
        # Use stratified K-fold to maintain class balance
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        optimal_weights_per_fold = []
        
        for fold_idx, (train_indices, test_indices) in enumerate(skf.split(X, y)):
            print(f"  Processing fold {fold_idx + 1}/{n_folds}...")
            
            # Optimize weights on training fold
            optimal_weights = self.optimize_weights_on_fold(train_indices.tolist())
            optimal_weights_per_fold.append(optimal_weights)
            
            # Evaluate on test fold
            test_accuracy = self.weighted_sum_accuracy(optimal_weights, test_indices.tolist())
            
            # Also get training accuracy for comparison
            train_accuracy = self.weighted_sum_accuracy(optimal_weights, train_indices.tolist())
            
            fold_results.append({
                'fold': fold_idx + 1,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'optimal_weights': dict(zip(self.criteria_names, optimal_weights))
            })
            
            print(f"    Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")
        
        # Calculate overall statistics
        test_accuracies = [r['test_accuracy'] for r in fold_results]
        train_accuracies = [r['train_accuracy'] for r in fold_results]
        
        # Average optimal weights across folds
        avg_optimal_weights = np.mean(optimal_weights_per_fold, axis=0)
        avg_optimal_weights = avg_optimal_weights / np.sum(avg_optimal_weights)  # Normalize
        
        # Test averaged weights on full dataset
        all_indices = list(range(len(self.comparison_data)))
        full_dataset_accuracy = self.weighted_sum_accuracy(avg_optimal_weights, all_indices)
        
        # Current weights baseline
        current_weights_array = [self.current_weights[criterion] for criterion in self.criteria_names]
        current_accuracy = self.weighted_sum_accuracy(np.array(current_weights_array), all_indices)
        
        results = {
            'cv_test_mean': np.mean(test_accuracies),
            'cv_test_std': np.std(test_accuracies),
            'cv_train_mean': np.mean(train_accuracies),
            'cv_train_std': np.std(train_accuracies),
            'test_accuracies': test_accuracies,
            'train_accuracies': train_accuracies,
            'fold_results': fold_results,
            'avg_optimal_weights': dict(zip(self.criteria_names, avg_optimal_weights)),
            'full_dataset_accuracy': full_dataset_accuracy,
            'current_accuracy': current_accuracy,
            'improvement_over_current': full_dataset_accuracy - current_accuracy,
            'overfitting_score': np.mean(train_accuracies) - np.mean(test_accuracies)
        }
        
        return results
    
    def compare_with_previous_results(self, cv_results: Dict) -> str:
        """Compare CV results with previous non-CV optimization."""
        # Load previous results
        try:
            with open("/home/ubuntu/wellactually/results/optimized_weights.json", 'r') as f:
                previous_results = json.load(f)
            previous_optimal_accuracy = previous_results.get('improvement', 0) + cv_results['current_accuracy']
        except:
            previous_optimal_accuracy = None
        
        comparison = []
        comparison.append("="*70)
        comparison.append("WEIGHTED SUM CROSS-VALIDATION RESULTS")
        comparison.append("="*70)
        comparison.append("")
        
        comparison.append("CROSS-VALIDATION PERFORMANCE:")
        comparison.append("-" * 40)
        comparison.append(f"CV Test Accuracy (mean ± std):  {cv_results['cv_test_mean']:.3f} ± {cv_results['cv_test_std']:.3f}")
        comparison.append(f"CV Train Accuracy (mean ± std): {cv_results['cv_train_mean']:.3f} ± {cv_results['cv_train_std']:.3f}")
        comparison.append(f"Overfitting Score (train - test): {cv_results['overfitting_score']:+.3f}")
        comparison.append("")
        
        comparison.append("INDIVIDUAL FOLD RESULTS:")
        comparison.append("-" * 30)
        for result in cv_results['fold_results']:
            comparison.append(f"Fold {result['fold']}: Train={result['train_accuracy']:.3f}, Test={result['test_accuracy']:.3f}")
        comparison.append("")
        
        comparison.append("COMPARISON WITH PREVIOUS RESULTS:")
        comparison.append("-" * 40)
        comparison.append(f"Current weights accuracy:           {cv_results['current_accuracy']:.3f}")
        comparison.append(f"CV optimal weights (test):         {cv_results['cv_test_mean']:.3f} ± {cv_results['cv_test_std']:.3f}")
        comparison.append(f"Averaged weights on full dataset:  {cv_results['full_dataset_accuracy']:.3f}")
        
        if previous_optimal_accuracy:
            comparison.append(f"Previous non-CV optimization:      {previous_optimal_accuracy:.3f}")
            comparison.append("")
            comparison.append("ANALYSIS:")
            comparison.append("-" * 15)
            
            cv_improvement = cv_results['cv_test_mean'] - cv_results['current_accuracy']
            previous_improvement = previous_optimal_accuracy - cv_results['current_accuracy']
            
            comparison.append(f"CV improvement over current:       {cv_improvement:+.3f}")
            comparison.append(f"Previous improvement claim:        {previous_improvement:+.3f}")
            comparison.append(f"Difference (previous was inflated): {previous_improvement - cv_improvement:+.3f}")
            
            if previous_improvement > cv_improvement + 0.02:
                comparison.append("\n⚠️  WARNING: Previous optimization was significantly overfitted!")
                comparison.append("   The non-CV approach inflated performance estimates.")
            elif abs(previous_improvement - cv_improvement) < 0.01:
                comparison.append("\n✓ Previous results were reasonably accurate (minimal overfitting)")
            
        comparison.append("")
        
        comparison.append("AVERAGED OPTIMAL WEIGHTS:")
        comparison.append("-" * 30)
        comparison.append(f"{'Criterion':<25} {'Current':<10} {'CV Optimal':<12} {'Change':<10}")
        comparison.append("-" * 57)
        
        for criterion in self.criteria_names:
            current = self.current_weights[criterion]
            optimal = cv_results['avg_optimal_weights'][criterion]
            change = optimal - current
            comparison.append(f"{criterion:<25} {current:<10.3f} {optimal:<12.3f} {change:+.3f}")
        
        comparison.append("")
        comparison.append("RECOMMENDATIONS:")
        comparison.append("-" * 20)
        
        if cv_results['cv_test_mean'] > cv_results['current_accuracy'] + 0.03:
            comparison.append("✓ Weighted sum optimization provides meaningful improvement")
            comparison.append(f"  Expected improvement: {cv_results['cv_test_mean'] - cv_results['current_accuracy']:+.3f}")
        elif cv_results['cv_test_mean'] > cv_results['current_accuracy'] + 0.01:
            comparison.append("~ Weighted sum optimization provides modest improvement")
        else:
            comparison.append("✗ Current weights are already well-calibrated")
            
        if cv_results['overfitting_score'] > 0.05:
            comparison.append("⚠️  High overfitting detected - use regularization or more data")
        elif cv_results['overfitting_score'] > 0.02:
            comparison.append("⚠️  Moderate overfitting - results may not fully generalize")
        else:
            comparison.append("✓ Low overfitting - results should generalize well")
        
        return "\n".join(comparison)
    
    def save_results(self, cv_results: Dict, report: str):
        """Save CV results and report."""
        results_dir = "/home/ubuntu/wellactually/results"
        
        # Save detailed CV results
        with open(f"{results_dir}/weighted_sum_cv_results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            cv_results_serializable = cv_results.copy()
            cv_results_serializable['test_accuracies'] = [float(x) for x in cv_results['test_accuracies']]
            cv_results_serializable['train_accuracies'] = [float(x) for x in cv_results['train_accuracies']]
            json.dump(cv_results_serializable, f, indent=2)
        
        # Save report
        with open(f"{results_dir}/weighted_sum_cv_report.txt", 'w') as f:
            f.write(report)
        
        print(f"\nResults saved to {results_dir}/weighted_sum_cv_results.json")
        print(f"Report saved to {results_dir}/weighted_sum_cv_report.txt")


def main():
    """Main function."""
    assessments_path = "/home/ubuntu/wellactually/data/processed/criteria_assessment/detailed_assessments.json"
    train_csv_path = "/home/ubuntu/wellactually/data/raw/train.csv"
    
    # Initialize tester
    tester = WeightedSumCVTester(assessments_path, train_csv_path)
    tester.load_data()
    
    if not tester.comparison_data:
        print("ERROR: No comparison data found!")
        return
    
    # Perform cross-validation
    cv_results = tester.cross_validate_weighted_sum(n_folds=5)
    
    # Generate comparison report
    report = tester.compare_with_previous_results(cv_results)
    print(report)
    
    # Save results
    tester.save_results(cv_results, report)


if __name__ == "__main__":
    main()