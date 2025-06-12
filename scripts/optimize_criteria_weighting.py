#!/usr/bin/env python3
"""
Optimize criteria weighting for repository assessments.

This script learns better combinations/weightings of criteria scores by:
1. Loading individual repository criteria scores from detailed_assessments.json
2. Using human-labeled pairwise comparisons from train.csv as ground truth
3. Training ML models to predict comparison outcomes from criteria differences
4. Finding optimal weightings that maximize agreement with human judgments
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, differential_evolution
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class CriteriaOptimizer:
    def __init__(self, assessments_path: str, train_csv_path: str):
        self.assessments_path = assessments_path
        self.train_csv_path = train_csv_path
        self.criteria_names = []
        self.repo_scores = {}
        self.comparison_data = []
        self.current_weights = {}
        
    def load_assessments(self):
        """Load repository criteria assessments."""
        print("Loading criteria assessments...")
        
        with open(self.assessments_path, 'r') as f:
            assessments = json.load(f)
        
        print(f"Loaded {len(assessments)} repository assessments")
        
        # Extract criteria names from first assessment
        if assessments:
            self.criteria_names = list(assessments[0]['criteria_scores'].keys())
            print(f"Found criteria: {self.criteria_names}")
        
        # Build repo scores lookup
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
            
            # Store current weights (assuming they're consistent)
            if not self.current_weights:
                self.current_weights = weights
        
        print(f"Processed {len(self.repo_scores)} repositories")
        return self
    
    def load_human_comparisons(self):
        """Load human-labeled comparisons for L1 (ethereum parent)."""
        print("Loading human comparisons...")
        
        df = pd.read_csv(self.train_csv_path)
        l1_df = df[df['parent'] == 'ethereum'].copy()
        
        print(f"Found {len(l1_df)} L1 comparisons")
        
        # Convert to comparison data with criteria features
        for _, row in l1_df.iterrows():
            repo_a = row['repo_a']
            repo_b = row['repo_b']
            choice = row['choice']
            
            # Check if both repos have criteria scores
            if repo_a in self.repo_scores and repo_b in self.repo_scores:
                scores_a = self.repo_scores[repo_a]['scores']
                scores_b = self.repo_scores[repo_b]['scores']
                
                # Create feature vector: only comparison-based features (no individual repo bias)
                features = {}
                for criterion in self.criteria_names:
                    if criterion in scores_a and criterion in scores_b:
                        # Difference (primary signal for comparison)
                        features[f'{criterion}_diff'] = scores_a[criterion] - scores_b[criterion]
                        
                        # Ratio (relative difference, handles scale better)
                        if scores_b[criterion] != 0:
                            features[f'{criterion}_ratio'] = scores_a[criterion] / scores_b[criterion]
                        else:
                            features[f'{criterion}_ratio'] = scores_a[criterion] + 1  # Handle division by zero
                        
                        # Sum (captures overall magnitude of both scores)
                        features[f'{criterion}_sum'] = scores_a[criterion] + scores_b[criterion]
                        
                        # Minimum (captures the baseline level)
                        features[f'{criterion}_min'] = min(scores_a[criterion], scores_b[criterion])
                        
                        # Maximum (captures the peak level)
                        features[f'{criterion}_max'] = max(scores_a[criterion], scores_b[criterion])
                
                # Convert choice to binary: 1 if A wins, 0 if B wins
                # Skip ties/equal for now to focus on clear preferences
                if choice == 1.0:  # A wins
                    target = 1
                elif choice == 2.0:  # B wins
                    target = 0
                else:
                    continue  # Skip ties
                
                self.comparison_data.append({
                    'repo_a': repo_a,
                    'repo_b': repo_b,
                    'features': features,
                    'target': target,
                    'juror': row['juror'],
                    'multiplier': row['multiplier']
                })
        
        print(f"Created {len(self.comparison_data)} comparison training examples")
        return self
    
    def validate_feature_design(self) -> Dict:
        """Validate that feature design prevents individual repository bias."""
        print("Validating feature design for bias prevention...")
        
        # Check that we only have comparison-based features
        feature_names = list(self.comparison_data[0]['features'].keys())
        
        # Categorize features
        diff_features = [f for f in feature_names if f.endswith('_diff')]
        ratio_features = [f for f in feature_names if f.endswith('_ratio')]
        sum_features = [f for f in feature_names if f.endswith('_sum')]
        min_features = [f for f in feature_names if f.endswith('_min')]
        max_features = [f for f in feature_names if f.endswith('_max')]
        individual_features = [f for f in feature_names if f.endswith('_a') or f.endswith('_b')]
        
        validation = {
            'total_features': len(feature_names),
            'diff_features': len(diff_features),
            'ratio_features': len(ratio_features),
            'sum_features': len(sum_features),
            'min_features': len(min_features),
            'max_features': len(max_features),
            'individual_features': len(individual_features),
            'bias_free': len(individual_features) == 0,
            'feature_breakdown': {
                'diff': diff_features,
                'ratio': ratio_features,
                'sum': sum_features,
                'min': min_features,
                'max': max_features,
                'individual': individual_features
            }
        }
        
        if validation['bias_free']:
            print("✓ Feature design is bias-free (no individual _a or _b features)")
        else:
            print(f"⚠️ Found {len(individual_features)} individual repository features that may cause bias")
            
        print(f"Feature breakdown: {validation['diff_features']} diff, {validation['ratio_features']} ratio, "
              f"{validation['sum_features']} sum, {validation['min_features']} min, {validation['max_features']} max")
        
        return validation
    
    def create_feature_matrix(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Create feature matrix and target vector for ML training."""
        if not self.comparison_data:
            raise ValueError("No comparison data available. Call load_human_comparisons() first.")
        
        # Validate feature design first
        validation = self.validate_feature_design()
        self._validation_results = validation  # Store for report generation
        
        # Get feature names from first example
        feature_names = list(self.comparison_data[0]['features'].keys())
        
        # Create feature matrix
        X = []
        y = []
        
        for comparison in self.comparison_data:
            feature_vector = [comparison['features'].get(fname, 0) for fname in feature_names]
            X.append(feature_vector)
            y.append(comparison['target'])
        
        return np.array(X), np.array(y), feature_names
    
    def train_ml_models(self) -> Dict:
        """Train various ML models to predict comparison outcomes and find best combination strategy."""
        print("Training comprehensive ML models...")
        
        X, y, feature_names = self.create_feature_matrix()
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features for some models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        
        # 1. Decision Tree (Interpretable non-linear model)
        print("Training Decision Tree...")
        dt = DecisionTreeClassifier(random_state=42, max_depth=8, min_samples_split=10, min_samples_leaf=5)
        dt.fit(X_train, y_train)
        dt_pred = dt.predict(X_test)
        models['Decision Tree'] = {
            'model': dt,
            'predictions': dt_pred,
            'accuracy': accuracy_score(y_test, dt_pred),
            'feature_importance': dict(zip(feature_names, dt.feature_importances_)),
            'cv_score': cross_val_score(dt, X_train, y_train, cv=5).mean(),
            'is_nonlinear': True,
            'interpretability': 'High'
        }
        
        # 2. Random Forest (Ensemble of trees)
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=12, min_samples_split=5)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        models['Random Forest'] = {
            'model': rf,
            'predictions': rf_pred,
            'accuracy': accuracy_score(y_test, rf_pred),
            'feature_importance': dict(zip(feature_names, rf.feature_importances_)),
            'cv_score': cross_val_score(rf, X_train, y_train, cv=5).mean(),
            'is_nonlinear': True,
            'interpretability': 'Medium'
        }
        
        # 3. Extra Trees (More randomized ensemble)
        print("Training Extra Trees...")
        et = ExtraTreesClassifier(n_estimators=200, random_state=42, max_depth=12)
        et.fit(X_train, y_train)
        et_pred = et.predict(X_test)
        models['Extra Trees'] = {
            'model': et,
            'predictions': et_pred,
            'accuracy': accuracy_score(y_test, et_pred),
            'feature_importance': dict(zip(feature_names, et.feature_importances_)),
            'cv_score': cross_val_score(et, X_train, y_train, cv=5).mean(),
            'is_nonlinear': True,
            'interpretability': 'Medium'
        }
        
        # 4. XGBoost (Gradient boosting with advanced features)
        print("Training XGBoost...")
        try:
            xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            models['XGBoost'] = {
                'model': xgb_model,
                'predictions': xgb_pred,
                'accuracy': accuracy_score(y_test, xgb_pred),
                'feature_importance': dict(zip(feature_names, xgb_model.feature_importances_)),
                'cv_score': cross_val_score(xgb_model, X_train, y_train, cv=5).mean(),
                'is_nonlinear': True,
                'interpretability': 'Medium'
            }
        except Exception as e:
            print(f"XGBoost training failed: {e}")
        
        # 5. Gradient Boosting (Scikit-learn implementation)
        print("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=150, random_state=42, max_depth=6, learning_rate=0.1)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        models['Gradient Boosting'] = {
            'model': gb,
            'predictions': gb_pred,
            'accuracy': accuracy_score(y_test, gb_pred),
            'feature_importance': dict(zip(feature_names, gb.feature_importances_)),
            'cv_score': cross_val_score(gb, X_train, y_train, cv=5).mean(),
            'is_nonlinear': True,
            'interpretability': 'Medium'
        }
        
        # 6. Neural Network (Multi-layer perceptron)
        print("Training Neural Network...")
        try:
            nn = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500, alpha=0.01)
            nn.fit(X_train_scaled, y_train)
            nn_pred = nn.predict(X_test_scaled)
            # Neural networks don't have simple feature importance, use permutation importance proxy
            models['Neural Network'] = {
                'model': nn,
                'predictions': nn_pred,
                'accuracy': accuracy_score(y_test, nn_pred),
                'feature_importance': {},  # We'll calculate this separately if needed
                'cv_score': cross_val_score(nn, X_train_scaled, y_train, cv=5).mean(),
                'is_nonlinear': True,
                'interpretability': 'Low',
                'scaler': scaler
            }
        except Exception as e:
            print(f"Neural Network training failed: {e}")
        
        # 7. Support Vector Machine with RBF kernel (Non-linear)
        print("Training SVM...")
        try:
            svm = SVC(kernel='rbf', random_state=42, probability=True, C=1.0, gamma='scale')
            svm.fit(X_train_scaled, y_train)
            svm_pred = svm.predict(X_test_scaled)
            models['SVM (RBF)'] = {
                'model': svm,
                'predictions': svm_pred,
                'accuracy': accuracy_score(y_test, svm_pred),
                'feature_importance': {},  # SVM doesn't have feature importance
                'cv_score': cross_val_score(svm, X_train_scaled, y_train, cv=3).mean(),  # Reduced CV for speed
                'is_nonlinear': True,
                'interpretability': 'Low',
                'scaler': scaler
            }
        except Exception as e:
            print(f"SVM training failed: {e}")
        
        # 8. AdaBoost (Adaptive boosting)
        print("Training AdaBoost...")
        ada = AdaBoostClassifier(n_estimators=100, random_state=42, learning_rate=1.0)
        ada.fit(X_train, y_train)
        ada_pred = ada.predict(X_test)
        models['AdaBoost'] = {
            'model': ada,
            'predictions': ada_pred,
            'accuracy': accuracy_score(y_test, ada_pred),
            'feature_importance': dict(zip(feature_names, ada.feature_importances_)),
            'cv_score': cross_val_score(ada, X_train, y_train, cv=5).mean(),
            'is_nonlinear': True,
            'interpretability': 'Medium'
        }
        
        # 9. K-Nearest Neighbors (Instance-based learning)
        print("Training K-Nearest Neighbors...")
        knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
        knn.fit(X_train_scaled, y_train)
        knn_pred = knn.predict(X_test_scaled)
        models['KNN'] = {
            'model': knn,
            'predictions': knn_pred,
            'accuracy': accuracy_score(y_test, knn_pred),
            'feature_importance': {},
            'cv_score': cross_val_score(knn, X_train_scaled, y_train, cv=5).mean(),
            'is_nonlinear': True,
            'interpretability': 'Low',
            'scaler': scaler
        }
        
        # 10. Logistic Regression (Linear baseline)
        print("Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        models['Logistic Regression'] = {
            'model': lr,
            'predictions': lr_pred,
            'accuracy': accuracy_score(y_test, lr_pred),
            'feature_importance': dict(zip(feature_names, np.abs(lr.coef_[0]))),
            'cv_score': cross_val_score(lr, X_train_scaled, y_train, cv=5).mean(),
            'is_nonlinear': False,
            'interpretability': 'High',
            'scaler': scaler
        }
        
        # 11. Naive Bayes (Probabilistic baseline)
        print("Training Naive Bayes...")
        nb = GaussianNB()
        nb.fit(X_train_scaled, y_train)
        nb_pred = nb.predict(X_test_scaled)
        models['Naive Bayes'] = {
            'model': nb,
            'predictions': nb_pred,
            'accuracy': accuracy_score(y_test, nb_pred),
            'feature_importance': {},
            'cv_score': cross_val_score(nb, X_train_scaled, y_train, cv=5).mean(),
            'is_nonlinear': False,
            'interpretability': 'Medium',
            'scaler': scaler
        }
        
        # Store test data for evaluation
        models['test_data'] = {'X_test': X_test, 'y_test': y_test, 'X_test_scaled': X_test_scaled}
        
        return models
    
    def optimize_weighted_sum(self) -> Dict:
        """Optimize weights for simple weighted sum approach."""
        print("Optimizing weighted sum...")
        
        def weighted_sum_accuracy(weights):
            """Calculate accuracy using weighted sum for comparison prediction."""
            correct = 0
            total = 0
            
            for comparison in self.comparison_data:
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
        
        # Objective function to minimize (negative accuracy)
        def objective(weights):
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            return -weighted_sum_accuracy(weights)
        
        # Optimize using differential evolution
        bounds = [(0, 1) for _ in self.criteria_names]
        result = differential_evolution(objective, bounds, seed=42, maxiter=100)
        
        optimal_weights = result.x / np.sum(result.x)  # Normalize
        optimal_accuracy = weighted_sum_accuracy(optimal_weights)
        
        # Also try current weights for comparison
        current_weights_array = [self.current_weights[criterion] for criterion in self.criteria_names]
        current_accuracy = weighted_sum_accuracy(current_weights_array)
        
        return {
            'optimal_weights': dict(zip(self.criteria_names, optimal_weights)),
            'optimal_accuracy': optimal_accuracy,
            'current_weights': self.current_weights,
            'current_accuracy': current_accuracy,
            'improvement': optimal_accuracy - current_accuracy
        }
    
    def analyze_feature_importance(self, models: Dict) -> pd.DataFrame:
        """Analyze and compare feature importance across models."""
        importance_data = []
        
        for model_name, model_info in models.items():
            if model_name == 'test_data':
                continue
                
            for feature, importance in model_info['feature_importance'].items():
                importance_data.append({
                    'model': model_name,
                    'feature': feature,
                    'importance': importance
                })
        
        return pd.DataFrame(importance_data)
    
    def create_criteria_table(self) -> pd.DataFrame:
        """Create table of criteria scores for all repositories."""
        data = []
        
        for repo_url, repo_data in self.repo_scores.items():
            row = {'repository_url': repo_url, 'target_score': repo_data['target_score']}
            row.update(repo_data['scores'])
            data.append(row)
        
        return pd.DataFrame(data)
    
    def extract_decision_tree_rules(self, dt_model, feature_names: List[str], max_rules: int = 10) -> List[str]:
        """Extract interpretable rules from decision tree."""
        from sklearn.tree import export_text
        
        tree_rules = export_text(dt_model, feature_names=feature_names, max_depth=4)
        
        # Parse and simplify the most important rules
        rules = []
        lines = tree_rules.split('\n')
        
        for line in lines[:max_rules]:
            if 'class:' in line and not line.strip().startswith('|'):
                # This is a leaf node with a prediction
                rules.append(line.strip())
        
        return rules[:max_rules]
    
    def generate_report(self, models: Dict, weight_optimization: Dict, importance_df: pd.DataFrame) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("="*80)
        report.append("CRITERIA COMBINATION OPTIMIZATION REPORT (BIAS-FREE)")
        report.append("="*80)
        report.append("")
        
        # Data summary
        report.append(f"Repositories analyzed: {len(self.repo_scores)}")
        report.append(f"Comparison training examples: {len(self.comparison_data)}")
        report.append(f"Criteria evaluated: {len(self.criteria_names)}")
        
        # Feature design validation
        if hasattr(self, '_validation_results'):
            validation = self._validation_results
            report.append("")
            report.append("FEATURE DESIGN VALIDATION:")
            report.append("-" * 30)
            if validation['bias_free']:
                report.append("✓ BIAS-FREE: No individual repository features (_a, _b)")
            else:
                report.append(f"⚠️ BIAS RISK: {validation['individual_features']} individual repository features found")
            
            report.append(f"Total features: {validation['total_features']}")
            report.append(f"  - Difference features: {validation['diff_features']}")
            report.append(f"  - Ratio features: {validation['ratio_features']}")
            report.append(f"  - Sum features: {validation['sum_features']}")
            report.append(f"  - Min/Max features: {validation['min_features']}/{validation['max_features']}")
        
        report.append("")
        
        # Model performance comparison
        report.append("MODEL PERFORMANCE COMPARISON:")
        report.append("-" * 70)
        report.append(f"{'Model':<20} {'CV Score':<10} {'Test Acc':<10} {'Type':<10} {'Interpretability':<15}")
        report.append("-" * 70)
        
        # Sort models by CV score
        model_performance = []
        for model_name, model_info in models.items():
            if model_name == 'test_data':
                continue
            model_performance.append((
                model_name,
                model_info['cv_score'],
                model_info['accuracy'],
                'Non-linear' if model_info.get('is_nonlinear', False) else 'Linear',
                model_info.get('interpretability', 'Medium')
            ))
        
        model_performance.sort(key=lambda x: x[1], reverse=True)
        
        for name, cv_score, test_acc, model_type, interpretability in model_performance:
            report.append(f"{name:<20} {cv_score:<10.3f} {test_acc:<10.3f} {model_type:<10} {interpretability:<15}")
        
        report.append("")
        
        # Best performing model analysis
        best_model_name = model_performance[0][0]
        best_model = models[best_model_name]
        
        report.append(f"BEST PERFORMING MODEL: {best_model_name}")
        report.append("-" * 40)
        report.append(f"Cross-validation score: {best_model['cv_score']:.3f}")
        report.append(f"Test accuracy: {best_model['accuracy']:.3f}")
        report.append(f"Model type: {'Non-linear' if best_model.get('is_nonlinear', False) else 'Linear'}")
        report.append(f"Interpretability: {best_model.get('interpretability', 'Medium')}")
        report.append("")
        
        # Feature importance for best model (if available)
        if best_model['feature_importance']:
            report.append(f"TOP FEATURES ({best_model_name}):")
            report.append("-" * 30)
            top_features = sorted(best_model['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:8]
            for feature, importance in top_features:
                report.append(f"{feature:<35} {importance:.3f}")
            report.append("")
        
        # Decision tree rules (if decision tree is among top performers)
        if 'Decision Tree' in [model[0] for model in model_performance[:3]]:
            report.append("DECISION TREE RULES (Top 5):")
            report.append("-" * 30)
            dt_model = models['Decision Tree']['model']
            X, _, feature_names = self.create_feature_matrix()
            rules = self.extract_decision_tree_rules(dt_model, feature_names, 5)
            for i, rule in enumerate(rules[:5], 1):
                report.append(f"{i}. {rule}")
            report.append("")
        
        # Weighted sum comparison
        report.append("WEIGHTED SUM OPTIMIZATION:")
        report.append("-" * 35)
        report.append(f"Current weighted sum accuracy: {weight_optimization['current_accuracy']:.3f}")
        report.append(f"Optimal weighted sum accuracy: {weight_optimization['optimal_accuracy']:.3f}")
        report.append(f"Improvement: {weight_optimization['improvement']:+.3f}")
        report.append("")
        
        report.append("OPTIMIZED WEIGHTS:")
        report.append("-" * 20)
        report.append(f"{'Criterion':<25} {'Current':<10} {'Optimal':<10} {'Change':<10}")
        report.append("-" * 55)
        
        for criterion in self.criteria_names:
            current = weight_optimization['current_weights'][criterion]
            optimal = weight_optimization['optimal_weights'][criterion]
            change = optimal - current
            report.append(f"{criterion:<25} {current:<10.3f} {optimal:<10.3f} {change:+.3f}")
        
        report.append("")
        
        # Performance comparison: ML vs Weighted Sum
        best_ml_score = model_performance[0][1]
        weighted_sum_score = weight_optimization['optimal_accuracy']
        
        report.append("APPROACH COMPARISON:")
        report.append("-" * 25)
        report.append(f"Best ML model ({best_model_name}): {best_ml_score:.3f}")
        report.append(f"Optimal weighted sum: {weighted_sum_score:.3f}")
        report.append(f"Current weighted sum: {weight_optimization['current_accuracy']:.3f}")
        report.append("")
        
        if best_ml_score > weighted_sum_score + 0.02:
            advantage = "ML model significantly better"
        elif weighted_sum_score > best_ml_score + 0.02:
            advantage = "Weighted sum significantly better"
        else:
            advantage = "Performance roughly equivalent"
        
        report.append(f"Conclusion: {advantage}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 20)
        
        if best_ml_score > weight_optimization['current_accuracy'] + 0.05:
            report.append(f"1. RECOMMENDED: Use {best_model_name} for criteria combination")
            report.append(f"   - Provides {best_ml_score - weight_optimization['current_accuracy']:+.3f} improvement over current approach")
            report.append(f"   - {'High' if best_model.get('interpretability') == 'High' else 'Moderate'} interpretability")
        else:
            report.append("1. Current weighted sum approach is reasonably effective")
            
        if weight_optimization['improvement'] > 0.02:
            report.append(f"2. If keeping weighted sum, use optimized weights for {weight_optimization['improvement']:+.3f} improvement")
        
        # Identify most important criteria changes
        weight_changes = [(criterion, abs(weight_optimization['optimal_weights'][criterion] - 
                                        weight_optimization['current_weights'][criterion]))
                         for criterion in self.criteria_names]
        weight_changes.sort(key=lambda x: x[1], reverse=True)
        
        report.append("3. Key criteria insights:")
        for criterion, change in weight_changes[:3]:
            current = weight_optimization['current_weights'][criterion]
            optimal = weight_optimization['optimal_weights'][criterion]
            direction = "increase" if optimal > current else "decrease"
            report.append(f"   - {criterion}: {direction} weight by {change:.3f}")
        
        # Non-linear insights
        nonlinear_models = [m for m in model_performance if m[3] == 'Non-linear']
        if nonlinear_models and nonlinear_models[0][1] > weight_optimization['optimal_accuracy'] + 0.03:
            report.append("4. Non-linear relationships detected - criteria interactions matter")
            report.append("   Consider using tree-based or ensemble methods for production")
        
        return "\n".join(report)
    
    def plot_weight_comparison(self, weight_optimization: Dict, save_path: str = None):
        """Plot current vs optimal weights."""
        criteria = list(self.criteria_names)
        current = [weight_optimization['current_weights'][c] for c in criteria]
        optimal = [weight_optimization['optimal_weights'][c] for c in criteria]
        
        x = np.arange(len(criteria))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, current, width, label='Current Weights', alpha=0.8)
        ax.bar(x + width/2, optimal, width, label='Optimal Weights', alpha=0.8)
        
        ax.set_xlabel('Criteria')
        ax.set_ylabel('Weight')
        ax.set_title('Current vs Optimal Criteria Weights')
        ax.set_xticks(x)
        ax.set_xticklabels(criteria, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, save_path: str = None):
        """Plot feature importance comparison across models."""
        # Focus on difference features for clearer interpretation
        diff_features = importance_df[importance_df['feature'].str.contains('_diff')]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pivot_df = diff_features.pivot(index='feature', columns='model', values='importance')
        pivot_df.plot(kind='bar', ax=ax)
        
        ax.set_xlabel('Features (Criteria Differences)')
        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance Across ML Models')
        ax.legend(title='Model')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main optimization function."""
    # Paths
    assessments_path = "/home/ubuntu/wellactually/data/processed/criteria_assessment/detailed_assessments.json"
    train_csv_path = "/home/ubuntu/wellactually/data/raw/train.csv"
    
    # Initialize optimizer
    optimizer = CriteriaOptimizer(assessments_path, train_csv_path)
    
    # Load data
    optimizer.load_assessments()
    optimizer.load_human_comparisons()
    
    if not optimizer.comparison_data:
        print("ERROR: No matching comparison data found!")
        return
    
    # Train ML models
    models = optimizer.train_ml_models()
    
    # Optimize weighted sum
    weight_optimization = optimizer.optimize_weighted_sum()
    
    # Analyze feature importance
    importance_df = optimizer.analyze_feature_importance(models)
    
    # Generate report
    report = optimizer.generate_report(models, weight_optimization, importance_df)
    print(report)
    
    # Save results
    results_dir = "/home/ubuntu/wellactually/results"
    
    # Save report
    with open(f"{results_dir}/criteria_optimization_report.txt", 'w') as f:
        f.write(report)
    
    # Save criteria table
    criteria_table = optimizer.create_criteria_table()
    criteria_table.to_csv(f"{results_dir}/criteria_scores_table.csv", index=False)
    
    # Save feature importance
    importance_df.to_csv(f"{results_dir}/feature_importance_analysis.csv", index=False)
    
    # Save optimized weights
    weight_data = {
        'current_weights': weight_optimization['current_weights'],
        'optimal_weights': weight_optimization['optimal_weights'],
        'improvement': weight_optimization['improvement']
    }
    
    with open(f"{results_dir}/optimized_weights.json", 'w') as f:
        json.dump(weight_data, f, indent=2)
    
    # Create plots
    plots_dir = "/home/ubuntu/wellactually/plots"
    optimizer.plot_weight_comparison(weight_optimization, f"{plots_dir}/weight_comparison.png")
    optimizer.plot_feature_importance(importance_df, f"{plots_dir}/feature_importance.png")
    
    print(f"\nResults saved to {results_dir}/")
    print(f"Plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()