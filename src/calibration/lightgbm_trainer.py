# src/uncertainty_calibration/lightgbm_trainer.py
#!/usr/bin/env python3
"""
LightGBM trainer for uncertainty calibration.
Core training logic for the calibration model.
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class LightGBMCalibrationTrainer:
    """
    Trains LightGBM model for uncertainty calibration.
    Maps (raw_uncertainty, model_metadata) -> P(answer_is_correct)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize trainer with configuration."""
        
        # Default LightGBM parameters optimized for calibration
        self.default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 10,
            'min_gain_to_split': 0.0,
            'lambda_l1': 0.0,
            'lambda_l2': 0.0,
            'verbose': -1,
            'random_state': 42,
            'force_row_wise': True  # Suppress warnings
        }
        
        # Training parameters
        self.training_params = {
            'num_boost_round': 1000,
            'early_stopping_rounds': 100,
            'verbose_eval': False
        }
        
        # Update with user config if provided
        if config:
            self.default_params.update(config.get('lgb_params', {}))
            self.training_params.update(config.get('training_params', {}))
        
        self.model = None
        self.feature_names = None
        self.training_history = {}
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
              feature_columns: List[str], target_column: str = 'is_correct') -> lgb.Booster:
        """
        Train LightGBM calibration model.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            feature_columns: List of feature column names
            target_column: Target column name
            
        Returns:
            Trained LightGBM model
        """
        
        logger.info("Starting LightGBM calibration training...")
        
        # Prepare data
        X_train = train_df[feature_columns]
        y_train = train_df[target_column].astype(int)
        X_val = val_df[feature_columns]
        y_val = val_df[target_column].astype(int)
        
        self.feature_names = feature_columns
        
        # Identify categorical features
        categorical_features = self._identify_categorical_features(X_train)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            categorical_feature=categorical_features,
            feature_name=feature_columns
        )
        
        val_data = lgb.Dataset(
            X_val,
            label=y_val, 
            categorical_feature=categorical_features,
            feature_name=feature_columns,
            reference=train_data
        )
        
        # Training callbacks
        callbacks = [
            lgb.early_stopping(self.training_params['early_stopping_rounds']),
            lgb.log_evaluation(0)  # Suppress verbose output
        ]
        
        # Train model
        self.model = lgb.train(
            self.default_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=self.training_params['num_boost_round'],
            callbacks=callbacks
        )
        
        # Store training history
        self.training_history = {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
            'feature_importance': self.get_feature_importance(),
            'training_params': self.default_params.copy(),
            'num_features': len(feature_columns),
            'train_samples': len(train_df),
            'val_samples': len(val_df)
        }
        
        logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")
        logger.info(f"Best validation score: {self.model.best_score}")
        
        return self.model
    
    def predict_calibrated_confidence(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated confidence scores.
        
        Args:
            features: Feature dataframe
            
        Returns:
            Array of confidence scores (probability of being correct)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        if self.feature_names:
            features = features[self.feature_names]
        
        # Get probability predictions
        probabilities = self.model.predict(
            features, 
            num_iteration=self.model.best_iteration
        )
        
        return probabilities
    
    def evaluate_on_validation(self, val_df: pd.DataFrame,
                             feature_columns: List[str],
                             target_column: str = 'is_correct') -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            val_df: Validation dataframe
            feature_columns: Feature columns
            target_column: Target column
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        X_val = val_df[feature_columns]
        y_val = val_df[target_column].astype(int)
        
        # Get predictions
        y_pred_proba = self.predict_calibrated_confidence(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'log_loss': log_loss(y_val, y_pred_proba),
            'brier_score': brier_score_loss(y_val, y_pred_proba),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'accuracy': (y_pred == y_val).mean(),
            'calibration_error': self._calculate_ece(y_val, y_pred_proba)
        }
        
        logger.info("Validation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def cross_validate(self, df: pd.DataFrame, feature_columns: List[str],
                      target_column: str = 'is_correct', cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            df: Full dataframe
            feature_columns: Feature columns
            target_column: Target column
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        X = df[feature_columns]
        y = df[target_column].astype(int)
        
        # Stratified CV to maintain class balance
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            # Split data
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model for this fold
            fold_model = self._train_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            
            # Evaluate
            y_pred_proba = fold_model.predict(X_val_fold, num_iteration=fold_model.best_iteration)
            score = log_loss(y_val_fold, y_pred_proba)
            cv_scores.append(score)
        
        cv_results = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'cv_folds': cv_folds
        }
        
        logger.info(f"Cross-validation completed:")
        logger.info(f"  Mean log loss: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        
        return cv_results
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Model must be trained to get feature importance")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names or [f'feature_{i}' for i in range(len(importance))],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_path: str):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        self.model.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'model_params': self.default_params
        }
        
        metadata_path = model_path.with_suffix('.metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path: str):
        """Load trained model from disk."""
        model_path = Path(model_path)
        
        # Load LightGBM model
        self.model = lgb.Booster(model_file=str(model_path))
        
        # Load metadata
        metadata_path = model_path.with_suffix('.metadata.joblib')
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.feature_names = metadata.get('feature_names')
            self.training_history = metadata.get('training_history', {})
        
        logger.info(f"Model loaded from {model_path}")
    
    def _identify_categorical_features(self, X: pd.DataFrame) -> List[str]:
        """Identify categorical features for LightGBM."""
        categorical_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_features.append(col)
            elif 'encoded' in col:  # Label encoded features
                categorical_features.append(col)
            elif col.startswith(('size_', 'unc_')):  # One-hot encoded (but handle as categorical)
                continue  # One-hot encoded features are handled automatically
        
        return categorical_features
    
    def _train_fold(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series) -> lgb.Booster:
        """Train model for a single CV fold."""
        
        categorical_features = self._identify_categorical_features(X_train)
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_features)
        
        model = lgb.train(
            self.default_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=self.training_params['num_boost_round'],
            callbacks=[lgb.early_stopping(self.training_params['early_stopping_rounds'])]
        )
        
        return model
    
    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece

def train_calibration_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                          feature_columns: List[str], config: Optional[Dict] = None) -> LightGBMCalibrationTrainer:
    """
    Convenience function to train calibration model.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe  
        feature_columns: Feature column names
        config: Optional configuration
        
    Returns:
        Trained LightGBMCalibrationTrainer
    """
    trainer = LightGBMCalibrationTrainer(config)
    trainer.train(train_df, val_df, feature_columns)
    return trainer