# src/uncertainty_calibration/two_model_trainer.py
#!/usr/bin/env python3
"""
Two-model trainer: separate choice classifier and confidence regressor.
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__) # Fixed: __name__ instead of name

class TwoModelTrainer:
    """
    Trains two separate models: choice classifier and confidence regressor.
    """


    def __init__(self, config: Optional[Dict] = None):
        """Initialize trainer."""

        # Default parameters
        self.choice_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1,
            'random_state': 42
        }

        self.confidence_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1,
            'random_state': 42
        }

        self.training_params = {
            'num_boost_round': 1000,
            'early_stopping_rounds': 100
        }

        # Update with config
        if config and 'lightgbm' in config:
            lgb_config = config['lightgbm']
            if 'params' in lgb_config:
                self.choice_params.update(lgb_config['params'])
                self.confidence_params.update(lgb_config['params'])
            if 'training' in lgb_config:
                self.training_params.update(lgb_config['training'])

        # Models
        self.choice_model = None
        self.confidence_model = None
        self.choice_encoder = LabelEncoder()
        self.feature_names = {}
        self.training_history = {}

    def train_choice_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                           feature_columns: List[str]) -> lgb.Booster:
        """Train choice classification model."""

        logger.info("Training choice classification model...")

        # Prepare data
        X_train = train_df[feature_columns]
        y_train = train_df['choice_target']
        X_val = val_df[feature_columns]
        y_val = val_df['choice_target']

        # Encode targets
        y_train_encoded = self.choice_encoder.fit_transform(y_train)
        y_val_encoded = self.choice_encoder.transform(y_val)

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train_encoded, feature_name=feature_columns)
        val_data = lgb.Dataset(X_val, label=y_val_encoded, feature_name=feature_columns)

        # Train model
        self.choice_model = lgb.train(
            self.choice_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=self.training_params['num_boost_round'],
            callbacks=[
                lgb.early_stopping(self.training_params['early_stopping_rounds']),
                lgb.log_evaluation(0)
            ]
        )

        self.feature_names['choice'] = feature_columns

        # Evaluate
        y_pred = self.choice_model.predict(X_val, num_iteration=self.choice_model.best_iteration)
        y_pred_labels = self.choice_encoder.inverse_transform(np.argmax(y_pred, axis=1))

        accuracy = accuracy_score(y_val, y_pred_labels)
        self.training_history['choice'] = {
            'best_iteration': self.choice_model.best_iteration,
            'validation_accuracy': accuracy,
            'num_features': len(feature_columns)
        }

        logger.info(f"Choice model training completed. Accuracy: {accuracy:.4f}")
        return self.choice_model

    def train_confidence_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                               feature_columns: List[str]) -> lgb.Booster:
        """Train confidence regression model."""

        logger.info("Training confidence regression model...")

        # Prepare data
        X_train = train_df[feature_columns]
        y_train = train_df['confidence_target']
        X_val = val_df[feature_columns]
        y_val = val_df['confidence_target']

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_columns)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_columns)

        # Train model
        self.confidence_model = lgb.train(
            self.confidence_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=self.training_params['num_boost_round'],
            callbacks=[
                lgb.early_stopping(self.training_params['early_stopping_rounds']),
                lgb.log_evaluation(0)
            ]
        )

        self.feature_names['confidence'] = feature_columns

        # Evaluate
        y_pred = self.confidence_model.predict(X_val, num_iteration=self.confidence_model.best_iteration)
        mse = mean_squared_error(y_val, y_pred)
        correlation = np.corrcoef(y_val, y_pred)[0, 1]

        self.training_history['confidence'] = {
            'best_iteration': self.confidence_model.best_iteration,
            'validation_mse': mse,
            'validation_correlation': correlation,
            'num_features': len(feature_columns)
        }

        logger.info(f"Confidence model training completed. MSE: {mse:.4f}, Correlation: {correlation:.4f}")
        return self.confidence_model

    def predict_choice(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict choice probabilities and labels."""
        if self.choice_model is None:
            raise ValueError("Choice model not trained")

        choice_features = features[self.feature_names['choice']]
        probabilities = self.choice_model.predict(choice_features,
                                                  num_iteration=self.choice_model.best_iteration)
        labels = self.choice_encoder.inverse_transform(np.argmax(probabilities, axis=1))

        return probabilities, labels

    def predict_confidence(self, features: pd.DataFrame) -> np.ndarray:
        """Predict confidence scores."""
        if self.confidence_model is None:
            raise ValueError("Confidence model not trained")

        confidence_features = features[self.feature_names['confidence']]
        confidence_scores = self.confidence_model.predict(confidence_features,
                                                         num_iteration=self.confidence_model.best_iteration)

        # Clip to [0, 1] range
        return np.clip(confidence_scores, 0.0, 1.0)

    def predict_both(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Predict both choice and confidence."""
        choice_probs, choice_labels = self.predict_choice(features)
        confidence_scores = self.predict_confidence(features)

        return {
            'choice_probabilities': choice_probs,
            'choice_labels': choice_labels,
            'confidence_scores': confidence_scores
        }

    def get_feature_importance(self, model_type: str = 'both') -> Dict[str, pd.DataFrame]:
        """Get feature importance from trained models."""
        results = {}

        if model_type in ['choice', 'both'] and self.choice_model:
            choice_importance = self.choice_model.feature_importance(importance_type='gain')
            results['choice'] = pd.DataFrame({
                'feature': self.feature_names['choice'],
                'importance': choice_importance
            }).sort_values('importance', ascending=False)

        if model_type in ['confidence', 'both'] and self.confidence_model:
            conf_importance = self.confidence_model.feature_importance(importance_type='gain')
            results['confidence'] = pd.DataFrame({
                'feature': self.feature_names['confidence'],
                'importance': conf_importance
            }).sort_values('importance', ascending=False)

        return results

    def save_models(self, model_dir: str):
        """Save both trained models."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save choice model
        if self.choice_model:
            choice_path = model_dir / "choice_model.lgb"
            self.choice_model.save_model(str(choice_path))

            # Save choice encoder
            encoder_path = model_dir / "choice_encoder.joblib"
            joblib.dump(self.choice_encoder, encoder_path)

        # Save confidence model
        if self.confidence_model:
            confidence_path = model_dir / "confidence_model.lgb"
            self.confidence_model.save_model(str(confidence_path))

        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'choice_params': self.choice_params,
            'confidence_params': self.confidence_params
        }

        metadata_path = model_dir / "two_model_metadata.joblib"
        joblib.dump(metadata, metadata_path)

        logger.info(f"Two models saved to {model_dir}")

    def load_models(self, model_dir: str):
        """Load both trained models."""
        model_dir = Path(model_dir)

        # Load choice model
        choice_path = model_dir / "choice_model.lgb"
        if choice_path.exists():
            self.choice_model = lgb.Booster(model_file=str(choice_path))

            # Load choice encoder
            encoder_path = model_dir / "choice_encoder.joblib"
            if encoder_path.exists():
                self.choice_encoder = joblib.load(encoder_path)

        # Load confidence model
        confidence_path = model_dir / "confidence_model.lgb"
        if confidence_path.exists():
            self.confidence_model = lgb.Booster(model_file=str(confidence_path))

        # Load metadata
        metadata_path = model_dir / "two_model_metadata.joblib"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.feature_names = metadata.get('feature_names', {})
            self.training_history = metadata.get('training_history', {})

        logger.info(f"Two models loaded from {model_dir}")
