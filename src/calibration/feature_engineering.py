# src/uncertainty_calibration/feature_engineering.py
#!/usr/bin/env python3
"""
Feature engineering for two-model uncertainty calibration approach.
Prepares features for both choice classification and confidence regression.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import LabelEncoder
import logging
from src.calibration.data_collection import CalibrationDataPoint
from src.shared.model_metadata import MODEL_PROVIDERS, MODEL_ARCHITECTURES

logger = logging.getLogger(__name__) # Fixed: __name__ instead of name

class TwoModelFeatureEngineer:
    """
    Feature engineering for two-model approach: choice classifier + confidence regressor.
    """


    def __init__(self):
        """Initialize feature engineer."""
        self.model_encoder = LabelEncoder()
        self.provider_encoder = LabelEncoder()
        self.architecture_encoder = LabelEncoder()
        self.is_fitted = False

    def prepare_choice_features(self, data_points: List[CalibrationDataPoint]) -> pd.DataFrame:
        """
        Prepare features for choice classification model.
        Target: 3-class {A, B, Equal}
        """
        rows = []

        for dp in data_points:
            row = {
                # Core features
                'raw_uncertainty': dp.raw_uncertainty,
                'model_name': dp.model_id,
                'param_count': dp.param_count,
                'temperature': dp.temperature,

                # Additional features
                'level': dp.level,
                'provider': MODEL_PROVIDERS.get(dp.model_id, 'unknown'),
                'architecture': MODEL_ARCHITECTURES.get(dp.model_id, 'unknown'),

                # Derived features
                'log_param_count': np.log10(dp.param_count + 1),
                'is_zero_temp': 1 if dp.temperature == 0.0 else 0,
                'temp_squared': dp.temperature ** 2,
                'model_size_category': self._categorize_model_size(dp.param_count),
                'uncertainty_bin': self._bin_uncertainty(dp.raw_uncertainty),

                # Choice target
                'choice_target': self._create_choice_target(dp.human_choice, dp.human_multiplier),

                # Metadata
                'question_id': dp.question_id,
                'human_multiplier': dp.human_multiplier
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Fit encoders on first call
        if not self.is_fitted:
            self._fit_encoders(df)
            self.is_fitted = True

        # Apply encodings
        df = self._apply_encodings(df)

        logger.info(f"Prepared {len(df)} choice feature rows")
        return df

    def prepare_confidence_features(self, data_points: List[CalibrationDataPoint]) -> pd.DataFrame:
        """
        Prepare features for confidence regression model.
        Target: continuous [0,1] representing P(choice is correct)
        """
        rows = []

        for dp in data_points:
            row = {
                # Core features
                'raw_uncertainty': dp.raw_uncertainty,
                'model_name': dp.model_id,
                'param_count': dp.param_count,
                'temperature': dp.temperature,

                # Additional features
                'level': dp.level,
                'provider': MODEL_PROVIDERS.get(dp.model_id, 'unknown'),
                'architecture': MODEL_ARCHITECTURES.get(dp.model_id, 'unknown'),

                # Derived features
                'log_param_count': np.log10(dp.param_count + 1),
                'is_zero_temp': 1 if dp.temperature == 0.0 else 0,
                'temp_squared': dp.temperature ** 2,
                'model_size_category': self._categorize_model_size(dp.param_count),
                'uncertainty_bin': self._bin_uncertainty(dp.raw_uncertainty),

                # Confidence-specific features
                'human_confidence_level': self._categorize_multiplier(dp.human_multiplier),
                'is_equal_case': 1 if 1.0 <= dp.human_multiplier <= 1.2 else 0,

                # Confidence target
                'confidence_target': self._create_confidence_target(dp.human_choice, dp.human_multiplier, dp.is_correct),

                # Metadata
                'question_id': dp.question_id,
                'human_multiplier': dp.human_multiplier,
                'is_correct': dp.is_correct
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Apply encodings (reuse fitted encoders)
        if self.is_fitted:
            df = self._apply_encodings(df)

        logger.info(f"Prepared {len(df)} confidence feature rows")
        return df

    def _create_choice_target(self, human_choice: float, human_multiplier: float) -> str:
        """Convert human judgment to 3-class choice target."""

        # Equal threshold for low multipliers
        if 1.0 <= human_multiplier <= 1.2:
            return "Equal"
        elif human_choice == 1.0:
            return "A"
        elif human_choice == 2.0:
            return "B"
        else:
            return "Equal"

    def _create_confidence_target(self, human_choice: float, human_multiplier: float,
                                  is_correct: bool) -> float:
        """Convert multiplier to confidence probability target."""

        # For equal cases, use model correctness
        if 1.0 <= human_multiplier <= 1.2:
            return 0.5  # Low confidence for equal cases

        # Odds-based conversion for clear preferences
        base_confidence = human_multiplier / (1 + human_multiplier)

        # Adjust based on whether model was correct
        if is_correct:
            return base_confidence
        else:
            return 1 - base_confidence

    def _categorize_model_size(self, param_count: float) -> str:
        """Categorize model by parameter count."""
        if param_count < 10:
            return 'small'
        elif param_count < 100:
            return 'medium'
        elif param_count < 500:
            return 'large'
        else:
            return 'xlarge'

    def _bin_uncertainty(self, uncertainty: float) -> str:
        """Bin uncertainty into categories."""
        if uncertainty < 0.2:
            return 'very_confident'
        elif uncertainty < 0.4:
            return 'confident'
        elif uncertainty < 0.6:
            return 'uncertain'
        elif uncertainty < 0.8:
            return 'very_uncertain'
        else:
            return 'random'

    def _categorize_multiplier(self, multiplier: float) -> str:
        """Categorize human confidence level."""
        if multiplier <= 1.2:
            return 'equal'
        elif multiplier <= 2.0:
            return 'low_confidence'
        elif multiplier <= 5.0:
            return 'medium_confidence'
        elif multiplier <= 10.0:
            return 'high_confidence'
        else:
            return 'very_high_confidence'

    def _fit_encoders(self, df: pd.DataFrame):
        """Fit label encoders on categorical features."""
        self.model_encoder.fit(df['model_name'])
        self.provider_encoder.fit(df['provider'])
        self.architecture_encoder.fit(df['architecture'])

    def _apply_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label encodings to categorical features."""
        df = df.copy()

        # Encode categorical features
        df['model_name_encoded'] = self.model_encoder.transform(df['model_name'])
        df['provider_encoded'] = self.provider_encoder.transform(df['provider'])
        df['architecture_encoded'] = self.architecture_encoder.transform(df['architecture'])

        # One-hot encode size and uncertainty categories
        df = pd.get_dummies(df, columns=['model_size_category', 'uncertainty_bin'],
                            prefix=['size', 'unc'])

        return df

    def get_choice_training_features(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns for choice classification."""
        core_features = [
            'raw_uncertainty', 'param_count', 'temperature', 'model_name_encoded',
            'level', 'provider_encoded', 'architecture_encoded',
            'log_param_count', 'is_zero_temp', 'temp_squared'
        ]

        # One-hot encoded features
        size_features = [col for col in df.columns if col.startswith('size_')]
        unc_features = [col for col in df.columns if col.startswith('unc_')]

        all_features = core_features + size_features + unc_features
        available_features = [f for f in all_features if f in df.columns]

        return available_features

    def get_confidence_training_features(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns for confidence regression."""
        core_features = [
            'raw_uncertainty', 'param_count', 'temperature', 'model_name_encoded',
            'level', 'provider_encoded', 'architecture_encoded',
            'log_param_count', 'is_zero_temp', 'temp_squared',
            'is_equal_case'  # Confidence-specific feature
        ]

        # One-hot encoded features
        size_features = [col for col in df.columns if col.startswith('size_')]
        unc_features = [col for col in df.columns if col.startswith('unc_')]

        all_features = core_features + size_features + unc_features
        available_features = [f for f in all_features if f in df.columns]

        return available_features

    def create_train_val_split(self, df: pd.DataFrame,
                               val_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create stratified train/validation split."""
        from sklearn.model_selection import train_test_split

        # Stratify by model and level
        df['strat_key'] = df['model_name'] + '_level' + df['level'].astype(str)

        train_df, val_df = train_test_split(
            df, test_size=val_fraction, random_state=42, stratify=df['strat_key']
        )

        # Remove stratification key
        train_df = train_df.drop('strat_key', axis=1)
        val_df = val_df.drop('strat_key', axis=1)

        return train_df, val_df
