# src/uncertainty_calibration/feature_engineering.py
#!/usr/bin/env python3
"""
Feature engineering for LightGBM uncertainty calibration.
Converts LLM responses to training features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import LabelEncoder
import logging

from uncertainty_calibration.data_collection import CalibrationDataPoint
from uncertainty_calibration.model_metadata import MODEL_PROVIDERS, MODEL_ARCHITECTURES

logger = logging.getLogger(__name__)

class CalibrationFeatureEngineer:
    """
    Converts calibration data points to features for LightGBM training.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.model_encoder = LabelEncoder()
        self.provider_encoder = LabelEncoder()
        self.architecture_encoder = LabelEncoder()
        self.is_fitted = False
    
    def prepare_features(self, data_points: List[CalibrationDataPoint]) -> pd.DataFrame:
        """
        Convert calibration data points to feature matrix.
        
        Args:
            data_points: List of CalibrationDataPoint objects
            
        Returns:
            DataFrame with features and target
        """
        if not data_points:
            raise ValueError("No data points provided")
        
        rows = []
        
        for dp in data_points:
            # Core features as specified in framework
            row = {
                # Primary features
                'raw_uncertainty': dp.raw_uncertainty,
                'model_name': dp.model_id,
                'param_count': dp.param_count,
                'temperature': dp.temperature,
                
                # Additional metadata features
                'level': dp.level,
                'provider': MODEL_PROVIDERS.get(dp.model_id, 'unknown'),
                'architecture': MODEL_ARCHITECTURES.get(dp.model_id, 'unknown'),
                
                # Derived features
                'log_param_count': np.log10(dp.param_count + 1),  # Log-scaled parameters
                'is_zero_temp': 1 if dp.temperature == 0.0 else 0,  # Zero temperature flag
                'temp_squared': dp.temperature ** 2,  # Non-linear temperature effect
                
                # Model size categories
                'model_size_category': self._categorize_model_size(dp.param_count),
                
                # Uncertainty bins
                'uncertainty_bin': self._bin_uncertainty(dp.raw_uncertainty),
                
                # Target variable
                'is_correct': int(dp.is_correct),
                
                # Metadata for analysis
                'question_id': dp.question_id,
                'model_prediction': dp.model_prediction,
                'correct_answer': dp.correct_answer
            }
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Fit encoders on first call
        if not self.is_fitted:
            self._fit_encoders(df)
            self.is_fitted = True
        
        # Apply encodings
        df = self._apply_encodings(df)
        
        logger.info(f"Prepared {len(df)} feature rows with {len(df.columns)} columns")
        return df
    
    def _categorize_model_size(self, param_count: float) -> str:
        """Categorize model by parameter count."""
        if param_count < 10:
            return 'small'  # < 10B parameters
        elif param_count < 100:
            return 'medium'  # 10B - 100B parameters
        elif param_count < 500:
            return 'large'  # 100B - 500B parameters
        else:
            return 'xlarge'  # > 500B parameters
    
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
        df = pd.get_dummies(df, columns=['model_size_category', 'uncertainty_bin'], prefix=['size', 'unc'])
        
        return df
    
    def get_training_features(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns for training."""
        # Core features from the framework
        core_features = [
            'raw_uncertainty',
            'param_count', 
            'temperature',
            'model_name_encoded'  # Categorical model name
        ]
        
        # Additional features that improve calibration
        additional_features = [
            'level',
            'provider_encoded',
            'architecture_encoded', 
            'log_param_count',
            'is_zero_temp',
            'temp_squared'
        ]
        
        # One-hot encoded features
        size_features = [col for col in df.columns if col.startswith('size_')]
        unc_features = [col for col in df.columns if col.startswith('unc_')]
        
        all_features = core_features + additional_features + size_features + unc_features
        
        # Filter to only include features that exist in the dataframe
        available_features = [f for f in all_features if f in df.columns]
        
        logger.info(f"Selected {len(available_features)} features for training")
        return available_features
    
    def create_train_val_split(self, df: pd.DataFrame, 
                             val_fraction: float = 0.2, 
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation split stratified by model and level.
        
        Args:
            df: Feature dataframe
            val_fraction: Fraction for validation
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df)
        """
        from sklearn.model_selection import train_test_split
        
        # Create stratification key combining model and level
        df['strat_key'] = df['model_name'] + '_level' + df['level'].astype(str)
        
        # Split maintaining distribution across models and levels
        train_df, val_df = train_test_split(
            df,
            test_size=val_fraction,
            random_state=random_state,
            stratify=df['strat_key']
        )
        
        # Remove stratification key
        train_df = train_df.drop('strat_key', axis=1)
        val_df = val_df.drop('strat_key', axis=1)
        
        logger.info(f"Created train split: {len(train_df)} samples")
        logger.info(f"Created validation split: {len(val_df)} samples")
        
        return train_df, val_df
    
    def analyze_feature_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature distributions for data quality checks."""
        
        analysis = {
            'total_samples': len(df),
            'features': {}
        }
        
        # Analyze key features
        key_features = ['raw_uncertainty', 'param_count', 'temperature', 'is_correct']
        
        for feature in key_features:
            if feature in df.columns:
                analysis['features'][feature] = {
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                    'null_count': df[feature].isnull().sum()
                }
        
        # Model distribution
        analysis['model_distribution'] = df['model_name'].value_counts().to_dict()
        
        # Level distribution
        analysis['level_distribution'] = df['level'].value_counts().to_dict()
        
        # Temperature distribution
        analysis['temperature_distribution'] = df['temperature'].value_counts().to_dict()
        
        # Correctness rate by model
        correctness_by_model = df.groupby('model_name')['is_correct'].mean().to_dict()
        analysis['correctness_by_model'] = correctness_by_model
        
        logger.info("Feature distribution analysis completed")
        return analysis
    
    def save_features(self, df: pd.DataFrame, output_path: str):
        """Save feature dataframe to CSV."""
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} feature rows to {output_path}")

def prepare_calibration_features(data_points: List[CalibrationDataPoint]) -> pd.DataFrame:
    """
    Convenience function to prepare features from calibration data points.
    
    Args:
        data_points: List of CalibrationDataPoint objects
        
    Returns:
        Feature dataframe ready for LightGBM training
    """
    engineer = CalibrationFeatureEngineer()
    df = engineer.prepare_features(data_points)
    return df