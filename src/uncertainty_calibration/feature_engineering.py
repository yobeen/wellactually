# src/uncertainty_calibration/feature_engineering.py
"""
Feature engineering for uncertainty calibration.
Converts raw model responses to training features for LightGBM.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from uncertainty_calibration.model_metadata import get_model_params, get_model_category

class CalibrationFeatureEngineer:
    """Converts model responses to features for calibration training."""
    
    def __init__(self):
        self.feature_columns = [
            'raw_uncertainty',
            'model_name', 
            'param_count',
            'temperature',
            'model_category',
            'uncertainty_normalized',
            'temp_uncertainty_interaction'
        ]
        self.target_column = 'is_correct'
    
    def prepare_features(self, responses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw model responses to training features.
        
        Args:
            responses_df: DataFrame with collected responses
            
        Returns:
            DataFrame with engineered features ready for training
        """
        df = responses_df.copy()
        
        # Core features
        df['model_name'] = df['model_id']
        df['param_count'] = df['model_id'].apply(get_model_params)
        df['model_category'] = df['model_id'].apply(get_model_category)
        
        # Normalize uncertainty within each model
        df['uncertainty_normalized'] = df.groupby('model_id')['raw_uncertainty'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        
        # Interaction features
        df['temp_uncertainty_interaction'] = df['temperature'] * df['raw_uncertainty']
        df['param_temp_interaction'] = df['param_count'] * df['temperature']
        df['log_param_count'] = np.log(df['param_count'] + 1)
        
        # Temperature bins
        df['temp_bin'] = pd.cut(df['temperature'], bins=[0, 0.1, 0.3, 0.6, 1.0], 
                               labels=['very_low', 'low', 'medium', 'high'])
        
        # Model size bins
        df['param_bin'] = pd.cut(df['param_count'], bins=[0, 10, 50, 200, float('inf')],
                                labels=['small', 'medium', 'large', 'xlarge'])
        
        # Uncertainty percentiles within model-temperature groups
        df['uncertainty_percentile'] = df.groupby(['model_id', 'temperature'])['raw_uncertainty'].rank(pct=True)
        
        # Ensemble disagreement features (if multiple models for same question)
        if 'question_id' in df.columns:
            question_stats = df.groupby('question_id').agg({
                'raw_uncertainty': ['mean', 'std', 'min', 'max'],
                'is_correct': 'mean'
            }).round(4)
            question_stats.columns = ['_'.join(col).strip() for col in question_stats.columns]
            question_stats = question_stats.add_prefix('question_')
            
            df = df.merge(question_stats, left_on='question_id', right_index=True, how='left')
            
            # Deviation from ensemble
            df['uncertainty_vs_ensemble'] = df['raw_uncertainty'] - df['question_raw_uncertainty_mean']
        
        return df
    
    def get_training_data(self, df: pd.DataFrame) -> tuple:
        """
        Extract features and target for training.
        
        Returns:
            (X, y) tuple for training
        """
        # Core features for LightGBM
        feature_cols = [
            'raw_uncertainty',
            'model_name',
            'param_count', 
            'temperature',
            'model_category',
            'uncertainty_normalized',
            'temp_uncertainty_interaction',
            'param_temp_interaction',
            'log_param_count',
            'temp_bin',
            'param_bin',
            'uncertainty_percentile'
        ]
        
        # Add ensemble features if available
        if 'question_raw_uncertainty_mean' in df.columns:
            feature_cols.extend([
                'question_raw_uncertainty_mean',
                'question_raw_uncertainty_std', 
                'question_is_correct_mean',
                'uncertainty_vs_ensemble'
            ])
        
        # Filter for available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols].copy()
        y = df[self.target_column].astype(int)
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, y
    
    def get_categorical_features(self) -> List[str]:
        """Get list of categorical feature names for LightGBM."""
        return ['model_name', 'model_category', 'temp_bin', 'param_bin']


def create_train_test_split(df: pd.DataFrame, test_size: float = 0.2, 
                           random_state: int = 42) -> tuple:
    """
    Create train/test split ensuring question-level separation.
    
    Args:
        df: Feature dataframe
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        (train_df, test_df) tuple
    """
    if 'question_id' in df.columns:
        # Split by questions to avoid data leakage
        unique_questions = df['question_id'].unique()
        np.random.seed(random_state)
        test_questions = np.random.choice(
            unique_questions, 
            size=int(len(unique_questions) * test_size),
            replace=False
        )
        
        train_df = df[~df['question_id'].isin(test_questions)]
        test_df = df[df['question_id'].isin(test_questions)]
    else:
        # Random split if no question IDs
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    return train_df, test_df


def analyze_feature_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze feature distributions for debugging."""
    analysis = {}
    
    # Basic stats
    analysis['total_examples'] = len(df)
    analysis['models'] = df['model_id'].unique().tolist()
    analysis['temperatures'] = sorted(df['temperature'].unique())
    analysis['accuracy_by_model'] = df.groupby('model_id')['is_correct'].mean().to_dict()
    analysis['accuracy_by_temp'] = df.groupby('temperature')['is_correct'].mean().to_dict()
    
    # Uncertainty distributions
    analysis['uncertainty_stats'] = df['raw_uncertainty'].describe().to_dict()
    analysis['uncertainty_by_model'] = df.groupby('model_id')['raw_uncertainty'].mean().to_dict()
    
    # Class balance
    analysis['class_balance'] = df['is_correct'].value_counts(normalize=True).to_dict()
    
    return analysis