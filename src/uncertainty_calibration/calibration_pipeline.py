# src/uncertainty_calibration/calibration_pipeline.py
"""
Calibration pipeline for applying trained models to new responses.
Provides the deployment interface for uncertainty calibration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

from uncertainty_calibration.lightgbm_trainer import LightGBMCalibrationTrainer
from uncertainty_calibration.model_metadata import get_model_params

class UncertaintyCalibrationPipeline:
    """Production pipeline for uncertainty calibration."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.trainer = LightGBMCalibrationTrainer()
        
        if model_path and Path(model_path).exists():
            self.trainer.load_model(model_path)
            self.is_trained = True
        else:
            self.is_trained = False
    
    def calibrate_single_response(self, 
                                response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calibrate a single model response.
        
        Args:
            response: Response dict with keys:
                - model_id: str
                - raw_uncertainty: float
                - temperature: float
                - prediction: str
                - content: str (optional)
                
        Returns:
            Calibrated response with additional fields:
                - calibrated_confidence: float
                - calibrated_uncertainty: float
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Load a model first.")
        
        # Extract features
        model_id = response['model_id']
        raw_uncertainty = response['raw_uncertainty']
        temperature = response.get('temperature', 0.0)
        param_count = get_model_params(model_id)
        
        # Get calibrated confidence
        calibrated_confidence = self.trainer.predict_calibrated_confidence(
            raw_uncertainty=raw_uncertainty,
            model_name=model_id,
            param_count=param_count,
            temperature=temperature
        )
        
        # Create calibrated response
        calibrated_response = response.copy()
        calibrated_response.update({
            'calibrated_confidence': float(calibrated_confidence),
            'calibrated_uncertainty': float(1 - calibrated_confidence),
            'param_count': param_count
        })
        
        return calibrated_response
    
    def calibrate_model_responses(self, 
                                model_responses: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Calibrate responses from multiple models.
        
        Args:
            model_responses: Dict mapping model_id to response dict
            
        Returns:
            Dict mapping model_id to calibrated response dict
        """
        calibrated_results = {}
        
        for model_id, response in model_responses.items():
            # Ensure model_id is in response
            response['model_id'] = model_id
            
            try:
                calibrated_response = self.calibrate_single_response(response)
                calibrated_results[model_id] = calibrated_response
            except Exception as e:
                # Keep original response if calibration fails
                calibrated_results[model_id] = response.copy()
                calibrated_results[model_id]['calibration_error'] = str(e)
        
        return calibrated_results
    
    def aggregate_calibrated_uncertainties(self, 
                                         calibrated_responses: Dict[str, Dict],
                                         method: str = 'mean') -> Dict[str, float]:
        """
        Aggregate calibrated uncertainties across models.
        
        Args:
            calibrated_responses: Dict of calibrated responses
            method: Aggregation method ('mean', 'weighted_mean', 'median')
            
        Returns:
            Dict with aggregated uncertainty metrics
        """
        confidences = []
        uncertainties = []
        weights = []
        
        for model_id, response in calibrated_responses.items():
            if 'calibrated_confidence' in response:
                confidences.append(response['calibrated_confidence'])
                uncertainties.append(response['calibrated_uncertainty'])
                
                # Weight by model size (larger models get higher weight)
                param_count = response.get('param_count', 7.0)
                weight = np.log(param_count + 1)
                weights.append(weight)
        
        if not confidences:
            return {'error': 'No valid calibrated responses'}
        
        confidences = np.array(confidences)
        uncertainties = np.array(uncertainties)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        if method == 'mean':
            agg_confidence = np.mean(confidences)
            agg_uncertainty = np.mean(uncertainties)
        elif method == 'weighted_mean':
            agg_confidence = np.average(confidences, weights=weights)
            agg_uncertainty = np.average(uncertainties, weights=weights)
        elif method == 'median':
            agg_confidence = np.median(confidences)
            agg_uncertainty = np.median(uncertainties)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Calculate ensemble metrics
        confidence_std = np.std(confidences)
        confidence_range = np.max(confidences) - np.min(confidences)
        
        return {
            'aggregated_confidence': float(agg_confidence),
            'aggregated_uncertainty': float(agg_uncertainty),
            'confidence_std': float(confidence_std),
            'confidence_range': float(confidence_range),
            'num_models': len(confidences),
            'individual_confidences': confidences.tolist(),
            'model_weights': weights.tolist() if method == 'weighted_mean' else None
        }
    
    def batch_calibrate(self, responses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calibrate a batch of responses in DataFrame format.
        
        Args:
            responses_df: DataFrame with columns:
                - model_id
                - raw_uncertainty
                - temperature (optional)
                - prediction (optional)
                
        Returns:
            DataFrame with added calibration columns
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Load a model first.")
        
        calibrated_df = responses_df.copy()
        
        # Add parameter counts
        calibrated_df['param_count'] = calibrated_df['model_id'].apply(get_model_params)
        
        # Ensure temperature column exists
        if 'temperature' not in calibrated_df.columns:
            calibrated_df['temperature'] = 0.0
        
        # Apply calibration to each row
        calibrated_confidences = []
        
        for _, row in calibrated_df.iterrows():
            try:
                confidence = self.trainer.predict_calibrated_confidence(
                    raw_uncertainty=row['raw_uncertainty'],
                    model_name=row['model_id'],
                    param_count=row['param_count'],
                    temperature=row['temperature']
                )
                calibrated_confidences.append(confidence)
            except Exception as e:
                # Use raw uncertainty as fallback
                calibrated_confidences.append(1 - row['raw_uncertainty'])
        
        calibrated_df['calibrated_confidence'] = calibrated_confidences
        calibrated_df['calibrated_uncertainty'] = 1 - calibrated_df['calibrated_confidence']
        
        return calibrated_df
    
    def get_model_reliability_scores(self, 
                                   responses_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate reliability scores for each model based on calibration.
        
        Args:
            responses_df: DataFrame with model responses and ground truth
            
        Returns:
            Dict mapping model_id to reliability score
        """
        if 'is_correct' not in responses_df.columns:
            raise ValueError("Need 'is_correct' column for reliability calculation")
        
        calibrated_df = self.batch_calibrate(responses_df)
        
        reliability_scores = {}
        
        for model_id in calibrated_df['model_id'].unique():
            model_data = calibrated_df[calibrated_df['model_id'] == model_id]
            
            if len(model_data) < 10:  # Need minimum samples
                continue
            
            # Calculate calibration error for this model
            y_true = model_data['is_correct']
            y_prob = model_data['calibrated_confidence']
            
            # Brier score (lower is better)
            brier_score = np.mean((y_prob - y_true) ** 2)
            
            # Convert to reliability score (higher is better)
            reliability_score = 1 - brier_score
            reliability_scores[model_id] = float(reliability_score)
        
        return reliability_scores


def create_pipeline_from_responses(training_responses: pd.DataFrame,
                                 model_save_path: str) -> UncertaintyCalibrationPipeline:
    """
    Create and train a calibration pipeline from training responses.
    
    Args:
        training_responses: DataFrame with training data
        model_save_path: Path to save trained model
        
    Returns:
        Trained calibration pipeline
    """
    from uncertainty_calibration.feature_engineering import create_train_test_split
    from uncertainty_calibration.lightgbm_trainer import train_calibration_model
    
    # Split data
    train_df, val_df = create_train_test_split(training_responses, test_size=0.2)
    
    # Train model
    trainer = train_calibration_model(train_df, val_df, model_save_path)
    
    # Create pipeline
    pipeline = UncertaintyCalibrationPipeline(model_save_path)
    
    return pipeline