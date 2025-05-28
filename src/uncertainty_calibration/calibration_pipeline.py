# src/uncertainty_calibration/calibration_pipeline.py
#!/usr/bin/env python3
"""
Production calibration pipeline for uncertainty estimation.
Orchestrates the complete uncertainty calibration workflow.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import joblib
from dataclasses import dataclass

from uncertainty_calibration.data_collection import UncertaintyDataCollector, CalibrationDataPoint
from uncertainty_calibration.feature_engineering import CalibrationFeatureEngineer
from uncertainty_calibration.lightgbm_trainer import LightGBMCalibrationTrainer
from uncertainty_calibration.evaluation import CalibrationEvaluator
from uncertainty_calibration.model_metadata import validate_model_id, get_model_metadata

logger = logging.getLogger(__name__)

@dataclass
class CalibrationPipelineConfig:
    """Configuration for calibration pipeline."""
    # Data paths
    train_data_path: str
    output_dir: str
    model_output_dir: str
    
    # Model selection
    models_to_use: List[str]
    temperatures: List[float]
    
    # Data collection
    max_samples_per_level: int = 30
    
    # Training
    validation_fraction: float = 0.2
    cross_validation: bool = True
    cv_folds: int = 5
    
    # Quality thresholds
    max_ece: float = 0.1
    min_roc_auc: float = 0.65

class UncertaintyCalibrationPipeline:
    """
    Complete pipeline for uncertainty calibration.
    
    Handles data collection, feature engineering, training, and evaluation.
    """
    
    def __init__(self, config: CalibrationPipelineConfig, llm_config: Dict):
        """Initialize pipeline with configuration."""
        self.config = config
        self.llm_config = llm_config
        
        # Initialize components
        self.data_collector = UncertaintyDataCollector(llm_config, config.models_to_use)
        self.feature_engineer = CalibrationFeatureEngineer()
        self.trainer = LightGBMCalibrationTrainer(llm_config.get('lightgbm'))
        self.evaluator = CalibrationEvaluator()
        
        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.model_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Pipeline state
        self.collected_data = None
        self.feature_df = None
        self.train_df = None
        self.val_df = None
        self.trained_model = None
        self.evaluation_results = None
        
        logger.info("Uncertainty calibration pipeline initialized")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete calibration pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        
        logger.info("Starting complete uncertainty calibration pipeline...")
        
        try:
            # Step 1: Load training data
            train_df = self.load_training_data()
            
            # Step 2: Collect LLM responses with temperature sweeps
            calibration_data = self.collect_calibration_data(train_df)
            
            # Step 3: Engineer features
            feature_df = self.engineer_features(calibration_data)
            
            # Step 4: Split data for training
            train_df, val_df = self.split_data(feature_df)
            
            # Step 5: Train calibration model
            trained_model = self.train_calibration_model(train_df, val_df)
            
            # Step 6: Evaluate calibration quality
            evaluation_results = self.evaluate_calibration(val_df)
            
            # Step 7: Save pipeline results
            self.save_pipeline_results()
            
            # Compile results
            results = {
                'status': 'success',
                'data_collection': {
                    'total_data_points': len(calibration_data),
                    'models_used': self.config.models_to_use,
                    'temperatures': self.config.temperatures
                },
                'feature_engineering': {
                    'total_features': len(self.feature_engineer.get_training_features(feature_df)),
                    'training_samples': len(train_df),
                    'validation_samples': len(val_df)
                },
                'training': {
                    'best_iteration': trained_model.training_history.get('best_iteration'),
                    'best_score': trained_model.training_history.get('best_score')
                },
                'evaluation': evaluation_results,
                'quality_check': self.check_quality_thresholds(evaluation_results)
            }
            
            logger.info("Pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def load_training_data(self) -> pd.DataFrame:
        """Load and validate training data."""
        
        logger.info(f"Loading training data from {self.config.train_data_path}")
        
        if not Path(self.config.train_data_path).exists():
            raise FileNotFoundError(f"Training data not found: {self.config.train_data_path}")
        
        df = pd.read_csv(self.config.train_data_path)
        
        # Validate required columns
        required_columns = ['repo_a', 'repo_b', 'parent', 'choice']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Loaded {len(df)} training samples")
        return df
    
    def collect_calibration_data(self, train_df: pd.DataFrame) -> List[CalibrationDataPoint]:
        """Collect LLM responses for calibration training."""
        
        logger.info("Starting LLM data collection for calibration...")
        
        # Validate models
        valid_models = [m for m in self.config.models_to_use if validate_model_id(m)]
        if not valid_models:
            raise ValueError("No valid models specified for data collection")
        
        # Update data collector with valid models
        self.data_collector.models = valid_models
        
        # Collect data
        calibration_data = self.data_collector.collect_training_data(
            train_df=train_df,
            temperatures=self.config.temperatures,
            max_samples_per_level=self.config.max_samples_per_level
        )
        
        if not calibration_data:
            raise RuntimeError("No calibration data collected")
        
        # Save collected data
        self.data_collector.save_collected_data(
            calibration_data,
            str(Path(self.config.output_dir) / "collected_calibration_data.csv")
        )
        
        self.collected_data = calibration_data
        logger.info(f"Collected {len(calibration_data)} calibration data points")
        
        return calibration_data
    
    def engineer_features(self, calibration_data: List[CalibrationDataPoint]) -> pd.DataFrame:
        """Engineer features from calibration data."""
        
        logger.info("Engineering features for calibration training...")
        
        # Convert to feature dataframe
        feature_df = self.feature_engineer.prepare_features(calibration_data)
        
        # Analyze feature distributions
        analysis = self.feature_engineer.analyze_feature_distributions(feature_df)
        logger.info(f"Feature analysis: {analysis['total_samples']} samples, "
                   f"{len(analysis['features'])} core features")
        
        # Save features
        self.feature_engineer.save_features(
            feature_df,
            str(Path(self.config.output_dir) / "calibration_features.csv")
        )
        
        self.feature_df = feature_df
        return feature_df
    
    def split_data(self, feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and validation sets."""
        
        logger.info("Splitting data for training and validation...")
        
        train_df, val_df = self.feature_engineer.create_train_val_split(
            feature_df,
            val_fraction=self.config.validation_fraction
        )
        
        self.train_df = train_df
        self.val_df = val_df
        
        return train_df, val_df
    
    def train_calibration_model(self, train_df: pd.DataFrame, 
                              val_df: pd.DataFrame) -> LightGBMCalibrationTrainer:
        """Train the calibration model."""
        
        logger.info("Training LightGBM calibration model...")
        
        # Get feature columns
        feature_columns = self.feature_engineer.get_training_features(train_df)
        
        # Train model
        self.trainer.train(train_df, val_df, feature_columns)
        
        # Cross-validation if enabled
        if self.config.cross_validation:
            logger.info("Running cross-validation...")
            cv_results = self.trainer.cross_validate(
                pd.concat([train_df, val_df]),
                feature_columns,
                cv_folds=self.config.cv_folds
            )
            logger.info(f"CV Log Loss: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        
        # Save model
        model_path = Path(self.config.model_output_dir) / "calibration_model.lgb"
        self.trainer.save_model(str(model_path))
        
        self.trained_model = self.trainer
        return self.trainer
    
    def evaluate_calibration(self, val_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate calibration quality."""
        
        logger.info("Evaluating calibration quality...")
        
        # Get predictions
        feature_columns = self.feature_engineer.get_training_features(val_df)
        y_true = val_df['is_correct'].values
        y_prob = self.trainer.predict_calibrated_confidence(val_df[feature_columns])
        
        # Overall evaluation
        overall_metrics = self.evaluator.evaluate_calibration(y_true, y_prob)
        
        # Per-model evaluation
        val_df_with_preds = val_df.copy()
        val_df_with_preds['predicted_prob'] = y_prob
        model_metrics = self.evaluator.evaluate_by_model(val_df_with_preds)
        
        # Create plots
        calibration_fig = self.evaluator.plot_calibration_curve(
            y_true, y_prob,
            title="Overall Calibration",
            save_path=str(Path(self.config.output_dir) / "calibration_curve.png")
        )
        
        model_comparison_fig = self.evaluator.plot_model_comparison(
            model_metrics,
            metric='ECE',
            title="Model Calibration Comparison",
            save_path=str(Path(self.config.output_dir) / "model_comparison.png")
        )
        
        # Feature importance
        feature_importance = self.trainer.get_feature_importance()
        feature_importance.to_csv(
            Path(self.config.output_dir) / "feature_importance.csv",
            index=False
        )
        
        evaluation_results = {
            'overall_metrics': overall_metrics,
            'model_metrics': model_metrics.to_dict('records'),
            'feature_importance': feature_importance.to_dict('records')
        }
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def check_quality_thresholds(self, evaluation_results: Dict) -> Dict[str, bool]:
        """Check if calibration meets quality thresholds."""
        
        overall_metrics = evaluation_results['overall_metrics']
        
        quality_checks = {
            'ece_threshold': overall_metrics['ECE'] <= self.config.max_ece,
            'roc_auc_threshold': overall_metrics['ROC_AUC'] >= self.config.min_roc_auc,
            'overall_quality': True
        }
        
        # Overall quality is True if all individual checks pass
        quality_checks['overall_quality'] = all([
            quality_checks['ece_threshold'],
            quality_checks['roc_auc_threshold']
        ])
        
        return quality_checks
    
    def save_pipeline_results(self):
        """Save all pipeline results."""
        
        # Save evaluation results
        if self.evaluation_results:
            results_path = Path(self.config.output_dir) / "evaluation_results.joblib"
            joblib.dump(self.evaluation_results, results_path)
        
        # Save pipeline configuration
        config_path = Path(self.config.output_dir) / "pipeline_config.joblib"
        joblib.dump(self.config, config_path)
        
        logger.info(f"Pipeline results saved to {self.config.output_dir}")
    
    def predict_calibrated_uncertainty(self, raw_uncertainties: List[float],
                                     model_names: List[str],
                                     temperatures: List[float]) -> np.ndarray:
        """
        Production inference: predict calibrated uncertainties.
        
        Args:
            raw_uncertainties: Raw uncertainty scores from models
            model_names: Model identifiers
            temperatures: Temperature values used
            
        Returns:
            Calibrated confidence scores
        """
        
        if self.trained_model is None:
            raise ValueError("Model must be trained before inference")
        
        # Prepare features for inference
        inference_data = []
        for raw_unc, model_name, temp in zip(raw_uncertainties, model_names, temperatures):
            model_metadata = get_model_metadata(model_name)
            
            inference_data.append({
                'raw_uncertainty': raw_unc,
                'model_name': model_name,
                'param_count': model_metadata['param_count'],
                'temperature': temp,
                'level': 1,  # Default level for inference
                'provider': model_metadata['provider'],
                'architecture': model_metadata['architecture']
            })
        
        inference_df = pd.DataFrame(inference_data)
        
        # Apply feature engineering (without fitting)
        inference_df = self.feature_engineer._apply_encodings(inference_df)
        
        # Get predictions
        feature_columns = self.feature_engineer.get_training_features(inference_df)
        calibrated_confidences = self.trained_model.predict_calibrated_confidence(
            inference_df[feature_columns]
        )
        
        return calibrated_confidences

def create_pipeline_from_config(config_path: str) -> UncertaintyCalibrationPipeline:
    """
    Create pipeline from configuration file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Initialized pipeline
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract pipeline configuration
    pipeline_config = CalibrationPipelineConfig(
        train_data_path="train.csv",  # Default
        output_dir=config['output_dir'],
        model_output_dir=config['model_output_dir'],
        models_to_use=list(config['models']['primary_models'].values()),
        temperatures=config['temperature_sweep']['temperatures'],
        max_samples_per_level=config['data_collection']['max_samples_per_level'],
        validation_fraction=config['lightgbm']['training']['validation_fraction'],
        cross_validation=config['evaluation']['cross_validation']['enabled'],
        cv_folds=config['evaluation']['cross_validation']['folds'],
        max_ece=config['quality_thresholds']['max_ece'],
        min_roc_auc=config['quality_thresholds']['min_roc_auc']
    )
    
    return UncertaintyCalibrationPipeline(pipeline_config, config)