#!/usr/bin/env python3
"""
Training script for LightGBM uncertainty calibration model.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import argparse
import yaml
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from uncertainty_calibration.data_collection import collect_calibration_dataset, CalibrationDataCollector
from uncertainty_calibration.feature_engineering import CalibrationFeatureEngineer, create_train_test_split
from uncertainty_calibration.lightgbm_trainer import LightGBMCalibrationTrainer
from uncertainty_calibration.calibration_pipeline import UncertaintyCalibrationPipeline
from uncertainty_calibration.evaluation import CalibrationEvaluator, evaluate_pipeline_calibration

def load_config(config_path: str = "configs/config.yaml") -> DictConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return OmegaConf.create(config_dict)

def create_sample_config():
    """Create a minimal config for testing."""
    config = {
        'api': {
            'openrouter': {
                'base_url': 'https://openrouter.ai/api/v1/chat/completions',
                'api_key_env': 'OPENROUTER_API_KEY'
            },
            'rate_limiting': {
                'requests_per_minute': 60,
                'max_retries': 3,
                'backoff_factor': 2,
                'timeout_seconds': 30
            }
        },
        'models': {
            'primary_models': {
                'gpt_4o': 'openai/gpt-4o',
                'llama_3_1_405b': 'meta-llama/llama-3.1-405b-instruct',
                'deepseek_v3': 'deepseek/deepseek-chat'
            }
        },
        'prompts': {
            'level_1': {
                'max_tokens': 10,
                'temperature': 0.7,
                'choice_options': ['A', 'B', 'Equal']
            }
        },
        'cost_management': {
            'track_costs': True,
            'total_budget_usd': 100.0,
            'cost_per_1k_tokens': 0.01
        }
    }
    return OmegaConf.create(config)

def collect_training_data(config: DictConfig, output_path: str = "data/calibration_training.csv"):
    """Collect training data using temperature sweep."""
    print("Collecting training data...")
    
    # This would normally collect real data, but for demo we'll create synthetic data
    collector = CalibrationDataCollector(config)
    
    # Create synthetic training data for demo
    np.random.seed(42)
    n_samples = 1000
    
    models = ['openai/gpt-4o', 'meta-llama/llama-3.1-405b-instruct', 'deepseek/deepseek-chat']
    temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    training_data = []
    
    for i in range(n_samples):
        model_id = np.random.choice(models)
        temperature = np.random.choice(temperatures)
        
        # Simulate uncertainty that correlates with correctness
        base_uncertainty = np.random.exponential(1.0)
        
        # Larger models tend to be more calibrated
        model_factor = {'openai/gpt-4o': 0.8, 'meta-llama/llama-3.1-405b-instruct': 0.9, 'deepseek/deepseek-chat': 1.2}[model_id]
        
        # Higher temperature increases uncertainty
        temp_factor = 1 + temperature * 0.5
        
        raw_uncertainty = base_uncertainty * model_factor * temp_factor
        
        # Probability correct inversely related to uncertainty (with noise)
        prob_correct = 1 / (1 + np.exp(raw_uncertainty - 2)) + np.random.normal(0, 0.1)
        prob_correct = np.clip(prob_correct, 0.1, 0.9)
        
        is_correct = np.random.random() < prob_correct
        
        training_data.append({
            'question_id': i % 100,  # 100 unique questions
            'model_id': model_id,
            'temperature': temperature,
            'raw_uncertainty': raw_uncertainty,
            'prediction': 'A' if np.random.random() > 0.5 else 'B',
            'correct_answer': 'A',
            'is_correct': is_correct
        })
    
    df = pd.DataFrame(training_data)
    df.to_csv(output_path, index=False)
    
    print(f"Created synthetic training data: {len(df)} examples")
    print(f"Saved to {output_path}")
    
    return df

def train_calibration_model(training_df: pd.DataFrame, 
                          model_save_path: str = "models/calibration_model.pkl"):
    """Train the LightGBM calibration model."""
    print("Training calibration model...")
    
    # Split data
    train_df, test_df = create_train_test_split(training_df, test_size=0.2)
    val_df, test_df = create_train_test_split(test_df, test_size=0.5)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Train model
    trainer = LightGBMCalibrationTrainer()
    model = trainer.train(train_df, val_df)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_df)
    print("\nTest Set Evaluation:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Feature importance
    importance = trainer.get_feature_importance()
    print("\nFeature Importance:")
    print(importance.head(10))
    
    # Cross-validation
    cv_results = trainer.cross_validate(training_df)
    print(f"\nCross-validation AUC: {cv_results['cv_auc_mean']:.4f} ± {cv_results['cv_auc_std']:.4f}")
    
    # Save model
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_save_path)
    
    return trainer, test_metrics

def evaluate_calibration_quality(model_path: str, test_df: pd.DataFrame):
    """Evaluate calibration quality of trained model."""
    print("Evaluating calibration quality...")
    
    # Load pipeline
    pipeline = UncertaintyCalibrationPipeline(model_path)
    
    # Comprehensive evaluation
    eval_results = evaluate_pipeline_calibration(pipeline, test_df)
    
    print("\nOverall Calibration Metrics:")
    for metric, value in eval_results['overall_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nBy-Model Metrics:")
    for model_id, metrics in eval_results['by_model_metrics'].items():
        print(f"\n{model_id}:")
        print(f"  ECE: {metrics['ECE']:.4f}")
        print(f"  Brier Score: {metrics['Brier_Score']:.4f}")
        print(f"  AUC: {metrics['AUC']:.4f}")
    
    # Plot reliability diagram
    evaluator = CalibrationEvaluator()
    calibrated_df = eval_results['calibrated_data']
    
    fig = evaluator.plot_reliability_diagram(
        calibrated_df['is_correct'].values,
        calibrated_df['calibrated_confidence'].values,
        title="LightGBM Calibration",
        save_path="plots/reliability_diagram.png"
    )
    
    # Plot by-model comparison
    fig = evaluator.plot_calibration_by_group(
        calibrated_df, 'model_id',
        title="Calibration by Model",
        save_path="plots/calibration_by_model.png"
    )
    
    print("\nPlots saved to plots/ directory")
    
    return eval_results

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM uncertainty calibration model")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--collect-data", action="store_true", help="Collect new training data")
    parser.add_argument("--train", action="store_true", help="Train calibration model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model")
    parser.add_argument("--data-path", default="data/calibration_training.csv", help="Training data path")
    parser.add_argument("--model-path", default="models/calibration_model.pkl", help="Model save path")
    
    args = parser.parse_args()
    
    # Create output directories
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    
    # Load or create config
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        print(f"Config file {args.config} not found, using sample config")
        config = create_sample_config()
    
    training_df = None
    
    # Collect data
    if args.collect_data:
        training_df = collect_training_data(config, args.data_path)
    
    # Load existing data if not collected
    if training_df is None and Path(args.data_path).exists():
        training_df = pd.read_csv(args.data_path)
        print(f"Loaded existing training data: {len(training_df)} examples")
    
    # Train model
    if args.train and training_df is not None:
        trainer, test_metrics = train_calibration_model(training_df, args.model_path)
    
    # Evaluate model
    if args.evaluate:
        if training_df is None:
            if Path(args.data_path).exists():
                training_df = pd.read_csv(args.data_path)
            else:
                print("No training data available for evaluation")
                return
        
        if Path(args.model_path).exists():
            # Use a portion of data for evaluation
            _, test_df = create_train_test_split(training_df, test_size=0.3)
            eval_results = evaluate_calibration_quality(args.model_path, test_df)
        else:
            print(f"Model file {args.model_path} not found")
    
    # Run all steps if no specific flags
    if not any([args.collect_data, args.train, args.evaluate]):
        print("Running complete pipeline...")
        
        # Collect data
        training_df = collect_training_data(config, args.data_path)
        
        # Train model
        trainer, test_metrics = train_calibration_model(training_df, args.model_path)
        
        # Evaluate model
        _, test_df = create_train_test_split(training_df, test_size=0.3)
        eval_results = evaluate_calibration_quality(args.model_path, test_df)
        
        print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
    