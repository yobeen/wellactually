# LightGBM Uncertainty Calibration Framework

A framework for calibrating uncertainty estimates from heterogeneous Large Language Models (LLMs) using gradient boosting.

## Overview

This framework solves the problem of incomparable uncertainty scores across different LLMs by training a LightGBM model that maps `[raw_uncertainty, model_metadata]` → `P(answer_is_correct)`.

## Key Features

- **Universal Calibration**: Maps raw perplexity scores from any model to meaningful confidence
- **Model-Aware**: Accounts for model size, architecture, and sampling parameters
- **Temperature Handling**: Automatically adjusts for different sampling temperatures
- **Ensemble Support**: Aggregates calibrated uncertainties across multiple models
- **Evaluation Metrics**: Comprehensive calibration quality assessment

## Quick Start

### 1. Training

```python
from uncertainty_calibration import UncertaintyCalibrationPipeline
from uncertainty_calibration.lightgbm_trainer import train_calibration_model

# Train from collected responses
pipeline = train_calibration_model(training_df, model_save_path="calibration_model.pkl")
```

### 2. Inference

```python
# Load trained pipeline
pipeline = UncertaintyCalibrationPipeline("calibration_model.pkl")

# Calibrate single response
response = {
    'model_id': 'openai/gpt-4o',
    'raw_uncertainty': 1.2,
    'temperature': 0.0,
    'prediction': 'A'
}

calibrated = pipeline.calibrate_single_response(response)
print(f"Confidence: {calibrated['calibrated_confidence']:.3f}")
```

### 3. Multi-Model Aggregation

```python
# Calibrate responses from multiple models
model_responses = {
    'gpt-4o': {'raw_uncertainty': 0.8, 'temperature': 0.0, 'prediction': 'A'},
    'llama-405b': {'raw_uncertainty': 1.5, 'temperature': 0.0, 'prediction': 'B'},
    'deepseek': {'raw_uncertainty': 2.1, 'temperature': 0.0, 'prediction': 'A'}
}

calibrated = pipeline.calibrate_model_responses(model_responses)
aggregated = pipeline.aggregate_calibrated_uncertainties(calibrated)

print(f"Ensemble confidence: {aggregated['aggregated_confidence']:.3f}")
```

## Data Collection

The framework includes a data collection pipeline that performs temperature sweeps across multiple models:

```python
from uncertainty_calibration.data_collection import collect_calibration_dataset

# Collect training data across temperature range [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
training_df = collect_calibration_dataset(config, "training_data.csv")
```

## Feature Engineering

Automatic feature engineering from raw responses:

- **Core Features**: `raw_uncertainty`, `model_name`, `param_count`, `temperature`
- **Derived Features**: Normalized uncertainty, temperature interactions, model categories
- **Ensemble Features**: Cross-model statistics for same questions

## Model Architecture

**LightGBM Configuration**:
- Objective: Binary classification (predict answer correctness)
- Categorical features: Model name, size category, temperature bins
- Early stopping with validation monitoring
- Feature importance analysis

## Evaluation

Comprehensive calibration quality metrics:

```python
from uncertainty_calibration.evaluation import CalibrationEvaluator

evaluator = CalibrationEvaluator()
metrics = evaluator.comprehensive_evaluation(y_true, y_prob)

print(f"Expected Calibration Error: {metrics['ECE']:.4f}")
print(f"Brier Score: {metrics['Brier_Score']:.4f}")
```

## Scripts

### Training Script

```bash
# Train complete pipeline
python scripts/train_calibrator.py

# Individual steps
python scripts/train_calibrator.py --collect-data
python scripts/train_calibrator.py --train
python scripts/train_calibrator.py --evaluate
```

### Demo Usage

```bash
# See calibration in action
python scripts/demo_usage.py
```

## Expected Benefits

1. **Unified Scale**: `uncertainty=0.7` means the same thing across all models
2. **Automatic Learning**: Discovers model-specific calibration behaviors
3. **Temperature Handling**: Accounts for sampling parameter effects
4. **Model Size Awareness**: Larger models automatically get different treatment
5. **Ensemble Ready**: Meaningful aggregation of multiple model outputs

## File Structure

```
src/uncertainty_calibration/
├── __init__.py                 # Package initialization
├── model_metadata.py           # Model parameter definitions
├── data_collection.py          # Temperature sweep data collection
├── feature_engineering.py      # Response to feature conversion
├── lightgbm_trainer.py        # LightGBM training logic
├── calibration_pipeline.py    # Production inference pipeline
└── evaluation.py              # Calibration quality metrics
```

## Dependencies

- `lightgbm` - Gradient boosting model
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - ML utilities
- `matplotlib` - Plotting

## Key Insight

The model learns that "uncertainty 0.7 from GPT-4 at temp=0.2" means something completely different than "uncertainty 0.7 from Llama-7B at temp=0.8", and maps both to appropriate confidence levels that actually predict answer correctness.