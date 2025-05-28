# LightGBM Uncertainty Calibration Framework

A production-ready framework for calibrating uncertainty scores from heterogeneous LLMs using gradient boosting. Transforms raw perplexity scores into reliable confidence estimates that accurately predict answer correctness.

## Overview

Instead of complex model-specific calibration methods, this framework uses LightGBM to learn the universal mapping:
```
f(raw_uncertainty, model_metadata) → P(answer_is_correct)
```

This automatically handles:
- Different uncertainty scales across model sizes
- Model-specific calibration behaviors  
- Temperature effects on uncertainty
- Non-linear relationships between features

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Data Exploration

```bash
python scripts/explore_data.py
```

### 3. Training Calibration Model

```bash
# Full training pipeline
python scripts/train_uncertainty_calibration.py

# Quick test with minimal data
python scripts/train_uncertainty_calibration.py --quick_test

# Custom configuration
python scripts/train_uncertainty_calibration.py \
    --config configs/uncertainty_calibration/llm.yaml \
    --models "openai/gpt-4o" "deepseek/deepseek-chat" \
    --max_samples 20
```

### 4. Using Trained Model

```bash
# Interactive mode
python scripts/calibrate_uncertainties.py \
    --model_path models/uncertainty_calibration/calibration_model.lgb \
    --interactive

# Batch processing
python scripts/calibrate_uncertainties.py \
    --model_path models/uncertainty_calibration/calibration_model.lgb \
    --input_csv input_uncertainties.csv \
    --output_csv calibrated_results.csv
```

## Project Structure

```
src/uncertainty_calibration/
├── __init__.py                 # Package initialization
├── model_metadata.py           # Model parameter definitions  
├── data_collection.py          # Temperature sweep data collection
├── feature_engineering.py      # Response to feature conversion
├── lightgbm_trainer.py        # LightGBM training logic
├── calibration_pipeline.py    # Production inference pipeline
└── evaluation.py              # Calibration quality metrics

configs/uncertainty_calibration/
└── llm.yaml                    # LLM and training configuration

scripts/
├── explore_data.py             # Data exploration
├── train_uncertainty_calibration.py  # Complete training pipeline
└── calibrate_uncertainties.py # Inference script
```

## Data Flow

### 1. Data Collection
- Load `train.csv` with human preference judgments
- Convert each row to appropriate prompts (Level 1-3)
- Query multiple LLMs at different temperatures [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- Extract perplexity/uncertainty scores from logprobs

### 2. Feature Engineering
Core features (as specified in framework):
- `raw_uncertainty`: Perplexity score from model
- `model_name`: Categorical model identifier
- `param_count`: Model parameters in billions
- `temperature`: Sampling temperature

Additional features:
- `level`: Data level (1=comparison, 2=originality, 3=dependency)
- `provider`: Model organization (openai, meta, etc.)
- `architecture`: Model architecture type
- `log_param_count`: Log-scaled parameters
- `is_zero_temp`: Zero temperature flag
- `temp_squared`: Non-linear temperature effect

### 3. Training
- LightGBM binary classifier predicts `P(answer_is_correct)`
- Cross-validation with stratification by model and level
- Early stopping to prevent overfitting
- Feature importance analysis

### 4. Evaluation
Calibration quality metrics:
- **Expected Calibration Error (ECE)**: Average difference between confidence and accuracy
- **Maximum Calibration Error (MCE)**: Worst-case calibration error
- **Brier Score**: Proper scoring rule decomposed into reliability + resolution
- **ROC AUC**: Discrimination ability
- **Sharpness**: How far predictions are from uninformative (0.5)

## Configuration

### Core Settings (`configs/uncertainty_calibration/llm.yaml`)

```yaml
# Model selection
models:
  primary_models:
    gpt_4o: "openai/gpt-4o"
    llama_4_maverick: "meta-llama/llama-4-maverick"
    deepseek_v3: "deepseek/deepseek-chat"

# Temperature sweep
temperature_sweep:
  temperatures: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Data collection
data_collection:
  max_samples_per_level: 30

# LightGBM parameters
lightgbm:
  params:
    objective: "binary"
    learning_rate: 0.05
    num_leaves: 31
```

### Quality Thresholds

```yaml
quality_thresholds:
  max_ece: 0.1          # Maximum Expected Calibration Error
  min_roc_auc: 0.65     # Minimum discrimination ability
```

## Data Levels

The framework handles three types of data from `train.csv`:

### Level 1: Seed Repository Comparisons
- `parent == "ethereum"`
- Compares two repositories: "Which contributes more to Ethereum ecosystem?"
- Output: A vs B vs Equal

### Level 2: Originality Assessment  
- `parent == "originality"`
- Assesses single repository originality on 1-10 scale
- Output: Originality bucket (1-10)

### Level 3: Dependency Comparisons
- `parent == <repository_url>`
- Compares dependencies within parent repository
- Output: A vs B vs Equal

## Production Usage

```python
from uncertainty_calibration import UncertaintyCalibrationPipeline

# Load trained pipeline
pipeline = create_pipeline_from_config("configs/uncertainty_calibration/llm.yaml")

# Calibrate new uncertainties
raw_uncertainties = [0.3, 0.7, 0.2]
model_names = ["openai/gpt-4o", "deepseek/deepseek-chat", "meta-llama/llama-4-maverick"]
temperatures = [0.0, 0.0, 0.2]

calibrated_confidences = pipeline.predict_calibrated_uncertainty(
    raw_uncertainties, model_names, temperatures
)

# Results: [0.82, 0.31, 0.76] - probabilities that answers are correct
```

## Expected Results

**Before Calibration**: Raw uncertainty 0.7 from different models means different things

**After Calibration**: All uncertainty scores map to meaningful confidence levels:
- 0.9 confidence → 90% chance the answer is correct
- 0.3 confidence → 30% chance the answer is correct

**Key Benefits**:
- **Unified Scale**: All models produce comparable uncertainty scores
- **Predictive**: Uncertainty accurately predicts correctness
- **Automatic**: No manual calibration curves or model-specific tuning
- **Extensible**: Easy to add new models or features

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Reduce `requests_per_minute` in config
2. **Out of Memory**: Reduce `max_samples_per_level` for large datasets
3. **Poor Calibration**: Check data quality, increase training samples
4. **Missing Models**: Verify model IDs in `model_metadata.py`

### Debugging

```bash
# Enable debug logging
export PYTHONPATH=src
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from uncertainty_calibration import run_calibration_pipeline
run_calibration_pipeline('configs/uncertainty_calibration/llm.yaml')
"
```

## Contributing

1. Add new models to `model_metadata.py`
2. Extend features in `feature_engineering.py`
3. Add evaluation metrics in `evaluation.py`
4. Update configuration in `llm.yaml`

## License

MIT License - see LICENSE file for details.