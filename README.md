# Cross-Model Uncertainty Aggregation Methods

A comprehensive framework for aggregating uncertainty measures from multiple Large Language Models (LLMs) to produce reliable, calibrated uncertainty estimates for question-answering tasks.

## Overview

This project implements multiple uncertainty aggregation methods to combine heterogeneous uncertainty scores from 10+ different LLMs, transforming them into meaningful, calibrated uncertainty estimates.

## Features

### Uncertainty Aggregation Methods
- **Calibration-Based Normalization**: Maps raw uncertainty to calibrated confidence using validation data
- **Rank-Based Normalization**: Converts scores to percentiles within historical distributions
- **Temperature Scaling**: Calibrates raw perplexity scores using learned temperature parameters
- **Meta-Learning Approach**: Trains models to predict optimal aggregated uncertainty
- **Bayesian Model Averaging**: Treats each model as expert opinion in Bayesian framework

### LLM Data Augmentation Pipeline
- **Level 1**: Seed repository comparisons (A vs B vs Equal)
- **Level 2**: Originality assessments (1-10 scale)
- **Level 3**: Dependency comparisons within repositories

## Project Structure

```
├── configs/                    # Hydra configuration files
│   ├── llm_augmentation/      # LLM pipeline configs
│   ├── uncertainty_aggregation/ # Uncertainty method configs
│   └── experiments/           # Experiment-specific configs
├── src/
│   ├── llm_augmentation/      # LLM data generation
│   │   ├── prompts/          # Level 1-3 prompt generators
│   │   └── engines/          # Multi-model API engine
│   ├── uncertainty_aggregation/ # Core aggregation methods
│   │   ├── methods/          # Individual aggregation methods
│   │   ├── pipelines/        # Training/inference pipelines
│   │   └── evaluation/       # Metrics and calibration
│   └── utils/                # Utility functions
├── data/
│   ├── raw/                  # Original data (train.csv)
│   ├── processed/            # Cleaned/preprocessed data
│   └── augmented/            # LLM-generated synthetic data
├── notebooks/                # Jupyter notebooks for analysis
├── tests/                    # Unit and integration tests
└── scripts/                  # Training and evaluation scripts
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository_url>
cd cross-model-uncertainty-aggregation

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

Place your `train.csv` file in `data/raw/`:

```bash
# Preprocess the data
python scripts/data_processing/preprocess_data.py
```

### 3. Training

```bash
# Run baseline experiment
python scripts/train.py experiment=baseline

# Run with custom config
python scripts/train.py uncertainty_aggregation.enabled_methods.meta_learning=false
```

### 4. Evaluation

```bash
# Evaluate trained models
python scripts/evaluate.py

# Run specific evaluation
python scripts/evaluate.py experiment=baseline
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/uncertainty_aggregation/default.yaml`: Uncertainty method settings
- `configs/llm_augmentation/default.yaml`: LLM pipeline settings

### Example Usage

```python
from src.uncertainty_aggregation.pipelines import InferencePipeline

# Initialize pipeline
pipeline = InferencePipeline.from_config(config)

# Aggregate uncertainties from multiple models
raw_uncertainties = [0.2, 0.3, 0.15, ...]  # From 10 models
aggregated_uncertainty = pipeline.aggregate(raw_uncertainties)
```

## Methods

### 1. Calibration-Based Normalization
Maps raw uncertainty scores to calibrated confidence using isotonic regression on validation data.

### 2. Rank-Based Normalization  
Converts each model's score to its percentile within the model's historical distribution.

### 3. Temperature Scaling
Learns temperature parameters to calibrate perplexity scores using cross-entropy loss.

### 4. Meta-Learning
Trains a meta-model to predict optimal aggregated uncertainty from individual model outputs.

### 5. Bayesian Model Averaging
Treats each model as an expert and uses Bayesian updating to combine opinions.

## Evaluation Metrics

- **Predictive Accuracy**: How well uncertainty predicts correctness
- **Calibration Error**: Expected Calibration Error (ECE) 
- **Sharpness**: How discriminative the uncertainty scores are
- **Ensemble Diversity**: Agreement between individual models

## Contributing

1. Follow PEP 8 style guidelines
2. Add unit tests for new functionality
3. Update configuration files as needed
4. Document new methods and APIs

## License

[License information]
