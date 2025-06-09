# Well Actually - LLM Assessment Framework

A comprehensive framework for evaluating software repositories in the Ethereum ecosystem using multiple LLM-based assessment tasks. Originally developed for uncertainty calibration, the project has evolved into a modular collection of specialized assessment pipelines with shared infrastructure.

## Overview

Well Actually provides three main assessment capabilities:

1. **Repository Comparison (L1)**: Compare two repositories for their contribution to the Ethereum ecosystem
2. **Originality Assessment (L2)**: Evaluate how original vs dependency-reliant a repository is (1-10 scale)
3. **Criteria Assessment**: Detailed evaluation against 11 specific importance criteria
4. **Dependency Comparison (L3)**: Compare dependencies within a repository context
5. **Uncertainty Calibration**: Transform raw LLM uncertainty scores into reliable confidence estimates

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Assessment Tasks

```bash
# Repository comparison validation
python scripts/validate_l1.py

# Originality assessment
python scripts/run_originality_assessment.py

# Criteria-based assessment
python scripts/run_criteria_assessment.py

# Test specific pipelines
python scripts/test_originality_pipeline.py
```

### 3. API Server

```bash
# Start FastAPI server
python src/api/main.py

# Or with uvicorn
uvicorn src.api.main:app --reload
```

### 4. Uncertainty Calibration (Original Framework)

```bash
# Train calibration model
python scripts/train.py

# Run inference
python scripts/inference.py
```

## New Project Structure

```
src/
├── shared/                     # Shared components
│   ├── multi_model_engine.py   # LLM query management
│   ├── cache_manager.py        # Response caching
│   ├── response_parser.py      # Response parsing utilities
│   ├── model_metadata.py       # Model configurations
│   └── model_answer_postprocessor.py
├── tasks/                      # Assessment tasks
│   ├── l1/                     # Repository comparisons
│   │   ├── level1_prompts.py
│   │   ├── l1_analysis.py
│   │   └── l1_*.py
│   ├── originality/            # Originality assessment
│   │   ├── originality_assessment_pipeline.py
│   │   ├── originality_prompt_generator.py
│   │   ├── originality_response_parser.py
│   │   └── level2_prompts.py
│   ├── criteria/               # Criteria-based assessment
│   │   ├── criteria_assessment_pipeline.py
│   │   ├── criteria_prompt_generator.py
│   │   ├── fuzzy_response_parser.py
│   │   └── *.py
│   └── l3/                     # Dependency comparisons
│       ├── level3_prompts.py
│       ├── dependency_context_extractor.py
│       └── dependency_response_parser.py
├── calibration/                # Original uncertainty calibration
│   ├── calibration_pipeline.py
│   ├── lightgbm_trainer.py
│   ├── feature_engineering.py
│   ├── data_collection.py
│   └── evaluation.py
├── api/                        # FastAPI endpoints
│   ├── main.py
│   ├── llm_orchestrator.py
│   └── *_handler.py
└── utils/                      # Utilities
    └── data_loader.py

configs/
├── config.yaml                 # Main configuration
├── uncertainty_calibration/
│   └── llm.yaml               # LLM settings
└── seed_repositories.yaml     # Ethereum repository metadata
```

## Assessment Tasks

### Repository Comparison (L1)
Compares two repositories against the Ethereum ecosystem:
- **Input**: Two repository URLs + "ethereum" context
- **Output**: A vs B vs Equal + confidence
- **Usage**: Ecosystem importance ranking

### Originality Assessment (L2)  
Evaluates repository originality across multiple criteria:
- **Input**: Single repository URL
- **Output**: Originality score (1-10) + detailed reasoning
- **Categories**: A-I (Execution Clients, Libraries, Tools, etc.)
- **Criteria**: 8 assessment dimensions with category-specific weights

### Criteria Assessment
Detailed evaluation against 11 importance criteria:
- **Input**: Repository URL + criteria weights
- **Output**: Per-criterion scores + overall assessment
- **Use case**: Comprehensive repository evaluation

### Dependency Comparison (L3)
Compares dependencies within a repository context:
- **Input**: Two dependencies + parent repository
- **Output**: Dependency preference + reasoning

## Configuration

### Core Settings (`configs/uncertainty_calibration/llm.yaml`)

```yaml
# Model selection (12+ supported models)
models:
  primary_models:
    gpt_4o: "openai/gpt-4o"
    llama_4_maverick: "meta-llama/llama-4-maverick"
    deepseek_v3: "deepseek/deepseek-chat"

# Temperature sweep for uncertainty
temperature_sweep:
  temperatures: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# API configuration
api:
  openrouter_api_key: ${oc.env:OPENROUTER_API_KEY}
  requests_per_minute: 60

# Caching
cache:
  enabled: true
  directory: "cache"
```

### Repository Metadata (`configs/seed_repositories.yaml`)
Contains 30+ Ethereum repositories with:
- Repository URLs and metadata
- Originality categories (A-I)
- Domain classifications
- Architecture types

## API Endpoints

```bash
# Repository comparison
POST /compare
{
  "repo_a": "https://github.com/ethereum/go-ethereum",
  "repo_b": "https://github.com/ethereum/solidity",
  "parent": "ethereum"
}

# Originality assessment  
POST /assess
{
  "repository_url": "https://github.com/ethereum/go-ethereum"
}

# Criteria assessment
POST /criteria
{
  "repository_url": "https://github.com/ethereum/go-ethereum",
  "criteria_weights": {...}
}
```

## Shared Infrastructure

### MultiModelEngine
- Manages queries across 12+ LLM models via OpenRouter
- Intelligent caching with correlation analysis  
- Rate limiting and provider filtering
- Temperature sweeps for uncertainty quantification

### Response Processing
- Standardized response parsing across tasks
- Model-specific answer postprocessing
- Uncertainty extraction from logprobs
- Robust error handling

## Data Structure

All assessments use structured data from `train.csv`:
- **Level 1**: `parent == "ethereum"` (repository comparisons)
- **Level 2**: `parent == "originality"` (originality assessment)
- **Level 3**: `parent == <repo_url>` (dependency comparisons)

## Environment Setup

Required environment variables:
```bash
export OPENROUTER_API_KEY="your_key_here"
export PYTHONPATH="src"
```

## Development

### Adding New Assessment Tasks
1. Create new folder in `src/tasks/`
2. Implement prompt generator and response parser
3. Add pipeline orchestrator
4. Update API handlers if needed

### Adding New Models
1. Update `src/shared/model_metadata.py`
2. Add model configuration to `llm.yaml`
3. Test with existing pipelines

## Original Uncertainty Calibration

The framework includes the original LightGBM-based uncertainty calibration:

```python
from src.calibration.calibration_pipeline import UncertaintyCalibrationPipeline

# Load trained pipeline
pipeline = UncertaintyCalibrationPipeline.from_config("configs/uncertainty_calibration/llm.yaml")

# Calibrate uncertainties
confidences = pipeline.predict_calibrated_uncertainty(
    raw_uncertainties=[0.3, 0.7, 0.2],
    model_names=["openai/gpt-4o", "deepseek/deepseek-chat"],
    temperatures=[0.0, 0.2]
)
```

## License

MIT License - see LICENSE file for details.