# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Well Actually** is an LLM Assessment Framework for evaluating software repositories in the Ethereum ecosystem. Originally developed for uncertainty calibration, it has evolved into a modular collection of specialized assessment tasks with shared infrastructure. The project supports repository comparison, originality assessment, criteria-based evaluation, dependency comparison, and uncertainty calibration.

## Refactored Architecture

The codebase has been recently refactored from a monolithic uncertainty calibration package into well-organized task-specific modules:

### Core Structure
- **`src/shared/`**: Shared components used across all assessment tasks
- **`src/tasks/`**: Task-specific assessment modules organized by type
- **`src/calibration/`**: Original uncertainty calibration functionality 
- **`src/api/`**: FastAPI server endpoints
- **`src/utils/`**: General utilities

### Task Organization
- **`src/tasks/l1/`**: Repository comparison (Level 1) - compares repos against Ethereum ecosystem
- **`src/tasks/originality/`**: Originality assessment (Level 2) - evaluates repo originality (1-10 scale)
- **`src/tasks/criteria/`**: Criteria-based assessment - detailed evaluation against 11 criteria
- **`src/tasks/l3/`**: Dependency comparison (Level 3) - compares dependencies within repos

## Development Commands

### Setup and Installation
```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="your_key_here"
export PYTHONPATH="src"
```

### Assessment Tasks
```bash
# Repository comparison validation
python scripts/validate_l1.py

# Originality assessment pipeline
python scripts/run_originality_assessment.py

# Criteria-based assessment
python scripts/run_criteria_assessment.py

# Test specific components
python scripts/test_originality_pipeline.py
```

### API Server
```bash
# Start FastAPI server
python src/api/main.py

# Or with uvicorn
uvicorn src.api.main:app --reload
```

### Original Uncertainty Calibration
```bash
# Train calibration models
python scripts/train.py

# Run calibration inference
python scripts/inference.py

# Data collection for calibration
python scripts/collect_responses.py
```

### Testing and Quality
```bash
pytest
black .
flake8 .
mypy .
```

## Key Components

### Shared Infrastructure (`src/shared/`)
- **`MultiModelEngine`**: Manages LLM queries across 12+ models via OpenRouter API
- **`CacheManager`**: Intelligent response caching with correlation analysis
- **`ResponseParser`**: Standardized response parsing utilities
- **`ModelMetadata`**: Model configuration and parameter definitions
- **`ModelAnswerPostprocessor`**: Model-specific answer normalization

### Assessment Task Architecture
Each task follows a consistent pattern:
- **Prompt Generator**: Creates task-specific prompts
- **Response Parser**: Parses and validates LLM responses
- **Assessment Pipeline**: Orchestrates the complete evaluation process

### Data Flow
1. **Input**: Repository URLs, comparison contexts, or assessment criteria
2. **Prompt Generation**: Task-specific prompts created based on input
3. **LLM Queries**: Multiple models queried with temperature sweeps
4. **Response Processing**: Responses parsed and uncertainty extracted
5. **Output**: Structured assessment results with confidence scores

## Configuration System

### Main Configuration
- **`configs/config.yaml`**: Main Hydra configuration entry point
- **`configs/uncertainty_calibration/llm.yaml`**: Comprehensive LLM settings (12+ models)
- **`configs/seed_repositories.yaml`**: 30+ Ethereum repositories with metadata

### Key Configuration Sections
- **Models**: Primary/secondary model definitions with API settings
- **Temperature Sweeps**: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] for uncertainty calculation
- **Caching**: File-based response caching configuration
- **API**: OpenRouter API settings and rate limiting

## Data Structure and Levels

The framework processes data from `train.csv` with three assessment levels:

### Level 1 (L1): Repository Comparisons
- **Context**: `parent == "ethereum"`
- **Task**: Compare two repositories for Ethereum ecosystem contribution
- **Output**: A vs B vs Equal + confidence scores

### Level 2 (L2): Originality Assessment  
- **Context**: `parent == "originality"`
- **Task**: Assess repository originality across 8 criteria
- **Output**: Originality score (1-10) + detailed reasoning
- **Categories**: A-I (Execution Clients, Libraries, Tools, etc.)

### Level 3 (L3): Dependency Comparisons
- **Context**: `parent == <repository_url>`
- **Task**: Compare dependencies within parent repository
- **Output**: Dependency preference + reasoning

### Criteria Assessment
- **Task**: Evaluate against 11 specific importance criteria
- **Output**: Per-criterion scores + overall weighted assessment

## API Endpoints

The FastAPI server provides three main endpoints:

```bash
POST /compare    # Repository comparison (L1/L3)
POST /assess     # Originality assessment (L2)  
POST /criteria   # Criteria-based evaluation
```

## Important Implementation Notes

### Import Structure
After refactoring, use these import patterns:
```python
# Shared components
from src.shared.multi_model_engine import MultiModelEngine
from src.shared.model_metadata import get_model_metadata

# Task-specific components
from src.tasks.l1.level1_prompts import Level1PromptGenerator
from src.tasks.originality.originality_assessment_pipeline import OriginalityAssessmentPipeline
from src.tasks.criteria.criteria_assessment_pipeline import CriteriaAssessmentPipeline

# Calibration components
from src.calibration.calibration_pipeline import UncertaintyCalibrationPipeline
```

### Development Guidelines
- **Always use MultiModelEngine** for LLM queries to ensure proper caching and rate limiting
- **Follow task-specific patterns** when adding new assessment types
- **Configuration over hardcoding** - use Hydra config files for all settings
- **Respect data level structure** - maintain L1/L2/L3 data organization
- **Cache management** - delete `correlation_cache.json` to force fresh queries
- **Environment variables** - always set `OPENROUTER_API_KEY` and `PYTHONPATH=src`

### Repository Assessment Context
- The framework specializes in **Ethereum ecosystem** repository evaluation
- **30+ seed repositories** are categorized and weighted in `seed_repositories.yaml`
- **Originality categories A-I** have specific weight matrices for different repo types
- **Human judgment data** in `train.csv` provides ground truth for validation

### Performance Considerations
- **Temperature sweeps** generate multiple queries per assessment for uncertainty calculation
- **Intelligent caching** prevents redundant API calls
- **Rate limiting** manages API quota across multiple models
- **Provider filtering** handles API failures gracefully

The refactored architecture provides clear separation of concerns while maintaining the rich functionality developed for Ethereum ecosystem repository assessment.