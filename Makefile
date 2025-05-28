.PHONY: help install install-dev test lint format clean train evaluate

help:  ## Show this help message
	@echo "Cross-Model Uncertainty Aggregation Methods"
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:  ## Run tests
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:  ## Run code linting
	flake8 src tests scripts
	mypy src

format:  ## Format code with black
	black src tests scripts
	isort src tests scripts

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/

train:  ## Run training pipeline
	python scripts/train.py

evaluate:  ## Run evaluation pipeline
	python scripts/evaluate.py

preprocess:  ## Preprocess raw data
	python scripts/data_processing/preprocess_data.py

# Experiment shortcuts
baseline:  ## Run baseline experiment
	python scripts/train.py experiment=baseline

# Docker commands (if needed later)
docker-build:  ## Build Docker image
	docker build -t cross-model-uncertainty .

docker-run:  ## Run Docker container
	docker run -it --rm cross-model-uncertainty
