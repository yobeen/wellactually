# src/uncertainty_calibration/model_metadata.py
"""
Model metadata for uncertainty calibration.
Maps model names to parameter counts and other metadata.
"""

# Model parameter counts in billions
MODEL_PARAMS = {
    # Primary models
    "openai/gpt-4o": 1800.0,  # Estimated
    "meta-llama/llama-4-maverick": 400.0,  # Estimated  
    "deepseek/deepseek-chat": 67.0,
    "x-ai/grok-3-beta": 314.0,
    
    # Secondary models
    "meta-llama/llama-3.1-405b-instruct": 405.0,
    "qwen/qwq-32b-preview": 32.0,
    "mistralai/mixtral-8x22b-instruct": 176.0,  # 8 x 22B
    
    # Specialist models
    "deepseek/deepseek-coder": 67.0,
    "google/gemma-3-27b-it": 27.0,
}

# Model categories for analysis
MODEL_CATEGORIES = {
    "large": ["openai/gpt-4o", "meta-llama/llama-3.1-405b-instruct", "meta-llama/llama-4-maverick", "x-ai/grok-3-beta"],
    "medium": ["deepseek/deepseek-chat", "deepseek/deepseek-coder", "mistralai/mixtral-8x22b-instruct"],
    "small": ["qwen/qwq-32b-preview", "google/gemma-3-27b-it"]
}

def get_model_params(model_id: str) -> float:
    """Get parameter count for a model."""
    return MODEL_PARAMS.get(model_id, 7.0)  # Default to 7B if unknown

def get_model_category(model_id: str) -> str:
    """Get size category for a model."""
    for category, models in MODEL_CATEGORIES.items():
        if model_id in models:
            return category
    return "unknown"