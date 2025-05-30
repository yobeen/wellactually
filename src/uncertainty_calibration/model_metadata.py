# src/uncertainty_calibration/model_metadata.py
#!/usr/bin/env python3
"""
Model metadata for LLM uncertainty calibration.
Maps model IDs to their parameter counts and other metadata.
"""

from typing import Dict, Optional

# Model parameter counts (in billions)
MODEL_PARAMS = {
    # OpenAI models
    "openai/gpt-4o": 200.0,  # GPT-4o
    
    # Meta LLaMA models
    "meta-llama/llama-4-maverick": 405.0,  # Llama 4 Maverick
    "meta-llama/llama-3.1-405b-instruct": 405.0,  # Llama 3.1 405B
    
    # DeepSeek models
    "deepseek/deepseek-chat": 236.0,  # DeepSeek V3
    "deepseek/deepseek-chat-v3-0324": 236.0,  # DeepSeek Chat V3
    "deepseek/deepseek-coder": 33.0,  # DeepSeek Coder V2
    "deepseek/deepseek-r1-0528": 671.0,  # DeepSeek R1
    
    # Qwen models
    "qwen/qwq-32b-preview": 32.0,  # QwQ 32B
    "qwen/qwen3-235b-a22b": 235.0,  # Qwen 3 235B
    
    # xAI models
    "x-ai/grok-3-beta": 314.0,  # Grok 3 Beta
    
    # Mistral models
    "mistralai/mixtral-8x22b-instruct": 176.0,  # Mixtral 8x22B (8*22B)
    
    # Google models
    "google/gemma-3-27b-it": 27.0,  # Gemma 3 27B
}

# Model organization/provider mapping
MODEL_PROVIDERS = {
    "openai/gpt-4o": "openai",
    "meta-llama/llama-4-maverick": "meta",
    "meta-llama/llama-3.1-405b-instruct": "meta", 
    "deepseek/deepseek-chat": "deepseek",
    "deepseek/deepseek-chat-v3-0324": "deepseek",
    "deepseek/deepseek-coder": "deepseek",
    "deepseek/deepseek-r1-0528": "deepseek",
    "x-ai/grok-3-beta": "xai",
    "qwen/qwq-32b-preview": "alibaba",
    "qwen/qwen3-235b-a22b": "alibaba",
    "mistralai/mixtral-8x22b-instruct": "mistral",
    "google/gemma-3-27b-it": "google",
}

# Model architecture types
MODEL_ARCHITECTURES = {
    "openai/gpt-4o": "transformer",
    "meta-llama/llama-4-maverick": "transformer",
    "meta-llama/llama-3.1-405b-instruct": "transformer",
    "deepseek/deepseek-chat": "transformer", 
    "deepseek/deepseek-chat-v3-0324": "transformer",
    "deepseek/deepseek-coder": "transformer",
    "deepseek/deepseek-r1-0528": "transformer",
    "x-ai/grok-3-beta": "transformer",
    "qwen/qwq-32b-preview": "transformer",
    "qwen/qwen3-235b-a22b": "transformer",
    "mistralai/mixtral-8x22b-instruct": "mixture_of_experts",
    "google/gemma-3-27b-it": "transformer",
}

def get_model_params(model_id: str) -> Optional[float]:
    """Get parameter count for a model."""
    return MODEL_PARAMS.get(model_id)

def get_model_provider(model_id: str) -> Optional[str]:
    """Get provider/organization for a model."""
    return MODEL_PROVIDERS.get(model_id)

def get_model_architecture(model_id: str) -> Optional[str]:
    """Get architecture type for a model."""
    return MODEL_ARCHITECTURES.get(model_id)

def get_all_models() -> list:
    """Get list of all available models."""
    return list(MODEL_PARAMS.keys())

def get_model_metadata(model_id: str) -> Dict:
    """Get complete metadata for a model."""
    return {
        "model_id": model_id,
        "param_count": get_model_params(model_id),
        "provider": get_model_provider(model_id),
        "architecture": get_model_architecture(model_id),
    }

def validate_model_id(model_id: str) -> bool:
    """Check if model ID is supported."""
    return model_id in MODEL_PARAMS

# Temperature ranges for data collection
TEMPERATURE_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Default sampling parameters
DEFAULT_SAMPLING_PARAMS = {
    "max_tokens": 10,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}