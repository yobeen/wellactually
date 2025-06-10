# src/api/config/settings.py
"""
Configuration settings for the FastAPI LLM service.
Loads existing LLM configuration and extends with API-specific settings.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)

class APISettings(BaseSettings):
    """API configuration settings."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    reload: bool = Field(default=False, env="API_RELOAD")
    log_level: str = Field(default="info", env="API_LOG_LEVEL")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_methods: List[str] = Field(default=["*"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")
    
    # LLM configuration path
    llm_config_path: str = Field(
        default="configs/uncertainty_calibration/llm.yaml",
        env="LLM_CONFIG_PATH"
    )
    
    # API-specific settings
    default_temperature: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")
    default_model: str = Field(default="deepseek/deepseek-r1-0528", env="DEFAULT_MODEL")
    request_timeout: int = Field(default=120, env="REQUEST_TIMEOUT")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    
    # Cache settings
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    cache_dir: str = Field(default="cache", env="CACHE_DIR")
    
    # Logging
    enable_request_logging: bool = Field(default=True, env="ENABLE_REQUEST_LOGGING")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Rate limiting (if needed)
    rate_limit_requests_per_minute: int = Field(default=60, env="RATE_LIMIT_RPM")
    
    def __init__(self, **kwargs):
        """Initialize settings and load LLM configuration."""
        super().__init__(**kwargs)
        self._llm_config = None
        self._load_llm_config()
    
    def _load_llm_config(self):
        """Load LLM configuration from YAML file."""
        config_path = Path(self.llm_config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"LLM config file not found: {config_path}")
        
        try:
            self._llm_config = OmegaConf.load(config_path)
            logger.info(f"Loaded LLM configuration from {config_path}")
            
            # Validate required configuration sections
            required_sections = ['api', 'models', 'cache']
            for section in required_sections:
                if section not in self._llm_config:
                    logger.warning(f"Missing required config section: {section}")
            
            # Override cache settings if specified in API settings
            if 'cache' in self._llm_config:
                if hasattr(self, 'enable_cache'):
                    self._llm_config.cache.enabled = self.enable_cache
                if hasattr(self, 'cache_dir'):
                    self._llm_config.cache.directory = self.cache_dir
            
        except Exception as e:
            logger.error(f"Failed to load LLM configuration: {e}")
            raise ValueError(f"Invalid LLM configuration file: {e}")
    
    @property
    def llm_config(self) -> DictConfig:
        """Get the loaded LLM configuration."""
        if self._llm_config is None:
            raise RuntimeError("LLM configuration not loaded")
        return self._llm_config
    
    def get_model_config(self, model_id: Optional[str] = None) -> dict:
        """Get configuration for a specific model."""
        if model_id is None:
            model_id = self.default_model
        
        models_config = self.llm_config.get('models', {})
        
        # Look in primary models first
        primary_models = models_config.get('primary_models', {})
        for key, value in primary_models.items():
            if value == model_id:
                return {
                    'model_id': model_id,
                    'config_key': key,
                    'type': 'primary'
                }
        
        # Then check secondary models
        secondary_models = models_config.get('secondary_models', {})
        for key, value in secondary_models.items():
            if value == model_id:
                return {
                    'model_id': model_id,
                    'config_key': key,
                    'type': 'secondary'
                }
        
        # Default configuration
        return {
            'model_id': model_id,
            'config_key': 'custom',
            'type': 'custom'
        }
    
    def get_api_config(self) -> dict:
        """Get API configuration from LLM config."""
        return self.llm_config.get('api', {})
    
    def get_cache_config(self) -> dict:
        """Get cache configuration."""
        cache_config = self.llm_config.get('cache', {})
        
        # Override with API settings
        cache_config['enabled'] = self.enable_cache
        cache_config['directory'] = self.cache_dir
        
        return cache_config
    
    def get_temperature_config(self) -> dict:
        """Get temperature configuration."""
        temp_config = self.llm_config.get('temperature_sweep', {})
        
        return {
            'enabled': temp_config.get('enabled', True),
            'temperatures': temp_config.get('temperatures', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            'default_temperature': self.default_temperature
        }
    
    def validate_environment(self) -> dict:
        """Validate that required environment variables are set."""
        validation_results = {
            'valid': True,
            'missing': [],
            'warnings': []
        }
        
        # Check for OpenRouter API key
        if not os.getenv('OPENROUTER_API_KEY'):
            validation_results['missing'].append('OPENROUTER_API_KEY')
            validation_results['valid'] = False
        
        # Check config file exists
        if not Path(self.llm_config_path).exists():
            validation_results['missing'].append(f'LLM config file: {self.llm_config_path}')
            validation_results['valid'] = False
        
        # Check cache directory is writable
        cache_path = Path(self.cache_dir)
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = cache_path / '.write_test'
            test_file.touch()
            test_file.unlink()
        except Exception:
            validation_results['warnings'].append(f'Cache directory not writable: {self.cache_dir}')
        
        return validation_results
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "allow"
    }