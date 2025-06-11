# src/shared/cache_manager.py
#!/usr/bin/env python3
"""
File-based cache manager for LLM API responses.
Caches raw API response dictionaries for reuse across runs.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages file-based caching of raw LLM API responses.
    """
    
    def __init__(self, cache_dir: str = "cache", enabled: bool = True):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache manager initialized: {self.cache_dir}")
        else:
            logger.info("Cache manager disabled")
    
    def _strip_bytes_fields(self, data: Any) -> Any:
        """
        Recursively remove 'bytes' fields from dictionary structures.
        
        Args:
            data: Data structure to clean
            
        Returns:
            Cleaned data structure without bytes fields
        """
        if isinstance(data, dict):
            return {k: self._strip_bytes_fields(v) for k, v in data.items() if k != "bytes"}
        elif isinstance(data, list):
            return [self._strip_bytes_fields(item) for item in data]
        else:
            return data
    
    def generate_cache_key(self, model_id: str, prompt: List[Dict[str, str]], 
                          temperature: float) -> str:
        """
        Generate deterministic cache key from request parameters.
        
        Args:
            model_id: Model identifier
            prompt: Prompt messages
            temperature: Sampling temperature
            
        Returns:
            Cache key string
        """
        # Create deterministic string from all parameters
        prompt_str = json.dumps(prompt, sort_keys=True, separators=(',', ':'))
        cache_input = f"{model_id}|{prompt_str}|{temperature}"
        
        # Generate hash
        cache_hash = hashlib.sha256(cache_input.encode('utf-8')).hexdigest()
        
        # Use first 16 characters for reasonable filename length
        return cache_hash[:16]
    
    def get_cache_file_path(self, model_id: str, cache_key: str) -> Path:
        """
        Get cache file path for model and key.
        
        Args:
            model_id: Model identifier
            cache_key: Cache key
            
        Returns:
            Path to cache file
        """
        # Sanitize model_id for filesystem (replace / with -)
        safe_model_id = model_id.replace("/", "-").replace(":", "-")
        
        model_cache_dir = self.cache_dir / safe_model_id
        if not model_cache_dir.exists():
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created new cache directory for model: {model_id}")
        
        return model_cache_dir / f"{cache_key}.json"
    
    def get_cached_response(self, model_id: str, prompt: List[Dict[str, str]], 
                           temperature: float, cache_key_suffix: str = "") -> Optional[Dict]:
        """
        Retrieve cached response if available.
        
        Args:
            model_id: Model identifier
            prompt: Prompt messages
            temperature: Sampling temperature
            cache_key_suffix: Additional suffix for cache key (e.g., for max_tokens)
            
        Returns:
            Raw API response dict or None if not cached
        """
        if not self.enabled:
            return None
        
        try:
            cache_key = self.generate_cache_key(model_id, prompt, temperature) + cache_key_suffix
            cache_file = self.get_cache_file_path(model_id, cache_key)
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Validate cache data structure
            if not self._validate_cache_data(cached_data):
                logger.warning(f"Invalid cache data in {cache_file}, removing")
                cache_file.unlink()
                return None
            
            logger.debug(f"Cache hit: {model_id} temp={temperature}")
            return cached_data['raw_api_response']
            
        except Exception as e:
            logger.warning(f"Error reading cache for {model_id}: {e}")
            return None
    
    def save_response_to_cache(self, model_id: str, prompt: List[Dict[str, str]], 
                              temperature: float, raw_api_response: Dict, cache_key_suffix: str = ""):
        """
        Save raw API response to cache.
        
        Args:
            model_id: Model identifier
            prompt: Prompt messages
            temperature: Sampling temperature
            raw_api_response: Raw API response dictionary
            cache_key_suffix: Additional suffix for cache key (e.g., for max_tokens)
        """
        if not self.enabled:
            return
        
        try:
            cache_key = self.generate_cache_key(model_id, prompt, temperature) + cache_key_suffix
            cache_file = self.get_cache_file_path(model_id, cache_key)
            
            # Create cache data structure
            prompt_hash = hashlib.sha256(
                json.dumps(prompt, sort_keys=True).encode('utf-8')
            ).hexdigest()[:16]
            
            # Strip bytes fields before caching
            cleaned_response = self._strip_bytes_fields(raw_api_response)
            
            cache_data = {
                "cache_key": cache_key,
                "model_id": model_id,
                "temperature": temperature,
                "prompt_hash": prompt_hash,
                "cached_at": datetime.now().isoformat(),
                "raw_api_response": cleaned_response
            }
            
            # Check if this is a new cache file
            is_new_file = not cache_file.exists()
            
            # Atomic write using temporary file
            self._atomic_write_json(cache_file, cache_data)
            
            if is_new_file:
                logger.info(f"Created new cache file: {cache_file}")
            logger.debug(f"Cached response: {model_id} temp={temperature}")
            
        except Exception as e:
            logger.warning(f"Error saving cache for {model_id}: {e}")
    
    def _atomic_write_json(self, file_path: Path, data: Dict):
        """
        Atomically write JSON data to file.
        
        Args:
            file_path: Target file path
            data: Data to write
        """
        # Write to temporary file first
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', 
                encoding='utf-8',
                dir=file_path.parent,
                delete=False,
                suffix='.tmp'
            ) as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                temp_file = f.name
            
            # Atomic move to final location
            temp_path = Path(temp_file)
            temp_path.replace(file_path)
            
        except Exception as e:
            # Clean up temp file on error
            if temp_file and Path(temp_file).exists():
                Path(temp_file).unlink()
            raise e
    
    def _validate_cache_data(self, cache_data: Dict) -> bool:
        """
        Validate cache data structure.
        
        Args:
            cache_data: Cache data to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'cache_key', 'model_id', 'temperature', 
            'cached_at', 'raw_api_response'
        ]
        
        for field in required_fields:
            if field not in cache_data:
                return False
        
        # Validate raw_api_response structure
        api_response = cache_data['raw_api_response']
        if not isinstance(api_response, dict):
            return False
        
        return True
    
    def clear_cache(self, model_id: Optional[str] = None):
        """
        Clear cache files.
        
        Args:
            model_id: Clear cache for specific model, or all if None
        """
        if not self.enabled or not self.cache_dir.exists():
            return
        
        try:
            if model_id:
                # Clear specific model cache
                safe_model_id = model_id.replace("/", "-").replace(":", "-")
                model_cache_dir = self.cache_dir / safe_model_id
                
                if model_cache_dir.exists():
                    for cache_file in model_cache_dir.glob("*.json"):
                        cache_file.unlink()
                    # Only remove directory if empty
                    try:
                        model_cache_dir.rmdir()
                    except OSError:
                        pass  # Directory not empty
                    logger.info(f"Cleared cache for model: {model_id}")
            else:
                # Clear all cache
                for model_dir in self.cache_dir.iterdir():
                    if model_dir.is_dir():
                        for cache_file in model_dir.glob("*.json"):
                            cache_file.unlink()
                        # Only remove directory if empty
                        try:
                            model_dir.rmdir()
                        except OSError:
                            pass  # Directory not empty
                logger.info("Cleared all cache")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled or not self.cache_dir.exists():
            return {"enabled": False, "total_files": 0, "total_size_mb": 0}
        
        stats = {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "models": {},
            "total_files": 0,
            "total_size_bytes": 0
        }
        
        try:
            for model_dir in self.cache_dir.iterdir():
                if model_dir.is_dir():
                    model_files = list(model_dir.glob("*.json"))
                    model_size = sum(f.stat().st_size for f in model_files)
                    
                    stats["models"][model_dir.name] = {
                        "files": len(model_files),
                        "size_bytes": model_size
                    }
                    
                    stats["total_files"] += len(model_files)
                    stats["total_size_bytes"] += model_size
            
            stats["total_size_mb"] = stats["total_size_bytes"] / (1024 * 1024)
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            stats["error"] = str(e)
        
        return stats