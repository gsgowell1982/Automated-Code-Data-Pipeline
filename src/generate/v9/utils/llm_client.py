"""
LLM Client module for Q&A Generation Engine v9.

Provides interface to Ollama API with retry and caching.
"""

import time
import logging
import requests
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Enhanced Ollama client with retry and availability caching.
    
    Usage:
        client = OllamaClient()
        if client.is_available():
            response = client.generate("Your prompt", temperature=0.7)
    """
    
    def __init__(self, api_url: str = None, model_name: str = None):
        self.api_url = api_url or Config.OLLAMA_API
        self.model_name = model_name or Config.MODEL_NAME
        self._available: Optional[bool] = None
        self._check_time: float = 0
        self._cache_duration: int = 60  # seconds
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available (with caching).
        
        Returns:
            bool: True if Ollama API is reachable
        """
        current_time = time.time()
        
        # Use cached result if recent
        if self._available is not None and current_time - self._check_time < self._cache_duration:
            return self._available
        
        try:
            response = requests.get(
                self.api_url.replace("/api/generate", "/api/tags"),
                timeout=5
            )
            self._available = response.status_code == 200
            self._check_time = current_time
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            self._available = False
            self._check_time = current_time
        
        return self._available
    
    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
        max_retries: int = None
    ) -> Optional[str]:
        """
        Generate response from LLM.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            temperature: Generation temperature (0.0-1.0)
            max_retries: Maximum retry attempts
            
        Returns:
            Generated text or None if failed
        """
        if not self.is_available():
            return None
        
        temperature = temperature or 0.7
        max_retries = max_retries or Config.LLM_RETRY_COUNT
        
        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                }
                
                if system:
                    payload["system"] = system
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=Config.LLM_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json().get("response", "").strip()
                    if result:
                        return result
                else:
                    logger.warning(f"LLM returned status {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"LLM timeout (attempt {attempt + 1}/{max_retries + 1})")
            except Exception as e:
                logger.warning(f"LLM error: {e}")
            
            # Exponential backoff
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        
        return None
    
    def generate_with_task(
        self,
        prompt: str,
        task: str,
        system: str = None
    ) -> Optional[str]:
        """
        Generate response with task-specific temperature.
        
        Args:
            prompt: The user prompt
            task: Task type ("question", "reasoning", "answer")
            system: Optional system prompt
            
        Returns:
            Generated text or None if failed
        """
        temperature = Config.LLM_TEMPERATURE.get(task, 0.7)
        return self.generate(prompt, system=system, temperature=temperature)
