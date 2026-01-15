"""
LLM client module for interacting with Ollama API.

Provides a unified interface for generating text responses from local LLMs
with retry logic, error handling, and configurable parameters.
"""

import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

import requests

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class GenerationResponse:
    """
    Represents a response from the LLM generation.
    
    Attributes:
        text: The generated text response
        success: Whether the generation was successful
        error: Error message if generation failed
        model: Model name used for generation
        duration_ms: Time taken for generation in milliseconds
    """
    text: str
    success: bool
    error: Optional[str] = None
    model: Optional[str] = None
    duration_ms: Optional[int] = None


class OllamaClient:
    """
    Client for interacting with Ollama API.
    
    Provides methods for generating text completions with configurable
    retry logic and error handling.
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        default_temperature: float = 0.7,
        context_window: int = 4096,
        max_predict: int = 1200,
        timeout: int = 600,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize the Ollama client.
        
        Args:
            api_url: Ollama API URL (defaults to config value)
            model_name: Model name to use (defaults to config value)
            default_temperature: Default temperature for generation
            context_window: Context window size
            max_predict: Maximum tokens to predict
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        config = get_config()
        self.api_url = api_url or config.ollama_api
        self.model_name = model_name or config.model_name
        self.default_temperature = default_temperature
        self.context_window = context_window
        self.max_predict = max_predict
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> GenerationResponse:
        """
        Generate a text completion from the LLM.
        
        Args:
            prompt: The input prompt for generation
            temperature: Temperature parameter (uses default if not specified)
            num_ctx: Context window size (uses default if not specified)
            num_predict: Max tokens to predict (uses default if not specified)
            system_prompt: Optional system prompt to prepend
            
        Returns:
            GenerationResponse: The generation response with text or error
        """
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "temperature": temperature or self.default_temperature,
            "options": {
                "num_ctx": num_ctx or self.context_window,
                "num_predict": num_predict or self.max_predict
            }
        }
        
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                duration_ms = int((time.time() - start_time) * 1000)
                
                return GenerationResponse(
                    text=result.get("response", ""),
                    success=True,
                    model=self.model_name,
                    duration_ms=duration_ms
                )
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} timed out: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.error(f"Unexpected error during generation: {e}")
                return GenerationResponse(
                    text="",
                    success=False,
                    error=str(e)
                )
        
        return GenerationResponse(
            text="",
            success=False,
            error=f"Failed after {self.max_retries} attempts"
        )
    
    def generate_with_tags(
        self,
        prompt: str,
        tags: list[str],
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate and extract tagged content from the response.
        
        Convenience method that generates text and extracts content
        marked with [[TAG]] patterns.
        
        Args:
            prompt: The input prompt
            tags: List of tags to extract (e.g., ["QUESTION", "ANSWER"])
            temperature: Temperature parameter
            system_prompt: Optional system prompt
            
        Returns:
            dict: Mapping of tag names to extracted content
        """
        from ..parser.response_parser import ResponseParser
        
        response = self.generate(
            prompt=prompt,
            temperature=temperature,
            system_prompt=system_prompt
        )
        
        if not response.success:
            return {tag: "" for tag in tags}
        
        parser = ResponseParser()
        return {tag: parser.extract_tag(response.text, tag) for tag in tags}
    
    def health_check(self) -> bool:
        """
        Check if the Ollama API is accessible.
        
        Returns:
            bool: True if the API is accessible, False otherwise
        """
        try:
            # Try to reach the Ollama API base URL
            base_url = self.api_url.replace("/api/generate", "")
            response = requests.get(base_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
