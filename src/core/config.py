"""
Configuration management module for the data generation pipeline.

Provides centralized configuration for:
- File paths and directories
- Ollama API settings
- Generation parameters
- Target code repositories
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


@dataclass
class Config:
    """
    Centralized configuration for the data generation pipeline.
    
    Attributes:
        base_dir: Base directory for the source code
        data_dir: Directory for generated data output
        repo_path: Path to the target repository for analysis
        ollama_api: URL for the Ollama API endpoint
        model_name: Name of the LLM model to use
        generation_temp: Temperature parameter for generation
        context_window: Context window size for the model
        max_predict: Maximum tokens to predict
        target_files: List of relative file paths to analyze
    """
    
    # Directory configurations
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir: str = field(default="")
    repo_path: str = field(default="")
    
    # Ollama API configurations
    ollama_api: str = field(default_factory=lambda: os.getenv("OLLAMA_API", "http://localhost:11434/api/generate"))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "qwen2.5:7b"))
    
    # Generation parameters
    generation_temp: float = field(default=0.7)
    context_window: int = field(default=4096)
    max_predict: int = field(default=1200)
    request_timeout: int = field(default=600)
    
    # Target files for analysis (relative to repo_path)
    target_files: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize derived paths after dataclass initialization."""
        if not self.data_dir:
            self.data_dir = os.path.abspath(os.path.join(self.base_dir, "..", "data"))
        if not self.repo_path:
            self.repo_path = os.path.abspath(
                os.path.join(self.base_dir, "..", "repos", "fastapi-realworld-example-app")
            )
        if not self.target_files:
            self.target_files = [
                os.path.join("app", "api", "routes", "authentication.py"),
                os.path.join("app", "api", "routes", "users.py"),
            ]
    
    def get_output_path(self, filename: str) -> str:
        """Get the full output path for a data file."""
        return os.path.join(self.data_dir, filename)
    
    def get_target_file_path(self, rel_path: str) -> str:
        """Get the full path for a target file in the repository."""
        return os.path.join(self.repo_path, rel_path)
    
    def ensure_data_dir(self) -> None:
        """Ensure the data directory exists."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)


# Singleton configuration instance
_config_instance: Optional[Config] = None


def get_config(**overrides) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        **overrides: Optional keyword arguments to override default config values
        
    Returns:
        Config: The configuration instance
    """
    global _config_instance
    if _config_instance is None or overrides:
        _config_instance = Config(**overrides)
    return _config_instance


# Language configuration for multi-language support
LANG_CONFIG = {
    "zh-cn": {
        "instruction": "请使用中文进行方案设计，并确保包含逻辑伪代码。",
        "schema_lang": "zh-cn"
    },
    "en": {
        "instruction": "Please provide the design solution in English, including logical pseudocode.",
        "schema_lang": "en"
    }
}


def get_lang_config(lang: str = "zh-cn") -> dict:
    """
    Get language-specific configuration.
    
    Args:
        lang: Language code (e.g., 'zh-cn', 'en')
        
    Returns:
        dict: Language configuration dictionary
    """
    return LANG_CONFIG.get(lang, LANG_CONFIG["zh-cn"])
