"""
Core module for shared utilities and configurations.

This module provides common functionality used across the data generation pipeline:
- Configuration management
- LLM client interactions
- Domain Business Rules (DBR) definitions
"""

from .config import Config, get_config
from .llm_client import OllamaClient
from .dbr_rules import DBRRegistry, DBR_01

__all__ = [
    "Config",
    "get_config",
    "OllamaClient",
    "DBRRegistry",
    "DBR_01",
]
