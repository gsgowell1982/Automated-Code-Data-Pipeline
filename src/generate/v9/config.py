"""
Configuration module for Q&A Generation Engine v9.

This module contains all configuration parameters that can be
modified without changing the core logic.
"""

from pathlib import Path


class Config:
    """Global configuration."""
    
    VERSION = "9.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).parent.resolve()
    WORKSPACE_ROOT = BASE_DIR.parent.parent.parent
    DATA_DIR = WORKSPACE_ROOT / "data"
    REPOS_DIR = WORKSPACE_ROOT / "repos"
    
    # Input/Output files
    RULE_METADATA_FILE = DATA_DIR / "dbr01_rule_metadata.json"
    AST_ANALYSIS_FILE = DATA_DIR / "fastapi_analysis_result.json"
    OUTPUT_FILE = DATA_DIR / "qwen_dbr_training_logic_v9.jsonl"
    
    # LLM Configuration
    OLLAMA_API = "http://localhost:11434/api/generate"
    MODEL_NAME = "qwen2.5:7b"
    LLM_TIMEOUT = 180
    LLM_RETRY_COUNT = 2
    
    # Temperature settings for different generation tasks
    LLM_TEMPERATURE = {
        "question": 0.85,    # Higher for diverse questions
        "reasoning": 0.5,    # Lower for consistent reasoning
        "answer": 0.6,       # Moderate for good answers
    }
    
    # Generation parameters
    SUPPORTED_LANGUAGES = ["en", "zh"]
    DEFAULT_QUESTIONS_PER_EVIDENCE = 5
    DEFAULT_TOTAL_LIMIT = None
    
    # Diversity parameters (v7)
    SIMILARITY_THRESHOLD = 0.6
    CANDIDATE_MULTIPLIER = 2.0
    
    # Consistency validation (v8)
    MAX_CONTRADICTION_SCORE = 0.3
    ENABLE_CODE_GROUNDING = True
    ENABLE_CONSISTENCY_VALIDATION = True
    
    @classmethod
    def get_output_path(cls, suffix: str = "") -> Path:
        """Get output file path with optional suffix."""
        if suffix:
            return cls.DATA_DIR / f"qwen_dbr_training_logic_v9_{suffix}.jsonl"
        return cls.OUTPUT_FILE
