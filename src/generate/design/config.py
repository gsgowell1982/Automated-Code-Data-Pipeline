"""
Configuration for Design Q&A Generation Engine.

Scenario 2: Generate design solutions and recommendations
for implementing authentication features based on DBR-01.
"""

from pathlib import Path


class DesignConfig:
    """Global configuration for design Q&A generation."""
    
    VERSION = "1.0.0"
    SCENARIO = "design_solution"
    
    # Paths
    BASE_DIR = Path(__file__).parent.resolve()
    WORKSPACE_ROOT = BASE_DIR.parent.parent.parent
    DATA_DIR = WORKSPACE_ROOT / "data"
    REPOS_DIR = WORKSPACE_ROOT / "repos"
    V9_DIR = BASE_DIR.parent / "v9"
    
    # Input files
    RULE_METADATA_FILE = DATA_DIR / "dbr01_rule_metadata.json"
    
    # Output file
    OUTPUT_FILE = DATA_DIR / "qwen_dbr_design_data_v1.jsonl"
    
    # LLM Configuration (reuse from v9)
    OLLAMA_API = "http://localhost:11434/api/generate"
    MODEL_NAME = "qwen2.5:7b"
    LLM_TIMEOUT = 180
    LLM_RETRY_COUNT = 2
    
    # Temperature settings
    LLM_TEMPERATURE = {
        "question": 0.85,
        "design": 0.7,      # Design solutions need creativity but accuracy
        "reasoning": 0.5,
    }
    
    # Generation parameters
    SUPPORTED_LANGUAGES = ["en", "zh"]
    DEFAULT_QUESTIONS_PER_SCENARIO = 5
    DEFAULT_TOTAL_LIMIT = None
    
    # Diversity parameters
    SIMILARITY_THRESHOLD = 0.55
    
    # Design-specific settings
    DESIGN_ASPECTS = [
        "architecture",
        "security",
        "data_model",
        "api_design",
        "error_handling",
        "validation",
        "session_management",
        "best_practices",
    ]
