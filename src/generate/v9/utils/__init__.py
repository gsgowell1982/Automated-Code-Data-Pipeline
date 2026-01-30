"""Utility modules for Q&A Generation Engine v9."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import OllamaClient
from utils.diversity import DiversityManager

__all__ = ["OllamaClient", "DiversityManager"]
