"""
Q&A Generation Engine v9.0 - Modular Architecture

This package provides a modular, maintainable Q&A generation system
for training LLMs on code analysis tasks.

Modules:
- config: Configuration parameters
- models: Data models and enums
- utils: Utility modules (LLM client, diversity manager)
- layers: Core processing layers
    - deterministic: AST, Call Graph, DBR Mapping
    - user_perspective: Business context transformation
    - llm_enhancement: Question, Reasoning, Answer generation
    - quality_assurance: Validation and quality checks
    - execution_flow: Code semantics analysis (v8)
    - consistency_validator: Logical consistency (v8)
- main: Main orchestrator

Usage:
    from v9 import HybridQAOrchestrator
    
    orchestrator = HybridQAOrchestrator(metadata_path, ast_path)
    orchestrator.initialize()
    pairs = orchestrator.run_pipeline(questions_per_evidence=5)
"""

from .config import Config
from .models import (
    UserRole, QuestionType, ExecutionSemantics,
    CodeContext, DBRLogic, CodeFacts, BusinessContext,
    GeneratedQuestion, ValidationResult, QAPair
)
from .main import HybridQAOrchestrator

__version__ = Config.VERSION
__all__ = [
    "Config",
    "UserRole", "QuestionType", "ExecutionSemantics",
    "CodeContext", "DBRLogic", "CodeFacts", "BusinessContext",
    "GeneratedQuestion", "ValidationResult", "QAPair",
    "HybridQAOrchestrator",
]
