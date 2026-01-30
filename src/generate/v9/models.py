"""
Data models for Q&A Generation Engine v9.

Contains all dataclasses and enums used across modules.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


# ============================================================================
# Enums
# ============================================================================

class UserRole(str, Enum):
    """User roles for question generation."""
    END_USER = "end_user"
    PRODUCT_MANAGER = "product_manager"
    QA_ENGINEER = "qa_engineer"
    SECURITY_AUDITOR = "security_auditor"
    NEW_DEVELOPER = "new_developer"


class QuestionType(str, Enum):
    """Types of questions for diversity tracking."""
    TROUBLESHOOTING = "troubleshooting"
    UNDERSTANDING = "understanding"
    EDGE_CASE = "edge_case"
    SECURITY = "security"
    WHAT_IF = "what_if"
    COMPARISON = "comparison"
    VALIDATION = "validation"
    DEEP_ANALYSIS = "deep_analysis"


class ExecutionSemantics(str, Enum):
    """Code execution semantics."""
    SEQUENTIAL = "sequential"
    EARLY_EXIT = "early_exit"
    ATOMIC = "atomic"
    CONDITIONAL = "conditional"


# ============================================================================
# Deterministic Layer Models
# ============================================================================

@dataclass
class CodeContext:
    """Deterministic code context from AST analysis."""
    file_path: str
    function_name: str
    line_start: int
    line_end: int
    code_snippet: str
    source_hash: str
    related_elements: List[str]
    call_chain: List[str] = field(default_factory=list)


@dataclass
class DBRLogic:
    """DBR rule mapping data."""
    rule_id: str
    subcategory_id: str
    trigger_type: str
    weight: float
    trigger_conditions: List[str] = field(default_factory=list)
    matched_patterns: List[str] = field(default_factory=list)


# ============================================================================
# Execution Flow Models (v8)
# ============================================================================

@dataclass
class ExecutionStep:
    """A single step in execution flow."""
    order: int
    operation: str
    semantics: ExecutionSemantics
    is_termination_point: bool = False
    condition: Optional[str] = None


@dataclass
class CodeFacts:
    """Deterministic facts extracted from code."""
    execution_order: List[ExecutionStep]
    is_synchronous: bool
    has_early_exit: bool
    atomicity_type: str
    termination_points: List[str]
    validation_checks: List[str]
    forbidden_claims: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_synchronous": self.is_synchronous,
            "has_early_exit": self.has_early_exit,
            "atomicity_type": self.atomicity_type,
            "execution_order": [f"{s.order}.{s.operation}" for s in self.execution_order],
            "validation_checks": self.validation_checks,
            "termination_points": self.termination_points,
            "forbidden_claims": self.forbidden_claims,
        }


# ============================================================================
# Business Context Models
# ============================================================================

@dataclass
class BusinessContext:
    """Business-level context for question generation."""
    scenario_name: str
    business_flow: str
    user_experience: str
    edge_cases: List[str]
    security_concerns: List[str]
    code_behavior: str = ""
    language: str = "en"
    
    def to_dict(self) -> Dict:
        return {
            "scenario_name": self.scenario_name,
            "business_flow": self.business_flow,
            "user_experience": self.user_experience,
            "edge_cases": self.edge_cases,
            "security_concerns": self.security_concerns,
            "code_behavior": self.code_behavior,
            "language": self.language,
        }


# ============================================================================
# Generation Output Models
# ============================================================================

@dataclass
class GeneratedQuestion:
    """A generated question."""
    question_id: str
    question_text: str
    source: str  # "llm" or "fallback"
    role: str
    language: str
    question_type: Optional[QuestionType] = None


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    score: float
    issues: List[str]
    
    
@dataclass 
class QAPair:
    """A complete Q&A pair."""
    sample_id: str
    instruction: str
    context: Dict
    auto_processing: Dict
    reasoning_trace: List[str]
    answer: str
    data_quality: Dict
    
    def to_dict(self) -> Dict:
        return {
            "sample_id": self.sample_id,
            "instruction": self.instruction,
            "context": self.context,
            "auto_processing": self.auto_processing,
            "reasoning_trace": self.reasoning_trace,
            "answer": self.answer,
            "data_quality": self.data_quality,
        }
