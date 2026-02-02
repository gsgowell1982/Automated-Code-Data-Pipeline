"""
Data models for Design Q&A Generation Engine.

Scenario 2: Design-focused models without code_snippet.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class DesignerRole(str, Enum):
    """Roles for design question generation."""
    BACKEND_DEVELOPER = "backend_developer"
    SYSTEM_ARCHITECT = "system_architect"
    SECURITY_ENGINEER = "security_engineer"
    TECH_LEAD = "tech_lead"
    API_DESIGNER = "api_designer"


class DesignAspect(str, Enum):
    """Aspects of system design."""
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    DATA_MODEL = "data_model"
    API_DESIGN = "api_design"
    ERROR_HANDLING = "error_handling"
    VALIDATION = "validation"
    SESSION_MANAGEMENT = "session_management"
    BEST_PRACTICES = "best_practices"


class QuestionIntent(str, Enum):
    """Intent types for design questions."""
    HOW_TO_DESIGN = "how_to_design"
    WHY_THIS_WAY = "why_this_way"
    WHAT_PATTERN = "what_pattern"
    TRADE_OFF = "trade_off"
    ALTERNATIVE = "alternative"
    BEST_PRACTICE = "best_practice"
    SECURITY_CONCERN = "security_concern"
    SCALABILITY = "scalability"


@dataclass
class DesignScenario:
    """A design scenario based on DBR rule."""
    scenario_id: str
    scenario_name: str
    description: str
    requirements: List[str]
    constraints: List[str]
    related_dbr: str
    design_aspects: List[DesignAspect]
    
    def to_dict(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "description": self.description,
            "requirements": self.requirements,
            "constraints": self.constraints,
            "related_dbr": self.related_dbr,
            "design_aspects": [a.value for a in self.design_aspects],
        }


@dataclass
class DesignSolution:
    """A design solution/recommendation."""
    approach: str
    rationale: str
    components: List[str]
    patterns_used: List[str]
    security_considerations: List[str]
    trade_offs: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "approach": self.approach,
            "rationale": self.rationale,
            "components": self.components,
            "patterns_used": self.patterns_used,
            "security_considerations": self.security_considerations,
            "trade_offs": self.trade_offs,
        }


@dataclass
class GeneratedDesignQuestion:
    """A generated design question."""
    question_id: str
    question_text: str
    source: str  # "llm" or "fallback"
    role: str
    intent: QuestionIntent
    aspect: DesignAspect
    language: str


@dataclass
class DesignQAPair:
    """A complete design Q&A pair (no code_snippet)."""
    sample_id: str
    instruction: str
    context: Dict  # file_path removed, has design_scenario instead
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
