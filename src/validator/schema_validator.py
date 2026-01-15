"""
Schema validation module for training data samples.

Provides Pydantic models and validation logic for ensuring
generated training samples conform to the required schema
and quality standards.
"""

import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ScenarioType(Enum):
    """Type of training scenario."""
    RULES_BASED = "scenario_1"  # Code-driven Q&A
    DESIGN_BASED = "scenario_2"  # Architecture design


class Context(BaseModel):
    """
    Context information for a training sample.
    
    For Scenario 1 (rules-based): Includes file_path and code_snippet
    For Scenario 2 (design-based): Includes architecture_context
    """
    file_path: Optional[str] = None
    related_dbr: str
    code_snippet: Optional[str] = None
    architecture_context: Optional[str] = None
    design_standard: Optional[str] = None


class AutoProcessing(BaseModel):
    """
    Metadata about automated processing steps applied.
    """
    parser: str
    dbr_logic: Optional[str] = None
    data_cleaning: Optional[str] = None
    feature: Optional[str] = None


class DataQuality(BaseModel):
    """
    Quality indicators for the training sample.
    """
    consistency_check: bool = True
    language: str = "zh-cn"
    temperature: float = 0.7


class TrainingSample(BaseModel):
    """
    Complete training sample schema for JSONL output.
    
    This schema aligns with the design document specifications
    for both Scenario 1 (code-driven) and Scenario 2 (design-based)
    training data.
    """
    sample_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str
    context: Context
    auto_processing: AutoProcessing
    reasoning_trace: List[str]
    answer: str
    data_quality: DataQuality = Field(default_factory=DataQuality)
    
    @field_validator('instruction')
    @classmethod
    def instruction_not_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 5:
            raise ValueError('Instruction must be at least 5 characters')
        return v.strip()
    
    @field_validator('reasoning_trace')
    @classmethod
    def reasoning_not_empty(cls, v: List[str]) -> List[str]:
        if not v or len(v) == 0:
            raise ValueError('Reasoning trace must have at least one step')
        return v
    
    @field_validator('answer')
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 10:
            raise ValueError('Answer must be at least 10 characters')
        return v.strip()


@dataclass
class ValidationResult:
    """
    Result of a validation check.
    
    Attributes:
        valid: Whether the sample passed validation
        errors: List of validation error messages
        warnings: List of validation warnings
        sample_id: ID of the validated sample
    """
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sample_id: Optional[str] = None


class SchemaValidator:
    """
    Validator for training data samples.
    
    Provides comprehensive validation including:
    - Schema conformance
    - Data quality checks
    - DBR alignment verification
    - Scenario-specific requirements
    """
    
    # Forbidden words in questions (should not expose implementation details)
    FORBIDDEN_QUESTION_WORDS = [
        "login", "register", "update_current_user", "函数",
        "function", "method", "class"
    ]
    
    def __init__(
        self,
        scenario: ScenarioType = ScenarioType.RULES_BASED,
        strict_mode: bool = False
    ):
        """
        Initialize the validator.
        
        Args:
            scenario: The scenario type being validated
            strict_mode: If True, warnings become errors
        """
        self.scenario = scenario
        self.strict_mode = strict_mode
    
    def validate_sample(self, sample: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single training sample.
        
        Args:
            sample: The sample dictionary to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        result = ValidationResult(valid=True, sample_id=sample.get("sample_id"))
        
        # Try to parse with Pydantic model
        try:
            parsed = TrainingSample(**sample)
        except Exception as e:
            result.valid = False
            result.errors.append(f"Schema validation failed: {str(e)}")
            return result
        
        # Additional validation checks
        self._check_forbidden_words(sample, result)
        self._check_scenario_requirements(sample, result)
        self._check_reasoning_quality(sample, result)
        self._check_code_snippet_quality(sample, result)
        
        # In strict mode, warnings become errors
        if self.strict_mode and result.warnings:
            result.errors.extend(result.warnings)
            result.warnings = []
            result.valid = False
        
        return result
    
    def _check_forbidden_words(
        self,
        sample: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Check that the instruction doesn't contain forbidden words."""
        instruction = sample.get("instruction", "").lower()
        
        for word in self.FORBIDDEN_QUESTION_WORDS:
            if word.lower() in instruction:
                result.warnings.append(
                    f"Instruction contains implementation detail: '{word}'"
                )
    
    def _check_scenario_requirements(
        self,
        sample: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Check scenario-specific requirements."""
        context = sample.get("context", {})
        
        if self.scenario == ScenarioType.RULES_BASED:
            # Scenario 1 requires code_snippet
            if not context.get("code_snippet"):
                result.errors.append(
                    "Scenario 1 requires code_snippet in context"
                )
                result.valid = False
            
            if not context.get("file_path"):
                result.warnings.append(
                    "Scenario 1 should include file_path in context"
                )
        
        elif self.scenario == ScenarioType.DESIGN_BASED:
            # Scenario 2 requires architecture context
            if not context.get("architecture_context"):
                result.warnings.append(
                    "Scenario 2 should include architecture_context"
                )
    
    def _check_reasoning_quality(
        self,
        sample: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Check the quality of reasoning trace."""
        reasoning = sample.get("reasoning_trace", [])
        
        if len(reasoning) < 2:
            result.warnings.append(
                "Reasoning trace has fewer than 2 steps"
            )
        
        # Check for very short steps
        short_steps = [s for s in reasoning if len(s) < 10]
        if short_steps:
            result.warnings.append(
                f"Found {len(short_steps)} very short reasoning steps"
            )
    
    def _check_code_snippet_quality(
        self,
        sample: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Check the quality of code snippets."""
        context = sample.get("context", {})
        code = context.get("code_snippet", "")
        
        if self.scenario == ScenarioType.RULES_BASED and code:
            # Check for proper Python code indicators
            if not any(kw in code for kw in ['def ', 'class ', 'if ', 'async ', 'await']):
                result.warnings.append(
                    "Code snippet may not contain valid Python code"
                )
    
    def validate_batch(
        self,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate a batch of training samples.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Summary of validation results
        """
        results = []
        valid_count = 0
        error_count = 0
        warning_count = 0
        
        for sample in samples:
            result = self.validate_sample(sample)
            results.append(result)
            
            if result.valid:
                valid_count += 1
            else:
                error_count += 1
            
            warning_count += len(result.warnings)
        
        return {
            "total": len(samples),
            "valid": valid_count,
            "invalid": error_count,
            "warnings": warning_count,
            "results": results
        }


def create_rules_sample(
    instruction: str,
    reasoning_steps: List[str],
    code_snippet: str,
    answer: str,
    file_path: str,
    dbr_id: str = "DBR-01",
    intent_desc: str = "",
    language: str = "zh-cn",
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Factory function to create a Scenario 1 (rules-based) training sample.
    
    Args:
        instruction: The question/instruction
        reasoning_steps: List of reasoning step strings
        code_snippet: The relevant code
        answer: The answer text
        file_path: Path to the source file
        dbr_id: The DBR identifier
        intent_desc: Description of the intent
        language: Language code
        temperature: Generation temperature
        
    Returns:
        Dictionary conforming to TrainingSample schema
    """
    # Build combined answer with sections
    combined_answer = (
        "### 推理链与合规逻辑\n"
        + "\n".join([f"- {step}" for step in reasoning_steps])
        + "\n\n### 业务方案解答\n"
        + answer
        + "\n\n### 核心源代码实现\n"
        + f"```python\n{code_snippet}\n```"
    )
    
    return {
        "sample_id": str(uuid.uuid4()),
        "instruction": instruction,
        "context": {
            "file_path": file_path,
            "related_dbr": dbr_id,
            "code_snippet": code_snippet
        },
        "auto_processing": {
            "parser": "multilingual_evidence_aligned_parser",
            "dbr_logic": f"{dbr_id} Trigger: {intent_desc}",
            "data_cleaning": "Step-placeholder removal, Markdown code normalization"
        },
        "reasoning_trace": reasoning_steps,
        "answer": combined_answer,
        "data_quality": {
            "consistency_check": True,
            "language": language,
            "temperature": temperature
        }
    }


def create_design_sample(
    instruction: str,
    reasoning_steps: List[str],
    design_solution: str,
    dbr_id: str = "DBR-01",
    architecture_context: str = "FastAPI + SQLAlchemy + Repository Pattern",
    design_standard: str = "Security-First API Design",
    language: str = "zh-cn",
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Factory function to create a Scenario 2 (design-based) training sample.
    
    Args:
        instruction: The design requirement
        reasoning_steps: List of reasoning step strings
        design_solution: The design solution text
        dbr_id: The DBR identifier
        architecture_context: Description of the architecture
        design_standard: The design standard being followed
        language: Language code
        temperature: Generation temperature
        
    Returns:
        Dictionary conforming to TrainingSample schema
    """
    return {
        "sample_id": str(uuid.uuid4()),
        "instruction": instruction,
        "context": {
            "related_dbr": dbr_id,
            "architecture_context": architecture_context,
            "design_standard": design_standard
        },
        "auto_processing": {
            "parser": "design_logic_generator_v5",
            "feature": "Regex-based-formatting & Logic-pumping"
        },
        "reasoning_trace": reasoning_steps,
        "answer": design_solution,
        "data_quality": {
            "consistency_check": True,
            "language": language,
            "temperature": temperature
        }
    }
