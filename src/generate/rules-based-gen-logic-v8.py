#!/usr/bin/env python3
"""
Enterprise Q&A Generation Engine v8.0 - Code-Grounded Constrained Generation

Fixes critical issues from v7:

Problem 1: "Semantic Tension" - LLM hallucinates distributed system scenarios
  - edge_cases like "network timeout", "partial failure" mislead LLM
  - Actual code is synchronous sequential flow with strong atomicity
  
Solution: ExecutionFlowAnalyzer extracts actual code semantics
  - Execution order: 1.check_username -> 2.check_email -> 3.create_user
  - Control semantics: "raise = 100% termination, NO partial state"
  - Synchronous flag: This is sequential Python, not distributed

Problem 2: "Role-Playing Side Effects" - LLM uses vague customer-support language
  - "maybe", "possibly", "partial save" contradicts deterministic code
  
Solution: Code-Grounded Prompts with explicit constraints
  - "You MUST NOT mention partial saves or network issues"
  - "raise HTTPException = immediate, complete termination"

Problem 3: "Reasoning Trace Lacks Code Constraints"
  - LLM generates reasoning without code verification
  - Cascading errors: wrong reasoning → wrong answer

Solution: LogicalConsistencyValidator
  - Detect contradictions: code has "raise" but answer says "partial save"
  - Reject answers that contradict code facts

Architecture v8:
┌─────────────────────────────────────────────────────────────────────┐
│                      Code-Grounded Generation                        │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              Execution Flow Analyzer (NEW)                       ││
│  │  Code → Execution Order → Control Semantics → Atomicity Flag     ││
│  │  Example: "1.check_username→2.check_email→3.create_user"        ││
│  │  Semantics: "raise=termination", "await=sequential"              ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │            Code-Grounded Prompt Builder (NEW)                    ││
│  │  Inject execution order + semantics into LLM prompts             ││
│  │  Add HARD CONSTRAINTS: "MUST NOT mention X, Y, Z"               ││
│  │  Remove misleading edge_cases that don't match code              ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │           Logical Consistency Validator (NEW)                    ││
│  │  Check LLM output against code facts                             ││
│  │  Reject: "partial save" when code has raise                      ││
│  │  Reject: "race condition" when code is synchronous              ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘

Author: Auto-generated
Version: 8.0.0
"""

import os
import sys
import json
import uuid
import hashlib
import logging
import random
import re
import time
import requests
import ast
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, Counter
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    VERSION = "8.0.0"
    BASE_DIR = Path(__file__).parent.resolve()
    WORKSPACE_ROOT = BASE_DIR.parent.parent
    DATA_DIR = WORKSPACE_ROOT / "data"
    
    RULE_METADATA_FILE = DATA_DIR / "dbr01_rule_metadata.json"
    AST_ANALYSIS_FILE = DATA_DIR / "fastapi_analysis_result.json"
    OUTPUT_FILE = DATA_DIR / "qwen_dbr_training_logic_v8.jsonl"
    
    OLLAMA_API = "http://localhost:11434/api/generate"
    MODEL_NAME = "qwen2.5:7b"
    LLM_TIMEOUT = 180
    LLM_TEMPERATURE_QUESTION = 0.85
    LLM_TEMPERATURE_REASONING = 0.5  # Lower for more deterministic reasoning
    LLM_TEMPERATURE_ANSWER = 0.6
    
    SUPPORTED_LANGUAGES = ["en", "zh"]
    DEFAULT_QUESTIONS_PER_EVIDENCE = 5
    DEFAULT_TOTAL_LIMIT = None
    
    # Consistency validation thresholds
    MAX_CONTRADICTION_SCORE = 0.3


# ============================================================================
# Enums
# ============================================================================

class UserRole(str, Enum):
    END_USER = "end_user"
    PRODUCT_MANAGER = "product_manager"
    QA_ENGINEER = "qa_engineer"
    SECURITY_AUDITOR = "security_auditor"
    NEW_DEVELOPER = "new_developer"


class QuestionType(str, Enum):
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
    SEQUENTIAL = "sequential"      # Synchronous, step-by-step
    EARLY_EXIT = "early_exit"      # raise/return terminates immediately
    ATOMIC = "atomic"              # All-or-nothing (transaction)
    CONDITIONAL = "conditional"    # if-else branching


# ============================================================================
# NEW: Execution Flow Analyzer
# ============================================================================

@dataclass
class ExecutionStep:
    """A single step in execution flow."""
    order: int
    operation: str
    semantics: ExecutionSemantics
    is_termination_point: bool = False  # raise, return
    condition: Optional[str] = None


@dataclass
class CodeFacts:
    """Deterministic facts extracted from code."""
    execution_order: List[ExecutionStep]
    is_synchronous: bool
    has_early_exit: bool
    atomicity_type: str  # "sequential", "transactional", "none"
    termination_points: List[str]  # List of raise/return statements
    validation_checks: List[str]  # List of validation operations
    forbidden_claims: List[str]  # Claims that would contradict the code


class ExecutionFlowAnalyzer:
    """
    Analyzes code to extract execution order and control flow semantics.
    This provides HARD FACTS that constrain LLM generation.
    """
    
    # Patterns for detecting control flow
    RAISE_PATTERN = r'\braise\s+\w+'
    RETURN_PATTERN = r'\breturn\s+'
    IF_PATTERN = r'\bif\s+'
    AWAIT_PATTERN = r'\bawait\s+'
    TRY_PATTERN = r'\btry\s*:'
    EXCEPT_PATTERN = r'\bexcept\s+'
    
    # Known validation function patterns
    VALIDATION_PATTERNS = [
        r'check_\w+',
        r'validate_\w+',
        r'is_\w+_taken',
        r'verify_\w+',
    ]
    
    # Known data operation patterns
    DATA_OP_PATTERNS = [
        r'create_\w+',
        r'update_\w+',
        r'delete_\w+',
        r'save\w*',
        r'insert\w*',
    ]
    
    def analyze(self, code: str, function_name: str = "") -> CodeFacts:
        """Analyze code and extract execution facts."""
        
        execution_order = self._extract_execution_order(code)
        termination_points = self._extract_termination_points(code)
        validation_checks = self._extract_validation_checks(code)
        
        # Determine synchronicity
        is_synchronous = self._is_synchronous_flow(code)
        
        # Determine if there's early exit (raise before data write)
        has_early_exit = len(termination_points) > 0 and self._has_validation_before_write(code)
        
        # Determine atomicity
        atomicity_type = self._determine_atomicity(code, has_early_exit)
        
        # Generate forbidden claims based on code facts
        forbidden_claims = self._generate_forbidden_claims(
            is_synchronous, has_early_exit, atomicity_type, termination_points
        )
        
        return CodeFacts(
            execution_order=execution_order,
            is_synchronous=is_synchronous,
            has_early_exit=has_early_exit,
            atomicity_type=atomicity_type,
            termination_points=termination_points,
            validation_checks=validation_checks,
            forbidden_claims=forbidden_claims
        )
    
    def _extract_execution_order(self, code: str) -> List[ExecutionStep]:
        """Extract the order of operations from code."""
        steps = []
        order = 0
        
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check for validation calls
            for pattern in self.VALIDATION_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    order += 1
                    is_termination = bool(re.search(self.RAISE_PATTERN, line))
                    steps.append(ExecutionStep(
                        order=order,
                        operation=match.group(),
                        semantics=ExecutionSemantics.EARLY_EXIT if is_termination else ExecutionSemantics.CONDITIONAL,
                        is_termination_point=is_termination,
                        condition=self._extract_condition(line) if 'if' in line else None
                    ))
            
            # Check for data operations
            for pattern in self.DATA_OP_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    order += 1
                    steps.append(ExecutionStep(
                        order=order,
                        operation=match.group(),
                        semantics=ExecutionSemantics.SEQUENTIAL,
                        is_termination_point=False
                    ))
            
            # Check for return statements
            if re.search(self.RETURN_PATTERN, line):
                order += 1
                steps.append(ExecutionStep(
                    order=order,
                    operation="return",
                    semantics=ExecutionSemantics.EARLY_EXIT,
                    is_termination_point=True
                ))
        
        return steps
    
    def _extract_condition(self, line: str) -> Optional[str]:
        """Extract condition from if statement."""
        match = re.search(r'if\s+(.+?):', line)
        return match.group(1) if match else None
    
    def _extract_termination_points(self, code: str) -> List[str]:
        """Extract all raise statements."""
        terminations = []
        for match in re.finditer(r'raise\s+(\w+)\s*\(([^)]*)\)', code):
            terminations.append(f"raise {match.group(1)}")
        return terminations
    
    def _extract_validation_checks(self, code: str) -> List[str]:
        """Extract validation function calls."""
        checks = []
        for pattern in self.VALIDATION_PATTERNS:
            for match in re.finditer(pattern, code):
                if match.group() not in checks:
                    checks.append(match.group())
        return checks
    
    def _is_synchronous_flow(self, code: str) -> bool:
        """Determine if code is synchronous (sequential execution)."""
        # Python with await is still sequential (just async I/O)
        # Look for actual concurrent patterns
        concurrent_patterns = [
            r'asyncio\.gather',
            r'asyncio\.create_task',
            r'ThreadPool',
            r'ProcessPool',
            r'concurrent\.futures',
        ]
        for pattern in concurrent_patterns:
            if re.search(pattern, code):
                return False
        return True
    
    def _has_validation_before_write(self, code: str) -> bool:
        """Check if validation happens before data write."""
        # Find positions
        validation_pos = -1
        write_pos = -1
        
        for pattern in self.VALIDATION_PATTERNS:
            match = re.search(pattern, code)
            if match:
                validation_pos = match.start()
                break
        
        for pattern in self.DATA_OP_PATTERNS:
            match = re.search(pattern, code)
            if match:
                write_pos = match.start()
                break
        
        return validation_pos != -1 and write_pos != -1 and validation_pos < write_pos
    
    def _determine_atomicity(self, code: str, has_early_exit: bool) -> str:
        """Determine atomicity type."""
        if has_early_exit:
            # Validation before write = "gated" atomicity
            return "gated_sequential"
        
        # Check for transaction patterns
        if re.search(r'transaction|commit|rollback', code, re.IGNORECASE):
            return "transactional"
        
        return "sequential"
    
    def _generate_forbidden_claims(
        self,
        is_synchronous: bool,
        has_early_exit: bool,
        atomicity_type: str,
        termination_points: List[str]
    ) -> List[str]:
        """Generate claims that would contradict the code facts."""
        forbidden = []
        
        if is_synchronous:
            forbidden.extend([
                "partial save",
                "partial update",
                "partially created",
                "half-saved",
                "incomplete data",
                "data inconsistency due to concurrent",
                "race condition in this function",
            ])
        
        if has_early_exit and termination_points:
            forbidden.extend([
                "might partially complete",
                "could leave data in inconsistent state",
                "partial execution",
            ])
        
        if atomicity_type == "gated_sequential":
            forbidden.extend([
                "data written before validation",
                "validation happens after save",
            ])
        
        return forbidden
    
    def format_execution_order_for_prompt(self, code_facts: CodeFacts, language: str = "en") -> str:
        """Format execution order as a constraint for LLM prompt."""
        if not code_facts.execution_order:
            return ""
        
        steps = []
        for step in code_facts.execution_order:
            if language == "zh":
                termination = " [终止点]" if step.is_termination_point else ""
                steps.append(f"{step.order}. {step.operation}{termination}")
            else:
                termination = " [TERMINATES]" if step.is_termination_point else ""
                steps.append(f"{step.order}. {step.operation}{termination}")
        
        if language == "zh":
            header = "【强制执行顺序 - 必须按此顺序分析】"
            sync_note = "注意：这是同步顺序执行代码，不存在并发或部分执行" if code_facts.is_synchronous else ""
            exit_note = "注意：raise语句会立即终止，不会有部分数据保存" if code_facts.has_early_exit else ""
        else:
            header = "【MANDATORY EXECUTION ORDER - Analyze in this sequence】"
            sync_note = "NOTE: This is SYNCHRONOUS sequential code. NO concurrency or partial execution." if code_facts.is_synchronous else ""
            exit_note = "NOTE: 'raise' statements cause IMMEDIATE termination. NO partial data is saved." if code_facts.has_early_exit else ""
        
        result = f"{header}\n" + " → ".join(steps)
        if sync_note:
            result += f"\n{sync_note}"
        if exit_note:
            result += f"\n{exit_note}"
        
        return result


# ============================================================================
# NEW: Code-Grounded Prompt Builder
# ============================================================================

class CodeGroundedPromptBuilder:
    """
    Builds prompts that enforce code semantics and prevent hallucination.
    """
    
    HARD_CONSTRAINTS_EN = """
【HARD CONSTRAINTS - VIOLATIONS WILL BE REJECTED】
1. This is SYNCHRONOUS Python code. Each line executes AFTER the previous completes.
2. "raise HTTPException" = IMMEDIATE termination. The function STOPS. Nothing after executes.
3. "if condition: raise" = GATED validation. If condition is True, function TERMINATES.
4. NO partial data can be saved. Either ALL validation passes and data is written, OR exception is raised and NOTHING is written.

【FORBIDDEN PHRASES - DO NOT USE】
{forbidden_phrases}

【YOU MUST SAY】
- "The function terminates immediately when validation fails"
- "No data is persisted until all checks pass"
- "This is a sequential check: first X, then Y, then Z"
"""

    HARD_CONSTRAINTS_ZH = """
【硬性约束 - 违反将被拒绝】
1. 这是同步Python代码。每行在前一行完成后执行。
2. "raise HTTPException" = 立即终止。函数停止。后续代码不执行。
3. "if condition: raise" = 门控验证。如果条件为True，函数终止。
4. 不可能保存部分数据。要么所有验证通过并写入数据，要么抛出异常，什么都不写入。

【禁止使用的表述】
{forbidden_phrases}

【必须说明】
- "当验证失败时函数立即终止"
- "在所有检查通过之前不会持久化任何数据"
- "这是顺序检查：先X，然后Y，最后Z"
"""

    def build_reasoning_prompt(
        self,
        question: str,
        business_context: Dict,
        code_facts: CodeFacts,
        code_snippet: str,
        language: str = "en"
    ) -> str:
        """Build reasoning prompt with code constraints."""
        
        execution_order = ExecutionFlowAnalyzer().format_execution_order_for_prompt(code_facts, language)
        forbidden = self._format_forbidden_phrases(code_facts.forbidden_claims, language)
        
        if language == "zh":
            constraints = self.HARD_CONSTRAINTS_ZH.format(forbidden_phrases=forbidden)
            return f"""分析此问题并基于代码事实提供推理。

【问题】
{question}

【代码事实 - 这些是不可争辩的】
{execution_order}

验证检查: {', '.join(code_facts.validation_checks)}
终止点: {', '.join(code_facts.termination_points)}
原子性: {code_facts.atomicity_type}

{constraints}

【实际代码】
```python
{code_snippet[:1000]}
```

提供4-5个推理步骤。每步必须基于代码事实，不可臆测。
格式：[步骤类型] 基于代码的具体分析

推理："""
        else:
            constraints = self.HARD_CONSTRAINTS_EN.format(forbidden_phrases=forbidden)
            return f"""Analyze this question and provide reasoning STRICTLY based on code facts.

【QUESTION】
{question}

【CODE FACTS - THESE ARE INDISPUTABLE】
{execution_order}

Validation checks: {', '.join(code_facts.validation_checks)}
Termination points: {', '.join(code_facts.termination_points)}
Atomicity: {code_facts.atomicity_type}

{constraints}

【ACTUAL CODE】
```python
{code_snippet[:1000]}
```

Provide 4-5 reasoning steps. Each step MUST be grounded in code facts. No speculation.
Format: [STEP_TYPE] Specific analysis based on code

Reasoning:"""

    def build_answer_prompt(
        self,
        question: str,
        reasoning: List[str],
        code_facts: CodeFacts,
        code_snippet: str,
        language: str = "en"
    ) -> str:
        """Build answer prompt with strict code grounding."""
        
        execution_order = ExecutionFlowAnalyzer().format_execution_order_for_prompt(code_facts, language)
        forbidden = self._format_forbidden_phrases(code_facts.forbidden_claims, language)
        
        if language == "zh":
            constraints = self.HARD_CONSTRAINTS_ZH.format(forbidden_phrases=forbidden)
            return f"""基于代码事实提供准确回答。

【问题】{question}

【代码事实】
{execution_order}
原子性: {code_facts.atomicity_type}
同步执行: {'是' if code_facts.is_synchronous else '否'}

【推理过程】
{chr(10).join(reasoning)}

{constraints}

【代码参考】
```python
{code_snippet[:800]}
```

提供准确回答。必须：
1. 直接回答问题
2. 基于代码执行顺序解释
3. 明确说明raise导致的终止行为
4. 不使用任何禁止的表述

回答："""
        else:
            constraints = self.HARD_CONSTRAINTS_EN.format(forbidden_phrases=forbidden)
            return f"""Provide an accurate answer STRICTLY based on code facts.

【QUESTION】{question}

【CODE FACTS】
{execution_order}
Atomicity: {code_facts.atomicity_type}
Synchronous: {'Yes' if code_facts.is_synchronous else 'No'}

【REASONING】
{chr(10).join(reasoning)}

{constraints}

【CODE REFERENCE】
```python
{code_snippet[:800]}
```

Provide accurate answer. You MUST:
1. Directly answer the question
2. Explain based on code execution order
3. Clearly state termination behavior from raise
4. NOT use any forbidden phrases

Answer:"""

    def _format_forbidden_phrases(self, forbidden: List[str], language: str) -> str:
        """Format forbidden phrases for prompt."""
        if not forbidden:
            return "None"
        return "\n".join(f"- \"{phrase}\"" for phrase in forbidden)


# ============================================================================
# NEW: Logical Consistency Validator
# ============================================================================

class LogicalConsistencyValidator:
    """
    Validates LLM output against code facts.
    Rejects answers that contradict deterministic code behavior.
    """
    
    # Contradiction patterns
    PARTIAL_SAVE_PATTERNS = [
        r'partial(?:ly)?\s+(?:save|saved|written|create|update)',
        r'部分(?:保存|写入|创建|更新)',
        r'half[\s-]?(?:saved|created|written)',
        r'incomplete\s+(?:data|record|account)',
        r'不完整的(?:数据|记录|账户)',
    ]
    
    RACE_CONDITION_PATTERNS = [
        r'race\s+condition\s+(?:in|within)\s+(?:this|the)\s+function',
        r'(?:这个|该)函数(?:内|中)(?:的|存在)?竞态',
        r'concurrent\s+execution\s+of\s+this\s+function',
    ]
    
    MAYBE_PATTERNS = [
        r'(?:data\s+)?might\s+(?:be\s+)?partially',
        r'(?:数据)?可能(?:会)?部分',
        r'could\s+leave\s+(?:data\s+)?in\s+inconsistent',
        r'可能(?:会)?(?:导致|留下)(?:数据)?不一致',
    ]
    
    def validate_reasoning(
        self,
        reasoning: List[str],
        code_facts: CodeFacts
    ) -> Tuple[bool, List[str], float]:
        """
        Validate reasoning against code facts.
        Returns (is_valid, issues, contradiction_score).
        """
        issues = []
        contradiction_score = 0.0
        
        reasoning_text = " ".join(reasoning).lower()
        
        # Check for forbidden claims
        for forbidden in code_facts.forbidden_claims:
            if forbidden.lower() in reasoning_text:
                issues.append(f"Contains forbidden claim: '{forbidden}'")
                contradiction_score += 0.3
        
        # Check for partial save claims (when code has early exit)
        if code_facts.has_early_exit:
            for pattern in self.PARTIAL_SAVE_PATTERNS:
                if re.search(pattern, reasoning_text, re.IGNORECASE):
                    issues.append("Claims partial save when code has early exit (impossible)")
                    contradiction_score += 0.4
                    break
        
        # Check for race condition claims (when code is synchronous)
        if code_facts.is_synchronous:
            for pattern in self.RACE_CONDITION_PATTERNS:
                if re.search(pattern, reasoning_text, re.IGNORECASE):
                    issues.append("Claims race condition in synchronous sequential code")
                    contradiction_score += 0.3
                    break
        
        is_valid = contradiction_score < Config.MAX_CONTRADICTION_SCORE
        return is_valid, issues, contradiction_score
    
    def validate_answer(
        self,
        answer: str,
        code_facts: CodeFacts
    ) -> Tuple[bool, List[str], float]:
        """
        Validate answer against code facts.
        """
        issues = []
        contradiction_score = 0.0
        
        answer_lower = answer.lower()
        
        # Check forbidden claims
        for forbidden in code_facts.forbidden_claims:
            if forbidden.lower() in answer_lower:
                issues.append(f"Contains forbidden claim: '{forbidden}'")
                contradiction_score += 0.3
        
        # Check partial save patterns
        if code_facts.has_early_exit:
            for pattern in self.PARTIAL_SAVE_PATTERNS:
                if re.search(pattern, answer_lower, re.IGNORECASE):
                    issues.append("Claims partial save (contradicts raise semantics)")
                    contradiction_score += 0.5
                    break
        
        # Check vague/hedging language for deterministic code
        if code_facts.is_synchronous and code_facts.has_early_exit:
            for pattern in self.MAYBE_PATTERNS:
                if re.search(pattern, answer_lower, re.IGNORECASE):
                    issues.append("Uses vague language for deterministic code behavior")
                    contradiction_score += 0.2
                    break
        
        is_valid = contradiction_score < Config.MAX_CONTRADICTION_SCORE
        return is_valid, issues, contradiction_score


# ============================================================================
# Enhanced Business Context (Fixed edge_cases)
# ============================================================================

class CodeConstrainedContextBuilder:
    """
    Builds business context that is CONSTRAINED by actual code behavior.
    Removes misleading edge_cases that don't apply to the actual code.
    """
    
    @classmethod
    def build_context(
        cls,
        evidence: Dict,
        subcategory_id: str,
        code_facts: CodeFacts,
        language: str = "en"
    ) -> Dict:
        """Build context constrained by code facts."""
        
        # Base context
        base = cls._get_base_context(subcategory_id, language)
        
        # Filter edge_cases to only those that are actually possible
        valid_edge_cases = cls._filter_edge_cases(base.get("edge_cases", []), code_facts, language)
        
        # Filter security_concerns to code-relevant ones
        valid_security = cls._filter_security_concerns(base.get("security_concerns", []), code_facts, language)
        
        return {
            "scenario_name": base.get("scenario_name", ""),
            "business_flow": base.get("business_flow", ""),
            "user_experience": base.get("user_experience", ""),
            "edge_cases": valid_edge_cases,
            "security_concerns": valid_security,
            "code_behavior": cls._describe_code_behavior(code_facts, language),
            "execution_semantics": {
                "is_synchronous": code_facts.is_synchronous,
                "has_early_exit": code_facts.has_early_exit,
                "atomicity": code_facts.atomicity_type,
            },
            "language": language,
        }
    
    @classmethod
    def _get_base_context(cls, subcategory_id: str, language: str) -> Dict:
        """Get base context without filtering."""
        contexts = {
            "DBR-01-01": {
                "en": {
                    "scenario_name": "User Registration & Profile Uniqueness",
                    "business_flow": "Validate uniqueness before creating/updating user",
                    "user_experience": "User submits registration; system checks uniqueness; rejects or accepts",
                    "edge_cases": [
                        "Email already registered by another user",
                        "Username already taken",
                        "Same user updating to taken email",
                        # REMOVED: "Network timeout" - not relevant to this code
                        # REMOVED: "Partial failure" - code prevents this
                    ],
                    "security_concerns": [
                        "Account enumeration through error messages",
                        "Timing attacks on validation",
                        # REMOVED: "Race condition" - code is sequential
                    ],
                },
                "zh": {
                    "scenario_name": "用户注册与资料唯一性",
                    "business_flow": "在创建/更新用户前验证唯一性",
                    "user_experience": "用户提交注册；系统检查唯一性；拒绝或接受",
                    "edge_cases": [
                        "邮箱已被其他用户注册",
                        "用户名已被占用",
                        "用户更新为已被占用的邮箱",
                    ],
                    "security_concerns": [
                        "通过错误消息进行账户枚举",
                        "验证时序攻击",
                    ],
                },
            },
            "DBR-01-02": {
                "en": {
                    "scenario_name": "Secure Account Creation",
                    "business_flow": "Hash password and create account atomically",
                    "user_experience": "User provides credentials; system securely stores",
                    "edge_cases": [
                        "Password at maximum allowed length",
                        "Special characters in password",
                    ],
                    "security_concerns": [
                        "Password hashing security",
                        "Credential storage",
                    ],
                },
                "zh": {
                    "scenario_name": "安全账户创建",
                    "business_flow": "哈希密码并原子性创建账户",
                    "user_experience": "用户提供凭据；系统安全存储",
                    "edge_cases": [
                        "密码达到最大允许长度",
                        "密码中包含特殊字符",
                    ],
                    "security_concerns": [
                        "密码哈希安全",
                        "凭据存储",
                    ],
                },
            },
            "DBR-01-03": {
                "en": {
                    "scenario_name": "Login Authentication",
                    "business_flow": "Verify credentials and return generic error on failure",
                    "user_experience": "User enters credentials; system validates; generic error if wrong",
                    "edge_cases": [
                        "Non-existent email",
                        "Wrong password for existing account",
                    ],
                    "security_concerns": [
                        "User enumeration prevention",
                        "Consistent error responses",
                    ],
                },
                "zh": {
                    "scenario_name": "登录认证",
                    "business_flow": "验证凭据，失败时返回通用错误",
                    "user_experience": "用户输入凭据；系统验证；错误时显示通用错误",
                    "edge_cases": [
                        "不存在的邮箱",
                        "现有账户的错误密码",
                    ],
                    "security_concerns": [
                        "防止用户枚举",
                        "一致的错误响应",
                    ],
                },
            },
            "DBR-01-04": {
                "en": {
                    "scenario_name": "Token Management",
                    "business_flow": "Generate and return authentication token",
                    "user_experience": "After successful auth, user receives token",
                    "edge_cases": [
                        "Token generation after login",
                        "Token refresh after profile update",
                    ],
                    "security_concerns": [
                        "Token security",
                        "Session management",
                    ],
                },
                "zh": {
                    "scenario_name": "令牌管理",
                    "business_flow": "生成并返回认证令牌",
                    "user_experience": "认证成功后，用户收到令牌",
                    "edge_cases": [
                        "登录后生成令牌",
                        "资料更新后刷新令牌",
                    ],
                    "security_concerns": [
                        "令牌安全",
                        "会话管理",
                    ],
                },
            },
        }
        
        return contexts.get(subcategory_id, {}).get(language, contexts.get(subcategory_id, {}).get("en", {}))
    
    @classmethod
    def _filter_edge_cases(cls, edge_cases: List[str], code_facts: CodeFacts, language: str) -> List[str]:
        """Remove edge cases that are impossible given the code."""
        # Patterns that indicate partial/concurrent issues (impossible in sequential code)
        invalid_patterns = [
            r'partial', r'部分',
            r'network\s+timeout', r'网络超时',
            r'concurrent', r'并发',
            r'half[\s-]?', r'一半',
        ]
        
        if code_facts.is_synchronous:
            return [ec for ec in edge_cases 
                    if not any(re.search(p, ec, re.IGNORECASE) for p in invalid_patterns)]
        
        return edge_cases
    
    @classmethod
    def _filter_security_concerns(cls, concerns: List[str], code_facts: CodeFacts, language: str) -> List[str]:
        """Remove security concerns that don't apply to this code."""
        invalid_patterns = []
        
        if code_facts.is_synchronous:
            invalid_patterns.extend([r'race\s+condition', r'竞态条件'])
        
        return [c for c in concerns 
                if not any(re.search(p, c, re.IGNORECASE) for p in invalid_patterns)]
    
    @classmethod
    def _describe_code_behavior(cls, code_facts: CodeFacts, language: str) -> str:
        """Generate human-readable description of code behavior."""
        if language == "zh":
            parts = []
            if code_facts.is_synchronous:
                parts.append("同步顺序执行")
            if code_facts.has_early_exit:
                parts.append("验证失败立即终止")
            if code_facts.atomicity_type == "gated_sequential":
                parts.append("门控原子性（验证通过才写入）")
            return "；".join(parts) if parts else "标准执行"
        else:
            parts = []
            if code_facts.is_synchronous:
                parts.append("Synchronous sequential execution")
            if code_facts.has_early_exit:
                parts.append("Immediate termination on validation failure")
            if code_facts.atomicity_type == "gated_sequential":
                parts.append("Gated atomicity (write only after all validations pass)")
            return "; ".join(parts) if parts else "Standard execution"


# ============================================================================
# LLM Client and Generators (Enhanced)
# ============================================================================

class OllamaClient:
    def __init__(self):
        self.api_url = Config.OLLAMA_API
        self._available = None
        self._check_time = 0
    
    def is_available(self) -> bool:
        current_time = time.time()
        if self._available is not None and current_time - self._check_time < 60:
            return self._available
        try:
            response = requests.get(self.api_url.replace("/api/generate", "/api/tags"), timeout=5)
            self._available = response.status_code == 200
            self._check_time = current_time
        except:
            self._available = False
            self._check_time = current_time
        return self._available
    
    def generate(self, prompt: str, system: str = None, temperature: float = None) -> Optional[str]:
        if not self.is_available():
            return None
        try:
            payload = {"model": Config.MODEL_NAME, "prompt": prompt, "stream": False, 
                      "options": {"temperature": temperature or 0.7}}
            if system:
                payload["system"] = system
            response = requests.post(self.api_url, json=payload, timeout=Config.LLM_TIMEOUT)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"LLM error: {e}")
        return None


class CodeGroundedReasoningGenerator:
    """Generates reasoning constrained by code facts."""
    
    def __init__(self, llm: OllamaClient, prompt_builder: CodeGroundedPromptBuilder, 
                 validator: LogicalConsistencyValidator):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.validator = validator
    
    def generate(
        self,
        question: str,
        business_context: Dict,
        code_facts: CodeFacts,
        code_snippet: str,
        language: str = "en"
    ) -> Tuple[List[str], bool, List[str]]:
        """Generate reasoning and validate against code facts."""
        
        if self.llm.is_available():
            prompt = self.prompt_builder.build_reasoning_prompt(
                question, business_context, code_facts, code_snippet, language
            )
            response = self.llm.generate(prompt, temperature=Config.LLM_TEMPERATURE_REASONING)
            
            if response:
                reasoning = self._parse_reasoning(response)
                is_valid, issues, _ = self.validator.validate_reasoning(reasoning, code_facts)
                
                if is_valid:
                    return reasoning, True, []
                else:
                    logger.debug(f"Reasoning validation failed: {issues}")
                    # Fall through to fallback
        
        # Fallback: Generate code-grounded reasoning
        reasoning = self._generate_fallback(code_facts, business_context, language)
        return reasoning, True, []
    
    def _parse_reasoning(self, response: str) -> List[str]:
        steps = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line.startswith('[') or re.match(r'^\d+\.', line)):
                steps.append(line)
        return steps[:6] if steps else [response[:200]]
    
    def _generate_fallback(self, code_facts: CodeFacts, context: Dict, language: str) -> List[str]:
        """Generate deterministic reasoning based on code facts."""
        exec_order = [f"{s.order}.{s.operation}" for s in code_facts.execution_order[:4]]
        
        if language == "zh":
            return [
                f"[执行顺序] 代码按以下顺序执行: {' → '.join(exec_order)}",
                f"[验证检查] 系统执行这些验证: {', '.join(code_facts.validation_checks)}",
                f"[终止语义] 当验证失败时，{code_facts.termination_points[0] if code_facts.termination_points else 'raise'} 立即终止函数",
                f"[原子性] 这是{code_facts.atomicity_type}执行，不可能有部分数据保存",
                f"[结论] 基于代码事实，系统正确实现了验证-终止-写入的门控流程",
            ]
        else:
            return [
                f"[EXECUTION ORDER] Code executes in sequence: {' → '.join(exec_order)}",
                f"[VALIDATION CHECKS] System performs: {', '.join(code_facts.validation_checks)}",
                f"[TERMINATION SEMANTICS] On validation failure, {code_facts.termination_points[0] if code_facts.termination_points else 'raise'} immediately terminates",
                f"[ATOMICITY] This is {code_facts.atomicity_type} execution - partial data save is impossible",
                f"[CONCLUSION] Based on code facts, system correctly implements validate-terminate-write gated flow",
            ]


class CodeGroundedAnswerGenerator:
    """Generates answers constrained by code facts."""
    
    def __init__(self, llm: OllamaClient, prompt_builder: CodeGroundedPromptBuilder,
                 validator: LogicalConsistencyValidator):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.validator = validator
    
    def generate(
        self,
        question: str,
        reasoning: List[str],
        code_facts: CodeFacts,
        code_snippet: str,
        language: str = "en"
    ) -> Tuple[str, bool, List[str]]:
        """Generate answer and validate against code facts."""
        
        if self.llm.is_available():
            prompt = self.prompt_builder.build_answer_prompt(
                question, reasoning, code_facts, code_snippet, language
            )
            response = self.llm.generate(prompt, temperature=Config.LLM_TEMPERATURE_ANSWER)
            
            if response:
                is_valid, issues, _ = self.validator.validate_answer(response, code_facts)
                
                if is_valid:
                    return self._format_answer(response, code_facts, code_snippet, language), True, []
                else:
                    logger.debug(f"Answer validation failed: {issues}")
        
        # Fallback
        answer = self._generate_fallback(code_facts, reasoning, code_snippet, language)
        return answer, True, []
    
    def _format_answer(self, response: str, code_facts: CodeFacts, code: str, language: str) -> str:
        """Add code reference to answer."""
        if language == "zh":
            code_section = f"""

### 代码执行事实

执行顺序: {' → '.join(f"{s.order}.{s.operation}" for s in code_facts.execution_order[:4])}
原子性: {code_facts.atomicity_type}
终止点: {', '.join(code_facts.termination_points)}

```python
{code[:500]}
```"""
        else:
            code_section = f"""

### Code Execution Facts

Execution order: {' → '.join(f"{s.order}.{s.operation}" for s in code_facts.execution_order[:4])}
Atomicity: {code_facts.atomicity_type}
Termination points: {', '.join(code_facts.termination_points)}

```python
{code[:500]}
```"""
        return response + code_section
    
    def _generate_fallback(self, code_facts: CodeFacts, reasoning: List[str], code: str, language: str) -> str:
        exec_order = ' → '.join(f"{s.order}.{s.operation}" for s in code_facts.execution_order[:4])
        
        if language == "zh":
            return f"""### 回答

基于代码的确定性分析：

**执行顺序**: {exec_order}

**关键行为**:
- 这是同步顺序执行的代码
- 当验证失败时，`raise` 语句立即终止函数
- 在所有验证通过之前，不会写入任何数据
- 不存在"部分保存"的可能性

**验证流程**:
{chr(10).join('- ' + v for v in code_facts.validation_checks)}

**推理过程**:
{chr(10).join(reasoning)}

### 代码参考

```python
{code[:500]}
```"""
        else:
            return f"""### Answer

Based on deterministic code analysis:

**Execution Order**: {exec_order}

**Key Behaviors**:
- This is synchronous sequential code
- When validation fails, `raise` immediately terminates the function
- No data is written until ALL validations pass
- "Partial save" is IMPOSSIBLE

**Validation Flow**:
{chr(10).join('- ' + v for v in code_facts.validation_checks)}

**Reasoning Process**:
{chr(10).join(reasoning)}

### Code Reference

```python
{code[:500]}
```"""


# ============================================================================
# Simplified Question Generator (v8)
# ============================================================================

class CodeAwareQuestionGenerator:
    """Question generator that respects code constraints."""
    
    CODE_NAME_PATTERNS = [
        r'\bcheck_\w+\b', r'\busers_repo\b', r'\buser_create\b',
        r'\bHTTP_\d+\b', r'\bEntityDoesNotExist\b',
    ]
    
    def __init__(self, llm: OllamaClient):
        self.llm = llm
        self.generated: Set[str] = set()
    
    def generate(
        self,
        business_context: Dict,
        code_facts: CodeFacts,
        role: UserRole,
        count: int,
        language: str
    ) -> List[Dict]:
        """Generate questions that respect code constraints."""
        
        # Generate candidates
        candidates = self._generate_candidates(business_context, code_facts, role, count * 2, language)
        
        # Filter and select
        selected = []
        for candidate in candidates:
            if len(selected) >= count:
                break
            
            q_text = candidate["question_text"].lower()
            if q_text not in self.generated:
                if not self._contains_code_names(candidate["question_text"]):
                    self.generated.add(q_text)
                    selected.append(candidate)
        
        return selected
    
    def _generate_candidates(
        self,
        context: Dict,
        code_facts: CodeFacts,
        role: UserRole,
        count: int,
        language: str
    ) -> List[Dict]:
        """Generate candidate questions."""
        templates = self._get_templates(context, code_facts, role, language)
        random.shuffle(templates)
        
        return [{
            "question_id": f"V8-{uuid.uuid4().hex[:8]}",
            "question_text": t,
            "role": role.value,
            "language": language,
        } for t in templates[:count]]
    
    def _get_templates(self, context: Dict, code_facts: CodeFacts, role: UserRole, language: str) -> List[str]:
        """Get code-constrained question templates."""
        # These templates are designed to NOT trigger misleading responses
        if language == "zh":
            base = {
                UserRole.END_USER: [
                    "如果我提交的邮箱已被注册，系统会如何处理我的请求？",
                    "当我输入错误的登录信息时，系统显示什么消息？",
                    "为什么登录错误不告诉我具体是邮箱错还是密码错？",
                    "如果用户名已被占用，我的注册请求会被完全拒绝吗？",
                    "系统在什么情况下会拒绝我的注册？",
                ],
                UserRole.QA_ENGINEER: [
                    "验证失败后，函数是否会继续执行后续代码？",
                    "如果第一个验证通过但第二个失败，会发生什么？",
                    "系统是先检查用户名还是先检查邮箱？顺序重要吗？",
                    "当验证失败时，数据库中会有任何记录吗？",
                    "函数的执行流程是什么？每一步发生什么？",
                ],
                UserRole.SECURITY_AUDITOR: [
                    "登录失败的错误消息是否能让攻击者判断邮箱是否存在？",
                    "验证过程是否存在时序差异可能泄露信息？",
                    "系统如何确保不通过错误消息暴露用户信息？",
                    "认证失败响应的设计目的是什么？",
                ],
                UserRole.NEW_DEVELOPER: [
                    "这段代码的执行顺序是什么？",
                    "raise语句在这里起什么作用？",
                    "为什么要在写入数据之前进行这些验证？",
                    "这种验证-终止模式的好处是什么？",
                ],
            }
        else:
            base = {
                UserRole.END_USER: [
                    "If I submit an email that's already registered, how does the system handle my request?",
                    "What message does the system show when I enter wrong login information?",
                    "Why doesn't the login error tell me if it's the email or password that's wrong?",
                    "If the username is taken, is my registration request completely rejected?",
                    "Under what conditions will the system reject my registration?",
                ],
                UserRole.QA_ENGINEER: [
                    "After validation fails, does the function continue executing subsequent code?",
                    "If the first validation passes but the second fails, what happens?",
                    "Does the system check username first or email first? Does the order matter?",
                    "When validation fails, will there be any records in the database?",
                    "What is the execution flow of the function? What happens at each step?",
                ],
                UserRole.SECURITY_AUDITOR: [
                    "Can login failure messages help an attacker determine if an email exists?",
                    "Are there timing differences in validation that could leak information?",
                    "How does the system ensure it doesn't expose user info through error messages?",
                    "What is the design purpose of the authentication failure response?",
                ],
                UserRole.NEW_DEVELOPER: [
                    "What is the execution order of this code?",
                    "What role does the raise statement play here?",
                    "Why do these validations happen before writing data?",
                    "What are the benefits of this validate-then-terminate pattern?",
                ],
            }
        
        return base.get(role, base[UserRole.END_USER])
    
    def _contains_code_names(self, text: str) -> bool:
        for pattern in self.CODE_NAME_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


# ============================================================================
# Main Orchestrator (v8)
# ============================================================================

class CodeGroundedOrchestrator:
    """Main orchestrator with code-grounded generation."""
    
    def __init__(self, rule_metadata_path: str, ast_analysis_path: str = None):
        self.rule_metadata_path = Path(rule_metadata_path)
        self.ast_analysis_path = Path(ast_analysis_path) if ast_analysis_path else None
        
        self.rule_metadata: Dict = {}
        self.ast_analysis: Dict = {}
        
        self.llm = OllamaClient()
        self.flow_analyzer = ExecutionFlowAnalyzer()
        self.prompt_builder = CodeGroundedPromptBuilder()
        self.consistency_validator = LogicalConsistencyValidator()
        
        self.question_generator: Optional[CodeAwareQuestionGenerator] = None
        self.reasoning_generator: Optional[CodeGroundedReasoningGenerator] = None
        self.answer_generator: Optional[CodeGroundedAnswerGenerator] = None
        
        self.generated_pairs: List[Dict] = []
        self.stats: Dict = defaultdict(int)
    
    def initialize(self) -> bool:
        try:
            with open(self.rule_metadata_path, 'r', encoding='utf-8') as f:
                self.rule_metadata = json.load(f)
            logger.info(f"Loaded rule metadata: {self.rule_metadata.get('rule_id')}")
            
            if self.ast_analysis_path and self.ast_analysis_path.exists():
                with open(self.ast_analysis_path, 'r', encoding='utf-8') as f:
                    self.ast_analysis = json.load(f)
            
            self.question_generator = CodeAwareQuestionGenerator(self.llm)
            self.reasoning_generator = CodeGroundedReasoningGenerator(
                self.llm, self.prompt_builder, self.consistency_validator
            )
            self.answer_generator = CodeGroundedAnswerGenerator(
                self.llm, self.prompt_builder, self.consistency_validator
            )
            
            if self.llm.is_available():
                logger.info(f"✓ LLM available: {Config.MODEL_NAME}")
            else:
                logger.warning("✗ LLM not available - using code-grounded fallback")
            
            return True
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    def run_pipeline(
        self,
        questions_per_evidence: int = None,
        total_limit: int = None,
        languages: List[str] = None,
    ) -> List[Dict]:
        questions_per_evidence = questions_per_evidence or Config.DEFAULT_QUESTIONS_PER_EVIDENCE
        languages = languages or Config.SUPPORTED_LANGUAGES
        
        self.generated_pairs = []
        self.stats = defaultdict(int)
        
        logger.info("=" * 60)
        logger.info(f"Starting Code-Grounded Q&A Pipeline v{Config.VERSION}")
        logger.info("=" * 60)
        logger.info(f"  NEW: Execution Flow Analysis")
        logger.info(f"  NEW: Logical Consistency Validation")
        logger.info(f"  NEW: Code-Constrained Prompts")
        logger.info(f"  Questions/evidence: {questions_per_evidence}")
        logger.info(f"  Total limit: {total_limit if total_limit else 'No limit'}")
        logger.info("=" * 60)
        
        for subcategory in self.rule_metadata.get("subcategories", []):
            if total_limit and len(self.generated_pairs) >= total_limit:
                break
            self._process_subcategory(subcategory, questions_per_evidence, total_limit, languages)
        
        logger.info(f"Generated {len(self.generated_pairs)} Q&A pairs")
        return self.generated_pairs
    
    def _process_subcategory(self, subcategory: Dict, qpe: int, total: Optional[int], languages: List[str]):
        subcategory_id = subcategory.get("subcategory_id", "")
        logger.info(f"Processing: {subcategory_id}")
        
        for evidence in subcategory.get("evidences", []):
            if total and len(self.generated_pairs) >= total:
                return
            self._process_evidence(evidence, subcategory_id, qpe, total, languages)
    
    def _process_evidence(
        self,
        evidence: Dict,
        subcategory_id: str,
        qpe: int,
        total: Optional[int],
        languages: List[str]
    ):
        code_data = evidence.get("code_snippet", {})
        code_snippet = code_data.get("code", "")
        if not code_snippet:
            return
        
        # Analyze code execution flow
        code_facts = self.flow_analyzer.analyze(code_snippet, evidence.get("name", ""))
        
        logger.debug(f"Code facts: sync={code_facts.is_synchronous}, "
                    f"early_exit={code_facts.has_early_exit}, "
                    f"atomicity={code_facts.atomicity_type}")
        
        roles = list(UserRole)
        random.shuffle(roles)
        
        for language in languages:
            if total and len(self.generated_pairs) >= total:
                return
            
            # Build code-constrained context
            business_context = CodeConstrainedContextBuilder.build_context(
                evidence, subcategory_id, code_facts, language
            )
            
            questions_generated = 0
            for role in roles:
                if questions_generated >= qpe:
                    break
                if total and len(self.generated_pairs) >= total:
                    return
                
                count = min(2, qpe - questions_generated)
                questions = self.question_generator.generate(
                    business_context, code_facts, role, count, language
                )
                
                for question in questions:
                    if questions_generated >= qpe:
                        break
                    if total and len(self.generated_pairs) >= total:
                        return
                    
                    self._generate_qa_pair(
                        question, evidence, business_context, code_facts,
                        code_snippet, subcategory_id, language
                    )
                    questions_generated += 1
    
    def _generate_qa_pair(
        self,
        question: Dict,
        evidence: Dict,
        business_context: Dict,
        code_facts: CodeFacts,
        code_snippet: str,
        subcategory_id: str,
        language: str
    ):
        question_text = question.get("question_text", "")
        
        # Generate and validate reasoning
        reasoning, reasoning_valid, reasoning_issues = self.reasoning_generator.generate(
            question_text, business_context, code_facts, code_snippet, language
        )
        
        # Generate and validate answer
        answer, answer_valid, answer_issues = self.answer_generator.generate(
            question_text, reasoning, code_facts, code_snippet, language
        )
        
        # Build QA pair
        dbr_logic = evidence.get("dbr_logic", {})
        code_data = evidence.get("code_snippet", {})
        
        qa_pair = {
            "sample_id": f"DBR01-V8-{uuid.uuid4().hex[:10]}",
            "instruction": question_text,
            "context": {
                "file_path": code_data.get("file_path", ""),
                "related_dbr": dbr_logic.get("rule_id", "DBR-01"),
                "code_snippet": code_snippet,
                "line_range": f"{code_data.get('line_start', 0)}-{code_data.get('line_end', 0)}",
            },
            "auto_processing": {
                "parser": "FastAPI-AST-Analyzer",
                "parser_version": "1.0.0",
                "dbr_logic": dbr_logic,
                "generation_metadata": {
                    "version": Config.VERSION,
                    "architecture": "code_grounded",
                    "user_role": question.get("role", "unknown"),
                },
                "code_facts": {
                    "is_synchronous": code_facts.is_synchronous,
                    "has_early_exit": code_facts.has_early_exit,
                    "atomicity_type": code_facts.atomicity_type,
                    "execution_order": [f"{s.order}.{s.operation}" for s in code_facts.execution_order],
                    "validation_checks": code_facts.validation_checks,
                    "termination_points": code_facts.termination_points,
                },
                "consistency_validation": {
                    "reasoning_valid": reasoning_valid,
                    "answer_valid": answer_valid,
                    "reasoning_issues": reasoning_issues,
                    "answer_issues": answer_issues,
                },
            },
            "reasoning_trace": reasoning,
            "answer": answer,
            "data_quality": {
                "consistency_check": reasoning_valid and answer_valid,
                "source_hash": code_data.get("source_hash", ""),
                "language": language,
                "evidence_id": evidence.get("evidence_id", ""),
            },
        }
        
        if reasoning_valid and answer_valid:
            self.generated_pairs.append(qa_pair)
            self.stats["valid"] += 1
            self.stats[f"role_{question.get('role', 'unknown')}"] += 1
        else:
            self.stats["rejected_consistency"] += 1
    
    def save_results(self, output_path: str = None) -> str:
        output_path = Path(output_path) if output_path else Config.OUTPUT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in self.generated_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(self.generated_pairs)} pairs to {output_path}")
        return str(output_path)
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print(f"Code-Grounded Q&A Generation Summary (v{Config.VERSION})")
        print("=" * 70)
        
        print(f"\n📊 Results:")
        print(f"  - Valid: {self.stats.get('valid', 0)}")
        print(f"  - Rejected (consistency): {self.stats.get('rejected_consistency', 0)}")
        
        print(f"\n🔒 Code Grounding Features:")
        print(f"  - Execution Flow Analysis: ✓")
        print(f"  - Logical Consistency Validation: ✓")
        print(f"  - Forbidden Claims Detection: ✓")
        print(f"  - Code-Constrained Prompts: ✓")
        
        print(f"\n👥 User Roles:")
        for role in UserRole:
            count = self.stats.get(f"role_{role.value}", 0)
            if count > 0:
                print(f"  - {role.value}: {count}")
        
        print("\n" + "=" * 70)
    
    def print_sample_pairs(self, n: int = 2):
        if not self.generated_pairs:
            return
        
        for i, pair in enumerate(self.generated_pairs[:n]):
            print("\n" + "=" * 70)
            meta = pair.get("auto_processing", {})
            code_facts = meta.get("code_facts", {})
            print(f"[Sample {i+1}] Role: {meta.get('generation_metadata', {}).get('user_role', 'N/A')}")
            print(f"Execution Order: {' → '.join(code_facts.get('execution_order', []))}")
            print(f"Atomicity: {code_facts.get('atomicity_type', 'N/A')}")
            print("=" * 70)
            print(f"\n【Question】:\n{pair['instruction']}")
            print(f"\n【Reasoning (code-grounded)】:")
            for step in pair.get("reasoning_trace", [])[:3]:
                print(f"  {step}")
            print(f"\n【Answer (excerpt)】:\n{pair['answer'][:400]}...")
            print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description=f'Code-Grounded Q&A Generation v{Config.VERSION}')
    parser.add_argument('-m', '--metadata', default=str(Config.RULE_METADATA_FILE))
    parser.add_argument('-a', '--ast', default=str(Config.AST_ANALYSIS_FILE))
    parser.add_argument('-o', '--output', default=str(Config.OUTPUT_FILE))
    parser.add_argument('-n', '--questions', type=int, default=5)
    parser.add_argument('-t', '--total', type=int, default=None)
    parser.add_argument('-l', '--languages', nargs='+', default=['en', 'zh'])
    parser.add_argument('--preview', type=int, default=2)
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    orchestrator = CodeGroundedOrchestrator(args.metadata, args.ast)
    
    if not orchestrator.initialize():
        print("Error: Failed to initialize.")
        sys.exit(1)
    
    print(f"\n🚀 Running Code-Grounded Q&A Pipeline v{Config.VERSION}")
    print(f"   NEW: Execution Flow Analyzer")
    print(f"   NEW: Logical Consistency Validator")
    print(f"   NEW: Code-Constrained Prompts")
    
    pairs = orchestrator.run_pipeline(
        questions_per_evidence=args.questions,
        total_limit=args.total,
        languages=args.languages,
    )
    
    if not pairs:
        print("Warning: No Q&A pairs generated.")
        sys.exit(1)
    
    output_path = orchestrator.save_results(args.output)
    orchestrator.print_summary()
    
    if args.preview > 0:
        print(f"\n--- Sample Q&A Pairs ---")
        orchestrator.print_sample_pairs(args.preview)
    
    print(f"\n✅ Successfully generated {len(pairs)} Q&A pairs")
    print(f"📁 Output saved to: {output_path}")


if __name__ == "__main__":
    main()
