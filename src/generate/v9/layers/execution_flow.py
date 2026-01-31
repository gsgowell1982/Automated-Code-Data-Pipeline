"""
Execution Flow Analyzer for Q&A Generation Engine v9.

Extracts execution order and control flow semantics from code.
This is critical for preventing LLM hallucinations about code behavior.

From v8: Fixes "semantic tension" by providing hard code facts.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ExecutionStep, ExecutionSemantics, CodeFacts


class ExecutionFlowAnalyzer:
    """
    Analyzes code to extract execution order and control flow semantics.
    
    This provides HARD FACTS that constrain LLM generation, preventing
    hallucinations about partial saves, race conditions, etc.
    
    Example output:
        Execution order: 1.check_username -> 2.check_email -> 3.create_user
        Semantics: "raise = 100% termination, NO partial state"
    """
    
    # Control flow patterns
    RAISE_PATTERN = r'\braise\s+\w+'
    RETURN_PATTERN = r'\breturn\s+'
    IF_PATTERN = r'\bif\s+'
    AWAIT_PATTERN = r'\bawait\s+'
    TRY_PATTERN = r'\btry\s*:'
    EXCEPT_PATTERN = r'\bexcept\s+'
    
    # Validation function patterns
    VALIDATION_PATTERNS = [
        r'check_\w+',
        r'validate_\w+',
        r'is_\w+_taken',
        r'verify_\w+',
    ]
    
    # Data operation patterns
    DATA_OP_PATTERNS = [
        r'create_\w+',
        r'update_\w+',
        r'delete_\w+',
        r'save\w*',
        r'insert\w*',
    ]
    
    def analyze(self, code: str, function_name: str = "") -> CodeFacts:
        """
        Analyze code and extract execution facts.
        
        Args:
            code: The code snippet to analyze
            function_name: Optional function name for context
            
        Returns:
            CodeFacts with all deterministic facts about the code
        """
        execution_order = self._extract_execution_order(code)
        termination_points = self._extract_termination_points(code)
        validation_checks = self._extract_validation_checks(code)
        
        is_synchronous = self._is_synchronous_flow(code)
        has_early_exit = (
            len(termination_points) > 0 and 
            self._has_validation_before_write(code)
        )
        atomicity_type = self._determine_atomicity(code, has_early_exit)
        
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
            return "gated_sequential"
        
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
    
    def format_for_prompt(self, code_facts: CodeFacts, language: str = "en") -> str:
        """
        Format execution order as a constraint for LLM prompt.
        
        Args:
            code_facts: The code facts to format
            language: Language code ("en" or "zh")
            
        Returns:
            Formatted string for inclusion in LLM prompt
        """
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
            header = "【强制执行顺序】"
            sync_note = "注意：同步顺序执行，不存在并发" if code_facts.is_synchronous else ""
            exit_note = "注意：raise立即终止，无部分保存" if code_facts.has_early_exit else ""
        else:
            header = "【EXECUTION ORDER】"
            sync_note = "NOTE: Synchronous sequential. NO concurrency." if code_facts.is_synchronous else ""
            exit_note = "NOTE: raise = IMMEDIATE termination. NO partial save." if code_facts.has_early_exit else ""
        
        result = f"{header}\n" + " → ".join(steps)
        if sync_note:
            result += f"\n{sync_note}"
        if exit_note:
            result += f"\n{exit_note}"
        
        return result
