"""
Quality Checker for Design Q&A Generation.

Validates design questions and solutions meet quality standards.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DesignQAPair, DesignSolution


class DesignQualityChecker:
    """
    Validates design Q&A pairs for quality.
    
    Checks:
    - Question relevance and format
    - Answer completeness
    - Reasoning coherence
    - No code implementation details (design-focused)
    """
    
    # Patterns that indicate code-level content (should be avoided in design Q&A)
    CODE_PATTERNS = [
        r'\bdef\s+\w+',
        r'\bclass\s+\w+',
        r'\bimport\s+\w+',
        r'\basync\s+def',
        r'\bawait\s+',
        r'```python',
        r'```\w+',
    ]
    
    # Minimum lengths
    MIN_QUESTION_LENGTH = 20
    MIN_ANSWER_LENGTH = 150
    MIN_REASONING_STEPS = 3
    
    def __init__(self):
        self.code_pattern = re.compile('|'.join(self.CODE_PATTERNS), re.IGNORECASE)
    
    def validate(self, qa_pair: DesignQAPair) -> Tuple[bool, float, List[str]]:
        """
        Validate a design Q&A pair.
        
        Args:
            qa_pair: The Q&A pair to validate
            
        Returns:
            Tuple of (is_valid, score, issues)
        """
        issues = []
        scores = []
        
        # Question validation
        question = qa_pair.instruction
        
        # Length check
        if len(question) < self.MIN_QUESTION_LENGTH:
            issues.append("Question too short")
            scores.append(0.3)
        else:
            scores.append(1.0)
        
        # Question mark check
        if not (question.endswith('?') or question.endswith('？')):
            issues.append("Missing question mark")
            scores.append(0.8)
        else:
            scores.append(1.0)
        
        # Design-focused check (should not contain code)
        if self._contains_code(question):
            issues.append("Question contains code patterns")
            scores.append(0.5)
        else:
            scores.append(1.0)
        
        # Answer validation
        answer = qa_pair.answer
        
        if len(answer) < self.MIN_ANSWER_LENGTH:
            issues.append("Answer too short")
            scores.append(0.3)
        else:
            scores.append(1.0)
        
        # Reasoning validation
        reasoning = qa_pair.reasoning_trace
        
        if len(reasoning) < self.MIN_REASONING_STEPS:
            issues.append("Insufficient reasoning steps")
            scores.append(0.5)
        else:
            scores.append(1.0)
        
        # Context validation (should have design_scenario)
        if "design_scenario" not in qa_pair.context:
            issues.append("Missing design scenario in context")
            scores.append(0.7)
        else:
            scores.append(1.0)
        
        # Calculate final score
        avg_score = sum(scores) / len(scores)
        is_valid = avg_score >= 0.7 and len(issues) <= 2
        
        return is_valid, avg_score, issues
    
    def validate_question(self, question: str) -> Tuple[bool, List[str]]:
        """Validate a question independently."""
        issues = []
        
        if len(question) < self.MIN_QUESTION_LENGTH:
            issues.append("Question too short")
        
        if not (question.endswith('?') or question.endswith('？')):
            issues.append("Missing question mark")
        
        if self._contains_code(question):
            issues.append("Question contains code")
        
        is_valid = len(issues) == 0 or (len(issues) == 1 and "Missing question mark" in issues)
        return is_valid, issues
    
    def validate_solution(self, solution: DesignSolution) -> Tuple[bool, List[str]]:
        """Validate a design solution."""
        issues = []
        
        if not solution.approach:
            issues.append("Missing approach")
        
        if not solution.rationale:
            issues.append("Missing rationale")
        
        if len(solution.components) < 2:
            issues.append("Insufficient components")
        
        if len(solution.patterns_used) < 1:
            issues.append("No design patterns specified")
        
        return len(issues) == 0, issues
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code patterns."""
        return bool(self.code_pattern.search(text))
