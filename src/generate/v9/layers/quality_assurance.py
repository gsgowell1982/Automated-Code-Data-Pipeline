"""
Quality Assurance Layer for Q&A Generation Engine v9.

Ensures generated Q&A pairs meet quality standards.

Components:
- Code reference validation
- DBR alignment check
- Hash validation
- Code name leakage detection
"""

import re
import hashlib
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import QAPair, ValidationResult, CodeFacts


class QualityAssuranceLayer:
    """
    Validates and ensures quality of generated Q&A pairs.
    
    Checks:
    - Question quality (length, format)
    - Answer quality (length, completeness)
    - Code name leakage (questions should not contain function names)
    - Source hash integrity
    - Consistency with code facts
    """
    
    # Code name patterns that should NOT appear in questions
    CODE_NAME_PATTERNS = [
        r'\bcheck_\w+\b',
        r'\busers_repo\b',
        r'\buser_create\b',
        r'\buser_update\b',
        r'\bHTTP_\d+\b',
        r'\bEntityDoesNotExist\b',
        r'\bwrong_login_error\b',
        r'\bcreate_access_token\b',
        r'\b[a-z]+_[a-z]+_[a-z]+\b',  # snake_case with 3+ parts
    ]
    
    def __init__(self, min_question_length: int = 15, min_answer_length: int = 100):
        self.min_question_length = min_question_length
        self.min_answer_length = min_answer_length
    
    def validate(self, qa_pair: QAPair, code_facts: CodeFacts = None) -> ValidationResult:
        """
        Validate a Q&A pair.
        
        Args:
            qa_pair: The Q&A pair to validate
            code_facts: Optional code facts for consistency check
            
        Returns:
            ValidationResult with is_valid, score, and issues
        """
        issues = []
        scores = []
        
        # Question validation
        question = qa_pair.instruction
        
        # Length check
        if len(question) < self.min_question_length:
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
        
        # Code name leakage check (critical)
        if self._contains_code_names(question):
            issues.append("Question contains code names")
            scores.append(0.0)  # Automatic fail
        else:
            scores.append(1.0)
        
        # Answer validation
        answer = qa_pair.answer
        
        if len(answer) < self.min_answer_length:
            issues.append("Answer too short")
            scores.append(0.3)
        else:
            scores.append(1.0)
        
        # Code snippet validation
        code = qa_pair.context.get("code_snippet", "")
        if len(code) < 50:
            issues.append("Code snippet too short")
            scores.append(0.5)
        else:
            scores.append(1.0)
        
        # Hash validation
        source_hash = qa_pair.data_quality.get("source_hash", "")
        if source_hash and code:
            computed = hashlib.md5(code.encode()).hexdigest()
            if computed != source_hash:
                issues.append("Source hash mismatch")
                scores.append(0.5)
            else:
                scores.append(1.0)
        else:
            scores.append(0.9)
        
        # Calculate final score and validity
        avg_score = sum(scores) / len(scores)
        
        # Code name leakage is automatic rejection
        is_valid = (
            avg_score >= 0.7 and 
            "Question contains code names" not in issues
        )
        
        return ValidationResult(
            is_valid=is_valid,
            score=avg_score,
            issues=issues
        )
    
    def validate_question(self, question: str) -> Tuple[bool, List[str]]:
        """
        Validate a question independently.
        
        Args:
            question: Question text
            
        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []
        
        if len(question) < self.min_question_length:
            issues.append("Question too short")
        
        if not (question.endswith('?') or question.endswith('？')):
            issues.append("Missing question mark")
        
        if self._contains_code_names(question):
            issues.append("Question contains code names")
        
        is_valid = len(issues) == 0 or (
            len(issues) == 1 and issues[0] == "Missing question mark"
        )
        
        return is_valid, issues
    
    def validate_answer(self, answer: str, code_facts: CodeFacts = None) -> Tuple[bool, List[str]]:
        """
        Validate an answer independently.
        
        Args:
            answer: Answer text
            code_facts: Optional code facts for consistency check
            
        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []
        
        if len(answer) < self.min_answer_length:
            issues.append("Answer too short")
        
        # Check for forbidden claims if code facts available
        if code_facts:
            from .consistency_validator import ConsistencyValidator
            validator = ConsistencyValidator()
            result = validator.validate_answer(answer, code_facts)
            issues.extend(result.issues)
        
        return len(issues) == 0, issues
    
    def _contains_code_names(self, text: str) -> bool:
        """Check if text contains code-level names."""
        for pattern in self.CODE_NAME_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def verify_hash(self, code: str, expected_hash: str) -> bool:
        """Verify code hash."""
        if not expected_hash:
            return True
        computed = hashlib.md5(code.encode()).hexdigest()
        return computed == expected_hash
    
    def check_dbr_alignment(self, qa_pair: QAPair, expected_dbr: str) -> bool:
        """Check if Q&A pair aligns with expected DBR rule."""
        actual_dbr = qa_pair.context.get("related_dbr", "")
        return actual_dbr == expected_dbr or actual_dbr.startswith(expected_dbr)
