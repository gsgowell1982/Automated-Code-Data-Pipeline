"""
Consistency Validator for Q&A Generation Engine v9.

Validates LLM output against code facts to detect contradictions.

From v8: Fixes "reasoning without code verification" by checking
LLM output against deterministic code facts.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models import CodeFacts, ValidationResult


class ConsistencyValidator:
    """
    Validates LLM output against code facts.
    
    Detects and rejects answers that contradict deterministic code behavior,
    such as claiming "partial save" when code has early exit.
    """
    
    # Contradiction patterns
    PARTIAL_SAVE_PATTERNS = [
        r'(?<!is\s)(?<!are\s)(?<!be\s)partial(?:ly)?\s+(?:save|saved|written|create|update)',
        r'(?<!不可能)部分(?:保存|写入|创建|更新)',
        r'half[\s-]?(?:saved|created|written)',
        r'incomplete\s+(?:data|record|account)',
        r'不完整的(?:数据|记录|账户)',
    ]
    
    RACE_CONDITION_PATTERNS = [
        r'race\s+condition\s+(?:in|within)\s+(?:this|the)\s+function',
        r'(?:这个|该)函数(?:内|中)(?:的|存在)?竞态',
        r'concurrent\s+execution\s+of\s+this\s+function',
    ]
    
    VAGUE_PATTERNS = [
        r'(?:data\s+)?might\s+(?:be\s+)?partially',
        r'(?:数据)?可能(?:会)?部分',
        r'could\s+leave\s+(?:data\s+)?in\s+inconsistent',
        r'可能(?:会)?(?:导致|留下)(?:数据)?不一致',
    ]
    
    # Negation context patterns (these make "partial save" OK)
    NEGATION_CONTEXT = [
        r'(?:no|not|cannot|impossible|never|防止|不可能|不会|无法).*partial',
        r'partial.*(?:is|are)\s+(?:not|impossible)',
        r'"partial.*"\s+is\s+(?:not|impossible)',
    ]
    
    def validate_reasoning(
        self,
        reasoning: List[str],
        code_facts: CodeFacts
    ) -> ValidationResult:
        """
        Validate reasoning against code facts.
        
        Args:
            reasoning: List of reasoning steps
            code_facts: Deterministic facts about the code
            
        Returns:
            ValidationResult with is_valid, score, and issues
        """
        issues = []
        contradiction_score = 0.0
        
        reasoning_text = " ".join(reasoning).lower()
        
        # Check for forbidden claims
        for forbidden in code_facts.forbidden_claims:
            if self._contains_claim(reasoning_text, forbidden):
                issues.append(f"Contains forbidden claim: '{forbidden}'")
                contradiction_score += 0.3
        
        # Check for partial save claims (when code has early exit)
        if code_facts.has_early_exit:
            if self._contains_partial_save_claim(reasoning_text):
                issues.append("Claims partial save when code has early exit")
                contradiction_score += 0.4
        
        # Check for race condition claims (when code is synchronous)
        if code_facts.is_synchronous:
            if self._contains_race_condition_claim(reasoning_text):
                issues.append("Claims race condition in synchronous code")
                contradiction_score += 0.3
        
        is_valid = contradiction_score < Config.MAX_CONTRADICTION_SCORE
        score = max(0, 1.0 - contradiction_score)
        
        return ValidationResult(is_valid=is_valid, score=score, issues=issues)
    
    def validate_answer(
        self,
        answer: str,
        code_facts: CodeFacts
    ) -> ValidationResult:
        """
        Validate answer against code facts.
        
        Args:
            answer: The answer text
            code_facts: Deterministic facts about the code
            
        Returns:
            ValidationResult with is_valid, score, and issues
        """
        issues = []
        contradiction_score = 0.0
        
        answer_lower = answer.lower()
        
        # Check forbidden claims
        for forbidden in code_facts.forbidden_claims:
            if self._contains_claim(answer_lower, forbidden):
                issues.append(f"Contains forbidden claim: '{forbidden}'")
                contradiction_score += 0.3
        
        # Check partial save patterns (with negation context)
        if code_facts.has_early_exit:
            if self._contains_partial_save_claim(answer_lower):
                issues.append("Claims partial save (contradicts raise semantics)")
                contradiction_score += 0.5
        
        # Check vague language for deterministic code
        if code_facts.is_synchronous and code_facts.has_early_exit:
            if self._contains_vague_language(answer_lower):
                issues.append("Uses vague language for deterministic behavior")
                contradiction_score += 0.2
        
        is_valid = contradiction_score < Config.MAX_CONTRADICTION_SCORE
        score = max(0, 1.0 - contradiction_score)
        
        return ValidationResult(is_valid=is_valid, score=score, issues=issues)
    
    def _contains_claim(self, text: str, claim: str) -> bool:
        """Check if text contains a claim (without negation context)."""
        claim_lower = claim.lower()
        if claim_lower not in text:
            return False
        
        # Check for negation context
        for pattern in self.NEGATION_CONTEXT:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        return True
    
    def _contains_partial_save_claim(self, text: str) -> bool:
        """Check if text claims partial save can happen."""
        for pattern in self.PARTIAL_SAVE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                # Check for negation context
                negated = False
                for neg_pattern in self.NEGATION_CONTEXT:
                    if re.search(neg_pattern, text, re.IGNORECASE):
                        negated = True
                        break
                if not negated:
                    return True
        return False
    
    def _contains_race_condition_claim(self, text: str) -> bool:
        """Check if text claims race condition in the function."""
        for pattern in self.RACE_CONDITION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _contains_vague_language(self, text: str) -> bool:
        """Check if text uses vague language for deterministic code."""
        for pattern in self.VAGUE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def get_forbidden_phrases_prompt(self, code_facts: CodeFacts, language: str = "en") -> str:
        """
        Get forbidden phrases for inclusion in prompt.
        
        Args:
            code_facts: Deterministic facts about the code
            language: Language code
            
        Returns:
            Formatted string listing forbidden phrases
        """
        if not code_facts.forbidden_claims:
            return ""
        
        if language == "zh":
            header = "【禁止使用的表述】"
        else:
            header = "【FORBIDDEN PHRASES - DO NOT USE】"
        
        phrases = "\n".join(f"- \"{claim}\"" for claim in code_facts.forbidden_claims)
        return f"{header}\n{phrases}"
