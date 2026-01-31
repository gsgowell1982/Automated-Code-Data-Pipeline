"""
Diversity Management module for Q&A Generation Engine v9.

Ensures question diversity at scale (from v7).
"""

import hashlib
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models import UserRole, QuestionType


class DiversityManager:
    """
    Manages question diversity across the entire generation process.
    
    Features:
    - Global question deduplication (exact + semantic)
    - Question type balancing
    - Role distribution tracking
    - Diversity metrics calculation
    """
    
    def __init__(self):
        # Global question tracking
        self.all_questions: List[str] = []
        self.question_hashes: Set[str] = set()
        self.question_ngrams: Dict[str, Set[str]] = {}
        
        # Distribution tracking
        self.type_counts: Counter = Counter()
        self.role_counts: Counter = Counter()
        self.scenario_counts: Counter = Counter()
        self.language_counts: Counter = Counter()
        
        # Rejection tracking
        self.duplicates_rejected: int = 0
        self.similar_rejected: int = 0
    
    def reset(self):
        """Reset all tracking data."""
        self.all_questions = []
        self.question_hashes = set()
        self.question_ngrams = {}
        self.type_counts = Counter()
        self.role_counts = Counter()
        self.scenario_counts = Counter()
        self.language_counts = Counter()
        self.duplicates_rejected = 0
        self.similar_rejected = 0
    
    def is_diverse(self, question: str) -> Tuple[bool, str]:
        """
        Check if a question is sufficiently diverse from existing questions.
        
        Args:
            question: The question text to check
            
        Returns:
            Tuple of (is_diverse, rejection_reason)
        """
        q_normalized = self._normalize(question)
        q_hash = hashlib.md5(q_normalized.encode()).hexdigest()
        
        # Exact duplicate check
        if q_hash in self.question_hashes:
            self.duplicates_rejected += 1
            return False, "exact_duplicate"
        
        # Semantic similarity check
        q_ngrams = self._get_ngrams(q_normalized)
        
        for existing_q, existing_ngrams in self.question_ngrams.items():
            similarity = self._jaccard_similarity(q_ngrams, existing_ngrams)
            if similarity > Config.SIMILARITY_THRESHOLD:
                self.similar_rejected += 1
                return False, f"too_similar:{similarity:.2f}"
        
        return True, "ok"
    
    def add_question(
        self,
        question: str,
        question_type: QuestionType,
        role: UserRole,
        scenario: str,
        language: str
    ):
        """
        Register a question for diversity tracking.
        
        Args:
            question: Question text
            question_type: Type of question
            role: User role
            scenario: Business scenario
            language: Language code
        """
        q_normalized = self._normalize(question)
        q_hash = hashlib.md5(q_normalized.encode()).hexdigest()
        q_ngrams = self._get_ngrams(q_normalized)
        
        self.all_questions.append(question)
        self.question_hashes.add(q_hash)
        self.question_ngrams[q_normalized] = q_ngrams
        
        self.type_counts[question_type.value if question_type else "unknown"] += 1
        self.role_counts[role.value if isinstance(role, UserRole) else role] += 1
        self.scenario_counts[scenario] += 1
        self.language_counts[language] += 1
    
    def get_underrepresented_types(self) -> List[QuestionType]:
        """Get question types that are underrepresented."""
        if not self.type_counts:
            return list(QuestionType)
        
        avg_count = sum(self.type_counts.values()) / len(QuestionType)
        underrepresented = []
        
        for qtype in QuestionType:
            if self.type_counts.get(qtype.value, 0) < avg_count * 0.5:
                underrepresented.append(qtype)
        
        return underrepresented if underrepresented else list(QuestionType)
    
    def get_underrepresented_roles(self) -> List[UserRole]:
        """Get user roles that are underrepresented."""
        if not self.role_counts:
            return list(UserRole)
        
        avg_count = sum(self.role_counts.values()) / len(UserRole)
        underrepresented = []
        
        for role in UserRole:
            if self.role_counts.get(role.value, 0) < avg_count * 0.5:
                underrepresented.append(role)
        
        return underrepresented if underrepresented else list(UserRole)
    
    def get_metrics(self) -> Dict:
        """
        Calculate comprehensive diversity metrics.
        
        Returns:
            Dictionary with diversity metrics
        """
        total = len(self.all_questions)
        if total == 0:
            return {"error": "No questions generated", "total_questions": 0}
        
        # Type distribution score (entropy-based)
        type_distribution = self._calculate_distribution_score(
            self.type_counts, len(QuestionType)
        )
        
        # Role distribution score
        role_distribution = self._calculate_distribution_score(
            self.role_counts, len(UserRole)
        )
        
        # Unique ratio
        total_attempted = total + self.duplicates_rejected + self.similar_rejected
        unique_ratio = total / max(total_attempted, 1)
        
        # Coverage scores
        type_coverage = len([t for t in QuestionType 
                           if self.type_counts.get(t.value, 0) > 0]) / len(QuestionType)
        role_coverage = len([r for r in UserRole 
                           if self.role_counts.get(r.value, 0) > 0]) / len(UserRole)
        
        overall_score = (
            type_distribution + role_distribution + 
            unique_ratio + type_coverage + role_coverage
        ) / 5
        
        return {
            "total_questions": total,
            "unique_ratio": unique_ratio,
            "duplicates_rejected": self.duplicates_rejected,
            "similar_rejected": self.similar_rejected,
            "type_distribution_score": type_distribution,
            "role_distribution_score": role_distribution,
            "type_coverage": type_coverage,
            "role_coverage": role_coverage,
            "type_counts": dict(self.type_counts),
            "role_counts": dict(self.role_counts),
            "language_counts": dict(self.language_counts),
            "overall_diversity_score": overall_score,
        }
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _get_ngrams(self, text: str, n: int = 3) -> Set[str]:
        """Get character n-grams from text."""
        ngrams = set()
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i+n])
        return ngrams
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _calculate_distribution_score(self, counts: Counter, expected_categories: int) -> float:
        """Calculate distribution score (1.0 = perfectly even)."""
        if not counts or expected_categories == 0:
            return 0.0
        
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        # Calculate entropy
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize by max entropy
        max_entropy = math.log2(expected_categories)
        return entropy / max_entropy if max_entropy > 0 else 0.0
