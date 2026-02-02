"""
Utility modules for Design Q&A Generation.

Provides local implementations to avoid import conflicts.
"""

import sys
import time
import hashlib
import math
import re
import logging
import requests
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DesignConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """LLM client for Ollama API."""
    
    def __init__(self, api_url: str = None, model_name: str = None):
        self.api_url = api_url or DesignConfig.OLLAMA_API
        self.model_name = model_name or DesignConfig.MODEL_NAME
        self._available: Optional[bool] = None
        self._check_time: float = 0
    
    def is_available(self) -> bool:
        current_time = time.time()
        if self._available is not None and current_time - self._check_time < 60:
            return self._available
        
        try:
            response = requests.get(
                self.api_url.replace("/api/generate", "/api/tags"),
                timeout=5
            )
            self._available = response.status_code == 200
            self._check_time = current_time
        except Exception:
            self._available = False
            self._check_time = current_time
        
        return self._available
    
    def generate_with_task(self, prompt: str, task: str, system: str = None) -> Optional[str]:
        if not self.is_available():
            return None
        
        temperature = DesignConfig.LLM_TEMPERATURE.get(task, 0.7)
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            }
            if system:
                payload["system"] = system
            
            response = requests.post(
                self.api_url, json=payload, timeout=DesignConfig.LLM_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"LLM error: {e}")
        
        return None


class QuestionType(str, Enum):
    """Question types for diversity tracking."""
    TROUBLESHOOTING = "troubleshooting"
    UNDERSTANDING = "understanding"
    EDGE_CASE = "edge_case"
    SECURITY = "security"
    WHAT_IF = "what_if"
    COMPARISON = "comparison"
    VALIDATION = "validation"
    DEEP_ANALYSIS = "deep_analysis"


class UserRole(str, Enum):
    """User roles (placeholder for compatibility)."""
    END_USER = "end_user"
    PRODUCT_MANAGER = "product_manager"
    QA_ENGINEER = "qa_engineer"
    SECURITY_AUDITOR = "security_auditor"
    NEW_DEVELOPER = "new_developer"


class DiversityManagerV2:
    """Diversity manager with semantic fingerprinting."""
    
    def __init__(self):
        self.all_questions: List[str] = []
        self.question_hashes: Set[str] = set()
        self.question_ngrams: Dict[str, Set[str]] = {}
        self.semantic_fingerprints: Set[str] = set()
        self.type_counts: Counter = Counter()
        self.role_counts: Counter = Counter()
        self.scenario_counts: Counter = Counter()
        self.language_counts: Counter = Counter()
        self.intent_counts: Counter = Counter()
        self.focus_counts: Counter = Counter()
        self.duplicates_rejected: int = 0
        self.similar_rejected: int = 0
        self.semantic_duplicate_rejected: int = 0
    
    def reset(self):
        self.all_questions = []
        self.question_hashes = set()
        self.question_ngrams = {}
        self.semantic_fingerprints = set()
        self.type_counts = Counter()
        self.role_counts = Counter()
        self.scenario_counts = Counter()
        self.language_counts = Counter()
        self.intent_counts = Counter()
        self.focus_counts = Counter()
        self.duplicates_rejected = 0
        self.similar_rejected = 0
        self.semantic_duplicate_rejected = 0
    
    def is_diverse(self, question: str, evidence_id: str = "", language: str = "en") -> Tuple[bool, str]:
        q_normalized = self._normalize(question)
        q_hash = hashlib.md5(q_normalized.encode()).hexdigest()
        
        if q_hash in self.question_hashes:
            self.duplicates_rejected += 1
            return False, "exact_duplicate"
        
        q_ngrams = self._get_ngrams(q_normalized)
        for existing_q, existing_ngrams in self.question_ngrams.items():
            similarity = self._jaccard_similarity(q_ngrams, existing_ngrams)
            if similarity > 0.55:
                self.similar_rejected += 1
                return False, f"surface_similar:{similarity:.2f}"
        
        return True, "ok"
    
    def add_question(
        self,
        question: str,
        question_type,
        role,
        scenario: str,
        language: str,
        evidence_id: str = ""
    ):
        q_normalized = self._normalize(question)
        q_hash = hashlib.md5(q_normalized.encode()).hexdigest()
        q_ngrams = self._get_ngrams(q_normalized)
        
        self.all_questions.append(question)
        self.question_hashes.add(q_hash)
        self.question_ngrams[q_normalized] = q_ngrams
        
        self.type_counts[question_type.value if hasattr(question_type, 'value') else str(question_type)] += 1
        self.role_counts[role.value if hasattr(role, 'value') else str(role)] += 1
        self.scenario_counts[scenario] += 1
        self.language_counts[language] += 1
    
    def get_metrics(self) -> Dict:
        total = len(self.all_questions)
        if total == 0:
            return {"error": "No questions", "total_questions": 0}
        
        total_attempted = total + self.duplicates_rejected + self.similar_rejected
        unique_ratio = total / max(total_attempted, 1)
        
        overall_score = unique_ratio * 0.5 + 0.5
        
        return {
            "total_questions": total,
            "unique_ratio": unique_ratio,
            "duplicates_rejected": self.duplicates_rejected,
            "similar_rejected": self.similar_rejected,
            "type_counts": dict(self.type_counts),
            "role_counts": dict(self.role_counts),
            "language_counts": dict(self.language_counts),
            "overall_diversity_score": overall_score,
        }
    
    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _get_ngrams(self, text: str, n: int = 3) -> Set[str]:
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


__all__ = ["OllamaClient", "DiversityManagerV2", "QuestionType", "UserRole"]
