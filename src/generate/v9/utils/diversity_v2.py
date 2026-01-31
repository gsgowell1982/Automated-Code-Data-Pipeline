"""
Enhanced Diversity Management v2 for Q&A Generation Engine v9.

Key improvements over v1:
1. Semantic dimension tracking - detect paraphrased questions
2. Question angle/perspective categories - ensure different viewpoints
3. Focus topic tracking - avoid repetition on same specific topic
4. Intent fingerprinting - detect questions with same underlying intent

Solves the problem of:
  "为什么我改用户名时会提示这个新名字已经被占用了？"
  "为什么我修改用户名后还是会收到用户名已存在的提示呢？"
being treated as different questions when they have identical intent.
"""

import hashlib
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models import UserRole, QuestionType


class QuestionAngle(str, Enum):
    """Different angles/perspectives for asking questions."""
    WHY_ERROR = "why_error"           # 为什么会报错/出错
    WHAT_HAPPENS = "what_happens"     # 会发生什么
    HOW_IT_WORKS = "how_it_works"     # 如何工作/流程是什么
    IS_IT_SECURE = "is_it_secure"     # 是否安全
    WHAT_IF_REMOVE = "what_if_remove" # 如果去掉会怎样
    CAN_I_DO = "can_i_do"             # 我能不能/是否可以
    HOW_TO_FIX = "how_to_fix"         # 如何解决/修复
    WHEN_DOES = "when_does"           # 什么时候/何时
    WHERE_IS = "where_is"             # 在哪里/哪个地方
    COMPARE = "compare"               # 对比/区别
    EDGE_CASE = "edge_case"           # 边界情况
    DEEP_ANALYSIS = "deep_analysis"   # 深度分析


class QuestionFocus(str, Enum):
    """Specific topics/focuses of questions."""
    USERNAME = "username"
    EMAIL = "email"
    PASSWORD = "password"
    LOGIN = "login"
    REGISTRATION = "registration"
    PROFILE_UPDATE = "profile_update"
    ERROR_MESSAGE = "error_message"
    VALIDATION = "validation"
    TOKEN = "token"
    SESSION = "session"
    DATABASE = "database"
    SECURITY = "security"
    USER_EXPERIENCE = "user_experience"
    SYSTEM_BEHAVIOR = "system_behavior"


class SemanticIntent:
    """Represents the semantic intent of a question."""
    
    # Intent patterns for Chinese
    INTENT_PATTERNS_ZH = {
        "ask_why_error": [
            r'为什么.*(?:报错|错误|提示|显示|出现)',
            r'为何.*(?:报错|失败|不行)',
            r'怎么.*(?:会|就).*(?:错|失败)',
        ],
        "ask_what_happens": [
            r'(?:会|将|能).*(?:发生|怎样|如何)',
            r'(?:如果|假如|当).*(?:会|将).*(?:怎|什么)',
            r'.*(?:结果|后果).*(?:是|会)',
        ],
        "ask_is_possible": [
            r'(?:能不能|可不可以|是否可以|能否)',
            r'.*(?:行不行|可以吗|能吗)',
        ],
        "ask_how_works": [
            r'(?:如何|怎么|怎样).*(?:工作|运行|处理|实现)',
            r'.*(?:流程|过程|步骤).*(?:是|怎)',
            r'.*(?:原理|机制).*(?:是|怎)',
        ],
        "ask_is_secure": [
            r'(?:是否|是不是).*(?:安全|有.*漏洞|有.*风险)',
            r'.*(?:安全|风险|攻击).*(?:吗|呢)',
        ],
        "ask_difference": [
            r'.*(?:区别|不同|差异).*(?:是|在)',
            r'.*(?:和|与|跟).*(?:区别|不同)',
        ],
    }
    
    # Intent patterns for English
    INTENT_PATTERNS_EN = {
        "ask_why_error": [
            r'why.*(?:error|fail|reject|denied|show)',
            r'why.*(?:get|see|receive).*(?:error|message)',
        ],
        "ask_what_happens": [
            r'what.*(?:happen|occur|result)',
            r'(?:if|when).*what.*(?:happen|will)',
        ],
        "ask_is_possible": [
            r'(?:can|could|is it possible).*(?:i|user|we)',
            r'(?:is|are).*(?:able|possible|allowed)',
        ],
        "ask_how_works": [
            r'how.*(?:does|do|work|handle|process)',
            r'what.*(?:flow|process|mechanism)',
        ],
        "ask_is_secure": [
            r'(?:is|are).*(?:secure|safe|vulnerable)',
            r'.*(?:security|attack|risk).*(?:\?)',
        ],
        "ask_difference": [
            r'(?:what|how).*(?:differ|different)',
            r'.*(?:vs|versus|compare)',
        ],
    }
    
    # Focus extraction patterns
    FOCUS_PATTERNS = {
        QuestionFocus.USERNAME: [r'用户名', r'username', r'user\s*name', r'名字', r'昵称'],
        QuestionFocus.EMAIL: [r'邮箱', r'email', r'邮件', r'e-mail'],
        QuestionFocus.PASSWORD: [r'密码', r'password', r'口令', r'pwd'],
        QuestionFocus.LOGIN: [r'登录', r'login', r'登入', r'sign\s*in'],
        QuestionFocus.REGISTRATION: [r'注册', r'register', r'sign\s*up', r'创建账户'],
        QuestionFocus.PROFILE_UPDATE: [r'修改', r'更新', r'update', r'改', r'edit', r'profile'],
        QuestionFocus.ERROR_MESSAGE: [r'错误', r'error', r'提示', r'message', r'报错'],
        QuestionFocus.VALIDATION: [r'验证', r'校验', r'valid', r'check', r'检查'],
        QuestionFocus.TOKEN: [r'令牌', r'token', r'jwt'],
        QuestionFocus.SESSION: [r'会话', r'session', r'登录状态'],
        QuestionFocus.SECURITY: [r'安全', r'secur', r'攻击', r'attack', r'漏洞'],
    }
    
    @classmethod
    def extract_intent(cls, question: str, language: str = "en") -> str:
        """Extract the primary intent of a question."""
        patterns = cls.INTENT_PATTERNS_ZH if language == "zh" else cls.INTENT_PATTERNS_EN
        question_lower = question.lower()
        
        for intent, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    return intent
        
        return "general_inquiry"
    
    @classmethod
    def extract_focus(cls, question: str) -> List[QuestionFocus]:
        """Extract the focus topics of a question."""
        focuses = []
        question_lower = question.lower()
        
        for focus, patterns in cls.FOCUS_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    focuses.append(focus)
                    break
        
        return focuses if focuses else [QuestionFocus.SYSTEM_BEHAVIOR]
    
    @classmethod
    def get_semantic_fingerprint(cls, question: str, language: str = "en") -> str:
        """
        Generate a semantic fingerprint for a question.
        Questions with same fingerprint are semantically equivalent.
        """
        intent = cls.extract_intent(question, language)
        focuses = cls.extract_focus(question)
        focus_str = "+".join(sorted(f.value for f in focuses))
        return f"{intent}:{focus_str}"


class DiversityManagerV2:
    """
    Enhanced Diversity Manager with semantic awareness.
    
    Key Features:
    1. Semantic fingerprinting - detect paraphrased questions
    2. Angle coverage - ensure different question perspectives
    3. Focus distribution - balance topics
    4. Combined scoring - surface + semantic similarity
    """
    
    def __init__(self):
        # Surface-level tracking
        self.all_questions: List[str] = []
        self.question_hashes: Set[str] = set()
        self.question_ngrams: Dict[str, Set[str]] = {}
        
        # Semantic-level tracking (NEW)
        self.semantic_fingerprints: Set[str] = set()
        self.intent_counts: Counter = Counter()
        self.focus_counts: Counter = Counter()
        self.angle_counts: Counter = Counter()
        
        # Per-evidence tracking to ensure diversity within same evidence
        self.evidence_fingerprints: Dict[str, Set[str]] = defaultdict(set)
        
        # Distribution tracking
        self.type_counts: Counter = Counter()
        self.role_counts: Counter = Counter()
        self.scenario_counts: Counter = Counter()
        self.language_counts: Counter = Counter()
        
        # Rejection tracking
        self.duplicates_rejected: int = 0
        self.similar_rejected: int = 0
        self.semantic_duplicate_rejected: int = 0
        
        # Configuration
        self.surface_similarity_threshold = 0.55  # Slightly lower than v1
        self.min_angles_per_evidence = 3
        self.min_focuses_per_evidence = 2
    
    def reset(self):
        """Reset all tracking data."""
        self.all_questions = []
        self.question_hashes = set()
        self.question_ngrams = {}
        self.semantic_fingerprints = set()
        self.intent_counts = Counter()
        self.focus_counts = Counter()
        self.angle_counts = Counter()
        self.evidence_fingerprints = defaultdict(set)
        self.type_counts = Counter()
        self.role_counts = Counter()
        self.scenario_counts = Counter()
        self.language_counts = Counter()
        self.duplicates_rejected = 0
        self.similar_rejected = 0
        self.semantic_duplicate_rejected = 0
    
    def is_diverse(
        self, 
        question: str, 
        evidence_id: str = "",
        language: str = "en"
    ) -> Tuple[bool, str]:
        """
        Check if a question is sufficiently diverse.
        
        Now includes:
        1. Exact duplicate check
        2. Surface similarity check (n-gram Jaccard)
        3. Semantic fingerprint check (NEW)
        4. Per-evidence diversity check (NEW)
        
        Args:
            question: Question text
            evidence_id: ID of the evidence this question is for
            language: Language code
            
        Returns:
            Tuple of (is_diverse, rejection_reason)
        """
        q_normalized = self._normalize(question)
        q_hash = hashlib.md5(q_normalized.encode()).hexdigest()
        
        # 1. Exact duplicate check
        if q_hash in self.question_hashes:
            self.duplicates_rejected += 1
            return False, "exact_duplicate"
        
        # 2. Semantic fingerprint check (NEW - most important)
        fingerprint = SemanticIntent.get_semantic_fingerprint(question, language)
        
        # Check global semantic duplicates
        if fingerprint in self.semantic_fingerprints:
            self.semantic_duplicate_rejected += 1
            return False, f"semantic_duplicate:{fingerprint}"
        
        # Check per-evidence semantic duplicates (stricter)
        if evidence_id and fingerprint in self.evidence_fingerprints.get(evidence_id, set()):
            self.semantic_duplicate_rejected += 1
            return False, f"evidence_semantic_duplicate:{fingerprint}"
        
        # 3. Surface similarity check
        q_ngrams = self._get_ngrams(q_normalized)
        
        for existing_q, existing_ngrams in self.question_ngrams.items():
            similarity = self._jaccard_similarity(q_ngrams, existing_ngrams)
            if similarity > self.surface_similarity_threshold:
                self.similar_rejected += 1
                return False, f"surface_similar:{similarity:.2f}"
        
        return True, "ok"
    
    def add_question(
        self,
        question: str,
        question_type: QuestionType,
        role: UserRole,
        scenario: str,
        language: str,
        evidence_id: str = "",
        angle: Optional[QuestionAngle] = None
    ):
        """
        Register a question for diversity tracking.
        
        Now tracks semantic dimensions in addition to surface features.
        """
        q_normalized = self._normalize(question)
        q_hash = hashlib.md5(q_normalized.encode()).hexdigest()
        q_ngrams = self._get_ngrams(q_normalized)
        
        # Surface tracking
        self.all_questions.append(question)
        self.question_hashes.add(q_hash)
        self.question_ngrams[q_normalized] = q_ngrams
        
        # Semantic tracking (NEW)
        fingerprint = SemanticIntent.get_semantic_fingerprint(question, language)
        self.semantic_fingerprints.add(fingerprint)
        
        if evidence_id:
            self.evidence_fingerprints[evidence_id].add(fingerprint)
        
        intent = SemanticIntent.extract_intent(question, language)
        focuses = SemanticIntent.extract_focus(question)
        
        self.intent_counts[intent] += 1
        for focus in focuses:
            self.focus_counts[focus.value] += 1
        
        if angle:
            self.angle_counts[angle.value] += 1
        
        # Standard tracking
        self.type_counts[question_type.value if question_type else "unknown"] += 1
        self.role_counts[role.value if isinstance(role, UserRole) else role] += 1
        self.scenario_counts[scenario] += 1
        self.language_counts[language] += 1
    
    def get_recommended_angle(self, evidence_id: str = "") -> QuestionAngle:
        """
        Get recommended question angle based on current distribution.
        
        Returns an underrepresented angle to improve diversity.
        """
        if not self.angle_counts:
            return QuestionAngle.WHY_ERROR
        
        # Find least used angles
        all_angles = list(QuestionAngle)
        min_count = min(self.angle_counts.get(a.value, 0) for a in all_angles)
        
        underrepresented = [
            a for a in all_angles 
            if self.angle_counts.get(a.value, 0) <= min_count + 1
        ]
        
        import random
        return random.choice(underrepresented) if underrepresented else random.choice(all_angles)
    
    def get_recommended_focus(self, evidence_id: str = "") -> QuestionFocus:
        """
        Get recommended question focus based on current distribution.
        """
        if not self.focus_counts:
            return QuestionFocus.SYSTEM_BEHAVIOR
        
        all_focuses = list(QuestionFocus)
        min_count = min(self.focus_counts.get(f.value, 0) for f in all_focuses)
        
        underrepresented = [
            f for f in all_focuses 
            if self.focus_counts.get(f.value, 0) <= min_count + 1
        ]
        
        import random
        return random.choice(underrepresented) if underrepresented else random.choice(all_focuses)
    
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
    
    def get_coverage_for_evidence(self, evidence_id: str) -> Dict:
        """
        Get diversity coverage metrics for a specific evidence.
        
        Useful for determining what angles/focuses are missing.
        """
        fingerprints = self.evidence_fingerprints.get(evidence_id, set())
        
        intents = set()
        focuses = set()
        
        for fp in fingerprints:
            if ":" in fp:
                intent, focus_str = fp.split(":", 1)
                intents.add(intent)
                for f in focus_str.split("+"):
                    focuses.add(f)
        
        return {
            "question_count": len(fingerprints),
            "unique_intents": len(intents),
            "unique_focuses": len(focuses),
            "intents": list(intents),
            "focuses": list(focuses),
        }
    
    def get_metrics(self) -> Dict:
        """
        Calculate comprehensive diversity metrics.
        
        Now includes semantic diversity metrics.
        """
        total = len(self.all_questions)
        if total == 0:
            return {"error": "No questions generated", "total_questions": 0}
        
        # Type distribution score
        type_distribution = self._calculate_distribution_score(
            self.type_counts, len(QuestionType)
        )
        
        # Role distribution score
        role_distribution = self._calculate_distribution_score(
            self.role_counts, len(UserRole)
        )
        
        # Intent distribution score (NEW)
        intent_distribution = self._calculate_distribution_score(
            self.intent_counts, 6  # Number of intent categories
        )
        
        # Focus distribution score (NEW)
        focus_distribution = self._calculate_distribution_score(
            self.focus_counts, len(QuestionFocus)
        )
        
        # Unique ratios
        total_attempted = total + self.duplicates_rejected + self.similar_rejected + self.semantic_duplicate_rejected
        unique_ratio = total / max(total_attempted, 1)
        semantic_unique_ratio = len(self.semantic_fingerprints) / max(total, 1)
        
        # Coverage scores
        type_coverage = len([t for t in QuestionType 
                           if self.type_counts.get(t.value, 0) > 0]) / len(QuestionType)
        role_coverage = len([r for r in UserRole 
                           if self.role_counts.get(r.value, 0) > 0]) / len(UserRole)
        
        # Overall score with semantic weight
        overall_score = (
            type_distribution * 0.15 +
            role_distribution * 0.15 +
            intent_distribution * 0.25 +  # NEW - weighted higher
            focus_distribution * 0.20 +   # NEW
            unique_ratio * 0.10 +
            semantic_unique_ratio * 0.15  # NEW
        )
        
        return {
            "total_questions": total,
            "unique_semantic_fingerprints": len(self.semantic_fingerprints),
            "unique_ratio": unique_ratio,
            "semantic_unique_ratio": semantic_unique_ratio,
            "duplicates_rejected": self.duplicates_rejected,
            "similar_rejected": self.similar_rejected,
            "semantic_duplicate_rejected": self.semantic_duplicate_rejected,
            "type_distribution_score": type_distribution,
            "role_distribution_score": role_distribution,
            "intent_distribution_score": intent_distribution,
            "focus_distribution_score": focus_distribution,
            "type_coverage": type_coverage,
            "role_coverage": role_coverage,
            "intent_counts": dict(self.intent_counts),
            "focus_counts": dict(self.focus_counts),
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


# Backward compatibility alias
DiversityManager = DiversityManagerV2
