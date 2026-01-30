#!/usr/bin/env python3
"""
Enterprise Q&A Generation Engine v7.0 - Scalable Diversity System

Based on v6's hybrid architecture, v7 adds:
1. Total Q&A count control (--total parameter)
2. Enhanced diversity mechanisms for large-scale generation
3. Diversity metrics and validation

Problem:
- When questions_per_evidence is large, questions may become repetitive
- Need to maintain diversity, representativeness, and realism at scale

Solution:
┌─────────────────────────────────────────────────────────────────────┐
│                    Diversity Enhancement System                      │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                Global Question Deduplication                     ││
│  │  - Exact match detection                                         ││
│  │  - Semantic similarity scoring (Jaccard, N-gram)                ││
│  │  - Cross-evidence duplicate prevention                           ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │               Question Type Balancer                             ││
│  │  - Rotate through question types evenly                          ││
│  │  - Track type distribution                                       ││
│  │  - Enforce minimum coverage per type                             ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              Dynamic Template Expander                           ││
│  │  - Generate more candidates than needed                          ││
│  │  - Select most diverse subset                                    ││
│  │  - Context-aware template variation                              ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              Diversity Metrics Reporter                          ││
│  │  - Unique question ratio                                         ││
│  │  - Type distribution score                                       ││
│  │  - Role coverage score                                           ││
│  │  - Semantic diversity index                                      ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘

Parameters:
- --total: Maximum total Q&A pairs to generate (new)
- --questions: Questions per evidence (existing)
- When both specified: stops when either limit is reached

Author: Auto-generated
Version: 7.0.0
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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, Counter
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Global configuration."""
    VERSION = "7.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).parent.resolve()
    WORKSPACE_ROOT = BASE_DIR.parent.parent
    DATA_DIR = WORKSPACE_ROOT / "data"
    REPOS_DIR = WORKSPACE_ROOT / "repos"
    
    # Input/Output
    RULE_METADATA_FILE = DATA_DIR / "dbr01_rule_metadata.json"
    AST_ANALYSIS_FILE = DATA_DIR / "fastapi_analysis_result.json"
    OUTPUT_FILE = DATA_DIR / "qwen_dbr_training_logic_v7.jsonl"
    
    # LLM Configuration
    OLLAMA_API = "http://localhost:11434/api/generate"
    MODEL_NAME = "qwen2.5:7b"
    LLM_TIMEOUT = 180
    LLM_TEMPERATURE_QUESTION = 0.85
    LLM_TEMPERATURE_REASONING = 0.6
    LLM_TEMPERATURE_ANSWER = 0.7
    
    # Generation Parameters
    SUPPORTED_LANGUAGES = ["en", "zh"]
    DEFAULT_QUESTIONS_PER_EVIDENCE = 5
    DEFAULT_TOTAL_LIMIT = None  # No limit by default
    LLM_RETRY_COUNT = 2
    
    # Diversity Parameters
    SIMILARITY_THRESHOLD = 0.6  # Questions with similarity > this are considered duplicates
    MIN_QUESTION_TYPES_COVERAGE = 0.8  # At least 80% of question types should be covered
    CANDIDATE_MULTIPLIER = 2.0  # Generate 2x candidates for selection


# ============================================================================
# Enums
# ============================================================================

class UserRole(str, Enum):
    """User roles for question generation."""
    END_USER = "end_user"
    PRODUCT_MANAGER = "product_manager"
    QA_ENGINEER = "qa_engineer"
    SECURITY_AUDITOR = "security_auditor"
    NEW_DEVELOPER = "new_developer"


class QuestionType(str, Enum):
    """Types of questions for diversity tracking."""
    TROUBLESHOOTING = "troubleshooting"      # "I encountered this issue..."
    UNDERSTANDING = "understanding"           # "How does X work?"
    EDGE_CASE = "edge_case"                  # "What happens if...?"
    SECURITY = "security"                     # "Could an attacker...?"
    WHAT_IF = "what_if"                      # "What would happen if we remove...?"
    COMPARISON = "comparison"                 # "What's the difference between...?"
    VALIDATION = "validation"                 # "Is it correct that...?"
    DEEP_ANALYSIS = "deep_analysis"          # "What are the implications of...?"


# ============================================================================
# Diversity Enhancement System
# ============================================================================

class DiversityManager:
    """
    Manages question diversity across the entire generation process.
    Ensures questions remain diverse, representative, and realistic at scale.
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
        
        # Diversity metrics
        self.duplicates_rejected = 0
        self.similar_rejected = 0
    
    def is_diverse(self, question: str, question_type: QuestionType = None) -> Tuple[bool, str]:
        """
        Check if a question is sufficiently diverse from existing questions.
        Returns (is_diverse, rejection_reason).
        """
        # Normalize
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
        """Register a question for diversity tracking."""
        q_normalized = self._normalize(question)
        q_hash = hashlib.md5(q_normalized.encode()).hexdigest()
        q_ngrams = self._get_ngrams(q_normalized)
        
        self.all_questions.append(question)
        self.question_hashes.add(q_hash)
        self.question_ngrams[q_normalized] = q_ngrams
        
        self.type_counts[question_type.value if question_type else "unknown"] += 1
        self.role_counts[role.value] += 1
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
    
    def get_diversity_metrics(self) -> Dict:
        """Calculate diversity metrics."""
        total = len(self.all_questions)
        if total == 0:
            return {"error": "No questions generated"}
        
        # Type distribution score (entropy-based)
        type_distribution = self._calculate_distribution_score(self.type_counts, len(QuestionType))
        
        # Role distribution score
        role_distribution = self._calculate_distribution_score(self.role_counts, len(UserRole))
        
        # Unique ratio (accounting for rejections)
        total_attempted = total + self.duplicates_rejected + self.similar_rejected
        unique_ratio = total / max(total_attempted, 1)
        
        # Coverage scores
        type_coverage = len([t for t in QuestionType if self.type_counts.get(t.value, 0) > 0]) / len(QuestionType)
        role_coverage = len([r for r in UserRole if self.role_counts.get(r.value, 0) > 0]) / len(UserRole)
        
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
            "overall_diversity_score": (type_distribution + role_distribution + unique_ratio + type_coverage + role_coverage) / 5,
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
        """Calculate distribution score (1.0 = perfectly even, 0.0 = all in one)."""
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


# ============================================================================
# Deterministic Layer (from v6)
# ============================================================================

@dataclass
class CodeContext:
    """Deterministic code context from AST analysis."""
    file_path: str
    function_name: str
    line_start: int
    line_end: int
    code_snippet: str
    source_hash: str
    related_elements: List[str]
    call_chain: List[str] = field(default_factory=list)


class DeterministicLayer:
    """Provides deterministic, verifiable data from AST analysis."""
    
    def __init__(self, rule_metadata: Dict, ast_analysis: Dict = None):
        self.rule_metadata = rule_metadata
        self.ast_analysis = ast_analysis or {}
        self._call_graph = self._build_call_graph()
    
    def _build_call_graph(self) -> Dict[str, List[str]]:
        """Build call graph from AST analysis."""
        call_graph = defaultdict(list)
        for module in self.ast_analysis.get("modules", []):
            for func in module.get("functions", []):
                func_name = func.get("name", "")
                calls = func.get("calls", [])
                call_graph[func_name] = calls
        return call_graph
    
    def get_code_context(self, evidence: Dict) -> CodeContext:
        """Extract deterministic code context from evidence."""
        code_data = evidence.get("code_snippet", {})
        location = evidence.get("location", {})
        
        return CodeContext(
            file_path=code_data.get("file_path", location.get("file_path", "")),
            function_name=evidence.get("name", ""),
            line_start=code_data.get("line_start", location.get("line_start", 0)),
            line_end=code_data.get("line_end", location.get("line_end", 0)),
            code_snippet=code_data.get("code", ""),
            source_hash=code_data.get("source_hash", ""),
            related_elements=evidence.get("related_elements", []),
            call_chain=self._get_call_chain(evidence.get("name", ""))
        )
    
    def _get_call_chain(self, func_name: str, depth: int = 3) -> List[str]:
        """Get call chain for a function."""
        chain = []
        visited = set()
        
        def traverse(name, current_depth):
            if current_depth <= 0 or name in visited:
                return
            visited.add(name)
            chain.append(name)
            for called in self._call_graph.get(name, []):
                traverse(called, current_depth - 1)
        
        traverse(func_name, depth)
        return chain
    
    def get_dbr_logic(self, evidence: Dict) -> Dict:
        """Get DBR rule mapping."""
        dbr_logic = evidence.get("dbr_logic", {})
        return {
            "rule_id": dbr_logic.get("rule_id", "DBR-01"),
            "subcategory_id": dbr_logic.get("subcategory_id", ""),
            "trigger_type": dbr_logic.get("trigger_type", "explicit"),
            "weight": dbr_logic.get("weight", 1.0),
            "trigger_conditions": dbr_logic.get("trigger_conditions", []),
            "matched_patterns": dbr_logic.get("matched_patterns", []),
        }
    
    def verify_source_hash(self, code_snippet: str, expected_hash: str) -> bool:
        """Verify code snippet integrity."""
        if not expected_hash:
            return True
        computed = hashlib.md5(code_snippet.encode()).hexdigest()
        return computed == expected_hash


# ============================================================================
# Business Context Transformer (from v6)
# ============================================================================

class BusinessContextTransformer:
    """Transforms code-level evidence into business-friendly context."""
    
    SCENARIO_CONTEXTS = {
        "DBR-01-01": {
            "en": {
                "scenario_name": "User Registration & Profile Uniqueness",
                "business_flow": "New user registration and profile update process",
                "user_experience": "When users register or update their profile, the system validates that their chosen username and email are not already in use.",
                "success_outcome": "User successfully creates account or updates profile",
                "failure_outcome": "User sees an error message indicating the identifier is already taken",
                "edge_cases": [
                    "Two users attempting to register with the same email simultaneously",
                    "User trying to change their email to one already registered",
                    "Reusing a username that was previously deleted",
                    "Network timeout during the validation process",
                    "Database connection failure during registration",
                    "Very long usernames or emails at system limits",
                ],
                "security_concerns": [
                    "Account enumeration through different error message patterns",
                    "Race conditions that might allow duplicate accounts",
                    "Information disclosure about existing user accounts",
                    "Timing attacks revealing validation results",
                    "Automated account creation attempts",
                ],
                "business_rules": [
                    "Each email address can only be associated with one account",
                    "Usernames must be unique across the platform",
                    "Validation must occur before any data is persisted",
                ],
            },
            "zh": {
                "scenario_name": "用户注册与资料唯一性验证",
                "business_flow": "新用户注册和资料更新流程",
                "user_experience": "当用户注册或更新资料时，系统验证其选择的用户名和邮箱未被其他账户使用。",
                "success_outcome": "用户成功创建账户或更新资料",
                "failure_outcome": "用户看到错误消息，提示标识符已被占用",
                "edge_cases": [
                    "两个用户同时尝试使用相同邮箱注册",
                    "用户尝试将邮箱更改为已注册的邮箱",
                    "重用之前被删除的用户名",
                    "验证过程中网络超时",
                    "注册期间数据库连接失败",
                    "用户名或邮箱长度达到系统限制",
                ],
                "security_concerns": [
                    "通过不同错误消息模式进行账户枚举",
                    "可能导致重复账户的竞态条件",
                    "关于现有用户账户的信息泄露",
                    "揭示验证结果的时序攻击",
                    "自动化账户创建尝试",
                ],
                "business_rules": [
                    "每个邮箱地址只能关联一个账户",
                    "用户名在整个平台必须唯一",
                    "验证必须在任何数据持久化之前进行",
                ],
            }
        },
        "DBR-01-02": {
            "en": {
                "scenario_name": "Secure Account Creation & Credential Storage",
                "business_flow": "Account creation with secure credential handling",
                "user_experience": "When a new account is created, the system securely processes and stores the user's password.",
                "success_outcome": "Account is created with properly secured credentials",
                "failure_outcome": "Account creation fails and user is notified",
                "edge_cases": [
                    "Database failure during account creation",
                    "Server crash mid-registration",
                    "Extremely long or special character passwords",
                    "Concurrent account creation attempts",
                    "Password that matches common breach lists",
                    "Unicode or emoji characters in passwords",
                ],
                "security_concerns": [
                    "Password storage security and hashing",
                    "Atomicity of account creation transaction",
                    "Recovery from partial failures",
                    "Protection against credential stuffing",
                    "Secure random salt generation",
                ],
                "business_rules": [
                    "Passwords must never be stored in plain text",
                    "Account creation must be atomic",
                    "Failed creation must not leave partial data",
                ],
            },
            "zh": {
                "scenario_name": "安全账户创建与凭据存储",
                "business_flow": "带有安全凭据处理的账户创建",
                "user_experience": "创建新账户时，系统安全地处理和存储用户的密码。",
                "success_outcome": "账户以正确保护的凭据创建成功",
                "failure_outcome": "账户创建失败并通知用户",
                "edge_cases": [
                    "账户创建期间数据库故障",
                    "注册过程中服务器崩溃",
                    "超长或包含特殊字符的密码",
                    "并发账户创建尝试",
                    "密码与常见泄露列表匹配",
                    "密码中包含Unicode或表情符号",
                ],
                "security_concerns": [
                    "密码存储安全性和哈希处理",
                    "账户创建事务的原子性",
                    "部分失败后的恢复",
                    "防止凭据填充攻击",
                    "安全随机盐值生成",
                ],
                "business_rules": [
                    "密码绝不能以明文存储",
                    "账户创建必须是原子的",
                    "失败的创建不能留下部分数据",
                ],
            }
        },
        "DBR-01-03": {
            "en": {
                "scenario_name": "Login Authentication & Security Feedback",
                "business_flow": "User login and authentication process",
                "user_experience": "When users attempt to log in, the system verifies their credentials and provides appropriate feedback.",
                "success_outcome": "User is authenticated and granted access",
                "failure_outcome": "User receives a generic error message",
                "edge_cases": [
                    "Login attempt with non-existent email",
                    "Login with incorrect password for existing account",
                    "Multiple rapid login attempts",
                    "Login during account lockout period",
                    "Login from unusual location or device",
                    "Session timeout during login process",
                ],
                "security_concerns": [
                    "User enumeration through different error responses",
                    "Timing attacks that reveal account existence",
                    "Brute force password guessing prevention",
                    "Account lockout bypass attempts",
                    "Credential stuffing attacks",
                ],
                "business_rules": [
                    "Error messages must not reveal whether email exists",
                    "Same error response for wrong email and wrong password",
                    "Response time should be consistent regardless of failure reason",
                ],
            },
            "zh": {
                "scenario_name": "登录认证与安全反馈",
                "business_flow": "用户登录和认证流程",
                "user_experience": "当用户尝试登录时，系统验证其凭据并提供适当的反馈。",
                "success_outcome": "用户通过认证并获得访问权限",
                "failure_outcome": "用户收到通用错误消息",
                "edge_cases": [
                    "使用不存在的邮箱尝试登录",
                    "使用错误密码登录现有账户",
                    "多次快速登录尝试",
                    "在账户锁定期间尝试登录",
                    "从异常位置或设备登录",
                    "登录过程中会话超时",
                ],
                "security_concerns": [
                    "通过不同错误响应进行用户枚举",
                    "揭示账户存在的时序攻击",
                    "暴力破解密码的预防",
                    "账户锁定绕过尝试",
                    "凭据填充攻击",
                ],
                "business_rules": [
                    "错误消息不得透露邮箱是否存在",
                    "错误邮箱和错误密码返回相同的错误响应",
                    "无论失败原因响应时间应保持一致",
                ],
            }
        },
        "DBR-01-04": {
            "en": {
                "scenario_name": "Session Token Management & Refresh",
                "business_flow": "Authentication token lifecycle management",
                "user_experience": "After successful authentication, the system manages session tokens to maintain secure user sessions.",
                "success_outcome": "User maintains secure authenticated session",
                "failure_outcome": "Session expires and user must re-authenticate",
                "edge_cases": [
                    "Token expiration during active operation",
                    "Same user logged in on multiple devices",
                    "Token theft or session hijacking attempts",
                    "Server restart affecting active sessions",
                    "Clock skew between client and server",
                    "Concurrent requests with same token",
                ],
                "security_concerns": [
                    "Token security and proper lifetime management",
                    "Session fixation attack prevention",
                    "Concurrent session handling",
                    "Token refresh timing and security",
                    "Secure token storage on client side",
                ],
                "business_rules": [
                    "New token generated after successful authentication",
                    "Sensitive operations trigger token refresh",
                    "Tokens have appropriate expiration times",
                ],
            },
            "zh": {
                "scenario_name": "会话令牌管理与刷新",
                "business_flow": "认证令牌生命周期管理",
                "user_experience": "成功认证后，系统管理会话令牌以维护安全的用户会话。",
                "success_outcome": "用户维持安全的已认证会话",
                "failure_outcome": "会话过期，用户需要重新认证",
                "edge_cases": [
                    "活动操作期间令牌过期",
                    "同一用户在多台设备上登录",
                    "令牌盗窃或会话劫持尝试",
                    "服务器重启影响活动会话",
                    "客户端和服务器之间的时钟偏差",
                    "使用同一令牌的并发请求",
                ],
                "security_concerns": [
                    "令牌安全性和适当的生命周期管理",
                    "会话固定攻击预防",
                    "并发会话处理",
                    "令牌刷新时机和安全性",
                    "客户端令牌的安全存储",
                ],
                "business_rules": [
                    "成功认证后生成新令牌",
                    "敏感操作触发令牌刷新",
                    "令牌具有适当的过期时间",
                ],
            }
        },
    }
    
    ROLE_CONTEXTS = {
        UserRole.END_USER: {
            "en": "a regular user who encountered an issue or has questions about how things work",
            "zh": "一个遇到问题或对事物如何工作有疑问的普通用户",
        },
        UserRole.PRODUCT_MANAGER: {
            "en": "a product manager who needs to understand user flows and business logic",
            "zh": "一个需要了解用户流程和业务逻辑的产品经理",
        },
        UserRole.QA_ENGINEER: {
            "en": "a QA engineer testing edge cases and potential failure scenarios",
            "zh": "一个测试边界情况和潜在失败场景的QA工程师",
        },
        UserRole.SECURITY_AUDITOR: {
            "en": "a security auditor examining potential vulnerabilities",
            "zh": "一个检查潜在漏洞的安全审计员",
        },
        UserRole.NEW_DEVELOPER: {
            "en": "a new developer trying to understand how the system works",
            "zh": "一个试图理解系统如何工作的新开发者",
        },
    }
    
    @classmethod
    def transform(cls, evidence: Dict, subcategory_id: str, language: str = "en") -> Dict:
        """Transform code evidence into business context."""
        scenario = cls.SCENARIO_CONTEXTS.get(subcategory_id, {}).get(language, {})
        if not scenario:
            scenario = cls.SCENARIO_CONTEXTS.get(subcategory_id, {}).get("en", {})
        
        evidence_desc = evidence.get("description", "")
        abstracted_desc = cls._abstract_description(evidence_desc, language)
        
        return {
            "scenario_name": scenario.get("scenario_name", "Authentication Process"),
            "business_flow": scenario.get("business_flow", ""),
            "user_experience": scenario.get("user_experience", ""),
            "success_outcome": scenario.get("success_outcome", ""),
            "failure_outcome": scenario.get("failure_outcome", ""),
            "edge_cases": scenario.get("edge_cases", []),
            "security_concerns": scenario.get("security_concerns", []),
            "business_rules": scenario.get("business_rules", []),
            "specific_behavior": abstracted_desc,
            "language": language,
        }
    
    @classmethod
    def _abstract_description(cls, text: str, language: str) -> str:
        """Remove code-level details from description."""
        patterns = {
            r'\bcheck_username_is_taken\b': 'username availability check' if language == 'en' else '用户名可用性检查',
            r'\bcheck_email_is_taken\b': 'email uniqueness verification' if language == 'en' else '邮箱唯一性验证',
            r'\busers_repo\.create_user\b': 'account creation process' if language == 'en' else '账户创建流程',
            r'\busers_repo\.update_user\b': 'profile update process' if language == 'en' else '资料更新流程',
            r'\bcreate_access_token_for_user\b': 'session token generation' if language == 'en' else '会话令牌生成',
            r'\buser\.check_password\b': 'password verification' if language == 'en' else '密码验证',
            r'\bwrong_login_error\b': 'authentication error response' if language == 'en' else '认证错误响应',
            r'\buser_create\b': 'registration data' if language == 'en' else '注册数据',
            r'\buser_update\b': 'update data' if language == 'en' else '更新数据',
            r'\bHTTP_400_BAD_REQUEST\b': 'validation error' if language == 'en' else '验证错误',
            r'\bHTTP_401_UNAUTHORIZED\b': 'authentication failure' if language == 'en' else '认证失败',
            r'\bEntityDoesNotExist\b': 'resource not found' if language == 'en' else '资源不存在',
            r'\busers_repo\b': 'user data service' if language == 'en' else '用户数据服务',
        }
        
        result = text
        for pattern, replacement in patterns.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
    
    @classmethod
    def get_role_context(cls, role: UserRole, language: str = "en") -> str:
        return cls.ROLE_CONTEXTS.get(role, {}).get(language, "")


# ============================================================================
# Ollama LLM Client (from v6)
# ============================================================================

class OllamaClient:
    """Enhanced Ollama client."""
    
    def __init__(self):
        self.api_url = Config.OLLAMA_API
        self._available = None
        self._check_time = 0
    
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
    
    def generate(self, prompt: str, system: str = None, temperature: float = None) -> Optional[str]:
        if not self.is_available():
            return None
        
        temperature = temperature or 0.7
        
        try:
            payload = {"model": Config.MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
            if system:
                payload["system"] = system
            
            response = requests.post(self.api_url, json=payload, timeout=Config.LLM_TIMEOUT)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"LLM error: {e}")
        
        return None


# ============================================================================
# Enhanced Question Generator with Diversity (v7)
# ============================================================================

class DiverseQuestionGenerator:
    """
    Question generator with built-in diversity mechanisms.
    Generates candidates and selects the most diverse subset.
    """
    
    CODE_NAME_PATTERNS = [
        r'\bcheck_\w+\b', r'\busers_repo\b', r'\buser_create\b', r'\buser_update\b',
        r'\bHTTP_\d+\b', r'\bEntityDoesNotExist\b', r'\bwrong_login_error\b',
        r'\b[a-z]+_[a-z]+_[a-z]+\b',
    ]
    
    # Question type indicators for classification
    TYPE_INDICATORS = {
        QuestionType.TROUBLESHOOTING: ["problem", "issue", "error", "wrong", "trouble", "问题", "错误", "故障"],
        QuestionType.UNDERSTANDING: ["how does", "how do", "what is", "explain", "如何", "什么是", "解释"],
        QuestionType.EDGE_CASE: ["what happens if", "what if", "when", "如果", "当", "会发生什么"],
        QuestionType.SECURITY: ["attack", "secure", "vulnerability", "exploit", "攻击", "安全", "漏洞"],
        QuestionType.WHAT_IF: ["what would happen", "remove", "without", "如果去掉", "没有"],
        QuestionType.COMPARISON: ["difference", "compare", "versus", "区别", "比较"],
        QuestionType.VALIDATION: ["is it correct", "should", "supposed to", "是否正确", "应该"],
        QuestionType.DEEP_ANALYSIS: ["implications", "consequences", "impact", "影响", "后果"],
    }
    
    def __init__(self, llm_client: OllamaClient, diversity_manager: DiversityManager):
        self.llm = llm_client
        self.diversity_manager = diversity_manager
    
    def generate_diverse_questions(
        self,
        business_context: Dict,
        role: UserRole,
        requested_count: int,
        language: str = "en"
    ) -> List[Dict]:
        """Generate questions with diversity guarantees."""
        
        # Generate more candidates than needed
        candidate_count = int(requested_count * Config.CANDIDATE_MULTIPLIER) + 2
        
        # Try LLM first
        candidates = self._generate_llm_questions(business_context, role, candidate_count, language)
        
        # Add fallback questions
        fallback_questions = self._generate_fallback_questions(business_context, role, candidate_count, language)
        candidates.extend(fallback_questions)
        
        # Select most diverse subset
        selected = self._select_diverse_subset(candidates, requested_count)
        
        return selected
    
    def _generate_llm_questions(
        self,
        context: Dict,
        role: UserRole,
        count: int,
        language: str
    ) -> List[Dict]:
        """Generate questions using LLM."""
        if not self.llm.is_available():
            return []
        
        system_prompt = self._get_system_prompt(language)
        gen_prompt = self._get_generation_prompt(context, role, count, language)
        
        response = self.llm.generate(gen_prompt, system=system_prompt, temperature=Config.LLM_TEMPERATURE_QUESTION)
        
        if response:
            return self._parse_questions(response, role, language)
        return []
    
    def _get_system_prompt(self, language: str) -> str:
        if language == "zh":
            return """你正在模拟真实用户对Web应用认证系统提出问题。
规则：
1. 生成真实用户会问的问题 - 不知道内部函数名或代码结构
2. 问题必须自然且口语化
3. 禁止使用任何技术代码术语
4. 生成多样的问题类型：故障排除、理解流程、边界情况、安全关注、假设场景"""
        else:
            return """You are simulating real users asking questions about authentication systems.
Rules:
1. Generate questions real users would ask - they don't know internal code
2. Questions must be natural and conversational
3. NEVER use technical code terms
4. Generate diverse question types: troubleshooting, understanding, edge cases, security, what-if"""
    
    def _get_generation_prompt(self, context: Dict, role: UserRole, count: int, language: str) -> str:
        edge_cases = context.get("edge_cases", [])
        security = context.get("security_concerns", [])
        
        if language == "zh":
            return f"""场景：{context.get('scenario_name', '')}
用户体验：{context.get('user_experience', '')}
边界情况：{', '.join(edge_cases[:4])}
安全方面：{', '.join(security[:3])}

你的角色：{BusinessContextTransformer.get_role_context(role, language)}

生成{count}个多样化的问题。包含不同类型：故障排除、理解、边界情况、安全、假设场景。
每行一个问题："""
        else:
            return f"""Scenario: {context.get('scenario_name', '')}
User Experience: {context.get('user_experience', '')}
Edge Cases: {', '.join(edge_cases[:4])}
Security: {', '.join(security[:3])}

Your Role: {BusinessContextTransformer.get_role_context(role, language)}

Generate {count} diverse questions. Include different types: troubleshooting, understanding, edge cases, security, what-if.
One question per line:"""
    
    def _parse_questions(self, response: str, role: UserRole, language: str) -> List[Dict]:
        """Parse and validate LLM questions."""
        questions = []
        
        for line in response.strip().split('\n'):
            line = line.strip()
            line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line)
            
            if not line or len(line) < 15:
                continue
            
            if not (line.endswith('?') or line.endswith('？')):
                continue
            
            if self._contains_code_names(line):
                continue
            
            question_type = self._classify_question_type(line)
            
            questions.append({
                "question_id": f"LLM-{uuid.uuid4().hex[:8]}",
                "question_text": line,
                "source": "llm",
                "role": role.value,
                "language": language,
                "question_type": question_type,
            })
        
        return questions
    
    def _generate_fallback_questions(
        self,
        context: Dict,
        role: UserRole,
        count: int,
        language: str
    ) -> List[Dict]:
        """Generate diverse fallback questions."""
        templates = self._get_expanded_templates(context, role, language)
        
        questions = []
        random.shuffle(templates)
        
        for template in templates[:count]:
            question_type = self._classify_question_type(template)
            questions.append({
                "question_id": f"FB-{uuid.uuid4().hex[:8]}",
                "question_text": template,
                "source": "fallback",
                "role": role.value,
                "language": language,
                "question_type": question_type,
            })
        
        return questions
    
    def _get_expanded_templates(self, context: Dict, role: UserRole, language: str) -> List[str]:
        """Get expanded template set for better diversity."""
        scenario = context.get("scenario_name", "authentication")
        edge_cases = context.get("edge_cases", [])
        security = context.get("security_concerns", [])
        
        # Base templates by role
        base_templates = {
            "en": {
                UserRole.END_USER: [
                    "I tried to register but got an error. I've never used this site before. What's happening?",
                    "When I log in, I get the same error whether I use wrong password or wrong email. Is this normal?",
                    "I was updating my profile and something went wrong. Were my changes partially saved?",
                    "Why doesn't the login error tell me specifically what went wrong?",
                    "I'm worried my account might have been compromised. What should I do?",
                    "The system unexpectedly logged me out. Did someone access my account?",
                    "What happens if my internet disconnects during registration?",
                    "I forgot my password but I'm not sure if my account even exists.",
                    "Can I use the same email for multiple accounts?",
                    "My registration seemed to hang. Should I try again or wait?",
                ],
                UserRole.PRODUCT_MANAGER: [
                    f"Can you explain the complete user experience for {scenario}?",
                    "When a user encounters a duplicate email, what's the full flow?",
                    "What's the user experience if the registration process fails midway?",
                    "How does our session management affect users on multiple devices?",
                    "Why was the decision made not to tell users which credential was wrong?",
                    "What metrics should we track for registration success rates?",
                    "How do we handle users who abandon registration midway?",
                    "What's our fallback if the primary validation service is down?",
                ],
                UserRole.QA_ENGINEER: [
                    "What happens if two users register with the same email within milliseconds?",
                    "What's the expected behavior when the database becomes unavailable?",
                    "What race condition scenarios should we be testing?",
                    "Are there scenarios where a user could have a partially created account?",
                    "What happens if token generation fails after successful authentication?",
                    "How do we test for timing-based vulnerabilities?",
                    "What's the maximum username length and what happens if exceeded?",
                    "How does the system handle Unicode characters in usernames?",
                ],
                UserRole.SECURITY_AUDITOR: [
                    "Could an attacker determine valid email addresses by analyzing error messages?",
                    "Is there a timing difference that could reveal account existence?",
                    "What prevents username enumeration through the login endpoint?",
                    "How does the system prevent session fixation attacks?",
                    "Is there risk of information leakage through error message patterns?",
                    "What rate limiting is in place for failed login attempts?",
                    "How are passwords protected in transit and at rest?",
                    "Could automated tools exploit any predictable behavior?",
                ],
                UserRole.NEW_DEVELOPER: [
                    "From a user's perspective, how does the authentication flow work?",
                    "Why do different login failures return the same error?",
                    "How does the system ensure usernames and emails stay unique?",
                    "What security measures are used in the password storage process?",
                    "How are sessions managed after successful login?",
                    "What happens to a user's session when they change their password?",
                    "How does the system handle session expiration gracefully?",
                    "What's the user experience for first-time vs returning users?",
                ],
            },
            "zh": {
                UserRole.END_USER: [
                    "我尝试注册但显示错误。我从没用过这个网站。发生了什么？",
                    "登录时无论密码错还是邮箱错都显示相同错误。这正常吗？",
                    "更新资料时出了问题。我的更改是否只保存了一部分？",
                    "为什么登录错误不具体告诉我哪里错了？",
                    "我担心账户可能被盗用了。我应该怎么做？",
                    "系统意外把我登出了。有人访问了我的账户吗？",
                    "如果注册时网络中断会怎样？",
                    "我忘记密码了，但不确定我的账户是否存在。",
                    "我可以用同一个邮箱注册多个账户吗？",
                    "我的注册似乎卡住了。应该再试一次还是等待？",
                ],
                UserRole.PRODUCT_MANAGER: [
                    f"能解释一下{scenario}的完整用户体验吗？",
                    "当用户遇到重复邮箱时，整个流程是什么？",
                    "如果注册过程中途失败，用户体验是什么？",
                    "会话管理如何影响多设备用户？",
                    "为什么决定不告诉用户具体哪个凭据错了？",
                    "我们应该跟踪哪些注册成功率指标？",
                    "如何处理中途放弃注册的用户？",
                    "如果主验证服务宕机，我们的后备方案是什么？",
                ],
                UserRole.QA_ENGINEER: [
                    "如果两个用户在几毫秒内用相同邮箱注册会发生什么？",
                    "当数据库不可用时预期行为是什么？",
                    "我们应该测试哪些竞态条件场景？",
                    "是否存在用户可能得到部分创建账户的场景？",
                    "成功认证后令牌生成失败会怎样？",
                    "如何测试基于时序的漏洞？",
                    "用户名最大长度是多少？超过会怎样？",
                    "系统如何处理用户名中的Unicode字符？",
                ],
                UserRole.SECURITY_AUDITOR: [
                    "攻击者能否通过分析错误消息确定有效邮箱？",
                    "是否存在可能揭示账户存在的时序差异？",
                    "什么机制防止通过登录端点枚举用户名？",
                    "系统如何防止会话固定攻击？",
                    "错误消息模式是否存在信息泄露风险？",
                    "对失败登录尝试有什么速率限制？",
                    "密码在传输和存储时如何保护？",
                    "自动化工具能否利用任何可预测的行为？",
                ],
                UserRole.NEW_DEVELOPER: [
                    "从用户角度看，认证流程如何工作？",
                    "为什么不同登录失败都返回相同错误？",
                    "系统如何确保用户名和邮箱唯一？",
                    "密码存储过程使用什么安全措施？",
                    "成功登录后会话如何管理？",
                    "用户更改密码后会话会发生什么？",
                    "系统如何优雅地处理会话过期？",
                    "首次用户和回访用户的体验有什么不同？",
                ],
            }
        }
        
        templates = base_templates.get(language, base_templates["en"]).get(role, [])
        
        # Dynamically expand with edge cases
        for edge_case in edge_cases[:3]:
            if language == "zh":
                templates.append(f"如果{edge_case}会发生什么？")
                templates.append(f"系统如何处理{edge_case}的情况？")
            else:
                templates.append(f"What happens if {edge_case.lower()}?")
                templates.append(f"How does the system handle {edge_case.lower()}?")
        
        # Expand with security concerns
        for concern in security[:2]:
            if language == "zh":
                templates.append(f"关于{concern}，系统有什么防护措施？")
            else:
                templates.append(f"What protections are in place against {concern.lower()}?")
        
        return templates
    
    def _select_diverse_subset(self, candidates: List[Dict], target_count: int) -> List[Dict]:
        """Select the most diverse subset from candidates."""
        if len(candidates) <= target_count:
            return candidates
        
        selected = []
        selected_texts = set()
        
        # First, ensure type diversity
        type_buckets = defaultdict(list)
        for candidate in candidates:
            qtype = candidate.get("question_type", QuestionType.UNDERSTANDING)
            type_buckets[qtype].append(candidate)
        
        # Take one from each type first
        for qtype in QuestionType:
            if type_buckets[qtype] and len(selected) < target_count:
                candidate = type_buckets[qtype].pop(0)
                q_text = candidate["question_text"].lower()
                
                # Check diversity
                is_diverse, _ = self.diversity_manager.is_diverse(candidate["question_text"])
                if is_diverse and q_text not in selected_texts:
                    selected.append(candidate)
                    selected_texts.add(q_text)
        
        # Fill remaining with diverse selections
        remaining = [c for bucket in type_buckets.values() for c in bucket]
        random.shuffle(remaining)
        
        for candidate in remaining:
            if len(selected) >= target_count:
                break
            
            q_text = candidate["question_text"].lower()
            is_diverse, _ = self.diversity_manager.is_diverse(candidate["question_text"])
            
            if is_diverse and q_text not in selected_texts:
                selected.append(candidate)
                selected_texts.add(q_text)
        
        return selected
    
    def _contains_code_names(self, text: str) -> bool:
        for pattern in self.CODE_NAME_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _classify_question_type(self, question: str) -> QuestionType:
        """Classify question into a type."""
        question_lower = question.lower()
        
        for qtype, indicators in self.TYPE_INDICATORS.items():
            if any(ind in question_lower for ind in indicators):
                return qtype
        
        return QuestionType.UNDERSTANDING


# ============================================================================
# Reasoning and Answer Generators (from v6)
# ============================================================================

class LLMReasoningGenerator:
    """Generates human-like reasoning chains."""
    
    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
    
    def generate_reasoning(
        self,
        question: str,
        business_context: Dict,
        code_context: CodeContext,
        language: str = "en"
    ) -> List[str]:
        # Try LLM
        if self.llm.is_available():
            response = self.llm.generate(
                self._get_prompt(question, business_context, code_context, language),
                temperature=Config.LLM_TEMPERATURE_REASONING
            )
            if response:
                return self._parse_reasoning(response)
        
        # Fallback
        return self._generate_fallback(business_context, code_context, language)
    
    def _get_prompt(self, question: str, context: Dict, code: CodeContext, language: str) -> str:
        if language == "zh":
            return f"""分析此问题并提供推理步骤。
问题：{question}
场景：{context.get('scenario_name', '')}
用户体验：{context.get('user_experience', '')}

提供4-5个推理步骤：[理解]、[识别]、[追踪]、[安全]、[结论]"""
        else:
            return f"""Analyze this question and provide reasoning steps.
Question: {question}
Scenario: {context.get('scenario_name', '')}
User Experience: {context.get('user_experience', '')}

Provide 4-5 reasoning steps: [UNDERSTAND], [IDENTIFY], [TRACE], [SECURITY], [CONCLUDE]"""
    
    def _parse_reasoning(self, response: str) -> List[str]:
        steps = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line.startswith('[') or re.match(r'^\d+\.', line)):
                steps.append(line)
        return steps[:6] if steps else [response[:200]]
    
    def _generate_fallback(self, context: Dict, code: CodeContext, language: str) -> List[str]:
        if language == "zh":
            return [
                f"[理解] 问题涉及{context.get('scenario_name', '认证流程')}",
                f"[识别] 系统行为：{context.get('user_experience', '')}",
                f"[追踪] 相关代码位于 {code.file_path} (第{code.line_start}-{code.line_end}行)",
                f"[安全] 安全考虑：{context.get('security_concerns', ['安全措施已实施'])[0]}",
                f"[结论] 系统正确实现了该场景的业务逻辑",
            ]
        else:
            return [
                f"[UNDERSTAND] Question relates to {context.get('scenario_name', 'authentication')}",
                f"[IDENTIFY] System behavior: {context.get('user_experience', '')}",
                f"[TRACE] Relevant code in {code.file_path} (lines {code.line_start}-{code.line_end})",
                f"[SECURITY] Security consideration: {context.get('security_concerns', ['Security measures'])[0]}",
                f"[CONCLUDE] System correctly implements the business logic",
            ]


class LLMAnswerGenerator:
    """Generates comprehensive answers."""
    
    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
    
    def generate_answer(
        self,
        question: str,
        business_context: Dict,
        code_context: CodeContext,
        reasoning: List[str],
        language: str = "en"
    ) -> str:
        if self.llm.is_available():
            response = self.llm.generate(
                self._get_prompt(question, business_context, code_context, reasoning, language),
                temperature=Config.LLM_TEMPERATURE_ANSWER
            )
            if response:
                return self._format_answer(response, code_context, language)
        
        return self._generate_fallback(business_context, code_context, reasoning, language)
    
    def _get_prompt(self, question: str, context: Dict, code: CodeContext, reasoning: List[str], language: str) -> str:
        if language == "zh":
            return f"""为此问题提供全面回答。
问题：{question}
场景：{context.get('scenario_name', '')}
行为：{context.get('user_experience', '')}
推理：{chr(10).join(reasoning)}

提供全面回答：直接回应问题、解释行为、讨论安全影响。"""
        else:
            return f"""Provide a comprehensive answer.
Question: {question}
Scenario: {context.get('scenario_name', '')}
Behavior: {context.get('user_experience', '')}
Reasoning: {chr(10).join(reasoning)}

Provide comprehensive answer: address the question, explain behavior, discuss security."""
    
    def _format_answer(self, response: str, code: CodeContext, language: str) -> str:
        code_ref = f"""

### {'代码参考' if language == 'zh' else 'Code Reference'}

`{code.file_path}` ({'第' if language == 'zh' else 'lines'} {code.line_start}-{code.line_end}):

```python
{code.code_snippet[:600]}
```"""
        return response + code_ref
    
    def _generate_fallback(self, context: Dict, code: CodeContext, reasoning: List[str], language: str) -> str:
        if language == "zh":
            return f"""### 回答

关于{context.get('scenario_name', '该场景')}：

**系统行为**：{context.get('user_experience', '')}

**安全考虑**：
{chr(10).join('- ' + c for c in context.get('security_concerns', [])[:3])}

### 推理过程
{chr(10).join(reasoning)}

### 代码参考
`{code.file_path}` (第{code.line_start}-{code.line_end}行):
```python
{code.code_snippet[:500]}
```"""
        else:
            return f"""### Answer

Regarding {context.get('scenario_name', 'this scenario')}:

**System Behavior**: {context.get('user_experience', '')}

**Security Considerations**:
{chr(10).join('- ' + c for c in context.get('security_concerns', [])[:3])}

### Reasoning Process
{chr(10).join(reasoning)}

### Code Reference
`{code.file_path}` (lines {code.line_start}-{code.line_end}):
```python
{code.code_snippet[:500]}
```"""


# ============================================================================
# Quality Assurance (from v6)
# ============================================================================

class QualityAssurance:
    """Ensures generated Q&A pairs meet quality standards."""
    
    CODE_NAME_PATTERNS = [
        r'\bcheck_\w+\b', r'\busers_repo\b', r'\buser_create\b',
        r'\bHTTP_\d+\b', r'\bEntityDoesNotExist\b', r'\bwrong_login_error\b',
    ]
    
    def validate(self, qa_pair: Dict) -> Tuple[bool, float, List[str]]:
        issues = []
        scores = []
        
        question = qa_pair.get("instruction", "")
        answer = qa_pair.get("answer", "")
        
        # Question checks
        if len(question) < 15:
            issues.append("Question too short")
            scores.append(0.3)
        else:
            scores.append(1.0)
        
        if not (question.endswith('?') or question.endswith('？')):
            issues.append("Missing question mark")
            scores.append(0.8)
        else:
            scores.append(1.0)
        
        if self._contains_code_names(question):
            issues.append("Question contains code names")
            scores.append(0.0)
        else:
            scores.append(1.0)
        
        # Answer checks
        if len(answer) < 100:
            issues.append("Answer too short")
            scores.append(0.3)
        else:
            scores.append(1.0)
        
        avg_score = sum(scores) / len(scores)
        is_valid = avg_score >= 0.7 and "Question contains code names" not in issues
        
        return is_valid, avg_score, issues
    
    def _contains_code_names(self, text: str) -> bool:
        for pattern in self.CODE_NAME_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


# ============================================================================
# Main Orchestrator (v7)
# ============================================================================

class ScalableDiverseOrchestrator:
    """
    Main orchestrator with scalable diversity controls.
    Supports both total limit and per-evidence limits.
    """
    
    def __init__(self, rule_metadata_path: str, ast_analysis_path: str = None):
        self.rule_metadata_path = Path(rule_metadata_path)
        self.ast_analysis_path = Path(ast_analysis_path) if ast_analysis_path else None
        
        self.rule_metadata: Dict = {}
        self.ast_analysis: Dict = {}
        
        self.llm = OllamaClient()
        self.diversity_manager = DiversityManager()
        
        self.deterministic_layer: Optional[DeterministicLayer] = None
        self.question_generator: Optional[DiverseQuestionGenerator] = None
        self.reasoning_generator: Optional[LLMReasoningGenerator] = None
        self.answer_generator: Optional[LLMAnswerGenerator] = None
        self.quality_assurance: Optional[QualityAssurance] = None
        
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
                logger.info("Loaded AST analysis")
            
            self.deterministic_layer = DeterministicLayer(self.rule_metadata, self.ast_analysis)
            self.question_generator = DiverseQuestionGenerator(self.llm, self.diversity_manager)
            self.reasoning_generator = LLMReasoningGenerator(self.llm)
            self.answer_generator = LLMAnswerGenerator(self.llm)
            self.quality_assurance = QualityAssurance()
            
            if self.llm.is_available():
                logger.info(f"✓ LLM available: {Config.MODEL_NAME}")
            else:
                logger.warning("✗ LLM not available - using fallback templates")
            
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
        """
        Run generation pipeline with configurable limits.
        
        Args:
            questions_per_evidence: Max questions per evidence (default: 5)
            total_limit: Max total Q&A pairs (default: no limit)
            languages: Languages to generate (default: ['en', 'zh'])
        """
        questions_per_evidence = questions_per_evidence or Config.DEFAULT_QUESTIONS_PER_EVIDENCE
        total_limit = total_limit or Config.DEFAULT_TOTAL_LIMIT
        languages = languages or Config.SUPPORTED_LANGUAGES
        
        self.generated_pairs = []
        self.stats = defaultdict(int)
        self.diversity_manager = DiversityManager()  # Reset
        self.question_generator.diversity_manager = self.diversity_manager
        
        logger.info("=" * 60)
        logger.info(f"Starting Scalable Diverse Q&A Pipeline v{Config.VERSION}")
        logger.info("=" * 60)
        logger.info(f"  Questions/evidence: {questions_per_evidence}")
        logger.info(f"  Total limit: {total_limit if total_limit else 'No limit'}")
        logger.info(f"  Languages: {languages}")
        logger.info(f"  LLM: {Config.MODEL_NAME} ({'available' if self.llm.is_available() else 'fallback'})")
        logger.info("=" * 60)
        
        for subcategory in self.rule_metadata.get("subcategories", []):
            if total_limit and len(self.generated_pairs) >= total_limit:
                logger.info(f"Reached total limit ({total_limit})")
                break
            
            self._process_subcategory(
                subcategory, questions_per_evidence, total_limit, languages
            )
        
        logger.info(f"Generated {len(self.generated_pairs)} Q&A pairs")
        return self.generated_pairs
    
    def _process_subcategory(
        self,
        subcategory: Dict,
        questions_per_evidence: int,
        total_limit: Optional[int],
        languages: List[str],
    ):
        subcategory_id = subcategory.get("subcategory_id", "")
        logger.info(f"Processing: {subcategory_id}")
        
        for evidence in subcategory.get("evidences", []):
            if total_limit and len(self.generated_pairs) >= total_limit:
                return
            
            self._process_evidence(
                evidence, subcategory_id, questions_per_evidence, total_limit, languages
            )
    
    def _process_evidence(
        self,
        evidence: Dict,
        subcategory_id: str,
        questions_per_evidence: int,
        total_limit: Optional[int],
        languages: List[str],
    ):
        code_context = self.deterministic_layer.get_code_context(evidence)
        dbr_logic = self.deterministic_layer.get_dbr_logic(evidence)
        
        if not code_context.code_snippet:
            return
        
        hash_valid = self.deterministic_layer.verify_source_hash(
            code_context.code_snippet, code_context.source_hash
        )
        
        # Prioritize underrepresented roles
        roles = self.diversity_manager.get_underrepresented_roles()
        if not roles:
            roles = list(UserRole)
        random.shuffle(roles)
        
        for language in languages:
            if total_limit and len(self.generated_pairs) >= total_limit:
                return
            
            business_context = BusinessContextTransformer.transform(
                evidence, subcategory_id, language
            )
            
            questions_generated = 0
            
            for role in roles:
                if questions_generated >= questions_per_evidence:
                    break
                if total_limit and len(self.generated_pairs) >= total_limit:
                    return
                
                remaining = questions_per_evidence - questions_generated
                count = min(2, remaining)
                
                questions = self.question_generator.generate_diverse_questions(
                    business_context, role, count, language
                )
                
                for question in questions:
                    if questions_generated >= questions_per_evidence:
                        break
                    if total_limit and len(self.generated_pairs) >= total_limit:
                        return
                    
                    # Check diversity
                    is_diverse, reason = self.diversity_manager.is_diverse(question["question_text"])
                    if not is_diverse:
                        self.stats[f"rejected_{reason.split(':')[0]}"] += 1
                        continue
                    
                    self._generate_qa_pair(
                        question, evidence, business_context, code_context,
                        dbr_logic, subcategory_id, language, hash_valid
                    )
                    questions_generated += 1
    
    def _generate_qa_pair(
        self,
        question: Dict,
        evidence: Dict,
        business_context: Dict,
        code_context: CodeContext,
        dbr_logic: Dict,
        subcategory_id: str,
        language: str,
        hash_valid: bool
    ):
        question_text = question.get("question_text", "")
        question_type = question.get("question_type", QuestionType.UNDERSTANDING)
        
        reasoning = self.reasoning_generator.generate_reasoning(
            question_text, business_context, code_context, language
        )
        
        answer = self.answer_generator.generate_answer(
            question_text, business_context, code_context, reasoning, language
        )
        
        qa_pair = {
            "sample_id": f"DBR01-V7-{uuid.uuid4().hex[:10]}",
            "instruction": question_text,
            "context": {
                "file_path": code_context.file_path,
                "related_dbr": dbr_logic.get("rule_id", "DBR-01"),
                "code_snippet": code_context.code_snippet,
                "line_range": f"{code_context.line_start}-{code_context.line_end}",
                "function_name": code_context.function_name,
                "call_chain": code_context.call_chain[:3],
            },
            "auto_processing": {
                "parser": "FastAPI-AST-Analyzer",
                "parser_version": "1.0.0",
                "dbr_logic": dbr_logic,
                "generation_metadata": {
                    "version": Config.VERSION,
                    "architecture": "hybrid_diverse",
                    "question_source": question.get("source", "unknown"),
                    "user_role": question.get("role", "unknown"),
                    "question_type": question_type.value if isinstance(question_type, QuestionType) else str(question_type),
                    "llm_model": Config.MODEL_NAME if question.get("source") == "llm" else None,
                },
                "data_cleaning": {
                    "cleaned": True,
                    "source_verified": hash_valid,
                    "code_names_checked": True,
                    "diversity_checked": True,
                },
            },
            "reasoning_trace": reasoning,
            "answer": answer,
            "data_quality": {
                "consistency_check": hash_valid,
                "source_hash": code_context.source_hash,
                "language": language,
                "temperature": Config.LLM_TEMPERATURE_ANSWER,
                "evidence_id": evidence.get("evidence_id", ""),
            },
        }
        
        is_valid, score, issues = self.quality_assurance.validate(qa_pair)
        qa_pair["data_quality"]["quality_score"] = score
        qa_pair["data_quality"]["validation_issues"] = issues
        
        if is_valid:
            # Register with diversity manager
            self.diversity_manager.add_question(
                question_text,
                question_type,
                UserRole(question.get("role", "end_user")),
                business_context.get("scenario_name", ""),
                language
            )
            
            self.generated_pairs.append(qa_pair)
            self.stats["valid"] += 1
            self.stats[f"source_{question.get('source', 'unknown')}"] += 1
            self.stats[f"role_{question.get('role', 'unknown')}"] += 1
            self.stats[f"type_{question_type.value if isinstance(question_type, QuestionType) else str(question_type)}"] += 1
        else:
            self.stats["invalid"] += 1
    
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
        print(f"Scalable Diverse Q&A Generation Summary (v{Config.VERSION})")
        print("=" * 70)
        
        print(f"\nTotal Generated: {len(self.generated_pairs)}")
        print(f"  - Valid: {self.stats.get('valid', 0)}")
        print(f"  - Invalid: {self.stats.get('invalid', 0)}")
        
        # Diversity metrics
        diversity = self.diversity_manager.get_diversity_metrics()
        
        print(f"\n📊 Diversity Metrics:")
        print(f"  - Overall Diversity Score: {diversity.get('overall_diversity_score', 0):.2%}")
        print(f"  - Unique Ratio: {diversity.get('unique_ratio', 0):.2%}")
        print(f"  - Type Coverage: {diversity.get('type_coverage', 0):.2%}")
        print(f"  - Role Coverage: {diversity.get('role_coverage', 0):.2%}")
        print(f"  - Duplicates Rejected: {diversity.get('duplicates_rejected', 0)}")
        print(f"  - Similar Rejected: {diversity.get('similar_rejected', 0)}")
        
        print(f"\n📋 Question Types:")
        for qtype in QuestionType:
            count = self.stats.get(f"type_{qtype.value}", 0)
            if count > 0:
                print(f"  - {qtype.value}: {count}")
        
        print(f"\n👥 User Roles:")
        for role in UserRole:
            count = self.stats.get(f"role_{role.value}", 0)
            if count > 0:
                print(f"  - {role.value}: {count}")
        
        print(f"\n🌐 Languages:")
        for lang, count in diversity.get("language_counts", {}).items():
            print(f"  - {lang}: {count}")
        
        print("\n" + "=" * 70)
    
    def print_sample_pairs(self, n: int = 3):
        if not self.generated_pairs:
            return
        
        for i, pair in enumerate(self.generated_pairs[:n]):
            print("\n" + "=" * 70)
            meta = pair.get("auto_processing", {}).get("generation_metadata", {})
            print(f"[Sample {i+1}] Type: {meta.get('question_type', 'N/A')} | Role: {meta.get('user_role', 'N/A')}")
            print("=" * 70)
            print(f"\n【Question】:\n{pair['instruction']}")
            print(f"\n【Answer (excerpt)】:\n{pair['answer'][:500]}...")
            print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description=f'Scalable Diverse Q&A Generation v{Config.VERSION}'
    )
    
    parser.add_argument('-m', '--metadata', default=str(Config.RULE_METADATA_FILE))
    parser.add_argument('-a', '--ast', default=str(Config.AST_ANALYSIS_FILE))
    parser.add_argument('-o', '--output', default=str(Config.OUTPUT_FILE))
    
    # New parameters
    parser.add_argument(
        '-n', '--questions',
        type=int,
        default=5,
        help='Questions per evidence (default: 5)'
    )
    
    parser.add_argument(
        '-t', '--total',
        type=int,
        default=None,
        help='Total Q&A pairs limit (default: no limit)'
    )
    
    parser.add_argument('-l', '--languages', nargs='+', default=['en', 'zh'])
    parser.add_argument('--preview', type=int, default=3)
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    orchestrator = ScalableDiverseOrchestrator(args.metadata, args.ast)
    
    if not orchestrator.initialize():
        print("Error: Failed to initialize.")
        sys.exit(1)
    
    print(f"\n🚀 Running Q&A Generation Pipeline v{Config.VERSION}")
    print(f"   Questions per evidence: {args.questions}")
    print(f"   Total limit: {args.total if args.total else 'No limit'}")
    print(f"   Languages: {args.languages}")
    
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
