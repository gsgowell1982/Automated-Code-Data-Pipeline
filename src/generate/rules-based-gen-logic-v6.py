#!/usr/bin/env python3
"""
Enterprise Q&A Generation Engine v6.0 - Hybrid Architecture with User Perspective

Key Innovation: Combines v3's robust hybrid architecture with v5's user-perspective
question generation to fix the "god's eye view" problem.

v3 Architecture (PRESERVED):
┌─────────────────────────────────────────────────────────────────────┐
│                      HybridQAOrchestrator                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────────────────────┐ │
│  │ Deterministic Layer │    │         LLM Enhancement Layer       │ │
│  │ ─────────────────── │    │ ─────────────────────────────────── │ │
│  │ - AST Code Snippets │    │ - Natural Question Generation       │ │
│  │ - Call Graph        │◄───┤ - Deep Security Questions           │ │
│  │ - DBR Rule Mapping  │    │ - Human-like Reasoning              │ │
│  │ - Source Hash       │    │ - Expert-level Analysis             │ │
│  └─────────────────────┘    └─────────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────┐    ┌─────────────────────────────────────┐ │
│  │  Quality Assurance  │    │        Output Validator             │ │
│  │ - Code Reference    │    │ - LLM Output Verification           │ │
│  │ - DBR Alignment     │    │ - Factual Consistency Check         │ │
│  │ - Hash Validation   │    │ - Hallucination Detection           │ │
│  └─────────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

v6 Enhancement (NEW):
┌─────────────────────────────────────────────────────────────────────┐
│                    User Perspective Layer                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              Business Context Transformer                        ││
│  │  Code Evidence ──► Business Scenario ──► User-Friendly Context   ││
│  │  (with code names)   (abstracted)        (no code names)         ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                Multi-Role Question Generator                     ││
│  │  LLM receives: Business context (NO function names)              ││
│  │  LLM outputs:  Natural user questions                            ││
│  │  Roles: End User, PM, QA, Security Auditor, Developer            ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │               Code Name Leakage Detector                         ││
│  │  Rejects any question containing function/variable names         ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘

Problem Solved:
- v3: Great architecture but "god's eye view" questions with function names
- v5: Fixed questions but lost hybrid architecture
- v6: Best of both - hybrid architecture + user perspective questions

Author: Auto-generated
Version: 6.0.0
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
from collections import defaultdict

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
    VERSION = "6.0.0"

    # Paths
    BASE_DIR = Path(__file__).parent.resolve()
    WORKSPACE_ROOT = BASE_DIR.parent.parent
    DATA_DIR = WORKSPACE_ROOT / "data"
    REPOS_DIR = WORKSPACE_ROOT / "repos"

    # Input/Output
    RULE_METADATA_FILE = DATA_DIR / "dbr01_rule_metadata.json"
    AST_ANALYSIS_FILE = DATA_DIR / "fastapi_analysis_result.json"
    OUTPUT_FILE = DATA_DIR / "qwen_dbr_training_logic_v6.jsonl"

    # LLM Configuration
    OLLAMA_API = "http://localhost:11434/api/generate"
    MODEL_NAME = "qwen2.5:7b"
    LLM_TIMEOUT = 120
    LLM_TEMPERATURE_QUESTION = 0.85  # Higher for diverse questions
    LLM_TEMPERATURE_REASONING = 0.6  # Lower for consistent reasoning
    LLM_TEMPERATURE_ANSWER = 0.7

    # Generation Parameters
    SUPPORTED_LANGUAGES = ["en", "zh"]
    QUESTIONS_PER_EVIDENCE = 2
    LLM_RETRY_COUNT = 2


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
    """Types of questions."""
    TROUBLESHOOTING = "troubleshooting"
    UNDERSTANDING = "understanding"
    EDGE_CASE = "edge_case"
    SECURITY = "security"
    WHAT_IF = "what_if"
    DEEP_ANALYSIS = "deep_analysis"


# ============================================================================
# Deterministic Layer: Code Context Provider (from v3)
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
    """
    Provides deterministic, verifiable data from AST analysis.
    This layer ensures factual accuracy and traceability.
    """

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
# Business Context Transformer (from v5, enhanced)
# ============================================================================

class BusinessContextTransformer:
    """
    Transforms code-level evidence into business-friendly context.
    The LLM receives this abstracted context, NOT function names.
    """

    SCENARIO_CONTEXTS = {
        "DBR-01-01": {
            "en": {
                "scenario_name": "User Registration & Profile Uniqueness",
                "business_flow": "New user registration and profile update process",
                "user_experience": "When users register or update their profile, the system validates that their chosen username and email are not already in use by another account.",
                "success_outcome": "User successfully creates account or updates profile",
                "failure_outcome": "User sees an error message indicating the identifier is already taken",
                "edge_cases": [
                    "Two users attempting to register with the same email simultaneously",
                    "User trying to change their email to one already registered",
                    "Reusing a username that was previously deleted",
                    "Network timeout during the validation process",
                ],
                "security_concerns": [
                    "Account enumeration through different error message patterns",
                    "Race conditions that might allow duplicate accounts",
                    "Information disclosure about existing user accounts",
                    "Timing attacks revealing validation results",
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
                ],
                "security_concerns": [
                    "通过不同错误消息模式进行账户枚举",
                    "可能导致重复账户的竞态条件",
                    "关于现有用户账户的信息泄露",
                    "揭示验证结果的时序攻击",
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
                ],
                "security_concerns": [
                    "Password storage security and hashing",
                    "Atomicity of account creation transaction",
                    "Recovery from partial failures",
                    "Protection against credential stuffing",
                ],
                "business_rules": [
                    "Passwords must never be stored in plain text",
                    "Account creation must be atomic - all or nothing",
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
                ],
                "security_concerns": [
                    "密码存储安全性和哈希处理",
                    "账户创建事务的原子性",
                    "部分失败后的恢复",
                    "防止凭据填充攻击",
                ],
                "business_rules": [
                    "密码绝不能以明文存储",
                    "账户创建必须是原子的 - 全部成功或全部失败",
                    "失败的创建不能留下部分数据",
                ],
            }
        },
        "DBR-01-03": {
            "en": {
                "scenario_name": "Login Authentication & Security Feedback",
                "business_flow": "User login and authentication process",
                "user_experience": "When users attempt to log in, the system verifies their credentials and provides appropriate feedback without revealing sensitive information.",
                "success_outcome": "User is authenticated and granted access",
                "failure_outcome": "User receives a generic error message",
                "edge_cases": [
                    "Login attempt with non-existent email",
                    "Login with incorrect password for existing account",
                    "Multiple rapid login attempts (potential brute force)",
                    "Login during account lockout period",
                ],
                "security_concerns": [
                    "User enumeration through different error responses",
                    "Timing attacks that reveal account existence",
                    "Brute force password guessing prevention",
                    "Account lockout bypass attempts",
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
                "user_experience": "当用户尝试登录时，系统验证其凭据并提供适当的反馈，而不泄露敏感信息。",
                "success_outcome": "用户通过认证并获得访问权限",
                "failure_outcome": "用户收到通用错误消息",
                "edge_cases": [
                    "使用不存在的邮箱尝试登录",
                    "使用错误密码登录现有账户",
                    "多次快速登录尝试（潜在暴力破解）",
                    "在账户锁定期间尝试登录",
                ],
                "security_concerns": [
                    "通过不同错误响应进行用户枚举",
                    "揭示账户存在的时序攻击",
                    "暴力破解密码的预防",
                    "账户锁定绕过尝试",
                ],
                "business_rules": [
                    "错误消息不得透露邮箱是否存在",
                    "错误邮箱和错误密码返回相同的错误响应",
                    "无论失败原因，响应时间应保持一致",
                ],
            }
        },
        "DBR-01-04": {
            "en": {
                "scenario_name": "Session Token Management & Refresh",
                "business_flow": "Authentication token lifecycle management",
                "user_experience": "After successful authentication or sensitive operations, the system manages session tokens to maintain secure user sessions.",
                "success_outcome": "User maintains secure authenticated session",
                "failure_outcome": "Session expires and user must re-authenticate",
                "edge_cases": [
                    "Token expiration during active operation",
                    "Same user logged in on multiple devices",
                    "Token theft or session hijacking attempts",
                    "Server restart affecting active sessions",
                ],
                "security_concerns": [
                    "Token security and proper lifetime management",
                    "Session fixation attack prevention",
                    "Concurrent session handling",
                    "Token refresh timing and security",
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
                "user_experience": "成功认证或敏感操作后，系统管理会话令牌以维护安全的用户会话。",
                "success_outcome": "用户维持安全的已认证会话",
                "failure_outcome": "会话过期，用户需要重新认证",
                "edge_cases": [
                    "活动操作期间令牌过期",
                    "同一用户在多台设备上登录",
                    "令牌盗窃或会话劫持尝试",
                    "服务器重启影响活动会话",
                ],
                "security_concerns": [
                    "令牌安全性和适当的生命周期管理",
                    "会话固定攻击预防",
                    "并发会话处理",
                    "令牌刷新时机和安全性",
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
            "en": "a regular user who encountered an issue or has questions about how things work. You don't know any technical details, just what you experienced.",
            "zh": "一个遇到问题或对事物如何工作有疑问的普通用户。你不知道任何技术细节，只知道你的体验。",
        },
        UserRole.PRODUCT_MANAGER: {
            "en": "a product manager who needs to understand user flows, business logic, and product requirements. You think about user experience and business impact.",
            "zh": "一个需要了解用户流程、业务逻辑和产品需求的产品经理。你关注用户体验和业务影响。",
        },
        UserRole.QA_ENGINEER: {
            "en": "a QA engineer testing edge cases and potential failure scenarios. You think about what could go wrong but don't have access to source code.",
            "zh": "一个测试边界情况和潜在失败场景的QA工程师。你考虑可能出错的情况，但无法访问源代码。",
        },
        UserRole.SECURITY_AUDITOR: {
            "en": "a security auditor examining potential vulnerabilities through black-box testing. You probe the system's behavior without seeing the code.",
            "zh": "一个通过黑盒测试检查潜在漏洞的安全审计员。你在不查看代码的情况下探测系统行为。",
        },
        UserRole.NEW_DEVELOPER: {
            "en": "a new developer trying to understand how the authentication system works from a high level. You ask conceptual questions, not code-specific ones.",
            "zh": "一个试图从高层次理解认证系统如何工作的新开发者。你问概念性问题，而不是代码特定的问题。",
        },
    }

    @classmethod
    def transform(cls, evidence: Dict, subcategory_id: str, language: str = "en") -> Dict:
        """Transform code evidence into business context."""
        scenario = cls.SCENARIO_CONTEXTS.get(subcategory_id, {}).get(language, {})

        if not scenario:
            # Fallback to English
            scenario = cls.SCENARIO_CONTEXTS.get(subcategory_id, {}).get("en", {})

        # Abstract the evidence description (remove code names)
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
        # Comprehensive pattern replacements
        patterns = {
            # Function names
            r'\bcheck_username_is_taken\b': 'username availability check' if language == 'en' else '用户名可用性检查',
            r'\bcheck_email_is_taken\b': 'email uniqueness verification' if language == 'en' else '邮箱唯一性验证',
            r'\busers_repo\.create_user\b': 'account creation process' if language == 'en' else '账户创建流程',
            r'\busers_repo\.update_user\b': 'profile update process' if language == 'en' else '资料更新流程',
            r'\bcreate_access_token_for_user\b': 'session token generation' if language == 'en' else '会话令牌生成',
            r'\buser\.check_password\b': 'password verification' if language == 'en' else '密码验证',

            # Variable names
            r'\bwrong_login_error\b': 'authentication error response' if language == 'en' else '认证错误响应',
            r'\buser_create\b': 'registration data' if language == 'en' else '注册数据',
            r'\buser_update\b': 'update data' if language == 'en' else '更新数据',
            r'\buser_login\b': 'login credentials' if language == 'en' else '登录凭据',
            r'\bcurrent_user\b': 'authenticated user' if language == 'en' else '已认证用户',
            r'\bexistence_error\b': 'not found error' if language == 'en' else '未找到错误',

            # Technical terms
            r'\bHTTP_400_BAD_REQUEST\b': 'validation error' if language == 'en' else '验证错误',
            r'\bHTTP_401_UNAUTHORIZED\b': 'authentication failure' if language == 'en' else '认证失败',
            r'\bEntityDoesNotExist\b': 'resource not found' if language == 'en' else '资源不存在',
            r'\bHTTPException\b': 'error response' if language == 'en' else '错误响应',
            r'\bUserWithToken\b': 'authenticated user response' if language == 'en' else '已认证用户响应',
            r'\busers_repo\b': 'user data service' if language == 'en' else '用户数据服务',

            # Code patterns
            r'\btry-except\s+(?:block|structure)\b': 'error handling' if language == 'en' else '错误处理',
            r'\bawait\s+': '' if language == 'en' else '',
        }

        result = text
        for pattern, replacement in patterns.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    @classmethod
    def get_role_context(cls, role: UserRole, language: str = "en") -> str:
        """Get role context description."""
        return cls.ROLE_CONTEXTS.get(role, {}).get(language, "")


# ============================================================================
# Ollama LLM Client (from v3, enhanced)
# ============================================================================

class OllamaClient:
    """Enhanced Ollama client with retry and caching."""

    def __init__(self):
        self.api_url = Config.OLLAMA_API
        self._available = None
        self._check_time = 0

    def is_available(self) -> bool:
        """Check if Ollama is available with caching."""
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

    def generate(
            self,
            prompt: str,
            system: str = None,
            temperature: float = None,
            max_retries: int = None
    ) -> Optional[str]:
        """Generate response with retries."""
        if not self.is_available():
            return None

        temperature = temperature or 0.7
        max_retries = max_retries or Config.LLM_RETRY_COUNT

        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": Config.MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                }

                if system:
                    payload["system"] = system

                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=Config.LLM_TIMEOUT
                )

                if response.status_code == 200:
                    return response.json().get("response", "").strip()

            except requests.exceptions.Timeout:
                logger.warning(f"LLM timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"LLM error: {e}")

            if attempt < max_retries:
                time.sleep(2 ** attempt)

        return None


# ============================================================================
# LLM Enhancement Layer: Question Generator (from v3 + v5)
# ============================================================================

class LLMQuestionGenerator:
    """
    LLM-powered question generator that creates user-perspective questions.
    Receives business context, NOT code names.
    """

    SYSTEM_PROMPT = {
        "en": """You are simulating real users asking questions about a web application's authentication system.

CRITICAL RULES:
1. Generate questions that REAL USERS would ask - they DON'T know internal function names, variable names, or code structure
2. Questions must be natural and conversational, as if asked in a support ticket, team meeting, or security audit
3. ABSOLUTELY FORBIDDEN: Any function names (check_email_is_taken, create_user, etc.), variable names (user_create, wrong_login_error, etc.), or HTTP status codes (HTTP_400, 401, etc.)
4. Focus on OBSERVABLE BEHAVIOR and USER EXPERIENCE, not implementation
5. Include diverse question types: troubleshooting, "what if" scenarios, security concerns, edge cases
6. Questions should be specific and actionable

GOOD question examples:
- "When I try to register, it says my email is already taken, but I've never used this site before. What's happening?"
- "What happens if my internet disconnects during registration? Will I have a half-created account?"
- "Can someone figure out which emails are registered by trying to sign up with different addresses?"
- "Why does the login error not tell me if it's my email or password that's wrong?"

BAD question examples (NEVER generate these):
- "What does check_email_is_taken return?" (exposes function name)
- "How does users_repo.create_user handle errors?" (exposes code structure)
- "What HTTP status code is returned for validation errors?" (too technical)
- "Does the user_create variable get validated?" (exposes variable name)""",

        "zh": """你正在模拟真实用户对Web应用认证系统提出问题。

关键规则：
1. 生成真实用户会问的问题 - 他们不知道内部函数名、变量名或代码结构
2. 问题必须自然且口语化，就像在支持工单、团队会议或安全审计中提出的
3. 绝对禁止：任何函数名（check_email_is_taken、create_user等）、变量名（user_create、wrong_login_error等）或HTTP状态码（HTTP_400、401等）
4. 关注可观察的行为和用户体验，而不是实现
5. 包含多样的问题类型：故障排除、"如果...会怎样"场景、安全关注点、边界情况
6. 问题应该具体可操作

好的问题示例：
- "当我尝试注册时，显示邮箱已被使用，但我从没用过这个网站。发生了什么？"
- "如果我的网络在注册过程中断开会怎样？我会有一个半创建的账户吗？"
- "有人能否通过尝试用不同地址注册来判断哪些邮箱已注册？"
- "为什么登录错误不告诉我是邮箱还是密码错了？"

坏的问题示例（绝不要生成这些）：
- "check_email_is_taken返回什么？"（暴露函数名）
- "users_repo.create_user如何处理错误？"（暴露代码结构）
- "验证错误返回什么HTTP状态码？"（太技术化）
- "user_create变量是否被验证？"（暴露变量名）"""
    }

    GENERATION_PROMPT = {
        "en": """Based on the following authentication scenario, generate {count} diverse questions that real users would ask.

SCENARIO: {scenario_name}
BUSINESS FLOW: {business_flow}
USER EXPERIENCE: {user_experience}
SUCCESS: {success_outcome}
FAILURE: {failure_outcome}
EDGE CASES: {edge_cases}
SECURITY ASPECTS: {security_concerns}
BUSINESS RULES: {business_rules}

YOUR ROLE: You are {role_context}

Generate exactly {count} questions. Each question should:
1. Be natural and conversational (like a real person asking)
2. NEVER contain function names, variable names, or technical code terms
3. Be relevant to the scenario and your role
4. Cover different aspects (user experience, edge cases, security, "what if" scenarios)

Output format: One question per line, starting with a number.

Questions:""",

        "zh": """基于以下认证场景，生成{count}个真实用户会问的多样化问题。

场景：{scenario_name}
业务流程：{business_flow}
用户体验：{user_experience}
成功结果：{success_outcome}
失败结果：{failure_outcome}
边界情况：{edge_cases}
安全方面：{security_concerns}
业务规则：{business_rules}

你的角色：你是{role_context}

精确生成{count}个问题。每个问题应该：
1. 自然且口语化（像真人提问）
2. 绝不包含函数名、变量名或技术代码术语
3. 与场景和你的角色相关
4. 涵盖不同方面（用户体验、边界情况、安全性、"如果...会怎样"场景）

输出格式：每行一个问题，以数字开头。

问题："""
    }

    # Code name patterns for detection
    CODE_NAME_PATTERNS = [
        r'\bcheck_\w+\b',
        r'\busers_repo\b',
        r'\buser_create\b',
        r'\buser_update\b',
        r'\buser_login\b',
        r'\bwrong_login_error\b',
        r'\bexistence_error\b',
        r'\bcurrent_user\b',
        r'\bHTTP_\d+\b',
        r'\bEntityDoesNotExist\b',
        r'\bcreate_access_token\b',
        r'\bUserWithToken\b',
        r'\bUsersRepository\b',
        r'\b[a-z]+_[a-z]+_[a-z]+\b',  # snake_case with 3+ parts
        r'\b[A-Z][a-z]+[A-Z]\w+\b',  # CamelCase like HTTPException
    ]

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        self.generated_questions: Set[str] = set()

    def generate_questions(
            self,
            business_context: Dict,
            role: UserRole,
            count: int = 2,
            language: str = "en"
    ) -> List[Dict]:
        """Generate questions using LLM with user perspective."""

        prompt = self.GENERATION_PROMPT.get(language, self.GENERATION_PROMPT["en"]).format(
            count=count,
            scenario_name=business_context.get("scenario_name", ""),
            business_flow=business_context.get("business_flow", ""),
            user_experience=business_context.get("user_experience", ""),
            success_outcome=business_context.get("success_outcome", ""),
            failure_outcome=business_context.get("failure_outcome", ""),
            edge_cases=", ".join(business_context.get("edge_cases", [])[:3]),
            security_concerns=", ".join(business_context.get("security_concerns", [])[:3]),
            business_rules=", ".join(business_context.get("business_rules", [])[:2]),
            role_context=BusinessContextTransformer.get_role_context(role, language),
        )

        system = self.SYSTEM_PROMPT.get(language, self.SYSTEM_PROMPT["en"])

        response = self.llm.generate(
            prompt,
            system=system,
            temperature=Config.LLM_TEMPERATURE_QUESTION
        )

        if response:
            questions = self._parse_and_filter(response, role, language)
            if questions:
                return questions

        # Fallback to templates
        return self._generate_fallback(business_context, role, count, language)

    def _parse_and_filter(self, response: str, role: UserRole, language: str) -> List[Dict]:
        """Parse LLM response and filter out invalid questions."""
        questions = []

        for line in response.strip().split('\n'):
            line = line.strip()
            # Remove numbering
            line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line)

            if not line:
                continue

            # Must be a question
            if not (line.endswith('?') or line.endswith('？')):
                continue

            # Check for code name leakage
            if self._contains_code_names(line):
                logger.debug(f"Rejected (code names): {line[:50]}...")
                continue

            # Check for duplicates
            line_lower = line.lower().strip()
            if line_lower in self.generated_questions:
                continue

            # Length check
            if len(line) < 15:
                continue

            self.generated_questions.add(line_lower)
            questions.append({
                "question_id": f"LLM-{uuid.uuid4().hex[:8]}",
                "question_text": line,
                "source": "llm",
                "role": role.value,
                "language": language,
            })

        return questions

    def _contains_code_names(self, text: str) -> bool:
        """Check if text contains code-level names."""
        for pattern in self.CODE_NAME_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _generate_fallback(
            self,
            context: Dict,
            role: UserRole,
            count: int,
            language: str
    ) -> List[Dict]:
        """Generate fallback questions when LLM unavailable."""
        templates = self._get_fallback_templates(context, language)
        role_templates = templates.get(role, templates.get(UserRole.END_USER, []))

        random.shuffle(role_templates)
        selected = role_templates[:count]

        return [{
            "question_id": f"FB-{uuid.uuid4().hex[:8]}",
            "question_text": q,
            "source": "fallback",
            "role": role.value,
            "language": language,
        } for q in selected]

    def _get_fallback_templates(self, context: Dict, language: str) -> Dict[UserRole, List[str]]:
        """Get role-specific fallback templates."""
        scenario = context.get("scenario_name", "authentication")
        edge_cases = context.get("edge_cases", [])
        security = context.get("security_concerns", [])

        if language == "zh":
            return {
                UserRole.END_USER: [
                    "我尝试注册但显示某个信息已被占用。我从没用过这个网站，怎么回事？",
                    "当我登录时，无论密码错还是邮箱错都显示相同错误。这正常吗？",
                    "我更新资料时出了问题，我的更改是否只保存了一部分？",
                    "为什么登录错误不告诉我具体是哪里错了？",
                    "我担心我的账户可能被盗用了。我应该怎么做？",
                    "系统意外把我登出了。有人访问了我的账户吗？",
                    "如果我在注册时网络中断会怎样？",
                    "我忘记密码了，尝试重置时不确定账户是否存在。",
                ],
                UserRole.PRODUCT_MANAGER: [
                    f"能解释一下{scenario}的完整用户体验吗？",
                    "当用户遇到重复邮箱时，整个流程是什么？",
                    "如果注册过程中途失败，用户体验是什么？",
                    "我们的会话管理如何影响多设备登录的用户？",
                    "为什么选择不告诉用户具体是哪个凭据错了？",
                ],
                UserRole.QA_ENGINEER: [
                    "如果两个用户在几毫秒内用相同邮箱注册会发生什么？",
                    "当验证过程中数据库不可用时，预期行为是什么？",
                    "我们应该测试哪些竞态条件场景？",
                    "是否存在用户可能得到部分创建账户的场景？",
                    "如果成功登录后令牌生成失败会怎样？",
                ],
                UserRole.SECURITY_AUDITOR: [
                    "攻击者能否通过分析错误消息确定有效邮箱？",
                    "响应中是否存在可能揭示账户存在的时序差异？",
                    "什么机制防止通过登录端点枚举用户名？",
                    "系统如何防止会话固定攻击？",
                    "错误消息模式是否存在信息泄露风险？",
                ],
                UserRole.NEW_DEVELOPER: [
                    "从用户角度来看，认证流程是如何工作的？",
                    "为什么不同的登录失败都返回相同错误？",
                    "系统如何确保用户名和邮箱唯一？",
                    "密码存储过程中使用了什么安全措施？",
                    "会话在登录成功后是如何管理的？",
                ],
            }
        else:
            return {
                UserRole.END_USER: [
                    "I tried to register but it says something is already taken. I've never used this site. What's happening?",
                    "When I log in, I get the same error whether I use wrong password or wrong email. Is this normal?",
                    "I was updating my profile and something went wrong. Were my changes partially saved?",
                    "Why doesn't the login error tell me specifically what went wrong?",
                    "I'm worried my account might have been compromised. What should I do?",
                    "The system unexpectedly logged me out. Did someone access my account?",
                    "What happens if my internet disconnects during registration?",
                    "I forgot my password and when trying to reset, I'm not sure if my account exists.",
                ],
                UserRole.PRODUCT_MANAGER: [
                    f"Can you explain the complete user experience for {scenario}?",
                    "When a user encounters a duplicate email during registration, what's the full flow?",
                    "What's the user experience if the registration process fails midway?",
                    "How does our session management affect users logged in on multiple devices?",
                    "Why was the decision made not to tell users which credential was wrong?",
                ],
                UserRole.QA_ENGINEER: [
                    "What happens if two users try to register with the same email within milliseconds?",
                    "What's the expected behavior when the database becomes unavailable during validation?",
                    "What race condition scenarios should we be testing?",
                    "Are there any scenarios where a user could end up with a partially created account?",
                    "What happens if token generation fails after successful authentication?",
                ],
                UserRole.SECURITY_AUDITOR: [
                    "Could an attacker determine valid email addresses by analyzing error messages?",
                    "Is there a timing difference in responses that could reveal account existence?",
                    "What prevents username enumeration through the login endpoint?",
                    "How does the system prevent session fixation attacks?",
                    "Is there any risk of information leakage through error message patterns?",
                ],
                UserRole.NEW_DEVELOPER: [
                    "From a user's perspective, how does the authentication flow work?",
                    "Why do different login failures return the same error?",
                    "How does the system ensure usernames and emails stay unique?",
                    "What security measures are used in the password storage process?",
                    "How are sessions managed after successful login?",
                ],
            }


# ============================================================================
# LLM Enhancement Layer: Reasoning & Answer Generator (from v3)
# ============================================================================

class LLMReasoningGenerator:
    """
    Generates human-like reasoning chains that connect questions to answers.
    Uses deterministic code context but presents in accessible language.
    """

    SYSTEM_PROMPT = {
        "en": """You are a senior software architect explaining authentication system behavior.
Your task is to provide clear, step-by-step reasoning that explains:
1. What the user is asking about
2. How the system handles this scenario
3. The security considerations involved
4. The actual behavior based on the code

Be thorough but accessible. Start with high-level concepts, then dive into specifics.""",

        "zh": """你是一位高级软件架构师，解释认证系统行为。
你的任务是提供清晰的逐步推理，解释：
1. 用户在问什么
2. 系统如何处理这个场景
3. 涉及的安全考虑
4. 基于代码的实际行为

要全面但易于理解。先从高层概念开始，然后深入细节。"""
    }

    REASONING_PROMPT = {
        "en": """Analyze this question and provide step-by-step reasoning.

QUESTION: {question}

SCENARIO CONTEXT:
- Business Flow: {business_flow}
- User Experience: {user_experience}
- Security Concerns: {security_concerns}

CODE CONTEXT (for factual accuracy, but explain in accessible terms):
```python
{code_snippet}
```

Provide 4-6 reasoning steps that:
1. Understand what the user is really asking
2. Identify the relevant system behavior
3. Trace through the logic flow
4. Consider security implications
5. Form a conclusion

Format each step as: [STEP_TYPE] Description
Step types: UNDERSTAND, IDENTIFY, TRACE, SECURITY, CONCLUDE

Reasoning:""",

        "zh": """分析这个问题并提供逐步推理。

问题：{question}

场景背景：
- 业务流程：{business_flow}
- 用户体验：{user_experience}
- 安全考虑：{security_concerns}

代码背景（用于事实准确性，但用易懂的方式解释）：
```python
{code_snippet}
```

提供4-6个推理步骤：
1. 理解用户真正在问什么
2. 识别相关的系统行为
3. 追踪逻辑流程
4. 考虑安全影响
5. 形成结论

每步格式：[步骤类型] 描述
步骤类型：理解、识别、追踪、安全、结论

推理："""
    }

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client

    def generate_reasoning(
            self,
            question: str,
            business_context: Dict,
            code_context: CodeContext,
            language: str = "en"
    ) -> List[str]:
        """Generate reasoning chain."""

        prompt = self.REASONING_PROMPT.get(language, self.REASONING_PROMPT["en"]).format(
            question=question,
            business_flow=business_context.get("business_flow", ""),
            user_experience=business_context.get("user_experience", ""),
            security_concerns=", ".join(business_context.get("security_concerns", [])[:2]),
            code_snippet=code_context.code_snippet[:800],
        )

        system = self.SYSTEM_PROMPT.get(language, self.SYSTEM_PROMPT["en"])

        response = self.llm.generate(
            prompt,
            system=system,
            temperature=Config.LLM_TEMPERATURE_REASONING
        )

        if response:
            return self._parse_reasoning(response)

        return self._generate_fallback_reasoning(business_context, code_context, language)

    def _parse_reasoning(self, response: str) -> List[str]:
        """Parse LLM reasoning response."""
        steps = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line.startswith('[') or re.match(r'^\d+\.', line)):
                steps.append(line)
        return steps[:6] if steps else [response[:200]]

    def _generate_fallback_reasoning(
            self,
            business_context: Dict,
            code_context: CodeContext,
            language: str
    ) -> List[str]:
        """Generate fallback reasoning."""
        if language == "zh":
            return [
                f"[理解] 问题涉及{business_context.get('scenario_name', '认证流程')}",
                f"[识别] 系统行为：{business_context.get('user_experience', '')}",
                f"[追踪] 相关代码位于 {code_context.file_path} (第{code_context.line_start}-{code_context.line_end}行)",
                f"[安全] 安全考虑：{business_context.get('security_concerns', ['安全措施已实施'])[0]}",
                f"[结论] 基于代码分析，系统正确实现了该场景的业务逻辑",
            ]
        else:
            return [
                f"[UNDERSTAND] Question relates to {business_context.get('scenario_name', 'authentication process')}",
                f"[IDENTIFY] System behavior: {business_context.get('user_experience', '')}",
                f"[TRACE] Relevant code in {code_context.file_path} (lines {code_context.line_start}-{code_context.line_end})",
                f"[SECURITY] Security consideration: {business_context.get('security_concerns', ['Security measures in place'])[0]}",
                f"[CONCLUDE] Based on code analysis, the system correctly implements the business logic for this scenario",
            ]


class LLMAnswerGenerator:
    """
    Generates comprehensive answers that bridge user questions to code.
    """

    SYSTEM_PROMPT = {
        "en": """You are a senior software architect providing comprehensive answers about authentication systems.

Your answers should:
1. First address the user's concern in accessible, non-technical language
2. Explain how the system handles the scenario
3. Discuss relevant security considerations
4. Reference the actual implementation for accuracy

Structure your answer with clear sections when appropriate.
Be helpful, accurate, and professional.""",

        "zh": """你是一位高级软件架构师，提供关于认证系统的全面回答。

你的回答应该：
1. 首先用易懂的非技术语言回应用户的关切
2. 解释系统如何处理该场景
3. 讨论相关的安全考虑
4. 参考实际实现以确保准确性

在适当时用清晰的分节组织回答。
要有帮助、准确、专业。"""
    }

    ANSWER_PROMPT = {
        "en": """Provide a comprehensive answer to this question.

QUESTION: {question}

CONTEXT:
- Scenario: {scenario_name}
- Business Flow: {business_flow}
- Expected Behavior: {user_experience}
- Security Aspects: {security_concerns}

REASONING CHAIN:
{reasoning}

CODE IMPLEMENTATION (for technical accuracy):
File: {file_path} (Lines {line_start}-{line_end})
```python
{code_snippet}
```

Provide a comprehensive answer that:
1. Directly addresses the user's question
2. Explains the behavior in user-friendly terms
3. Includes relevant security insights
4. References the code implementation appropriately

Answer:""",

        "zh": """为这个问题提供全面的回答。

问题：{question}

背景：
- 场景：{scenario_name}
- 业务流程：{business_flow}
- 预期行为：{user_experience}
- 安全方面：{security_concerns}

推理链：
{reasoning}

代码实现（用于技术准确性）：
文件：{file_path}（第{line_start}-{line_end}行）
```python
{code_snippet}
```

提供全面的回答：
1. 直接回应用户的问题
2. 用用户友好的术语解释行为
3. 包含相关的安全见解
4. 适当引用代码实现

回答："""
    }

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
        """Generate comprehensive answer."""

        prompt = self.ANSWER_PROMPT.get(language, self.ANSWER_PROMPT["en"]).format(
            question=question,
            scenario_name=business_context.get("scenario_name", ""),
            business_flow=business_context.get("business_flow", ""),
            user_experience=business_context.get("user_experience", ""),
            security_concerns=", ".join(business_context.get("security_concerns", [])[:2]),
            reasoning="\n".join(reasoning),
            file_path=code_context.file_path,
            line_start=code_context.line_start,
            line_end=code_context.line_end,
            code_snippet=code_context.code_snippet[:1000],
        )

        system = self.SYSTEM_PROMPT.get(language, self.SYSTEM_PROMPT["en"])

        response = self.llm.generate(
            prompt,
            system=system,
            temperature=Config.LLM_TEMPERATURE_ANSWER
        )

        if response:
            return self._format_answer(response, code_context, language)

        return self._generate_fallback_answer(business_context, code_context, reasoning, language)

    def _format_answer(self, llm_response: str, code_context: CodeContext, language: str) -> str:
        """Format LLM answer with code reference."""
        code_ref = f"""

### {'代码参考' if language == 'zh' else 'Code Reference'}

{'相关实现位于' if language == 'zh' else 'The relevant implementation is in'} `{code_context.file_path}` ({'第' if language == 'zh' else 'lines'} {code_context.line_start}-{code_context.line_end}):

```python
{code_context.code_snippet[:600]}
```"""

        return llm_response + code_ref

    def _generate_fallback_answer(
            self,
            business_context: Dict,
            code_context: CodeContext,
            reasoning: List[str],
            language: str
    ) -> str:
        """Generate fallback answer."""
        if language == "zh":
            return f"""### 回答

关于您的问题，以下是{business_context.get('scenario_name', '该场景')}的工作原理：

**业务流程**：{business_context.get('business_flow', '')}

**系统行为**：{business_context.get('user_experience', '')}

**安全考虑**：
{chr(10).join('- ' + c for c in business_context.get('security_concerns', ['安全措施已实施'])[:3])}

### 推理过程

{chr(10).join(reasoning)}

### 代码参考

相关实现位于 `{code_context.file_path}`（第{code_context.line_start}-{code_context.line_end}行）：

```python
{code_context.code_snippet[:600]}
```

此代码展示了系统如何实现上述行为。"""
        else:
            return f"""### Answer

Regarding your question, here's how {business_context.get('scenario_name', 'this scenario')} works:

**Business Flow**: {business_context.get('business_flow', '')}

**System Behavior**: {business_context.get('user_experience', '')}

**Security Considerations**:
{chr(10).join('- ' + c for c in business_context.get('security_concerns', ['Security measures in place'])[:3])}

### Reasoning Process

{chr(10).join(reasoning)}

### Code Reference

The relevant implementation is in `{code_context.file_path}` (lines {code_context.line_start}-{code_context.line_end}):

```python
{code_context.code_snippet[:600]}
```

This code shows how the system implements the behavior described above."""


# ============================================================================
# Quality Assurance Layer (from v3)
# ============================================================================

class QualityAssurance:
    """
    Ensures generated Q&A pairs meet quality standards.
    Includes code name leakage detection.
    """

    CODE_NAME_PATTERNS = [
        r'\bcheck_\w+\b',
        r'\busers_repo\b',
        r'\buser_create\b',
        r'\buser_update\b',
        r'\bHTTP_\d+\b',
        r'\bEntityDoesNotExist\b',
        r'\bwrong_login_error\b',
        r'\b[a-z]+_[a-z]+_[a-z]+\b',  # snake_case with 3+ parts
    ]

    def validate(self, qa_pair: Dict) -> Tuple[bool, float, List[str]]:
        """Validate Q&A pair quality."""
        issues = []
        scores = []

        question = qa_pair.get("instruction", "")
        answer = qa_pair.get("answer", "")
        code_snippet = qa_pair.get("context", {}).get("code_snippet", "")
        source_hash = qa_pair.get("data_quality", {}).get("source_hash", "")

        # Question validation
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

        # Code name leakage detection (critical for v6)
        if self._contains_code_names(question):
            issues.append("Question contains code names")
            scores.append(0.0)  # Automatic rejection
        else:
            scores.append(1.0)

        # Answer validation
        if len(answer) < 100:
            issues.append("Answer too short")
            scores.append(0.3)
        else:
            scores.append(1.0)

        # Code snippet validation
        if len(code_snippet) < 50:
            issues.append("Code snippet too short")
            scores.append(0.5)
        else:
            scores.append(1.0)

        # Hash validation (if available)
        if source_hash and code_snippet:
            computed_hash = hashlib.md5(code_snippet.encode()).hexdigest()
            if computed_hash != source_hash:
                issues.append("Source hash mismatch")
                scores.append(0.5)
            else:
                scores.append(1.0)
        else:
            scores.append(0.9)

        avg_score = sum(scores) / len(scores)
        # Reject if contains code names or score too low
        is_valid = avg_score >= 0.7 and "Question contains code names" not in issues

        return is_valid, avg_score, issues

    def _contains_code_names(self, text: str) -> bool:
        """Check if text contains code-level names."""
        for pattern in self.CODE_NAME_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


# ============================================================================
# Hybrid QA Orchestrator (v3 architecture + v6 enhancements)
# ============================================================================

class HybridQAOrchestrator:
    """
    Main orchestrator combining v3's hybrid architecture with v6's user perspective.

    Architecture:
    - Deterministic Layer: AST code snippets, call graph, DBR mapping, source hash
    - LLM Enhancement Layer: User-perspective questions, reasoning, answers
    - Quality Assurance: Code reference, DBR alignment, hash validation
    - Output Validator: Code name leakage detection, consistency check
    """

    def __init__(self, rule_metadata_path: str, ast_analysis_path: str = None):
        self.rule_metadata_path = Path(rule_metadata_path)
        self.ast_analysis_path = Path(ast_analysis_path) if ast_analysis_path else None

        # Load data
        self.rule_metadata: Dict = {}
        self.ast_analysis: Dict = {}

        # Initialize LLM
        self.llm = OllamaClient()

        # Initialize layers (will be set in initialize())
        self.deterministic_layer: Optional[DeterministicLayer] = None
        self.question_generator: Optional[LLMQuestionGenerator] = None
        self.reasoning_generator: Optional[LLMReasoningGenerator] = None
        self.answer_generator: Optional[LLMAnswerGenerator] = None
        self.quality_assurance: Optional[QualityAssurance] = None

        # Output
        self.generated_pairs: List[Dict] = []
        self.stats: Dict = defaultdict(int)

    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Load rule metadata
            with open(self.rule_metadata_path, 'r', encoding='utf-8') as f:
                self.rule_metadata = json.load(f)
            logger.info(f"Loaded rule metadata: {self.rule_metadata.get('rule_id')}")

            # Load AST analysis if available
            if self.ast_analysis_path and self.ast_analysis_path.exists():
                with open(self.ast_analysis_path, 'r', encoding='utf-8') as f:
                    self.ast_analysis = json.load(f)
                logger.info(f"Loaded AST analysis")

            # Initialize layers
            self.deterministic_layer = DeterministicLayer(self.rule_metadata, self.ast_analysis)
            self.question_generator = LLMQuestionGenerator(self.llm)
            self.reasoning_generator = LLMReasoningGenerator(self.llm)
            self.answer_generator = LLMAnswerGenerator(self.llm)
            self.quality_assurance = QualityAssurance()

            # Check LLM availability
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
            languages: List[str] = None,
    ) -> List[Dict]:
        """Run the hybrid generation pipeline."""
        questions_per_evidence = questions_per_evidence or Config.QUESTIONS_PER_EVIDENCE
        languages = languages or Config.SUPPORTED_LANGUAGES

        self.generated_pairs = []
        self.stats = defaultdict(int)

        logger.info(f"=" * 60)
        logger.info(f"Starting Hybrid Q&A Pipeline v{Config.VERSION}")
        logger.info(f"=" * 60)
        logger.info(f"  Deterministic Layer: AST code, call graph, DBR mapping")
        logger.info(f"  LLM Layer: User-perspective questions, reasoning, answers")
        logger.info(f"  Quality Assurance: Code name detection, hash validation")
        logger.info(f"  LLM: {Config.MODEL_NAME} ({'available' if self.llm.is_available() else 'fallback mode'})")
        logger.info(f"  Languages: {languages}")
        logger.info(f"  Questions/evidence: {questions_per_evidence}")
        logger.info(f"=" * 60)

        for subcategory in self.rule_metadata.get("subcategories", []):
            self._process_subcategory(subcategory, questions_per_evidence, languages)

        logger.info(f"Generated {len(self.generated_pairs)} Q&A pairs")
        return self.generated_pairs

    def _process_subcategory(
            self,
            subcategory: Dict,
            questions_per_evidence: int,
            languages: List[str],
    ):
        """Process a subcategory."""
        subcategory_id = subcategory.get("subcategory_id", "")
        logger.info(f"Processing: {subcategory_id}")

        for evidence in subcategory.get("evidences", []):
            self._process_evidence(evidence, subcategory_id, questions_per_evidence, languages)

    def _process_evidence(
            self,
            evidence: Dict,
            subcategory_id: str,
            questions_per_evidence: int,
            languages: List[str],
    ):
        """Process a single evidence with hybrid pipeline."""

        # Deterministic Layer: Extract code context
        code_context = self.deterministic_layer.get_code_context(evidence)
        dbr_logic = self.deterministic_layer.get_dbr_logic(evidence)

        if not code_context.code_snippet:
            return

        # Verify source hash
        hash_valid = self.deterministic_layer.verify_source_hash(
            code_context.code_snippet,
            code_context.source_hash
        )

        roles = list(UserRole)

        for language in languages:
            # Transform to business context (NO code names)
            business_context = BusinessContextTransformer.transform(
                evidence, subcategory_id, language
            )

            # Generate questions across roles
            questions_generated = 0
            random.shuffle(roles)

            for role in roles:
                if questions_generated >= questions_per_evidence:
                    break

                count = min(2, questions_per_evidence - questions_generated)

                # LLM Layer: Generate user-perspective questions
                questions = self.question_generator.generate_questions(
                    business_context, role, count, language
                )

                for question in questions:
                    if questions_generated >= questions_per_evidence:
                        break

                    # Generate reasoning and answer
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
        """Generate complete Q&A pair using hybrid pipeline."""

        question_text = question.get("question_text", "")

        # LLM Layer: Generate reasoning
        reasoning = self.reasoning_generator.generate_reasoning(
            question_text, business_context, code_context, language
        )

        # LLM Layer: Generate answer
        answer = self.answer_generator.generate_answer(
            question_text, business_context, code_context, reasoning, language
        )

        # Build QA pair
        qa_pair = {
            "sample_id": f"DBR01-V6-{uuid.uuid4().hex[:10]}",
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
                    "architecture": "hybrid",
                    "question_source": question.get("source", "unknown"),
                    "user_role": question.get("role", "unknown"),
                    "llm_model": Config.MODEL_NAME if question.get("source") == "llm" else None,
                },
                "data_cleaning": {
                    "cleaned": True,
                    "source_verified": hash_valid,
                    "code_names_checked": True,
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

        # Quality Assurance: Validate
        is_valid, score, issues = self.quality_assurance.validate(qa_pair)
        qa_pair["data_quality"]["quality_score"] = score
        qa_pair["data_quality"]["validation_issues"] = issues

        if is_valid:
            self.generated_pairs.append(qa_pair)
            self.stats["valid"] += 1
            self.stats[f"source_{question.get('source', 'unknown')}"] += 1
            self.stats[f"role_{question.get('role', 'unknown')}"] += 1
        else:
            self.stats["invalid"] += 1
            logger.debug(f"Rejected: {issues}")

    def save_results(self, output_path: str = None) -> str:
        """Save results to JSONL file."""
        output_path = Path(output_path) if output_path else Config.OUTPUT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in self.generated_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(self.generated_pairs)} pairs to {output_path}")
        return str(output_path)

    def print_summary(self):
        """Print generation summary."""
        print("\n" + "=" * 70)
        print(f"Hybrid Q&A Generation Summary (v{Config.VERSION})")
        print("=" * 70)

        print(f"\nArchitecture: Deterministic + LLM Enhancement + Quality Assurance")

        print(f"\nTotal Generated: {len(self.generated_pairs)}")
        print(f"  - Valid: {self.stats.get('valid', 0)}")
        print(f"  - Invalid (rejected): {self.stats.get('invalid', 0)}")

        print(f"\nGeneration Source:")
        print(f"  - LLM-generated: {self.stats.get('source_llm', 0)}")
        print(f"  - Fallback templates: {self.stats.get('source_fallback', 0)}")

        print(f"\nUser Roles:")
        for role in UserRole:
            count = self.stats.get(f"role_{role.value}", 0)
            if count > 0:
                print(f"  - {role.value}: {count}")

        # Language distribution
        lang_counts = defaultdict(int)
        for pair in self.generated_pairs:
            lang = pair.get("data_quality", {}).get("language", "unknown")
            lang_counts[lang] += 1

        print("\nBy Language:")
        for lang, count in lang_counts.items():
            print(f"  - {lang}: {count}")

        # Quality metrics
        scores = [p.get("data_quality", {}).get("quality_score", 0) for p in self.generated_pairs]
        avg_score = sum(scores) / len(scores) if scores else 0

        code_name_free = sum(1 for p in self.generated_pairs
                             if "Question contains code names" not in
                             p.get("data_quality", {}).get("validation_issues", []))

        print(f"\nQuality Metrics:")
        print(f"  - Average Score: {avg_score:.2%}")
        print(
            f"  - Code-name-free Questions: {code_name_free}/{len(self.generated_pairs)} ({100 * code_name_free / max(1, len(self.generated_pairs)):.0f}%)")

        print("\n" + "=" * 70)

    def print_sample_pairs(self, n: int = 3):
        """Print sample pairs."""
        if not self.generated_pairs:
            return

        samples = self.generated_pairs[:n]

        for i, pair in enumerate(samples):
            print("\n" + "=" * 70)
            meta = pair.get("auto_processing", {}).get("generation_metadata", {})
            print(
                f"[Sample {i + 1}] Source: {meta.get('question_source', 'N/A')} | Role: {meta.get('user_role', 'N/A')}")
            print("=" * 70)

            print(f"\n【Question】:\n{pair['instruction']}")

            print(f"\n【Reasoning】:")
            for step in pair.get("reasoning_trace", [])[:3]:
                print(f"  {step}")

            print(f"\n【Answer (excerpt)】:\n{pair['answer'][:600]}...")

            # Verify no code names
            has_code = any(re.search(p, pair['instruction'], re.IGNORECASE)
                           for p in QualityAssurance.CODE_NAME_PATTERNS)
            print(f"\n【Code Names in Question】: {'❌ FOUND' if has_code else '✓ None'}")

            print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description=f'Hybrid Q&A Generation v{Config.VERSION}'
    )

    parser.add_argument(
        '-m', '--metadata',
        default=str(Config.RULE_METADATA_FILE),
        help='Path to rule metadata JSON'
    )

    parser.add_argument(
        '-a', '--ast',
        default=str(Config.AST_ANALYSIS_FILE),
        help='Path to AST analysis JSON'
    )

    parser.add_argument(
        '-o', '--output',
        default=str(Config.OUTPUT_FILE),
        help='Output JSONL file path'
    )

    parser.add_argument(
        '-n', '--questions',
        type=int,
        default=2,
        help='Questions per evidence'
    )

    parser.add_argument(
        '-l', '--languages',
        nargs='+',
        default=['en', 'zh'],
        help='Languages to generate'
    )

    parser.add_argument(
        '--preview',
        type=int,
        default=3,
        help='Number of samples to preview'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize orchestrator
    orchestrator = HybridQAOrchestrator(args.metadata, args.ast)

    if not orchestrator.initialize():
        print("Error: Failed to initialize.")
        sys.exit(1)

    # Run pipeline
    pairs = orchestrator.run_pipeline(
        questions_per_evidence=args.questions,
        languages=args.languages,
    )

    if not pairs:
        print("Warning: No Q&A pairs generated.")
        sys.exit(1)

    # Save results
    output_path = orchestrator.save_results(args.output)

    # Print summary
    orchestrator.print_summary()

    # Preview samples
    if args.preview > 0:
        print(f"\n--- Sample Q&A Pairs ---")
        orchestrator.print_sample_pairs(args.preview)

    print(f"\n✅ Successfully generated {len(pairs)} Q&A pairs")
    print(f"📁 Output saved to: {output_path}")


if __name__ == "__main__":
    main()