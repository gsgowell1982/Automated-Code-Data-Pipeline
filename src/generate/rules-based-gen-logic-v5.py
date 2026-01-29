#!/usr/bin/env python3
"""
Enterprise Q&A Generation Engine v5.0 - LLM-Powered Human Simulation

Key Innovation: Uses LLM (qwen2.5:7b) to simulate REAL USER questions,
not template-based generation. Fixes the "god's eye view" problem by
giving the LLM business context instead of code names.

Problems in v4:
- Fixed templates with string replacement = mechanical, predictable
- Lost LLM's natural language generation capability
- Questions lack diversity and naturalness

Solution in v5:
1. LLM generates questions from USER PERSPECTIVE:
   - Given: "This system validates email uniqueness during registration"
   - LLM generates: "我注册时邮箱显示已被使用，但我确定没注册过，怎么回事？"
   - NOT: "check_email_is_taken 函数是否有竞态条件？"

2. Business Context Injection:
   - Convert code evidence to business scenarios
   - Feed business context to LLM, NOT function names
   - LLM naturally generates user-perspective questions

3. Multi-Role Simulation:
   - End User: "Why can't I register with this email?"
   - Product Manager: "What happens in the registration flow if..."
   - QA Engineer: "How does the system handle concurrent registrations?"
   - Security Auditor: "What are the security implications of..."

4. Answer Bridging:
   - Questions from user perspective (no code names)
   - Answers map back to actual code for training

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                      LLMPoweredOrchestrator                         │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                  Business Context Builder                        ││
│  │  Code Evidence → Business Scenario → User-Friendly Description   ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                 LLM Question Generator                           ││
│  │  Qwen 2.5:7b simulates different user roles asking questions     ││
│  │  Input: Business context (NO code names)                         ││
│  │  Output: Natural, diverse, representative questions              ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                 LLM Answer Generator                             ││
│  │  Generates comprehensive answers that bridge:                    ││
│  │  User question → Business explanation → Code implementation      ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                   Quality & Diversity Gate                       ││
│  │  - Deduplication    - Perspective balance                        ││
│  │  - Naturalness check - Coverage verification                     ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘

Author: Auto-generated
Version: 5.0.0
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
    VERSION = "5.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).parent.resolve()
    WORKSPACE_ROOT = BASE_DIR.parent.parent
    DATA_DIR = WORKSPACE_ROOT / "data"
    
    # Input/Output
    RULE_METADATA_FILE = DATA_DIR / "dbr01_rule_metadata.json"
    AST_ANALYSIS_FILE = DATA_DIR / "fastapi_analysis_result.json"
    OUTPUT_FILE = DATA_DIR / "qwen_dbr_training_logic_v5.jsonl"
    
    # LLM Configuration
    OLLAMA_API = "http://localhost:11434/api/generate"
    MODEL_NAME = "qwen2.5:7b"
    LLM_TIMEOUT = 180
    LLM_TEMPERATURE = 0.8  # Higher for more diversity
    
    # Generation Parameters
    SUPPORTED_LANGUAGES = ["en", "zh"]
    QUESTIONS_PER_EVIDENCE = 5
    LLM_RETRY_COUNT = 2


# ============================================================================
# User Role Simulation
# ============================================================================

class UserRole(str, Enum):
    """Different user roles to simulate diverse perspectives."""
    END_USER = "end_user"           # Regular user experiencing issues
    PRODUCT_MANAGER = "product_manager"  # Asking about flows and requirements
    QA_ENGINEER = "qa_engineer"     # Testing edge cases
    SECURITY_AUDITOR = "security_auditor"  # Security implications
    NEW_DEVELOPER = "new_developer"  # Learning the codebase


# ============================================================================
# Business Context Builder
# ============================================================================

class BusinessContextBuilder:
    """
    Converts code evidence into business-friendly context for LLM.
    The LLM receives business scenarios, NOT function names.
    """
    
    # Subcategory to business scenario mapping
    SCENARIO_CONTEXTS = {
        "DBR-01-01": {
            "en": {
                "scenario_name": "User Registration & Profile Uniqueness",
                "user_experience": "When users register or update their profile, the system checks if their chosen username and email are already taken by someone else.",
                "what_happens": "If the username or email is already in use, the user sees an error message and cannot proceed.",
                "edge_cases": [
                    "Two users registering with the same email at the exact same time",
                    "User trying to change their email to one that's already taken",
                    "Username that was previously deleted being reused",
                ],
                "security_concerns": [
                    "Account enumeration through different error messages",
                    "Race conditions allowing duplicate accounts",
                    "Information leakage about existing users",
                ],
            },
            "zh": {
                "scenario_name": "用户注册与资料唯一性",
                "user_experience": "当用户注册或更新资料时，系统会检查所选的用户名和邮箱是否已被他人使用。",
                "what_happens": "如果用户名或邮箱已被使用，用户会看到错误消息，无法继续操作。",
                "edge_cases": [
                    "两个用户在完全相同的时间使用相同邮箱注册",
                    "用户尝试将邮箱更改为已被占用的邮箱",
                    "之前被删除的用户名被重新使用",
                ],
                "security_concerns": [
                    "通过不同的错误消息进行账户枚举",
                    "竞态条件导致重复账户",
                    "关于现有用户的信息泄露",
                ],
            }
        },
        "DBR-01-02": {
            "en": {
                "scenario_name": "Account Creation & Credential Security",
                "user_experience": "When a new account is created, the password is securely processed before being stored.",
                "what_happens": "The system hashes the password and stores it atomically, ensuring no partial state.",
                "edge_cases": [
                    "Database failure during account creation",
                    "Server crash mid-registration",
                    "Extremely long or special character passwords",
                ],
                "security_concerns": [
                    "Password storage security",
                    "Atomicity of account creation",
                    "Recovery from partial failures",
                ],
            },
            "zh": {
                "scenario_name": "账户创建与凭据安全",
                "user_experience": "创建新账户时，密码会在存储前经过安全处理。",
                "what_happens": "系统对密码进行哈希处理并原子性存储，确保不会出现部分状态。",
                "edge_cases": [
                    "账户创建过程中数据库故障",
                    "注册过程中服务器崩溃",
                    "超长或包含特殊字符的密码",
                ],
                "security_concerns": [
                    "密码存储安全",
                    "账户创建的原子性",
                    "部分失败后的恢复",
                ],
            }
        },
        "DBR-01-03": {
            "en": {
                "scenario_name": "Login & Authentication Feedback",
                "user_experience": "When users try to log in, the system verifies their credentials and provides feedback.",
                "what_happens": "If login fails, the system returns a generic error message without revealing whether the email exists or the password was wrong.",
                "edge_cases": [
                    "Login with non-existent email",
                    "Login with wrong password for existing account",
                    "Multiple rapid login attempts",
                ],
                "security_concerns": [
                    "User enumeration through login responses",
                    "Timing attacks revealing account existence",
                    "Brute force password guessing",
                ],
            },
            "zh": {
                "scenario_name": "登录与认证反馈",
                "user_experience": "当用户尝试登录时，系统验证其凭据并提供反馈。",
                "what_happens": "如果登录失败，系统返回通用错误消息，不透露邮箱是否存在或密码是否错误。",
                "edge_cases": [
                    "使用不存在的邮箱登录",
                    "使用错误密码登录现有账户",
                    "快速多次登录尝试",
                ],
                "security_concerns": [
                    "通过登录响应进行用户枚举",
                    "时序攻击揭示账户存在性",
                    "暴力破解密码",
                ],
            }
        },
        "DBR-01-04": {
            "en": {
                "scenario_name": "Session & Token Management",
                "user_experience": "After successful authentication, the system issues a token to maintain the user's session.",
                "what_happens": "A new token is generated after login and certain sensitive operations, keeping the session secure.",
                "edge_cases": [
                    "Token expiration during active session",
                    "Same user logged in on multiple devices",
                    "Token theft or leakage",
                ],
                "security_concerns": [
                    "Token security and lifetime",
                    "Session fixation attacks",
                    "Concurrent session management",
                ],
            },
            "zh": {
                "scenario_name": "会话与令牌管理",
                "user_experience": "成功认证后，系统发放令牌以维护用户会话。",
                "what_happens": "登录和某些敏感操作后会生成新令牌，保持会话安全。",
                "edge_cases": [
                    "活动会话期间令牌过期",
                    "同一用户在多台设备上登录",
                    "令牌被盗或泄露",
                ],
                "security_concerns": [
                    "令牌安全性和有效期",
                    "会话固定攻击",
                    "并发会话管理",
                ],
            }
        },
    }
    
    # User role perspectives
    ROLE_PERSPECTIVES = {
        UserRole.END_USER: {
            "en": "You are a regular user who encountered an issue or has a question about how the system works.",
            "zh": "你是一个普通用户，遇到了问题或对系统如何工作有疑问。",
        },
        UserRole.PRODUCT_MANAGER: {
            "en": "You are a product manager who needs to understand the user flow and business logic.",
            "zh": "你是一个产品经理，需要了解用户流程和业务逻辑。",
        },
        UserRole.QA_ENGINEER: {
            "en": "You are a QA engineer testing edge cases and potential failure scenarios.",
            "zh": "你是一个QA工程师，正在测试边界情况和潜在的失败场景。",
        },
        UserRole.SECURITY_AUDITOR: {
            "en": "You are a security auditor examining potential vulnerabilities without access to source code.",
            "zh": "你是一个安全审计员，在没有源代码访问权限的情况下检查潜在漏洞。",
        },
        UserRole.NEW_DEVELOPER: {
            "en": "You are a new developer trying to understand how authentication works in this system.",
            "zh": "你是一个新开发者，正在尝试理解这个系统的认证机制如何工作。",
        },
    }
    
    @classmethod
    def build_context(cls, evidence: Dict, subcategory_id: str, language: str = "en") -> Dict:
        """Build business context from code evidence."""
        scenario = cls.SCENARIO_CONTEXTS.get(subcategory_id, {}).get(language, {})
        
        # Get evidence details but abstract them
        evidence_name = evidence.get("name", "")
        evidence_desc = evidence.get("description", "")
        related_elements = evidence.get("related_elements", [])
        
        # Abstract the technical description
        abstracted_desc = cls._abstract_description(evidence_desc, language)
        
        return {
            "scenario_name": scenario.get("scenario_name", "Authentication Process"),
            "user_experience": scenario.get("user_experience", ""),
            "what_happens": scenario.get("what_happens", ""),
            "edge_cases": scenario.get("edge_cases", []),
            "security_concerns": scenario.get("security_concerns", []),
            "specific_behavior": abstracted_desc,
            "language": language,
        }
    
    @classmethod
    def _abstract_description(cls, desc: str, language: str) -> str:
        """Abstract technical terms from description."""
        # Pattern replacements
        patterns = {
            r'\bcheck_username_is_taken\b': 'username availability check' if language == 'en' else '用户名可用性检查',
            r'\bcheck_email_is_taken\b': 'email uniqueness verification' if language == 'en' else '邮箱唯一性验证',
            r'\busers_repo\.create_user\b': 'account creation' if language == 'en' else '账户创建',
            r'\busers_repo\.update_user\b': 'profile update' if language == 'en' else '资料更新',
            r'\bwrong_login_error\b': 'authentication error response' if language == 'en' else '认证错误响应',
            r'\buser_create\b': 'registration data' if language == 'en' else '注册数据',
            r'\buser_update\b': 'update data' if language == 'en' else '更新数据',
            r'\bHTTP_400_BAD_REQUEST\b': 'validation error' if language == 'en' else '验证错误',
            r'\bHTTP_401_UNAUTHORIZED\b': 'authentication failure' if language == 'en' else '认证失败',
            r'\bEntityDoesNotExist\b': 'account not found' if language == 'en' else '账户不存在',
            r'\bcreate_access_token_for_user\b': 'session token generation' if language == 'en' else '会话令牌生成',
            r'\bUserWithToken\b': 'authenticated user response' if language == 'en' else '已认证用户响应',
        }
        
        result = desc
        for pattern, replacement in patterns.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    @classmethod
    def get_role_perspective(cls, role: UserRole, language: str = "en") -> str:
        """Get role perspective description."""
        return cls.ROLE_PERSPECTIVES.get(role, {}).get(language, "")


# ============================================================================
# Ollama LLM Client
# ============================================================================

class OllamaClient:
    """Enhanced Ollama client with retry and better error handling."""
    
    def __init__(self):
        self.api_url = Config.OLLAMA_API
        self._available = None
        self._check_time = 0
    
    def is_available(self) -> bool:
        """Check if Ollama is available (with caching)."""
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
        
        temperature = temperature or Config.LLM_TEMPERATURE
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
                logger.warning(f"LLM timeout (attempt {attempt + 1}/{max_retries + 1})")
            except Exception as e:
                logger.warning(f"LLM error: {e}")
            
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None


# ============================================================================
# LLM-Powered Question Generator
# ============================================================================

class LLMQuestionGenerator:
    """
    Uses LLM to generate natural, diverse questions from user perspective.
    The LLM receives business context, NOT code names.
    """
    
    # System prompts for different scenarios
    SYSTEM_PROMPTS = {
        "en": """You are simulating real users asking questions about a web application's authentication system.

CRITICAL RULES:
1. Generate questions that a REAL USER would ask - they DON'T know function names, variable names, or internal code structure
2. Questions should be natural, as if asked in a support ticket, team meeting, or security review
3. NEVER use technical terms like: check_email_is_taken, users_repo, HTTP_400, EntityDoesNotExist, user_create, etc.
4. Focus on USER EXPERIENCE and OBSERVABLE BEHAVIOR, not implementation details
5. Include diverse question types: troubleshooting, understanding flows, edge cases, security concerns
6. Make questions specific and actionable, not vague

GOOD EXAMPLES:
- "I tried to register but it says my email is taken. I've never used this site before. What's going on?"
- "What error message do users see if they enter the wrong password multiple times?"
- "If our database goes down during user registration, what happens to their data?"
- "Can an attacker figure out which emails are registered by trying to register with them?"

BAD EXAMPLES (DO NOT GENERATE THESE):
- "What does check_email_is_taken return?" (exposes internal function name)
- "How does users_repo.create_user handle errors?" (exposes code structure)
- "What HTTP status code is returned?" (too technical)""",

        "zh": """你正在模拟真实用户对Web应用认证系统提出问题。

关键规则：
1. 生成真实用户会问的问题 - 他们不知道函数名、变量名或内部代码结构
2. 问题应该自然，就像在支持工单、团队会议或安全审查中提出的那样
3. 绝不使用技术术语如：check_email_is_taken、users_repo、HTTP_400、EntityDoesNotExist、user_create等
4. 关注用户体验和可观察的行为，而不是实现细节
5. 包含多样的问题类型：故障排除、理解流程、边界情况、安全关注点
6. 让问题具体可操作，不要含糊

好的例子：
- "我尝试注册但显示邮箱已被使用。我从没用过这个网站。怎么回事？"
- "如果用户多次输入错误密码，会看到什么错误消息？"
- "如果我们的数据库在用户注册过程中宕机，他们的数据会怎样？"
- "攻击者能否通过尝试注册来判断哪些邮箱已经注册？"

坏的例子（不要生成这些）：
- "check_email_is_taken 返回什么？"（暴露内部函数名）
- "users_repo.create_user 如何处理错误？"（暴露代码结构）
- "返回什么HTTP状态码？"（过于技术化）"""
    }
    
    QUESTION_GENERATION_PROMPT = {
        "en": """Based on the following authentication scenario, generate {count} diverse questions that REAL USERS would ask.

SCENARIO: {scenario_name}
USER EXPERIENCE: {user_experience}
WHAT HAPPENS: {what_happens}
EDGE CASES TO CONSIDER: {edge_cases}
SECURITY ASPECTS: {security_concerns}
SPECIFIC BEHAVIOR: {specific_behavior}

ROLE TO SIMULATE: {role_perspective}

Generate {count} questions, one per line. Each question should:
1. Be natural and conversational (as if from a real person)
2. NOT contain any function names, variable names, or code-level details
3. Be relevant to the scenario
4. Cover different aspects (user experience, edge cases, security)

Questions:""",

        "zh": """基于以下认证场景，生成{count}个真实用户会问的多样化问题。

场景：{scenario_name}
用户体验：{user_experience}
发生什么：{what_happens}
需要考虑的边界情况：{edge_cases}
安全方面：{security_concerns}
具体行为：{specific_behavior}

模拟角色：{role_perspective}

生成{count}个问题，每行一个。每个问题应该：
1. 自然且口语化（像真人提问那样）
2. 不包含任何函数名、变量名或代码级细节
3. 与场景相关
4. 涵盖不同方面（用户体验、边界情况、安全性）

问题："""
    }
    
    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        self.generated_questions: Set[str] = set()  # For deduplication
    
    def generate_questions(
        self,
        business_context: Dict,
        role: UserRole,
        count: int = 3,
        language: str = "en"
    ) -> List[Dict]:
        """Generate questions using LLM."""
        
        # Build prompt
        prompt_template = self.QUESTION_GENERATION_PROMPT.get(language, self.QUESTION_GENERATION_PROMPT["en"])
        system_prompt = self.SYSTEM_PROMPTS.get(language, self.SYSTEM_PROMPTS["en"])
        
        prompt = prompt_template.format(
            count=count,
            scenario_name=business_context.get("scenario_name", ""),
            user_experience=business_context.get("user_experience", ""),
            what_happens=business_context.get("what_happens", ""),
            edge_cases=", ".join(business_context.get("edge_cases", [])),
            security_concerns=", ".join(business_context.get("security_concerns", [])),
            specific_behavior=business_context.get("specific_behavior", ""),
            role_perspective=BusinessContextBuilder.get_role_perspective(role, language),
        )
        
        # Generate with LLM
        response = self.llm.generate(prompt, system=system_prompt)
        
        if response:
            questions = self._parse_questions(response, role, language)
            # Filter out duplicates and invalid questions
            unique_questions = []
            for q in questions:
                q_text = q.get("question_text", "").lower().strip()
                if q_text and q_text not in self.generated_questions:
                    if self._is_valid_question(q_text, language):
                        self.generated_questions.add(q_text)
                        unique_questions.append(q)
            return unique_questions
        
        # Fallback to template-based generation
        return self._generate_fallback_questions(business_context, role, count, language)
    
    def _parse_questions(self, response: str, role: UserRole, language: str) -> List[Dict]:
        """Parse LLM response into questions."""
        questions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Remove numbering
            line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•]\s*', '', line)
            
            if line and (line.endswith('?') or line.endswith('？')):
                questions.append({
                    "question_id": f"LLM-{uuid.uuid4().hex[:8]}",
                    "question_text": line,
                    "source": "llm",
                    "role": role.value,
                    "language": language,
                })
        
        return questions
    
    def _is_valid_question(self, question: str, language: str) -> bool:
        """Check if question is valid (no code names, natural language)."""
        # Code name patterns to reject
        code_patterns = [
            r'\bcheck_\w+',
            r'\busers_repo\b',
            r'\buser_create\b',
            r'\buser_update\b',
            r'\bHTTP_\d+',
            r'\bEntityDoesNotExist\b',
            r'\bcreate_access_token\b',
            r'\bUserWithToken\b',
            r'\bwrong_login_error\b',
            r'\b[a-z]+_[a-z]+_[a-z]+\b',  # snake_case with 3+ parts likely code
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                logger.debug(f"Rejected question with code pattern: {question[:50]}...")
                return False
        
        # Length check
        if len(question) < 15:
            return False
        
        return True
    
    def _generate_fallback_questions(
        self,
        business_context: Dict,
        role: UserRole,
        count: int,
        language: str
    ) -> List[Dict]:
        """Fallback to template-based generation when LLM unavailable."""
        scenario = business_context.get('scenario_name', 'the authentication system')
        edge_cases = business_context.get('edge_cases', ['unexpected situations'])
        security = business_context.get('security_concerns', ['security issues'])
        behavior = business_context.get('what_happens', 'the system processes your request')
        
        # More diverse and specific templates
        templates = {
            "en": {
                UserRole.END_USER: [
                    f"I tried to register but got an error saying something was already taken. I've never used this site before. What's happening?",
                    f"When I try to log in, I keep getting the same error whether I use a wrong password or wrong email. Is this normal?",
                    f"I forgot my password and when I tried to reset it, I'm not sure if my account exists. How can I tell?",
                    f"My friend and I tried to sign up at the same time with similar info. Could that cause issues?",
                    f"I updated my profile but something went wrong. Did my changes get saved partially?",
                    f"Why does the login error message not tell me if it's my email or password that's wrong?",
                    f"I'm worried my session might have been compromised. What should I do?",
                    f"The system logged me out unexpectedly. Did someone else access my account?",
                ],
                UserRole.PRODUCT_MANAGER: [
                    f"Can you explain the user journey when someone encounters a duplicate email during registration?",
                    f"What's the user experience if the registration process fails midway?",
                    f"How do we handle the case where a user wants to change their email to one they previously used?",
                    f"What feedback do users get when their login fails, and why was this design chosen?",
                    f"How does our session management affect users who are logged in on multiple devices?",
                    f"What happens to user data if there's a server issue during account creation?",
                ],
                UserRole.QA_ENGINEER: [
                    f"If two users submit registration with the same email within milliseconds, what happens?",
                    f"What's the expected behavior when the database becomes unavailable during a login attempt?",
                    f"How does the system behave if someone tries to update their email to one being registered simultaneously?",
                    f"What edge cases should we test around password validation during registration?",
                    f"Are there any scenarios where a user could end up with a partially created account?",
                    f"How do we test for race conditions in the profile update flow?",
                    f"What happens if the session token generation fails after successful authentication?",
                ],
                UserRole.SECURITY_AUDITOR: [
                    f"Can an attacker determine valid email addresses by analyzing registration error messages?",
                    f"Is there a timing difference in responses that could reveal whether an account exists?",
                    f"What prevents an attacker from enumerating valid usernames through the login endpoint?",
                    f"How does the system prevent session fixation attacks?",
                    f"Could brute forcing the login endpoint reveal information about account existence?",
                    f"What security measures prevent race condition exploits during account creation?",
                    f"Is there any risk of information leakage through error message patterns?",
                ],
                UserRole.NEW_DEVELOPER: [
                    f"How does the authentication flow work from a user's perspective?",
                    f"What happens behind the scenes when a user registers a new account?",
                    f"Why do we return the same error for different login failures?",
                    f"How does the system ensure usernames and emails stay unique?",
                    f"What security patterns are used in the password storage process?",
                    f"How are sessions managed after a user successfully logs in?",
                ],
            },
            "zh": {
                UserRole.END_USER: [
                    f"我尝试注册但收到错误说某个信息已被占用。我从没用过这个网站。这是怎么回事？",
                    f"当我尝试登录时，无论密码错还是邮箱错都显示相同的错误。这正常吗？",
                    f"我忘记了密码，尝试重置时不确定我的账户是否存在。我怎么能知道？",
                    f"我和朋友同时用类似的信息注册。这会导致问题吗？",
                    f"我更新了个人资料但出了问题。我的更改是否只保存了一部分？",
                    f"为什么登录错误消息不告诉我是邮箱还是密码错了？",
                    f"我担心我的会话可能被入侵了。我应该怎么做？",
                    f"系统意外把我登出了。是否有其他人访问了我的账户？",
                ],
                UserRole.PRODUCT_MANAGER: [
                    f"能解释一下当用户在注册时遇到重复邮箱的整个用户体验吗？",
                    f"如果注册过程中途失败，用户体验是什么？",
                    f"我们如何处理用户想将邮箱改为之前使用过的邮箱的情况？",
                    f"登录失败时用户得到什么反馈？为什么选择这种设计？",
                    f"我们的会话管理如何影响在多设备上登录的用户？",
                    f"如果账户创建期间服务器出问题，用户数据会怎样？",
                ],
                UserRole.QA_ENGINEER: [
                    f"如果两个用户在几毫秒内用相同邮箱提交注册会发生什么？",
                    f"当登录尝试期间数据库变得不可用时，预期的行为是什么？",
                    f"如果有人试图将邮箱更新为同时正在被注册的邮箱，系统会怎样？",
                    f"关于注册时的密码验证，我们应该测试哪些边界情况？",
                    f"是否存在用户可能得到部分创建的账户的场景？",
                    f"我们如何测试资料更新流程中的竞态条件？",
                    f"如果成功认证后会话令牌生成失败会发生什么？",
                ],
                UserRole.SECURITY_AUDITOR: [
                    f"攻击者能否通过分析注册错误消息来确定有效的邮箱地址？",
                    f"响应中是否存在可能揭示账户是否存在的时序差异？",
                    f"什么机制防止攻击者通过登录端点枚举有效用户名？",
                    f"系统如何防止会话固定攻击？",
                    f"暴力破解登录端点是否可能泄露账户存在的信息？",
                    f"哪些安全措施防止账户创建期间的竞态条件利用？",
                    f"错误消息模式是否存在信息泄露的风险？",
                ],
                UserRole.NEW_DEVELOPER: [
                    f"从用户的角度来看，认证流程是如何工作的？",
                    f"当用户注册新账户时，幕后发生了什么？",
                    f"为什么我们对不同的登录失败返回相同的错误？",
                    f"系统如何确保用户名和邮箱保持唯一？",
                    f"密码存储过程中使用了哪些安全模式？",
                    f"用户成功登录后会话是如何管理的？",
                ],
            }
        }
        
        role_templates = templates.get(language, templates["en"]).get(role, templates["en"][UserRole.END_USER])
        
        # Shuffle and select to avoid repetition
        available = list(role_templates)
        random.shuffle(available)
        selected = available[:min(count, len(available))]
        
        return [{
            "question_id": f"FB-{uuid.uuid4().hex[:8]}",
            "question_text": q,
            "source": "fallback",
            "role": role.value,
            "language": language,
        } for q in selected]


# ============================================================================
# LLM-Powered Answer Generator
# ============================================================================

class LLMAnswerGenerator:
    """
    Generates comprehensive answers that bridge user questions to code implementation.
    """
    
    SYSTEM_PROMPTS = {
        "en": """You are a senior software architect explaining authentication system behavior.

Your answers should:
1. First address the user's concern in plain, accessible language
2. Then explain the underlying logic (without exposing function names unless asked)
3. Discuss security implications where relevant
4. Be helpful, accurate, and professional

Structure your answer with clear sections when appropriate.""",

        "zh": """你是一位高级软件架构师，正在解释认证系统的行为。

你的回答应该：
1. 首先用通俗易懂的语言回应用户的关切
2. 然后解释底层逻辑（除非被问到，否则不要暴露函数名）
3. 在相关时讨论安全影响
4. 有帮助、准确、专业

在适当时使用清晰的分节来组织回答。"""
    }
    
    ANSWER_PROMPT = {
        "en": """Question: {question}

Context about the system:
- Scenario: {scenario_name}
- How it works: {what_happens}
- Security considerations: {security_concerns}

Actual implementation (for your reference, DO NOT expose function names in answer unless specifically asked):
{code_snippet}

Provide a comprehensive answer that:
1. Directly addresses the question
2. Explains the behavior in user-friendly terms
3. Discusses security implications if relevant
4. Is accurate based on the actual implementation

Answer:""",

        "zh": """问题：{question}

系统背景：
- 场景：{scenario_name}
- 工作原理：{what_happens}
- 安全考虑：{security_concerns}

实际实现（供您参考，除非特别询问，否则不要在回答中暴露函数名）：
{code_snippet}

提供一个全面的回答：
1. 直接回应问题
2. 用用户友好的术语解释行为
3. 如果相关，讨论安全影响
4. 基于实际实现保持准确

回答："""
    }
    
    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
    
    def generate_answer(
        self,
        question: Dict,
        business_context: Dict,
        code_snippet: str,
        language: str = "en"
    ) -> str:
        """Generate comprehensive answer using LLM."""
        
        prompt_template = self.ANSWER_PROMPT.get(language, self.ANSWER_PROMPT["en"])
        system_prompt = self.SYSTEM_PROMPTS.get(language, self.SYSTEM_PROMPTS["en"])
        
        prompt = prompt_template.format(
            question=question.get("question_text", ""),
            scenario_name=business_context.get("scenario_name", ""),
            what_happens=business_context.get("what_happens", ""),
            security_concerns=", ".join(business_context.get("security_concerns", [])),
            code_snippet=code_snippet[:1500],  # Limit code size
        )
        
        response = self.llm.generate(
            prompt, 
            system=system_prompt,
            temperature=0.6  # Lower temp for more consistent answers
        )
        
        if response:
            return self._format_answer(response, code_snippet, language)
        
        return self._generate_fallback_answer(question, business_context, code_snippet, language)
    
    def _format_answer(self, llm_response: str, code_snippet: str, language: str) -> str:
        """Format LLM answer with code reference."""
        
        # Add code reference section
        if language == "zh":
            code_section = f"""

### 相关实现代码

以下是相关的代码实现供技术参考：

```python
{code_snippet[:800]}
```

此代码展示了系统如何实现上述行为。"""
        else:
            code_section = f"""

### Related Implementation

For technical reference, here is the relevant code implementation:

```python
{code_snippet[:800]}
```

This code shows how the system implements the behavior described above."""
        
        return llm_response + code_section
    
    def _generate_fallback_answer(
        self,
        question: Dict,
        business_context: Dict,
        code_snippet: str,
        language: str
    ) -> str:
        """Generate fallback answer when LLM unavailable."""
        
        if language == "zh":
            return f"""### 回答

关于您的问题，以下是该认证场景的工作原理：

**场景**：{business_context.get('scenario_name', '认证流程')}

**系统行为**：{business_context.get('what_happens', '系统处理您的请求。')}

**安全考虑**：
{chr(10).join('- ' + c for c in business_context.get('security_concerns', ['安全措施已到位']))}

**边界情况**：
{chr(10).join('- ' + c for c in business_context.get('edge_cases', ['系统优雅地处理意外情况']))}

### 实现参考

```python
{code_snippet[:600]}
```

这段代码展示了系统如何处理此场景。"""
        else:
            return f"""### Answer

Regarding your question, here's how this authentication scenario works:

**Scenario**: {business_context.get('scenario_name', 'Authentication process')}

**System Behavior**: {business_context.get('what_happens', 'The system processes your request.')}

**Security Considerations**:
{chr(10).join('- ' + c for c in business_context.get('security_concerns', ['Security measures are in place']))}

**Edge Cases**:
{chr(10).join('- ' + c for c in business_context.get('edge_cases', ['The system handles unexpected situations gracefully']))}

### Implementation Reference

```python
{code_snippet[:600]}
```

This code shows how the system handles this scenario."""


# ============================================================================
# Quality Gate
# ============================================================================

class QualityGate:
    """Validates and filters generated Q&A pairs."""
    
    def validate(self, qa_pair: Dict) -> Tuple[bool, float, List[str]]:
        """Validate Q&A pair quality."""
        issues = []
        scores = []
        
        question = qa_pair.get("instruction", "")
        answer = qa_pair.get("answer", "")
        
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
        
        # Check for code names in question
        code_patterns = [r'\bcheck_\w+', r'\busers_repo\b', r'\bHTTP_\d+']
        for pattern in code_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                issues.append("Question contains code names")
                scores.append(0.2)
                break
        else:
            scores.append(1.0)
        
        # Answer validation
        if len(answer) < 100:
            issues.append("Answer too short")
            scores.append(0.3)
        else:
            scores.append(1.0)
        
        avg_score = sum(scores) / len(scores)
        is_valid = avg_score >= 0.7 and "Question contains code names" not in issues
        
        return is_valid, avg_score, issues


# ============================================================================
# Main Orchestrator
# ============================================================================

class LLMPoweredOrchestrator:
    """Main orchestrator using LLM for natural question generation."""
    
    def __init__(self, rule_metadata_path: str):
        self.rule_metadata_path = Path(rule_metadata_path)
        
        # Components
        self.llm = OllamaClient()
        self.question_generator = LLMQuestionGenerator(self.llm)
        self.answer_generator = LLMAnswerGenerator(self.llm)
        self.quality_gate = QualityGate()
        
        # Data
        self.rule_metadata: Dict = {}
        self.generated_pairs: List[Dict] = []
        self.stats: Dict = defaultdict(int)
    
    def initialize(self) -> bool:
        """Initialize the orchestrator."""
        try:
            with open(self.rule_metadata_path, 'r', encoding='utf-8') as f:
                self.rule_metadata = json.load(f)
            logger.info(f"Loaded rule metadata: {self.rule_metadata.get('rule_id')}")
            
            if self.llm.is_available():
                logger.info(f"✓ LLM available: {Config.MODEL_NAME}")
            else:
                logger.warning("✗ LLM not available - will use fallback templates")
            
            return True
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return False
    
    def run_pipeline(
        self,
        questions_per_evidence: int = None,
        languages: List[str] = None,
    ) -> List[Dict]:
        """Run the LLM-powered generation pipeline."""
        questions_per_evidence = questions_per_evidence or Config.QUESTIONS_PER_EVIDENCE
        languages = languages or Config.SUPPORTED_LANGUAGES
        
        self.generated_pairs = []
        self.stats = defaultdict(int)
        
        logger.info(f"Starting LLM-powered pipeline v{Config.VERSION}")
        logger.info(f"  LLM: {Config.MODEL_NAME} ({'available' if self.llm.is_available() else 'fallback mode'})")
        logger.info(f"  Languages: {languages}")
        logger.info(f"  Questions/evidence: {questions_per_evidence}")
        
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
        """Process a single evidence."""
        
        code_data = evidence.get("code_snippet", {})
        code_snippet = code_data.get("code", "")
        if not code_snippet:
            return
        
        # Rotate through different user roles
        roles = list(UserRole)
        
        for language in languages:
            # Build business context (NO code names!)
            business_context = BusinessContextBuilder.build_context(
                evidence, subcategory_id, language
            )
            
            # Generate questions using LLM (or fallback)
            # Distribute across all roles evenly
            questions_generated = 0
            role_list = list(roles)
            random.shuffle(role_list)  # Randomize role order for variety
            
            for role in role_list:
                if questions_generated >= questions_per_evidence:
                    break
                
                # Request 1-2 questions per role
                count = 1 if questions_per_evidence <= len(role_list) else 2
                
                questions = self.question_generator.generate_questions(
                    business_context,
                    role,
                    count=count,
                    language=language
                )
                
                for question in questions:
                    if questions_generated >= questions_per_evidence:
                        break
                    
                    self._generate_qa_pair(
                        question, evidence, business_context, 
                        code_snippet, subcategory_id, language
                    )
                    questions_generated += 1
    
    def _generate_qa_pair(
        self,
        question: Dict,
        evidence: Dict,
        business_context: Dict,
        code_snippet: str,
        subcategory_id: str,
        language: str
    ):
        """Generate a complete Q&A pair."""
        
        # Generate answer
        answer = self.answer_generator.generate_answer(
            question, business_context, code_snippet, language
        )
        
        # Build QA pair
        qa_pair = self._build_qa_pair(
            question, evidence, business_context, code_snippet, 
            answer, subcategory_id, language
        )
        
        # Validate
        is_valid, score, issues = self.quality_gate.validate(qa_pair)
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
    
    def _build_qa_pair(
        self,
        question: Dict,
        evidence: Dict,
        business_context: Dict,
        code_snippet: str,
        answer: str,
        subcategory_id: str,
        language: str
    ) -> Dict:
        """Build the final Q&A pair dictionary."""
        code_data = evidence.get("code_snippet", {})
        dbr_logic = evidence.get("dbr_logic", {})
        
        return {
            "sample_id": f"DBR01-V5-{uuid.uuid4().hex[:10]}",
            "instruction": question["question_text"],
            "context": {
                "file_path": code_data.get("file_path", ""),
                "related_dbr": dbr_logic.get("rule_id", "DBR-01"),
                "code_snippet": code_snippet,
                "line_range": f"{code_data.get('line_start', 0)}-{code_data.get('line_end', 0)}",
                "scenario": business_context.get("scenario_name", ""),
            },
            "auto_processing": {
                "parser": "FastAPI-AST-Analyzer",
                "parser_version": "1.0.0",
                "dbr_logic": {
                    "rule_id": dbr_logic.get("rule_id", "DBR-01"),
                    "subcategory_id": dbr_logic.get("subcategory_id", ""),
                    "trigger_type": dbr_logic.get("trigger_type", "explicit"),
                    "weight": dbr_logic.get("weight", 1.0),
                },
                "generation_metadata": {
                    "version": Config.VERSION,
                    "source": question.get("source", "unknown"),
                    "user_role": question.get("role", "unknown"),
                    "llm_model": Config.MODEL_NAME if question.get("source") == "llm" else None,
                },
            },
            "reasoning_trace": [
                f"[USER_ROLE] {question.get('role', 'unknown')}",
                f"[SCENARIO] {business_context.get('scenario_name', '')}",
                f"[BEHAVIOR] {business_context.get('what_happens', '')}",
            ],
            "answer": answer,
            "data_quality": {
                "consistency_check": True,
                "source_hash": code_data.get("source_hash", ""),
                "language": language,
                "temperature": Config.LLM_TEMPERATURE,
                "evidence_id": evidence.get("evidence_id", ""),
            },
        }
    
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
        print(f"Q&A Generation Summary (v{Config.VERSION} - LLM-Powered)")
        print("=" * 70)
        
        print(f"\nTotal Generated: {len(self.generated_pairs)}")
        print(f"  - Valid: {self.stats.get('valid', 0)}")
        print(f"  - Invalid: {self.stats.get('invalid', 0)}")
        
        print(f"\nGeneration Source:")
        print(f"  - LLM-generated: {self.stats.get('source_llm', 0)}")
        print(f"  - Fallback templates: {self.stats.get('source_fallback', 0)}")
        
        print(f"\nUser Roles Simulated:")
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
        
        # Quality
        scores = [p.get("data_quality", {}).get("quality_score", 0) for p in self.generated_pairs]
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"\nAverage Quality Score: {avg_score:.2%}")
        
        print("\n" + "=" * 70)
    
    def print_sample_pairs(self, n: int = 3):
        """Print sample pairs."""
        # Group by source
        llm_samples = [p for p in self.generated_pairs if p.get("auto_processing", {}).get("generation_metadata", {}).get("source") == "llm"]
        fb_samples = [p for p in self.generated_pairs if p.get("auto_processing", {}).get("generation_metadata", {}).get("source") == "fallback"]
        
        samples = []
        if llm_samples:
            samples.append(("LLM-Generated Question", llm_samples[0]))
        if len(llm_samples) > 1:
            samples.append(("LLM-Generated Question (Different Role)", llm_samples[min(len(llm_samples)-1, 5)]))
        if fb_samples:
            samples.append(("Fallback Template Question", fb_samples[0]))
        
        for label, pair in samples[:n]:
            print("\n" + "=" * 70)
            print(f"[{label}]")
            meta = pair.get("auto_processing", {}).get("generation_metadata", {})
            print(f"Source: {meta.get('source', 'N/A')} | Role: {meta.get('user_role', 'N/A')}")
            print("=" * 70)
            
            print(f"\n【Question】:\n{pair['instruction']}\n")
            print(f"【Answer (excerpt)】:\n{pair['answer'][:800]}...")
            
            print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Q&A Generation v5.0 (LLM-Powered Human Simulation)'
    )
    
    parser.add_argument(
        '-m', '--metadata',
        default=str(Config.RULE_METADATA_FILE),
        help='Path to rule metadata JSON'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=str(Config.OUTPUT_FILE),
        help='Output JSONL file path'
    )
    
    parser.add_argument(
        '-n', '--questions',
        type=int,
        default=5,
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
    orchestrator = LLMPoweredOrchestrator(args.metadata)
    
    if not orchestrator.initialize():
        print("Error: Failed to initialize.")
        sys.exit(1)
    
    # Run pipeline
    print(f"\n🚀 Running Q&A Generation Pipeline v{Config.VERSION}")
    print(f"   LLM Model: {Config.MODEL_NAME}")
    print(f"   Languages: {args.languages}")
    print(f"   Questions/evidence: {args.questions}")
    
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
