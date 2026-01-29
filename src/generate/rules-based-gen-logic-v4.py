#!/usr/bin/env python3
"""
Enterprise Q&A Generation Engine v4.0 - Dual-Perspective System

Key Innovation: Generates questions from TWO perspectives to avoid "hallucination dependency"

Problem in v3:
- Questions contain function names like "check_email_is_taken", "update_current_user"
- Real users don't know internal function names
- Model learns to only identify issues when seeing code names
- Can't handle vague business descriptions

Solution in v4:
1. Business-Level Questions (User Perspective):
   - "在用户注册时，如果邮箱已被使用，系统会如何处理？"
   - No function names, no variable names, pure business language
   
2. Code-Level Questions (Developer Perspective):  
   - "check_email_is_taken 函数在高并发下是否存在竞态条件？"
   - For code review and debugging scenarios

3. Business Abstraction Layer:
   - Maps: check_email_is_taken → "邮箱唯一性验证"
   - Maps: update_current_user → "用户资料更新流程"
   - Maps: wrong_login_error → "登录失败响应"

4. Answer Bridging:
   - Starts with business-level explanation
   - Then dives into code implementation
   - Teaches model to map between business and code

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    DualPerspectiveOrchestrator                      │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────┐    ┌─────────────────────────────────┐ │
│  │   Business Abstractor   │    │      Code Context Provider      │ │
│  │ ─────────────────────── │    │ ───────────────────────────── │ │
│  │ - Function → Business   │    │ - AST Code Snippets            │ │
│  │ - Variable → Concept    │    │ - Call Graph                   │ │
│  │ - Flow → User Journey   │    │ - DBR Mapping                  │ │
│  └─────────────────────────┘    └─────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────────┐    ┌─────────────────────────────────┐ │
│  │  Business Q Generator   │    │     Code Q Generator            │ │
│  │ - No function names     │    │ - With function names           │ │
│  │ - User perspective      │    │ - Developer perspective         │ │
│  │ - Scenario-based        │    │ - Code review style             │ │
│  └─────────────────────────┘    └─────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    Bridging Answer Composer                      │ │
│  │  Business Explanation → Code Mapping → Security Analysis         │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

Author: Auto-generated
Version: 4.0.0
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
    """Global configuration for v4 dual-perspective generation."""
    VERSION = "4.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).parent.resolve()
    WORKSPACE_ROOT = BASE_DIR.parent.parent
    DATA_DIR = WORKSPACE_ROOT / "data"
    
    # Input/Output
    RULE_METADATA_FILE = DATA_DIR / "dbr01_rule_metadata.json"
    AST_ANALYSIS_FILE = DATA_DIR / "fastapi_analysis_result.json"
    OUTPUT_FILE = DATA_DIR / "qwen_dbr_training_logic_v4.jsonl"
    
    # LLM Configuration
    OLLAMA_API = "http://localhost:11434/api/generate"
    MODEL_NAME = "qwen2.5:7b"
    LLM_TIMEOUT = 120
    LLM_TEMPERATURE = 0.7
    
    # Generation Parameters
    SUPPORTED_LANGUAGES = ["en", "zh"]
    BUSINESS_QUESTION_RATIO = 0.7  # 70% business, 30% code-level
    ENABLE_LLM = True


# ============================================================================
# Question Perspective Types
# ============================================================================

class QuestionPerspective(str, Enum):
    """Perspective for question generation."""
    BUSINESS = "business"       # User/Business analyst perspective (no code names)
    DEVELOPER = "developer"     # Developer/Code reviewer perspective (with code names)
    SECURITY = "security"       # Security auditor perspective (mixed)


class BusinessScenario(str, Enum):
    """Business scenarios for abstraction."""
    USER_REGISTRATION = "user_registration"
    USER_LOGIN = "user_login"
    PROFILE_UPDATE = "profile_update"
    SESSION_MANAGEMENT = "session_management"
    PASSWORD_MANAGEMENT = "password_management"
    ACCOUNT_SECURITY = "account_security"


# ============================================================================
# Business Abstraction Layer
# ============================================================================

class BusinessAbstractor:
    """
    Maps code-level concepts to business-level descriptions.
    
    This is crucial for generating questions that real users would ask,
    without exposing internal function names or variable names.
    """
    
    # Function name to business description mapping
    FUNCTION_TO_BUSINESS = {
        # EN mappings
        "en": {
            # Authentication functions
            "login": "user login process",
            "register": "new user registration",
            "check_username_is_taken": "username availability verification",
            "check_email_is_taken": "email uniqueness validation",
            "create_user": "account creation",
            "update_user": "profile information update",
            "get_user_by_email": "user lookup by email",
            "get_user_by_username": "user lookup by username",
            
            # User management
            "retrieve_current_user": "current session user retrieval",
            "update_current_user": "user profile modification",
            
            # Token/Session
            "create_access_token_for_user": "authentication token generation",
            "create_jwt_token": "session token creation",
            "get_username_from_token": "session validation",
            
            # Password
            "check_password": "password verification",
            "change_password": "password update",
            
            # Generic patterns
            "check_": "validation of",
            "get_": "retrieval of",
            "create_": "creation of",
            "update_": "modification of",
            "delete_": "removal of",
        },
        # CN mappings
        "zh": {
            # Authentication functions
            "login": "用户登录流程",
            "register": "新用户注册",
            "check_username_is_taken": "用户名可用性验证",
            "check_email_is_taken": "邮箱唯一性验证",
            "create_user": "账户创建",
            "update_user": "资料信息更新",
            "get_user_by_email": "通过邮箱查找用户",
            "get_user_by_username": "通过用户名查找用户",
            
            # User management
            "retrieve_current_user": "获取当前会话用户",
            "update_current_user": "修改用户资料",
            
            # Token/Session
            "create_access_token_for_user": "生成认证令牌",
            "create_jwt_token": "创建会话令牌",
            "get_username_from_token": "会话验证",
            
            # Password
            "check_password": "密码验证",
            "change_password": "密码更新",
            
            # Generic patterns
            "check_": "验证",
            "get_": "获取",
            "create_": "创建",
            "update_": "更新",
            "delete_": "删除",
        }
    }
    
    # Variable name to business concept mapping
    VARIABLE_TO_CONCEPT = {
        "en": {
            "wrong_login_error": "unified authentication failure response",
            "user_create": "new user registration data",
            "user_update": "profile update request",
            "user_login": "login credentials",
            "current_user": "currently authenticated user",
            "token": "session authentication token",
            "existence_error": "account not found error",
            "hashed_password": "securely stored password",
            "salt": "password security salt",
            "HTTP_400_BAD_REQUEST": "request validation failure",
            "HTTP_401_UNAUTHORIZED": "authentication failure",
            "EntityDoesNotExist": "requested resource not found",
            "USERNAME_TAKEN": "username already in use error",
            "EMAIL_TAKEN": "email already registered error",
            "INCORRECT_LOGIN_INPUT": "invalid credentials error",
        },
        "zh": {
            "wrong_login_error": "统一的认证失败响应",
            "user_create": "新用户注册数据",
            "user_update": "资料更新请求",
            "user_login": "登录凭据",
            "current_user": "当前已认证用户",
            "token": "会话认证令牌",
            "existence_error": "账户不存在错误",
            "hashed_password": "安全存储的密码",
            "salt": "密码安全盐值",
            "HTTP_400_BAD_REQUEST": "请求验证失败",
            "HTTP_401_UNAUTHORIZED": "认证失败",
            "EntityDoesNotExist": "请求的资源不存在",
            "USERNAME_TAKEN": "用户名已被使用错误",
            "EMAIL_TAKEN": "邮箱已被注册错误",
            "INCORRECT_LOGIN_INPUT": "凭据无效错误",
        }
    }
    
    # DBR subcategory to business scenario mapping
    SUBCATEGORY_TO_SCENARIO = {
        "DBR-01-01": {
            "en": {
                "scenario": BusinessScenario.USER_REGISTRATION,
                "description": "user registration and profile update processes",
                "user_action": "registering a new account or updating profile information",
                "security_concern": "ensuring each user has unique identifiers",
            },
            "zh": {
                "scenario": BusinessScenario.USER_REGISTRATION,
                "description": "用户注册和资料更新流程",
                "user_action": "注册新账户或更新个人资料",
                "security_concern": "确保每个用户拥有唯一标识符",
            }
        },
        "DBR-01-02": {
            "en": {
                "scenario": BusinessScenario.ACCOUNT_SECURITY,
                "description": "secure account creation and credential storage",
                "user_action": "completing the registration process",
                "security_concern": "protecting user credentials from unauthorized access",
            },
            "zh": {
                "scenario": BusinessScenario.ACCOUNT_SECURITY,
                "description": "安全的账户创建和凭据存储",
                "user_action": "完成注册流程",
                "security_concern": "保护用户凭据免受未授权访问",
            }
        },
        "DBR-01-03": {
            "en": {
                "scenario": BusinessScenario.USER_LOGIN,
                "description": "user authentication and login process",
                "user_action": "logging into their account",
                "security_concern": "preventing attackers from discovering valid accounts",
            },
            "zh": {
                "scenario": BusinessScenario.USER_LOGIN,
                "description": "用户认证和登录流程",
                "user_action": "登录账户",
                "security_concern": "防止攻击者发现有效账户",
            }
        },
        "DBR-01-04": {
            "en": {
                "scenario": BusinessScenario.SESSION_MANAGEMENT,
                "description": "user session and token management",
                "user_action": "performing authenticated operations",
                "security_concern": "maintaining secure session state",
            },
            "zh": {
                "scenario": BusinessScenario.SESSION_MANAGEMENT,
                "description": "用户会话和令牌管理",
                "user_action": "执行需要认证的操作",
                "security_concern": "维护安全的会话状态",
            }
        },
    }
    
    @classmethod
    def get_business_description(cls, func_name: str, language: str = "en") -> str:
        """Convert function name to business description."""
        mappings = cls.FUNCTION_TO_BUSINESS.get(language, cls.FUNCTION_TO_BUSINESS["en"])
        
        # Direct match
        if func_name in mappings:
            return mappings[func_name]
        
        # Partial match (for patterns like check_, get_, etc.)
        for pattern, description in mappings.items():
            if pattern.endswith("_") and func_name.startswith(pattern):
                suffix = func_name[len(pattern):].replace("_", " ")
                return f"{description} {suffix}"
        
        # Fallback: convert snake_case to readable text
        return func_name.replace("_", " ")
    
    @classmethod
    def get_concept_description(cls, var_name: str, language: str = "en") -> str:
        """Convert variable name to business concept."""
        mappings = cls.VARIABLE_TO_CONCEPT.get(language, cls.VARIABLE_TO_CONCEPT["en"])
        return mappings.get(var_name, var_name.replace("_", " "))
    
    @classmethod
    def get_scenario_context(cls, subcategory_id: str, language: str = "en") -> Dict:
        """Get business scenario context for a DBR subcategory."""
        scenario_data = cls.SUBCATEGORY_TO_SCENARIO.get(subcategory_id, {})
        return scenario_data.get(language, scenario_data.get("en", {
            "scenario": BusinessScenario.ACCOUNT_SECURITY,
            "description": "security process",
            "user_action": "performing an action",
            "security_concern": "maintaining system security",
        }))
    
    # Additional technical patterns to abstract
    TECHNICAL_PATTERNS = {
        "en": {
            # Method calls
            r"users_repo\.create_user": "the account creation service",
            r"users_repo\.update_user": "the profile update service",
            r"user\.check_password": "the password verification mechanism",
            r"jwt\.create_access_token_for_user": "the session token generator",
            # Standalone identifiers
            r"\busers_repo\b": "the data service layer",
            r"\buser_create\b": "registration request data",
            r"\buser_update\b": "update request data",
            r"\buser_login\b": "login credentials",
            # Parameters
            r"calling\s+\w+_repo\.\w+": "invoking the data layer",
            r"await\s+check_": "performing validation to",
            r"await\s+\w+_repo\.": "invoking the data service to",
            # HTTP codes
            r"HTTP_400_BAD_REQUEST": "a validation error",
            r"HTTP_401_UNAUTHORIZED": "an authentication failure",
            r"400\s*Bad\s*Request": "a validation error",
            r"401\s*Unauthorized": "an authentication failure",
            # General code patterns
            r"raises?\s+HTTPException": "returns an error response",
            r"try-except\s+(?:structure|block)": "error handling mechanism",
            r"input\s+parameter": "request data",
            r"Before\s+calling": "Before proceeding with",
            r"it\s+awaits": "it validates",
        },
        "zh": {
            # Method calls
            r"users_repo\.create_user": "账户创建服务",
            r"users_repo\.update_user": "资料更新服务",
            r"user\.check_password": "密码验证机制",
            r"jwt\.create_access_token_for_user": "会话令牌生成器",
            # Standalone identifiers
            r"\busers_repo\b": "数据服务层",
            r"\buser_create\b": "注册请求数据",
            r"\buser_update\b": "更新请求数据",
            r"\buser_login\b": "登录凭据",
            # Parameters
            r"calling\s+\w+_repo\.\w+": "调用数据层",
            r"await\s+check_": "验证",
            r"await\s+\w+_repo\.": "调用数据服务",
            # HTTP codes
            r"HTTP_400_BAD_REQUEST": "验证错误",
            r"HTTP_401_UNAUTHORIZED": "认证失败",
            # General code patterns
            r"raises?\s+HTTPException": "返回错误响应",
            r"try-except\s+(?:structure|block)": "错误处理机制",
            r"input\s+parameter": "请求数据",
            r"System\s+receives": "系统接收",
            r"Before\s+calling": "在执行",
            r"it\s+awaits": "系统验证",
        }
    }
    
    @classmethod
    def abstract_code_elements(cls, text: str, language: str = "en") -> str:
        """Remove code-specific names from text, replacing with business terms."""
        result = text
        
        # First, apply technical patterns (more specific patterns)
        for pattern, replacement in cls.TECHNICAL_PATTERNS.get(language, {}).items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Replace function names
        for func_name, business_desc in cls.FUNCTION_TO_BUSINESS.get(language, {}).items():
            if not func_name.endswith("_"):
                # Exact replacement with word boundaries
                pattern = r'\b' + re.escape(func_name) + r'\b'
                result = re.sub(pattern, business_desc, result, flags=re.IGNORECASE)
        
        # Replace variable names
        for var_name, concept in cls.VARIABLE_TO_CONCEPT.get(language, {}).items():
            pattern = r'\b' + re.escape(var_name) + r'\b'
            result = re.sub(pattern, concept, result, flags=re.IGNORECASE)
        
        # Clean up any remaining snake_case identifiers that look like code
        # But be careful not to remove legitimate business terms
        remaining_code_pattern = r'\b([a-z]+_[a-z_]+)\b'
        matches = re.findall(remaining_code_pattern, result)
        for match in matches:
            # Only replace if it looks like a function/variable name
            if len(match) > 3 and match not in ['bad_request', 'side_channel']:
                # Convert to readable text
                readable = match.replace('_', ' ')
                result = result.replace(match, readable)
        
        return result


# ============================================================================
# Business-Level Question Generator
# ============================================================================

class BusinessQuestionGenerator:
    """
    Generates questions from a business/user perspective.
    These questions contain NO function names or variable names.
    """
    
    # Business-level question templates (no code names)
    TEMPLATES = {
        "en": {
            "registration": [
                "When a new user tries to register with an email address that's already in use, how does the system handle this situation?",
                "What happens if someone attempts to create multiple accounts using the same username?",
                "During the registration process, what security checks are performed to ensure data integrity?",
                "If two users try to register with the same email at exactly the same time, could both registrations succeed?",
                "What information does a user receive when their registration is rejected due to duplicate credentials?",
                "How does the system prevent users from registering with commonly used or compromised passwords?",
            ],
            "login": [
                "When a user enters an incorrect password, what message do they see and why?",
                "If someone tries to log in with an email that doesn't exist in the system, how does the response differ from an incorrect password?",
                "Could an attacker determine which email addresses are registered by observing login error messages?",
                "What security measures prevent brute-force password guessing attacks during login?",
                "How does the system protect against automated login attempts from bots?",
                "When authentication fails, does the system reveal whether the username or password was incorrect?",
            ],
            "profile_update": [
                "If a user tries to change their email to one that's already taken by another account, what happens?",
                "When updating profile information, what validations are performed before saving the changes?",
                "Can a user change their username to one that was previously used by a deleted account?",
                "What happens if a user's session expires while they're in the middle of updating their profile?",
                "How does the system handle concurrent profile updates from the same user on different devices?",
            ],
            "session": [
                "After logging in successfully, how does the system maintain the user's authenticated state?",
                "What happens to a user's session after they complete a sensitive operation like changing their password?",
                "If a user is logged in on multiple devices, does logging out from one affect the others?",
                "How long does an authentication session remain valid, and what happens when it expires?",
                "What security measures protect the authentication token from being stolen or misused?",
            ],
            "security_general": [
                "What would be the security impact if the system skipped verifying email uniqueness during registration?",
                "How does the authentication system prevent information leakage about registered users?",
                "In a high-traffic scenario, could there be timing issues that allow duplicate accounts to be created?",
                "What happens if the database becomes temporarily unavailable during an authentication attempt?",
                "How does the system ensure that sensitive operations like password changes are atomic and cannot be partially completed?",
            ],
        },
        "zh": {
            "registration": [
                "当新用户尝试使用已被占用的邮箱地址注册时，系统如何处理这种情况？",
                "如果有人尝试使用相同的用户名创建多个账户会发生什么？",
                "在注册过程中，系统执行哪些安全检查来确保数据完整性？",
                "如果两个用户同时尝试使用相同的邮箱注册，是否可能两个注册都成功？",
                "当用户因重复凭据而注册被拒绝时，他们会收到什么信息？",
                "系统如何防止用户使用常见或已泄露的密码进行注册？",
            ],
            "login": [
                "当用户输入错误的密码时，他们会看到什么消息？为什么？",
                "如果有人尝试使用系统中不存在的邮箱登录，响应与密码错误有何不同？",
                "攻击者能否通过观察登录错误消息来确定哪些邮箱地址已注册？",
                "在登录过程中，有哪些安全措施可以防止暴力破解密码攻击？",
                "系统如何防止机器人的自动登录尝试？",
                "当认证失败时，系统是否会透露是用户名还是密码错误？",
            ],
            "profile_update": [
                "如果用户尝试将邮箱更改为已被其他账户使用的邮箱，会发生什么？",
                "更新个人资料时，在保存更改之前会执行哪些验证？",
                "用户能否将用户名更改为之前被删除账户使用过的用户名？",
                "如果用户的会话在更新资料过程中过期会发生什么？",
                "系统如何处理同一用户在不同设备上同时进行的资料更新？",
            ],
            "session": [
                "成功登录后，系统如何维护用户的认证状态？",
                "用户完成敏感操作（如更改密码）后，会话会发生什么变化？",
                "如果用户在多台设备上登录，从一台设备注销是否会影响其他设备？",
                "认证会话保持有效多长时间？过期后会发生什么？",
                "有哪些安全措施保护认证令牌免受盗窃或滥用？",
            ],
            "security_general": [
                "如果系统在注册时跳过验证邮箱唯一性，会产生什么安全影响？",
                "认证系统如何防止泄露已注册用户的信息？",
                "在高流量场景下，是否可能存在时序问题导致重复账户被创建？",
                "如果数据库在认证尝试期间暂时不可用会发生什么？",
                "系统如何确保密码更改等敏感操作是原子的，不会部分完成？",
            ],
        }
    }
    
    # Scenario to template category mapping
    SCENARIO_TO_CATEGORY = {
        BusinessScenario.USER_REGISTRATION: "registration",
        BusinessScenario.USER_LOGIN: "login",
        BusinessScenario.PROFILE_UPDATE: "profile_update",
        BusinessScenario.SESSION_MANAGEMENT: "session",
        BusinessScenario.PASSWORD_MANAGEMENT: "security_general",
        BusinessScenario.ACCOUNT_SECURITY: "security_general",
    }
    
    def __init__(self):
        self.used_templates: Set[str] = set()
    
    def generate_questions(
        self,
        evidence: Dict,
        subcategory_id: str,
        language: str = "en",
        count: int = 3
    ) -> List[Dict]:
        """Generate business-level questions without code names."""
        questions = []
        
        # Get scenario context
        scenario_ctx = BusinessAbstractor.get_scenario_context(subcategory_id, language)
        scenario = scenario_ctx.get("scenario", BusinessScenario.ACCOUNT_SECURITY)
        
        # Get template category
        category = self.SCENARIO_TO_CATEGORY.get(scenario, "security_general")
        templates = self.TEMPLATES.get(language, self.TEMPLATES["en"]).get(category, [])
        
        if not templates:
            templates = self.TEMPLATES.get(language, self.TEMPLATES["en"]).get("security_general", [])
        
        # Select diverse templates
        available = [t for t in templates if t not in self.used_templates]
        if len(available) < count:
            available = templates  # Reset if running low
        
        selected = random.sample(available, min(count, len(available)))
        
        for template in selected:
            self.used_templates.add(template)
            questions.append({
                "question_id": f"BQ-{uuid.uuid4().hex[:8]}",
                "question_text": template,
                "perspective": QuestionPerspective.BUSINESS.value,
                "scenario": scenario.value,
                "has_code_names": False,
            })
        
        return questions
    
    def generate_contextual_question(
        self,
        evidence: Dict,
        subcategory_id: str,
        language: str = "en"
    ) -> Dict:
        """Generate a contextual question based on evidence details."""
        scenario_ctx = BusinessAbstractor.get_scenario_context(subcategory_id, language)
        description = scenario_ctx.get("description", "")
        user_action = scenario_ctx.get("user_action", "")
        security_concern = scenario_ctx.get("security_concern", "")
        
        # Generate what-if question without code names
        if language == "zh":
            templates = [
                f"在{description}中，如果系统不验证数据唯一性会发生什么安全问题？",
                f"当用户{user_action}时，系统需要进行哪些安全检查？",
                f"关于{security_concern}，系统采取了哪些措施？如果这些措施被绕过会怎样？",
                f"如果同时有大量用户{user_action}，系统如何确保不会出现数据冲突？",
            ]
        else:
            templates = [
                f"In {description}, what security issues would arise if the system didn't validate data uniqueness?",
                f"When a user is {user_action}, what security checks should the system perform?",
                f"Regarding {security_concern}, what measures does the system take? What if these measures were bypassed?",
                f"If many users are {user_action} simultaneously, how does the system ensure no data conflicts occur?",
            ]
        
        question_text = random.choice(templates)
        
        return {
            "question_id": f"BQ-{uuid.uuid4().hex[:8]}",
            "question_text": question_text,
            "perspective": QuestionPerspective.BUSINESS.value,
            "scenario": scenario_ctx.get("scenario", BusinessScenario.ACCOUNT_SECURITY).value,
            "has_code_names": False,
        }


# ============================================================================
# Code-Level Question Generator
# ============================================================================

class CodeQuestionGenerator:
    """
    Generates questions from a developer/code reviewer perspective.
    These questions contain specific function names and variable names.
    Used for training the model on code review scenarios.
    """
    
    def generate_questions(
        self,
        evidence: Dict,
        subcategory_id: str,
        language: str = "en",
        count: int = 2
    ) -> List[Dict]:
        """Generate code-level questions with specific function names."""
        questions = []
        
        func_name = evidence.get("name", "")
        related_elements = evidence.get("related_elements", [])
        dbr_logic = evidence.get("dbr_logic", {})
        
        # Code-level question templates
        if language == "zh":
            templates = [
                f"在 `{func_name}` 的实现中，`{related_elements[0] if related_elements else 'validation'}` 的调用顺序是否存在竞态条件风险？",
                f"如果从 `{func_name}` 中移除 `{related_elements[1] if len(related_elements) > 1 else 'check'}` 调用，会引入什么漏洞？",
                f"`{func_name}` 函数中的异常处理是否会导致时序侧信道攻击？",
                f"在高并发场景下，`{func_name}` 的数据库事务处理是否保证了原子性？",
                f"审查 `{func_name}` 的实现，它是否正确实现了 {dbr_logic.get('rule_id', 'DBR-01')} 的要求？",
            ]
        else:
            templates = [
                f"In the implementation of `{func_name}`, is there a race condition risk in the order of `{related_elements[0] if related_elements else 'validation'}` calls?",
                f"What vulnerability would be introduced if the `{related_elements[1] if len(related_elements) > 1 else 'check'}` call were removed from `{func_name}`?",
                f"Could the exception handling in `{func_name}` lead to a timing side-channel attack?",
                f"Under high concurrency, does the database transaction handling in `{func_name}` guarantee atomicity?",
                f"Reviewing the implementation of `{func_name}`, does it correctly implement the requirements of {dbr_logic.get('rule_id', 'DBR-01')}?",
            ]
        
        selected = random.sample(templates, min(count, len(templates)))
        
        for template in selected:
            questions.append({
                "question_id": f"CQ-{uuid.uuid4().hex[:8]}",
                "question_text": template,
                "perspective": QuestionPerspective.DEVELOPER.value,
                "has_code_names": True,
                "code_elements": {
                    "function": func_name,
                    "related": related_elements[:3],
                },
            })
        
        return questions


# ============================================================================
# Bridging Answer Composer
# ============================================================================

class BridgingAnswerComposer:
    """
    Composes answers that bridge business and code perspectives.
    
    Structure:
    1. Business-level explanation (what the user experiences)
    2. Code mapping (how it's implemented)
    3. Security analysis (implications and recommendations)
    """
    
    def compose_answer(
        self,
        question: Dict,
        evidence: Dict,
        code_snippet: str,
        subcategory_id: str,
        language: str = "en"
    ) -> str:
        """Compose a bridging answer."""
        
        perspective = question.get("perspective", "business")
        has_code_names = question.get("has_code_names", False)
        
        # Get business context
        scenario_ctx = BusinessAbstractor.get_scenario_context(subcategory_id, language)
        
        # Get evidence details
        func_name = evidence.get("name", "")
        description = evidence.get("description" if language == "en" else "description_cn", "")
        related_elements = evidence.get("related_elements", [])
        code_data = evidence.get("code_snippet", {})
        file_path = code_data.get("file_path", "")
        line_start = code_data.get("line_start", 0)
        line_end = code_data.get("line_end", 0)
        
        if language == "zh":
            answer = self._compose_chinese_answer(
                question, evidence, code_snippet, scenario_ctx,
                func_name, description, related_elements,
                file_path, line_start, line_end, has_code_names
            )
        else:
            answer = self._compose_english_answer(
                question, evidence, code_snippet, scenario_ctx,
                func_name, description, related_elements,
                file_path, line_start, line_end, has_code_names
            )
        
        return answer
    
    def _compose_english_answer(
        self,
        question: Dict,
        evidence: Dict,
        code_snippet: str,
        scenario_ctx: Dict,
        func_name: str,
        description: str,
        related_elements: List[str],
        file_path: str,
        line_start: int,
        line_end: int,
        has_code_names: bool
    ) -> str:
        """Compose English answer."""
        
        # Business-level explanation
        business_desc = scenario_ctx.get("description", "the process")
        user_action = scenario_ctx.get("user_action", "performing the action")
        security_concern = scenario_ctx.get("security_concern", "security")
        
        # Abstract description for business perspective
        if not has_code_names:
            abstracted_desc = BusinessAbstractor.abstract_code_elements(description, "en")
        else:
            abstracted_desc = description
        
        answer = f"""### Business Context

When users are {user_action}, the system implements several security measures to ensure {security_concern}. This is a critical part of {business_desc}.

### How It Works

{abstracted_desc}

The system follows a defensive approach where validation occurs before any data persistence. This "validate-first" pattern ensures that:
1. Invalid requests are rejected early
2. Database integrity is maintained
3. Users receive immediate feedback

### Security Implications

"""
        
        # Add security analysis based on subcategory
        dbr_logic = evidence.get("dbr_logic", {})
        subcategory = dbr_logic.get("subcategory_id", "")
        
        security_notes = {
            "DBR-01-01": "By validating uniqueness before creation, the system prevents duplicate accounts which could lead to identity confusion, data integrity issues, or potential account takeover scenarios.",
            "DBR-01-02": "Using atomic transactions ensures that account creation either fully succeeds or fully fails, preventing partial states that could leave the system vulnerable.",
            "DBR-01-03": "Returning generic error messages prevents attackers from enumerating valid accounts through different response patterns.",
            "DBR-01-04": "Refreshing tokens after operations maintains session security and limits the exposure window if a token is compromised.",
        }
        
        answer += security_notes.get(subcategory, "This implementation follows security best practices.")
        
        # Code reference section (different detail level based on question type)
        if has_code_names:
            answer += f"""

### Implementation Details

**File**: `{file_path}` (Lines {line_start}-{line_end})
**Function**: `{func_name}`
**Key Elements**: `{', '.join(related_elements[:3])}`

```python
{code_snippet}
```

### Code Review Notes

- The function correctly implements validation before data modification
- Exception handling follows secure patterns
- Consider adding additional logging for security auditing
"""
        else:
            # Business perspective - NO raw code, only conceptual description
            answer += f"""

### How the System Handles This

The system implements robust validation logic in the backend that:
- Validates all incoming data before processing
- Checks for conflicts or duplicates before making changes
- Returns clear, user-friendly error messages when issues are detected
- Maintains data consistency through careful operation ordering

This is achieved through a layered architecture where:
1. **Input Validation Layer**: Verifies all required fields are present and properly formatted
2. **Business Rule Layer**: Applies domain-specific rules (like uniqueness constraints)
3. **Persistence Layer**: Only commits changes after all validations pass

### Why This Matters

This defensive design approach protects users by:
- Preventing accidental data corruption
- Providing immediate feedback on issues
- Ensuring the system remains in a consistent state even under heavy load

"""
        
        return answer
    
    def _compose_chinese_answer(
        self,
        question: Dict,
        evidence: Dict,
        code_snippet: str,
        scenario_ctx: Dict,
        func_name: str,
        description: str,
        related_elements: List[str],
        file_path: str,
        line_start: int,
        line_end: int,
        has_code_names: bool
    ) -> str:
        """Compose Chinese answer."""
        
        business_desc = scenario_ctx.get("description", "流程")
        user_action = scenario_ctx.get("user_action", "执行操作")
        security_concern = scenario_ctx.get("security_concern", "安全性")
        
        # Abstract description for business perspective
        if not has_code_names:
            abstracted_desc = BusinessAbstractor.abstract_code_elements(description, "zh")
        else:
            abstracted_desc = description
        
        answer = f"""### 业务背景

当用户{user_action}时，系统实施多项安全措施以确保{security_concern}。这是{business_desc}的关键部分。

### 工作原理

{abstracted_desc}

系统采用防御性方法，在任何数据持久化之前进行验证。这种"先验证"模式确保：
1. 无效请求被尽早拒绝
2. 数据库完整性得到维护
3. 用户获得即时反馈

### 安全影响

"""
        
        dbr_logic = evidence.get("dbr_logic", {})
        subcategory = dbr_logic.get("subcategory_id", "")
        
        security_notes = {
            "DBR-01-01": "通过在创建前验证唯一性，系统防止了重复账户的产生，避免身份混淆、数据完整性问题或潜在的账户接管场景。",
            "DBR-01-02": "使用原子事务确保账户创建要么完全成功要么完全失败，防止可能使系统易受攻击的部分状态。",
            "DBR-01-03": "返回通用错误消息可防止攻击者通过不同的响应模式枚举有效账户。",
            "DBR-01-04": "在操作后刷新令牌可维护会话安全性，并在令牌被泄露时限制暴露窗口。",
        }
        
        answer += security_notes.get(subcategory, "此实现遵循安全最佳实践。")
        
        if has_code_names:
            answer += f"""

### 实现细节

**文件**: `{file_path}` (第 {line_start}-{line_end} 行)
**函数**: `{func_name}`
**关键元素**: `{', '.join(related_elements[:3])}`

```python
{code_snippet}
```

### 代码审查说明

- 该函数正确实现了数据修改前的验证
- 异常处理遵循安全模式
- 建议添加额外的日志记录以便安全审计
"""
        else:
            # 业务视角 - 不显示原始代码，只描述概念
            answer += f"""

### 系统处理方式

系统在后端实现了强大的验证逻辑：
- 在处理前验证所有传入数据
- 在进行更改前检查冲突或重复项
- 检测到问题时返回清晰、用户友好的错误消息
- 通过精心设计的操作顺序维护数据一致性

这是通过分层架构实现的：
1. **输入验证层**：验证所有必填字段是否存在且格式正确
2. **业务规则层**：应用特定领域的规则（如唯一性约束）
3. **持久化层**：只有在所有验证通过后才提交更改

### 为什么这很重要

这种防御性设计方法通过以下方式保护用户：
- 防止意外数据损坏
- 提供问题的即时反馈
- 确保系统即使在高负载下也保持一致状态

"""
        
        return answer


# ============================================================================
# Ollama Client (Simplified)
# ============================================================================

class OllamaClient:
    """Simple Ollama client with availability check."""
    
    def __init__(self):
        self.api_url = Config.OLLAMA_API
        self._available = None
    
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            response = requests.get(
                self.api_url.replace("/api/generate", "/api/tags"),
                timeout=3
            )
            self._available = response.status_code == 200
        except:
            self._available = False
        return self._available
    
    def generate(self, prompt: str, system: str = None) -> Optional[str]:
        if not self.is_available():
            return None
        try:
            full_prompt = f"System: {system}\n\nUser: {prompt}" if system else prompt
            response = requests.post(
                self.api_url,
                json={
                    "model": Config.MODEL_NAME,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"temperature": Config.LLM_TEMPERATURE}
                },
                timeout=Config.LLM_TIMEOUT
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except:
            pass
        return None


# ============================================================================
# Quality Validator
# ============================================================================

class QualityValidator:
    """Validates generated Q&A pairs."""
    
    def validate(self, qa_pair: Dict) -> Tuple[bool, float, List[str]]:
        issues = []
        scores = []
        
        # Question checks
        question = qa_pair.get("instruction", "")
        if len(question) < 20:
            issues.append("Question too short")
            scores.append(0.3)
        else:
            scores.append(1.0)
        
        if not (question.endswith("?") or question.endswith("？")):
            issues.append("Missing question mark")
            scores.append(0.8)
        else:
            scores.append(1.0)
        
        # Answer checks
        answer = qa_pair.get("answer", "")
        if len(answer) < 100:
            issues.append("Answer too short")
            scores.append(0.3)
        else:
            scores.append(1.0)
        
        # Code snippet check
        code = qa_pair.get("context", {}).get("code_snippet", "")
        if len(code) < 50:
            issues.append("Code snippet too short")
            scores.append(0.5)
        else:
            scores.append(1.0)
        
        avg_score = sum(scores) / len(scores)
        is_valid = avg_score >= 0.7
        
        return is_valid, avg_score, issues


# ============================================================================
# Dual-Perspective Orchestrator
# ============================================================================

class DualPerspectiveOrchestrator:
    """
    Main orchestrator that generates Q&A from both business and code perspectives.
    """
    
    def __init__(self, rule_metadata_path: str):
        self.rule_metadata_path = Path(rule_metadata_path)
        
        # Components
        self.business_generator = BusinessQuestionGenerator()
        self.code_generator = CodeQuestionGenerator()
        self.answer_composer = BridgingAnswerComposer()
        self.validator = QualityValidator()
        self.llm = OllamaClient()
        
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
                logger.info("✗ LLM not available (using templates)")
            
            return True
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return False
    
    def run_pipeline(
        self,
        samples_per_evidence: int = 4,
        languages: List[str] = None,
        business_ratio: float = None
    ) -> List[Dict]:
        """Run the dual-perspective generation pipeline."""
        languages = languages or Config.SUPPORTED_LANGUAGES
        business_ratio = business_ratio or Config.BUSINESS_QUESTION_RATIO
        
        self.generated_pairs = []
        self.stats = defaultdict(int)
        
        logger.info(f"Starting dual-perspective pipeline")
        logger.info(f"  Business questions: {business_ratio:.0%}")
        logger.info(f"  Code questions: {1-business_ratio:.0%}")
        
        for subcategory in self.rule_metadata.get("subcategories", []):
            self._process_subcategory(
                subcategory, samples_per_evidence, languages, business_ratio
            )
        
        logger.info(f"Generated {len(self.generated_pairs)} Q&A pairs")
        return self.generated_pairs
    
    def _process_subcategory(
        self,
        subcategory: Dict,
        samples_per_evidence: int,
        languages: List[str],
        business_ratio: float
    ):
        """Process a subcategory."""
        subcategory_id = subcategory.get("subcategory_id", "")
        logger.info(f"Processing: {subcategory_id}")
        
        for evidence in subcategory.get("evidences", []):
            self._process_evidence(
                evidence, subcategory_id, samples_per_evidence, languages, business_ratio
            )
    
    def _process_evidence(
        self,
        evidence: Dict,
        subcategory_id: str,
        samples_per_evidence: int,
        languages: List[str],
        business_ratio: float
    ):
        """Process a single evidence with dual perspectives."""
        
        # Get code snippet
        code_data = evidence.get("code_snippet", {})
        code_snippet = code_data.get("code", "")
        if not code_snippet:
            return
        
        # Calculate question distribution
        business_count = int(samples_per_evidence * business_ratio)
        code_count = samples_per_evidence - business_count
        
        for language in languages:
            # Generate business-level questions
            business_questions = self.business_generator.generate_questions(
                evidence, subcategory_id, language, business_count
            )
            
            # Add contextual business question
            contextual_q = self.business_generator.generate_contextual_question(
                evidence, subcategory_id, language
            )
            business_questions.append(contextual_q)
            
            # Generate code-level questions
            code_questions = self.code_generator.generate_questions(
                evidence, subcategory_id, language, code_count
            )
            
            # Process all questions
            all_questions = business_questions + code_questions
            
            for question in all_questions:
                self._generate_qa_pair(
                    question, evidence, code_snippet, subcategory_id, language
                )
    
    def _generate_qa_pair(
        self,
        question: Dict,
        evidence: Dict,
        code_snippet: str,
        subcategory_id: str,
        language: str
    ):
        """Generate a complete Q&A pair."""
        
        # Compose answer
        answer = self.answer_composer.compose_answer(
            question, evidence, code_snippet, subcategory_id, language
        )
        
        # Build QA pair
        qa_pair = self._build_qa_pair(
            question, evidence, code_snippet, answer, subcategory_id, language
        )
        
        # Validate
        is_valid, score, issues = self.validator.validate(qa_pair)
        qa_pair["data_quality"]["quality_score"] = score
        qa_pair["data_quality"]["validation_issues"] = issues
        
        if is_valid:
            self.generated_pairs.append(qa_pair)
            self.stats[f"{question['perspective']}_questions"] += 1
            self.stats["valid"] += 1
        else:
            self.stats["invalid"] += 1
    
    def _build_qa_pair(
        self,
        question: Dict,
        evidence: Dict,
        code_snippet: str,
        answer: str,
        subcategory_id: str,
        language: str
    ) -> Dict:
        """Build the final Q&A pair dictionary."""
        code_data = evidence.get("code_snippet", {})
        dbr_logic = evidence.get("dbr_logic", {})
        
        return {
            "sample_id": f"DBR01-V4-{uuid.uuid4().hex[:10]}",
            "instruction": question["question_text"],
            "context": {
                "file_path": code_data.get("file_path", ""),
                "related_dbr": dbr_logic.get("rule_id", "DBR-01"),
                "code_snippet": code_snippet,
                "line_range": f"{code_data.get('line_start', 0)}-{code_data.get('line_end', 0)}",
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
                    "perspective": question.get("perspective", "business"),
                    "has_code_names": question.get("has_code_names", False),
                    "scenario": question.get("scenario", ""),
                },
            },
            "reasoning_trace": [
                f"[PERSPECTIVE] {question.get('perspective', 'business').title()} viewpoint",
                f"[CONTEXT] {BusinessAbstractor.get_scenario_context(subcategory_id, language).get('description', '')}",
                f"[SECURITY] {BusinessAbstractor.get_scenario_context(subcategory_id, language).get('security_concern', '')}",
            ],
            "answer": answer,
            "data_quality": {
                "consistency_check": True,
                "source_hash": code_data.get("source_hash", ""),
                "language": language,
                "temperature": Config.LLM_TEMPERATURE,
                "evidence_id": evidence.get("evidence_id", ""),
                "question_perspective": question.get("perspective", "business"),
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
        print("Q&A Generation Summary (v4.0 - Dual-Perspective)")
        print("=" * 70)
        
        print(f"\nTotal Generated: {len(self.generated_pairs)}")
        print(f"  - Business questions (no code names): {self.stats.get('business_questions', 0)}")
        print(f"  - Developer questions (with code names): {self.stats.get('developer_questions', 0)}")
        print(f"  - Valid: {self.stats.get('valid', 0)}")
        print(f"  - Invalid: {self.stats.get('invalid', 0)}")
        
        # Perspective distribution
        business_count = sum(1 for p in self.generated_pairs 
                           if p.get("data_quality", {}).get("question_perspective") == "business")
        code_count = len(self.generated_pairs) - business_count
        
        print(f"\nPerspective Distribution:")
        print(f"  - Business (user-facing): {business_count} ({100*business_count/max(1,len(self.generated_pairs)):.0f}%)")
        print(f"  - Developer (code review): {code_count} ({100*code_count/max(1,len(self.generated_pairs)):.0f}%)")
        
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
        """Print sample pairs showing both perspectives."""
        # Show one business and one code perspective
        business_samples = [p for p in self.generated_pairs 
                          if p.get("data_quality", {}).get("question_perspective") == "business"]
        code_samples = [p for p in self.generated_pairs 
                       if p.get("data_quality", {}).get("question_perspective") == "developer"]
        
        samples = []
        if business_samples:
            samples.append(("Business Perspective (No Code Names)", business_samples[0]))
        if code_samples:
            samples.append(("Developer Perspective (With Code Names)", code_samples[0]))
        
        for label, pair in samples[:n]:
            print("\n" + "=" * 70)
            print(f"[{label}]")
            print(f"ID: {pair['sample_id']}")
            print("=" * 70)
            
            meta = pair.get("auto_processing", {}).get("generation_metadata", {})
            print(f"\nHas Code Names: {meta.get('has_code_names', False)}")
            print(f"Perspective: {meta.get('perspective', 'N/A')}")
            
            print(f"\n【Question】:\n{pair['instruction']}\n")
            print(f"【Answer (excerpt)】:\n{pair['answer'][:700]}...")
            
            print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Q&A Generation v4.0 (Dual-Perspective System)'
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
        '-n', '--samples',
        type=int,
        default=4,
        help='Samples per evidence'
    )
    
    parser.add_argument(
        '-l', '--languages',
        nargs='+',
        default=['en', 'zh'],
        help='Languages to generate'
    )
    
    parser.add_argument(
        '--business-ratio',
        type=float,
        default=0.7,
        help='Ratio of business questions (0.0-1.0)'
    )
    
    parser.add_argument(
        '--preview',
        type=int,
        default=2,
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
    orchestrator = DualPerspectiveOrchestrator(args.metadata)
    
    if not orchestrator.initialize():
        print("Error: Failed to initialize.")
        sys.exit(1)
    
    # Run pipeline
    print(f"\n🚀 Running Q&A Generation Pipeline v4.0 (Dual-Perspective)")
    print(f"   Business questions: {args.business_ratio:.0%} (no code names)")
    print(f"   Developer questions: {1-args.business_ratio:.0%} (with code names)")
    print(f"   Languages: {args.languages}")
    print(f"   Samples/evidence: {args.samples}")
    
    pairs = orchestrator.run_pipeline(
        samples_per_evidence=args.samples,
        languages=args.languages,
        business_ratio=args.business_ratio
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
        print(f"\n--- Sample Q&A Pairs (showing both perspectives) ---")
        orchestrator.print_sample_pairs(args.preview)
    
    print(f"\n✅ Successfully generated {len(pairs)} Q&A pairs")
    print(f"📁 Output saved to: {output_path}")


if __name__ == "__main__":
    main()
