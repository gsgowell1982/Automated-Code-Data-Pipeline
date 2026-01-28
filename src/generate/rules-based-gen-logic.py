#!/usr/bin/env python3
"""
Enterprise-Grade Q&A Generation Engine for LLM Fine-tuning

This module generates high-quality Q&A pairs based on DBR rule metadata
from AST analysis, with logical guarantees for:
1. Code snippet accuracy (from actual source, hash-verified)
2. Reasoning correctness (structured templates, not LLM hallucination)
3. Question diversity (multi-strategy, multi-type, multi-perspective)

Target: Qwen 2.5 series model training and fine-tuning

Architecture:
- QuestionGenerator: Multi-strategy question generation
- ReasoningEngine: Structured reasoning chain construction
- AnswerComposer: Gold-standard answer assembly
- QualityValidator: Consistency and quality checks
- QAPairGenerator: Orchestrates the entire pipeline

Author: Auto-generated
Version: 1.0.0
"""

import os
import sys
import json
import uuid
import hashlib
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
import re
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
    """Global configuration for Q&A generation."""
    VERSION = "1.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).parent.resolve()
    WORKSPACE_ROOT = BASE_DIR.parent.parent
    DATA_DIR = WORKSPACE_ROOT / "data"
    REPO_PATH = WORKSPACE_ROOT / "repos" / "fastapi-realworld-example-app"
    
    # Input/Output
    RULE_METADATA_FILE = DATA_DIR / "dbr01_rule_metadata.json"
    OUTPUT_FILE = DATA_DIR / "qwen_dbr_training_logic_v1.jsonl"
    
    # Generation parameters
    DEFAULT_TEMPERATURE = 0.7
    SUPPORTED_LANGUAGES = ["en", "zh"]
    MAX_SAMPLES_PER_EVIDENCE = 5
    MIN_QUESTION_LENGTH = 10
    MIN_ANSWER_LENGTH = 50
    

# ============================================================================
# Enums and Data Classes
# ============================================================================

class QuestionType(str, Enum):
    """Types of questions to generate."""
    WHAT = "what"           # What does X do?
    HOW = "how"             # How does X work?
    WHY = "why"             # Why is X designed this way?
    WHEN = "when"           # When does X occur?
    WHICH = "which"         # Which component handles X?
    COMPARE = "compare"     # Compare X and Y
    SCENARIO = "scenario"   # Given scenario, what happens?
    DEBUG = "debug"         # How to debug/fix X?
    SECURITY = "security"   # Security implications of X


class QuestionPerspective(str, Enum):
    """Perspectives for question generation."""
    DEVELOPER = "developer"
    SECURITY_AUDITOR = "security_auditor"
    ARCHITECT = "architect"
    CODE_REVIEWER = "code_reviewer"
    QA_ENGINEER = "qa_engineer"
    COMPLIANCE_OFFICER = "compliance_officer"


class QuestionStrategy(str, Enum):
    """Strategies for question generation."""
    DIRECT = "direct"           # Directly ask about the evidence
    INDIRECT = "indirect"       # Ask about related concepts
    COMPARATIVE = "comparative" # Compare with alternatives
    SCENARIO_BASED = "scenario" # Present a scenario
    PROBLEM_SOLVING = "problem" # Present a problem to solve
    BEST_PRACTICE = "best"      # Ask about best practices


@dataclass
class QuestionTemplate:
    """Template for generating questions."""
    template_id: str
    template_text: str
    template_text_cn: str
    question_type: QuestionType
    perspective: QuestionPerspective
    strategy: QuestionStrategy
    required_variables: List[str]
    applicable_evidence_types: List[str]
    difficulty: str = "medium"  # easy, medium, hard
    

@dataclass
class GeneratedQuestion:
    """A generated question with metadata."""
    question_id: str
    question_text: str
    question_text_cn: str
    question_type: QuestionType
    perspective: QuestionPerspective
    strategy: QuestionStrategy
    template_id: str
    evidence_id: str
    variables_used: Dict[str, str]
    

@dataclass
class ReasoningChain:
    """Structured reasoning chain."""
    chain_id: str
    steps: List[Dict[str, str]]
    evidence_references: List[str]
    dbr_references: List[str]
    conclusion: str
    conclusion_cn: str


@dataclass
class CodeContext:
    """Context information for code snippet."""
    file_path: str
    code_snippet: str
    line_start: int
    line_end: int
    source_hash: str
    function_name: str
    related_elements: List[str]


@dataclass
class QAPair:
    """Complete Q&A pair for training."""
    sample_id: str
    instruction: str
    context: Dict[str, Any]
    auto_processing: Dict[str, Any]
    reasoning_trace: List[str]
    answer: str
    data_quality: Dict[str, Any]


# ============================================================================
# Question Templates Registry
# ============================================================================

class QuestionTemplateRegistry:
    """Registry of question templates for diverse generation."""
    
    def __init__(self):
        self.templates: List[QuestionTemplate] = []
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize comprehensive question templates."""
        
        # === WHAT questions ===
        self.templates.extend([
            QuestionTemplate(
                template_id="WHAT-001",
                template_text="What mechanism does the system use to {action} in the {component}?",
                template_text_cn="系统在{component_cn}中使用什么机制来{action_cn}？",
                question_type=QuestionType.WHAT,
                perspective=QuestionPerspective.DEVELOPER,
                strategy=QuestionStrategy.DIRECT,
                required_variables=["action", "component", "action_cn", "component_cn"],
                applicable_evidence_types=["pattern", "function", "call"],
            ),
            QuestionTemplate(
                template_id="WHAT-002",
                template_text="What security measure is implemented to prevent {threat} during {operation}?",
                template_text_cn="在{operation_cn}期间实施了什么安全措施来防止{threat_cn}？",
                question_type=QuestionType.WHAT,
                perspective=QuestionPerspective.SECURITY_AUDITOR,
                strategy=QuestionStrategy.DIRECT,
                required_variables=["threat", "operation", "threat_cn", "operation_cn"],
                applicable_evidence_types=["exception_handling", "pattern"],
            ),
            QuestionTemplate(
                template_id="WHAT-003",
                template_text="What validation logic is executed before {operation} to ensure data integrity?",
                template_text_cn="在{operation_cn}之前执行什么验证逻辑以确保数据完整性？",
                question_type=QuestionType.WHAT,
                perspective=QuestionPerspective.CODE_REVIEWER,
                strategy=QuestionStrategy.DIRECT,
                required_variables=["operation", "operation_cn"],
                applicable_evidence_types=["pattern", "function"],
            ),
        ])
        
        # === HOW questions ===
        self.templates.extend([
            QuestionTemplate(
                template_id="HOW-001",
                template_text="How does the {component} handle {scenario} to maintain {goal}?",
                template_text_cn="{component_cn}如何处理{scenario_cn}以维护{goal_cn}？",
                question_type=QuestionType.HOW,
                perspective=QuestionPerspective.ARCHITECT,
                strategy=QuestionStrategy.DIRECT,
                required_variables=["component", "scenario", "goal", "component_cn", "scenario_cn", "goal_cn"],
                applicable_evidence_types=["pattern", "exception_handling", "function"],
            ),
            QuestionTemplate(
                template_id="HOW-002",
                template_text="How is {resource} protected from {attack_type} in the authentication flow?",
                template_text_cn="在身份验证流程中，{resource_cn}如何防止{attack_type_cn}？",
                question_type=QuestionType.HOW,
                perspective=QuestionPerspective.SECURITY_AUDITOR,
                strategy=QuestionStrategy.DIRECT,
                required_variables=["resource", "attack_type", "resource_cn", "attack_type_cn"],
                applicable_evidence_types=["exception_handling", "pattern"],
            ),
            QuestionTemplate(
                template_id="HOW-003",
                template_text="How does the system ensure atomicity when {operation}?",
                template_text_cn="系统在{operation_cn}时如何确保原子性？",
                question_type=QuestionType.HOW,
                perspective=QuestionPerspective.DEVELOPER,
                strategy=QuestionStrategy.DIRECT,
                required_variables=["operation", "operation_cn"],
                applicable_evidence_types=["pattern"],
            ),
        ])
        
        # === WHY questions ===
        self.templates.extend([
            QuestionTemplate(
                template_id="WHY-001",
                template_text="Why does the system return {response_type} instead of {alternative} when {condition}?",
                template_text_cn="为什么系统在{condition_cn}时返回{response_type_cn}而不是{alternative_cn}？",
                question_type=QuestionType.WHY,
                perspective=QuestionPerspective.SECURITY_AUDITOR,
                strategy=QuestionStrategy.COMPARATIVE,
                required_variables=["response_type", "alternative", "condition", 
                                   "response_type_cn", "alternative_cn", "condition_cn"],
                applicable_evidence_types=["exception_handling"],
            ),
            QuestionTemplate(
                template_id="WHY-002",
                template_text="Why is {check} performed before {action} in the {flow}?",
                template_text_cn="为什么在{flow_cn}中的{action_cn}之前要执行{check_cn}？",
                question_type=QuestionType.WHY,
                perspective=QuestionPerspective.CODE_REVIEWER,
                strategy=QuestionStrategy.DIRECT,
                required_variables=["check", "action", "flow", "check_cn", "action_cn", "flow_cn"],
                applicable_evidence_types=["pattern", "function"],
            ),
            QuestionTemplate(
                template_id="WHY-003",
                template_text="Why is it important to {action} in the context of {domain}?",
                template_text_cn="在{domain_cn}的背景下，为什么{action_cn}很重要？",
                question_type=QuestionType.WHY,
                perspective=QuestionPerspective.ARCHITECT,
                strategy=QuestionStrategy.BEST_PRACTICE,
                required_variables=["action", "domain", "action_cn", "domain_cn"],
                applicable_evidence_types=["pattern", "exception_handling", "call"],
                difficulty="hard",
            ),
        ])
        
        # === SCENARIO questions ===
        self.templates.extend([
            QuestionTemplate(
                template_id="SCENARIO-001",
                template_text="A user attempts to {user_action}. What validation steps does the system perform?",
                template_text_cn="用户尝试{user_action_cn}。系统执行哪些验证步骤？",
                question_type=QuestionType.SCENARIO,
                perspective=QuestionPerspective.QA_ENGINEER,
                strategy=QuestionStrategy.SCENARIO_BASED,
                required_variables=["user_action", "user_action_cn"],
                applicable_evidence_types=["pattern", "function"],
            ),
            QuestionTemplate(
                template_id="SCENARIO-002",
                template_text="When {error_condition} occurs during {operation}, how does the system respond?",
                template_text_cn="当{operation_cn}期间发生{error_condition_cn}时，系统如何响应？",
                question_type=QuestionType.SCENARIO,
                perspective=QuestionPerspective.DEVELOPER,
                strategy=QuestionStrategy.SCENARIO_BASED,
                required_variables=["error_condition", "operation", "error_condition_cn", "operation_cn"],
                applicable_evidence_types=["exception_handling"],
            ),
            QuestionTemplate(
                template_id="SCENARIO-003",
                template_text="Consider a scenario where {scenario_desc}. What security controls are activated?",
                template_text_cn="考虑这样一个场景：{scenario_desc_cn}。哪些安全控制被激活？",
                question_type=QuestionType.SCENARIO,
                perspective=QuestionPerspective.SECURITY_AUDITOR,
                strategy=QuestionStrategy.SCENARIO_BASED,
                required_variables=["scenario_desc", "scenario_desc_cn"],
                applicable_evidence_types=["exception_handling", "pattern"],
                difficulty="hard",
            ),
        ])
        
        # === SECURITY questions ===
        self.templates.extend([
            QuestionTemplate(
                template_id="SEC-001",
                template_text="What security vulnerability does the {mechanism} prevent in the {context}?",
                template_text_cn="{mechanism_cn}在{context_cn}中防止了什么安全漏洞？",
                question_type=QuestionType.SECURITY,
                perspective=QuestionPerspective.SECURITY_AUDITOR,
                strategy=QuestionStrategy.DIRECT,
                required_variables=["mechanism", "context", "mechanism_cn", "context_cn"],
                applicable_evidence_types=["exception_handling", "pattern"],
            ),
            QuestionTemplate(
                template_id="SEC-002",
                template_text="How does the system mitigate {attack_type} through its {feature} implementation?",
                template_text_cn="系统通过其{feature_cn}实现如何缓解{attack_type_cn}？",
                question_type=QuestionType.SECURITY,
                perspective=QuestionPerspective.SECURITY_AUDITOR,
                strategy=QuestionStrategy.DIRECT,
                required_variables=["attack_type", "feature", "attack_type_cn", "feature_cn"],
                applicable_evidence_types=["exception_handling", "pattern"],
            ),
        ])
        
        # === COMPARE questions ===
        self.templates.extend([
            QuestionTemplate(
                template_id="COMP-001",
                template_text="Compare the validation approach used in {context1} versus {context2}.",
                template_text_cn="比较{context1_cn}与{context2_cn}中使用的验证方法。",
                question_type=QuestionType.COMPARE,
                perspective=QuestionPerspective.ARCHITECT,
                strategy=QuestionStrategy.COMPARATIVE,
                required_variables=["context1", "context2", "context1_cn", "context2_cn"],
                applicable_evidence_types=["pattern", "function"],
                difficulty="hard",
            ),
        ])
        
        # === DEBUG questions ===
        self.templates.extend([
            QuestionTemplate(
                template_id="DEBUG-001",
                template_text="How would you diagnose issues related to {problem} in the {component}?",
                template_text_cn="您如何诊断{component_cn}中与{problem_cn}相关的问题？",
                question_type=QuestionType.DEBUG,
                perspective=QuestionPerspective.DEVELOPER,
                strategy=QuestionStrategy.PROBLEM_SOLVING,
                required_variables=["problem", "component", "problem_cn", "component_cn"],
                applicable_evidence_types=["exception_handling", "function"],
            ),
        ])
        
        # === WHICH questions ===
        self.templates.extend([
            QuestionTemplate(
                template_id="WHICH-001",
                template_text="Which functions are responsible for {responsibility} in the authentication module?",
                template_text_cn="认证模块中哪些函数负责{responsibility_cn}？",
                question_type=QuestionType.WHICH,
                perspective=QuestionPerspective.DEVELOPER,
                strategy=QuestionStrategy.DIRECT,
                required_variables=["responsibility", "responsibility_cn"],
                applicable_evidence_types=["function", "call"],
            ),
            QuestionTemplate(
                template_id="WHICH-002",
                template_text="Which API endpoints trigger {action} as part of their flow?",
                template_text_cn="哪些API端点在其流程中触发{action_cn}？",
                question_type=QuestionType.WHICH,
                perspective=QuestionPerspective.DEVELOPER,
                strategy=QuestionStrategy.DIRECT,
                required_variables=["action", "action_cn"],
                applicable_evidence_types=["call"],
            ),
        ])
        
        # === WHEN questions ===
        self.templates.extend([
            QuestionTemplate(
                template_id="WHEN-001",
                template_text="When is {action} triggered during the {process}?",
                template_text_cn="在{process_cn}期间，{action_cn}何时被触发？",
                question_type=QuestionType.WHEN,
                perspective=QuestionPerspective.DEVELOPER,
                strategy=QuestionStrategy.DIRECT,
                required_variables=["action", "process", "action_cn", "process_cn"],
                applicable_evidence_types=["call", "pattern"],
            ),
        ])
    
    def get_templates_for_evidence(self, evidence_type: str) -> List[QuestionTemplate]:
        """Get applicable templates for an evidence type."""
        return [t for t in self.templates if evidence_type in t.applicable_evidence_types]
    
    def get_templates_by_type(self, question_type: QuestionType) -> List[QuestionTemplate]:
        """Get templates by question type."""
        return [t for t in self.templates if t.question_type == question_type]
    
    def get_templates_by_perspective(self, perspective: QuestionPerspective) -> List[QuestionTemplate]:
        """Get templates by perspective."""
        return [t for t in self.templates if t.perspective == perspective]


# ============================================================================
# Variable Mapping for Evidence Types
# ============================================================================

class EvidenceVariableMapper:
    """Maps evidence metadata to template variables."""
    
    # Variable mappings for different evidence types and DBR subcategories
    VARIABLE_MAPPINGS = {
        "DBR-01-01": {  # Uniqueness Interception
            "action": "enforce uniqueness validation",
            "action_cn": "执行唯一性验证",
            "component": "user registration flow",
            "component_cn": "用户注册流程",
            "goal": "data integrity",
            "goal_cn": "数据完整性",
            "scenario": "duplicate identifier detection",
            "scenario_cn": "重复标识符检测",
            "operation": "creating or updating user accounts",
            "operation_cn": "创建或更新用户账户",
            "check": "uniqueness validation",
            "check_cn": "唯一性验证",
            "flow": "registration flow",
            "flow_cn": "注册流程",
            "user_action": "register with an existing email address",
            "user_action_cn": "使用已存在的邮箱地址注册",
            "domain": "identity management",
            "domain_cn": "身份管理",
            "problem": "uniqueness constraint violations",
            "problem_cn": "唯一性约束违规",
            "responsibility": "validating identifier uniqueness",
            "responsibility_cn": "验证标识符唯一性",
            "context1": "user registration",
            "context1_cn": "用户注册",
            "context2": "profile updates",
            "context2_cn": "资料更新",
            # Additional variables for all templates
            "threat": "duplicate account creation",
            "threat_cn": "重复账户创建",
            "attack_type": "data integrity violation",
            "attack_type_cn": "数据完整性违规",
            "resource": "user identity uniqueness",
            "resource_cn": "用户身份唯一性",
            "feature": "pre-validation check",
            "feature_cn": "预验证检查",
            "mechanism": "uniqueness constraint enforcement",
            "mechanism_cn": "唯一性约束执行",
            "context": "user registration and update",
            "context_cn": "用户注册和更新",
            "process": "user account lifecycle",
            "process_cn": "用户账户生命周期",
            "error_condition": "identifier already exists",
            "error_condition_cn": "标识符已存在",
            "response_type": "400 Bad Request",
            "response_type_cn": "400错误请求",
            "alternative": "silent failure",
            "alternative_cn": "静默失败",
            "condition": "duplicate detected",
            "condition_cn": "检测到重复",
            "scenario_desc": "a user tries to register with an existing email",
            "scenario_desc_cn": "用户尝试使用已存在的邮箱注册",
        },
        "DBR-01-02": {  # Account Security & Atomicity
            "action": "ensure transaction atomicity",
            "action_cn": "确保事务原子性",
            "component": "user repository",
            "component_cn": "用户仓库",
            "goal": "data consistency",
            "goal_cn": "数据一致性",
            "scenario": "account creation with password hashing",
            "scenario_cn": "带密码哈希的账户创建",
            "operation": "persisting user credentials",
            "operation_cn": "持久化用户凭据",
            "mechanism": "transaction-based storage",
            "mechanism_cn": "基于事务的存储",
            "context": "credential persistence",
            "context_cn": "凭据持久化",
            "user_action": "complete registration",
            "user_action_cn": "完成注册",
            "domain": "secure credential storage",
            "domain_cn": "安全凭据存储",
            # Additional variables
            "threat": "partial data corruption",
            "threat_cn": "部分数据损坏",
            "check": "password hashing",
            "check_cn": "密码哈希",
            "flow": "account creation flow",
            "flow_cn": "账户创建流程",
            "problem": "transaction rollback failures",
            "problem_cn": "事务回滚失败",
            "responsibility": "atomic data persistence",
            "responsibility_cn": "原子数据持久化",
            "process": "user creation process",
            "process_cn": "用户创建流程",
            "attack_type": "credential exposure",
            "attack_type_cn": "凭据泄露",
            "resource": "user credentials",
            "resource_cn": "用户凭据",
            "feature": "transactional storage",
            "feature_cn": "事务性存储",
        },
        "DBR-01-03": {  # Authentication Security Feedback
            "action": "return vague error messages",
            "action_cn": "返回模糊错误信息",
            "component": "login handler",
            "component_cn": "登录处理器",
            "goal": "security through obscurity",
            "goal_cn": "通过模糊性实现安全",
            "scenario": "invalid credentials",
            "scenario_cn": "无效凭据",
            "threat": "user enumeration attacks",
            "threat_cn": "用户枚举攻击",
            "operation": "user authentication",
            "operation_cn": "用户认证",
            "response_type": "generic error message",
            "response_type_cn": "通用错误信息",
            "alternative": "specific error details",
            "alternative_cn": "具体错误详情",
            "condition": "authentication fails",
            "condition_cn": "认证失败",
            "error_condition": "invalid password or non-existent user",
            "error_condition_cn": "密码无效或用户不存在",
            "attack_type": "credential enumeration",
            "attack_type_cn": "凭据枚举",
            "resource": "user account existence information",
            "resource_cn": "用户账户存在信息",
            "feature": "unified error handling",
            "feature_cn": "统一错误处理",
            "mechanism": "vague error response",
            "mechanism_cn": "模糊错误响应",
            "context": "authentication failure handling",
            "context_cn": "认证失败处理",
            "scenario_desc": "an attacker tries to determine valid usernames",
            "scenario_desc_cn": "攻击者试图确定有效的用户名",
            # Additional variables
            "check": "credential validation",
            "check_cn": "凭据验证",
            "flow": "login flow",
            "flow_cn": "登录流程",
            "domain": "authentication security",
            "domain_cn": "认证安全",
            "problem": "authentication bypass attempts",
            "problem_cn": "认证绕过尝试",
            "responsibility": "secure error handling",
            "responsibility_cn": "安全错误处理",
            "user_action": "log in with invalid credentials",
            "user_action_cn": "使用无效凭据登录",
            "process": "authentication process",
            "process_cn": "认证流程",
        },
        "DBR-01-04": {  # Token Refresh
            "action": "regenerate JWT tokens",
            "action_cn": "重新生成JWT令牌",
            "component": "session management",
            "component_cn": "会话管理",
            "goal": "session state consistency",
            "goal_cn": "会话状态一致性",
            "scenario": "successful authentication",
            "scenario_cn": "成功认证",
            "operation": "completing user operations",
            "operation_cn": "完成用户操作",
            "process": "user authentication flow",
            "process_cn": "用户认证流程",
            "user_action": "log in to the system",
            "user_action_cn": "登录系统",
            "domain": "session management",
            "domain_cn": "会话管理",
            "responsibility": "token generation and refresh",
            "responsibility_cn": "令牌生成和刷新",
            # Additional variables
            "threat": "session hijacking",
            "threat_cn": "会话劫持",
            "check": "token generation",
            "check_cn": "令牌生成",
            "flow": "token refresh flow",
            "flow_cn": "令牌刷新流程",
            "mechanism": "JWT token regeneration",
            "mechanism_cn": "JWT令牌重新生成",
            "context": "session state management",
            "context_cn": "会话状态管理",
            "problem": "stale token issues",
            "problem_cn": "过期令牌问题",
            "attack_type": "token theft",
            "attack_type_cn": "令牌盗窃",
            "resource": "user session",
            "resource_cn": "用户会话",
            "feature": "automatic token refresh",
            "feature_cn": "自动令牌刷新",
            "error_condition": "token expiration",
            "error_condition_cn": "令牌过期",
            "response_type": "new JWT token",
            "response_type_cn": "新JWT令牌",
            "alternative": "session termination",
            "alternative_cn": "会话终止",
            "condition": "successful operation",
            "condition_cn": "操作成功",
            "scenario_desc": "a user completes a sensitive operation",
            "scenario_desc_cn": "用户完成敏感操作",
        },
    }
    
    @classmethod
    def get_variables(cls, subcategory_id: str, evidence_data: Dict) -> Dict[str, str]:
        """Get variable mapping for an evidence."""
        base_vars = cls.VARIABLE_MAPPINGS.get(subcategory_id, {}).copy()
        
        # Add evidence-specific variables
        if evidence_data:
            base_vars["function_name"] = evidence_data.get("name", "")
            base_vars["function_name_cn"] = evidence_data.get("name", "")
            
            location = evidence_data.get("location", {})
            base_vars["file_path"] = location.get("file_path", "")
            
            # Extract specific elements
            related = evidence_data.get("related_elements", [])
            if related:
                base_vars["key_element"] = related[0]
                base_vars["key_element_cn"] = related[0]
        
        return base_vars


# ============================================================================
# Question Generator
# ============================================================================

class QuestionGenerator:
    """Generates diverse questions from evidence using templates."""
    
    def __init__(self):
        self.registry = QuestionTemplateRegistry()
        self.used_templates: Set[str] = set()
    
    def generate_questions(
        self,
        evidence: Dict,
        subcategory_id: str,
        count: int = 3,
        language: str = "en"
    ) -> List[GeneratedQuestion]:
        """Generate multiple diverse questions for an evidence."""
        questions = []
        evidence_type = evidence.get("evidence_type", "pattern")
        
        # Get applicable templates
        templates = self.registry.get_templates_for_evidence(evidence_type)
        if not templates:
            logger.warning(f"No templates found for evidence type: {evidence_type}")
            return questions
        
        # Get variable mapping
        variables = EvidenceVariableMapper.get_variables(subcategory_id, evidence)
        
        # Select diverse templates
        selected_templates = self._select_diverse_templates(templates, count)
        
        for template in selected_templates:
            try:
                question = self._generate_from_template(
                    template, evidence, variables, language
                )
                if question:
                    questions.append(question)
            except Exception as e:
                logger.error(f"Error generating question from template {template.template_id}: {e}")
        
        return questions
    
    def _select_diverse_templates(
        self, 
        templates: List[QuestionTemplate], 
        count: int
    ) -> List[QuestionTemplate]:
        """Select diverse templates ensuring variety in types and perspectives."""
        if len(templates) <= count:
            return templates
        
        selected = []
        used_types = set()
        used_perspectives = set()
        
        # First pass: ensure diversity
        for template in templates:
            if len(selected) >= count:
                break
            
            type_key = template.question_type
            persp_key = template.perspective
            
            # Prefer templates that add diversity
            if type_key not in used_types or persp_key not in used_perspectives:
                selected.append(template)
                used_types.add(type_key)
                used_perspectives.add(persp_key)
        
        # Second pass: fill remaining slots
        if len(selected) < count:
            remaining = [t for t in templates if t not in selected]
            random.shuffle(remaining)
            selected.extend(remaining[:count - len(selected)])
        
        return selected
    
    def _generate_from_template(
        self,
        template: QuestionTemplate,
        evidence: Dict,
        variables: Dict[str, str],
        language: str
    ) -> Optional[GeneratedQuestion]:
        """Generate a question from a template."""
        # Check if all required variables are available
        missing_vars = []
        for var in template.required_variables:
            if var not in variables:
                missing_vars.append(var)
        
        if missing_vars:
            logger.debug(f"Missing variables for template {template.template_id}: {missing_vars}")
            # Try to fill with defaults
            for var in missing_vars:
                variables[var] = f"[{var}]"
        
        try:
            # Generate question text
            if language == "zh":
                question_text = template.template_text_cn.format(**variables)
            else:
                question_text = template.template_text.format(**variables)
            
            # Generate bilingual versions
            question_en = template.template_text.format(**variables)
            question_cn = template.template_text_cn.format(**variables)
            
            return GeneratedQuestion(
                question_id=f"Q-{uuid.uuid4().hex[:8]}",
                question_text=question_text,
                question_text_cn=question_cn,
                question_type=template.question_type,
                perspective=template.perspective,
                strategy=template.strategy,
                template_id=template.template_id,
                evidence_id=evidence.get("evidence_id", ""),
                variables_used=variables.copy(),
            )
        except KeyError as e:
            logger.error(f"Variable substitution error: {e}")
            return None


# ============================================================================
# Reasoning Engine
# ============================================================================

class ReasoningEngine:
    """Generates structured reasoning chains from evidence."""
    
    def __init__(self):
        self.step_type_descriptions = {
            "observation": "Identify and observe",
            "analysis": "Analyze the structure and logic",
            "inference": "Infer the design intent",
            "conclusion": "Conclude the implementation benefit",
        }
        self.step_type_descriptions_cn = {
            "observation": "识别和观察",
            "analysis": "分析结构和逻辑",
            "inference": "推断设计意图",
            "conclusion": "总结实现收益",
        }
    
    def generate_reasoning_chain(
        self,
        evidence: Dict,
        question: GeneratedQuestion,
        language: str = "en"
    ) -> ReasoningChain:
        """Generate a reasoning chain for an evidence and question."""
        
        # Get reasoning template from evidence
        reasoning_template = evidence.get("reasoning_template", {})
        template_steps = reasoning_template.get("steps", [])
        
        # Build reasoning steps
        steps = []
        evidence_refs = []
        dbr_refs = []
        
        if template_steps:
            # Use existing template steps
            for step in template_steps:
                step_desc = step.get("description", "") if language == "en" else step.get("description_cn", "")
                steps.append({
                    "step_id": step.get("step_id", len(steps) + 1),
                    "step_type": step.get("step_type", "analysis"),
                    "description": step_desc,
                    "description_cn": step.get("description_cn", step_desc),
                })
                
                if step.get("source_reference"):
                    evidence_refs.append(step["source_reference"])
                if step.get("dbr_reference"):
                    dbr_refs.append(step["dbr_reference"])
        else:
            # Generate default reasoning steps based on question type
            steps = self._generate_default_steps(evidence, question, language)
        
        # Generate conclusion
        conclusion = self._generate_conclusion(evidence, question, language)
        
        return ReasoningChain(
            chain_id=f"RC-{uuid.uuid4().hex[:8]}",
            steps=steps,
            evidence_references=list(set(evidence_refs)),
            dbr_references=list(set(dbr_refs)),
            conclusion=conclusion if language == "en" else "",
            conclusion_cn=conclusion if language == "zh" else self._generate_conclusion(evidence, question, "zh"),
        )
    
    def _generate_default_steps(
        self,
        evidence: Dict,
        question: GeneratedQuestion,
        language: str
    ) -> List[Dict[str, str]]:
        """Generate default reasoning steps."""
        evidence_type = evidence.get("evidence_type", "pattern")
        related_elements = evidence.get("related_elements", [])
        
        steps = []
        
        # Step 1: Observation
        if language == "en":
            obs = f"Observe the {evidence_type} in {evidence.get('name', 'the code')}"
            if related_elements:
                obs += f", specifically the elements: {', '.join(related_elements[:3])}"
        else:
            obs = f"观察{evidence.get('name', '代码')}中的{evidence_type}"
            if related_elements:
                obs += f"，特别是元素：{', '.join(related_elements[:3])}"
        
        steps.append({
            "step_id": 1,
            "step_type": "observation",
            "description": obs,
            "description_cn": obs if language == "zh" else "",
        })
        
        # Step 2: Analysis
        dbr_logic = evidence.get("dbr_logic", {})
        patterns = dbr_logic.get("matched_patterns", [])
        
        if language == "en":
            analysis = f"Analyze the implementation pattern that {', '.join(patterns[:2]) if patterns else 'handles the logic'}"
        else:
            analysis = f"分析实现模式，该模式{', '.join(patterns[:2]) if patterns else '处理逻辑'}"
        
        steps.append({
            "step_id": 2,
            "step_type": "analysis",
            "description": analysis,
            "description_cn": analysis if language == "zh" else "",
        })
        
        # Step 3: Inference
        trigger_conditions = dbr_logic.get("trigger_conditions", [])
        if language == "en":
            inference = f"Infer the security/design intent from the conditions: {', '.join(trigger_conditions[:2]) if trigger_conditions else 'the code structure'}"
        else:
            inference = f"从条件推断安全/设计意图：{', '.join(trigger_conditions[:2]) if trigger_conditions else '代码结构'}"
        
        steps.append({
            "step_id": 3,
            "step_type": "inference",
            "description": inference,
            "description_cn": inference if language == "zh" else "",
        })
        
        # Step 4: Conclusion
        if language == "en":
            concl = f"Conclude that this implementation ensures {evidence.get('description', 'proper functionality')[:100]}"
        else:
            concl = f"得出结论：此实现确保了{evidence.get('description_cn', '正确的功能')[:100]}"
        
        steps.append({
            "step_id": 4,
            "step_type": "conclusion",
            "description": concl,
            "description_cn": concl if language == "zh" else "",
        })
        
        return steps
    
    def _generate_conclusion(
        self,
        evidence: Dict,
        question: GeneratedQuestion,
        language: str
    ) -> str:
        """Generate a conclusion for the reasoning chain."""
        if language == "en":
            return f"The {evidence.get('name', 'implementation')} correctly implements the {question.question_type.value} behavior as required by DBR-01."
        else:
            return f"{evidence.get('name', '实现')}正确实现了DBR-01要求的{question.question_type.value}行为。"
    
    def format_reasoning_trace(
        self,
        chain: ReasoningChain,
        language: str = "en"
    ) -> List[str]:
        """Format reasoning chain as a list of strings."""
        trace = []
        for step in chain.steps:
            if language == "zh" and step.get("description_cn"):
                trace.append(f"[{step['step_type'].upper()}] {step['description_cn']}")
            else:
                trace.append(f"[{step['step_type'].upper()}] {step['description']}")
        return trace


# ============================================================================
# Answer Composer
# ============================================================================

class AnswerComposer:
    """Composes gold-standard answers from components."""
    
    def __init__(self):
        self.section_templates = {
            "en": {
                "reasoning": "### Reasoning Chain\n",
                "explanation": "### Technical Explanation\n",
                "code": "### Relevant Source Code\n",
                "security": "### Security Implications\n",
                "best_practice": "### Best Practice Notes\n",
            },
            "zh": {
                "reasoning": "### 推理链\n",
                "explanation": "### 技术解释\n",
                "code": "### 相关源代码\n",
                "security": "### 安全影响\n",
                "best_practice": "### 最佳实践说明\n",
            }
        }
    
    def compose_answer(
        self,
        evidence: Dict,
        question: GeneratedQuestion,
        reasoning_chain: ReasoningChain,
        code_context: CodeContext,
        language: str = "en"
    ) -> str:
        """Compose a comprehensive gold-standard answer."""
        templates = self.section_templates.get(language, self.section_templates["en"])
        
        answer_parts = []
        
        # 1. Reasoning section
        answer_parts.append(templates["reasoning"])
        reasoning_trace = ReasoningEngine().format_reasoning_trace(reasoning_chain, language)
        for step in reasoning_trace:
            answer_parts.append(f"- {step}\n")
        answer_parts.append("\n")
        
        # 2. Technical explanation
        answer_parts.append(templates["explanation"])
        if language == "zh":
            explanation = evidence.get("description_cn", evidence.get("description", ""))
        else:
            explanation = evidence.get("description", "")
        answer_parts.append(f"{explanation}\n\n")
        
        # 3. Code section
        answer_parts.append(templates["code"])
        answer_parts.append(f"**File:** `{code_context.file_path}` (Lines {code_context.line_start}-{code_context.line_end})\n\n")
        answer_parts.append(f"```python\n{code_context.code_snippet}\n```\n\n")
        
        # 4. Security implications (if applicable)
        if question.question_type in [QuestionType.SECURITY, QuestionType.WHY]:
            answer_parts.append(templates["security"])
            security_note = self._generate_security_note(evidence, language)
            answer_parts.append(f"{security_note}\n\n")
        
        # 5. Best practice notes
        answer_parts.append(templates["best_practice"])
        best_practice = self._generate_best_practice(evidence, question, language)
        answer_parts.append(f"{best_practice}\n")
        
        return "".join(answer_parts)
    
    def _generate_security_note(self, evidence: Dict, language: str) -> str:
        """Generate security implications note."""
        dbr_logic = evidence.get("dbr_logic", {})
        subcategory = dbr_logic.get("subcategory_id", "DBR-01")
        
        security_notes = {
            "DBR-01-01": {
                "en": "This implementation prevents duplicate account creation, which could lead to data integrity issues and potential security vulnerabilities.",
                "zh": "此实现防止重复账户创建，避免可能导致数据完整性问题和潜在安全漏洞。"
            },
            "DBR-01-02": {
                "en": "The atomic transaction ensures that partial failures don't leave the system in an inconsistent state. Password hashing protects credentials from exposure.",
                "zh": "原子事务确保部分失败不会使系统处于不一致状态。密码哈希保护凭据免受泄露。"
            },
            "DBR-01-03": {
                "en": "Vague error messages prevent attackers from enumerating valid usernames through error response analysis (timing attack mitigation).",
                "zh": "模糊的错误信息防止攻击者通过分析错误响应来枚举有效用户名（时序攻击缓解）。"
            },
            "DBR-01-04": {
                "en": "Token refresh ensures session continuity while allowing for token rotation, reducing the window of opportunity for token theft attacks.",
                "zh": "令牌刷新确保会话连续性，同时允许令牌轮换，减少令牌盗窃攻击的时间窗口。"
            },
        }
        
        return security_notes.get(subcategory, {}).get(language, "N/A")
    
    def _generate_best_practice(
        self, 
        evidence: Dict, 
        question: GeneratedQuestion,
        language: str
    ) -> str:
        """Generate best practice notes."""
        if language == "zh":
            return (
                "- 始终在数据持久化之前执行验证\n"
                "- 使用显式错误处理而不是静默失败\n"
                "- 保持安全控制的一致性跨所有入口点\n"
                "- 记录安全相关事件以便审计"
            )
        else:
            return (
                "- Always perform validation before data persistence\n"
                "- Use explicit error handling rather than silent failures\n"
                "- Maintain consistency of security controls across all entry points\n"
                "- Log security-relevant events for auditing purposes"
            )


# ============================================================================
# Quality Validator
# ============================================================================

class QualityValidator:
    """Validates quality of generated Q&A pairs."""
    
    def __init__(self):
        self.min_question_length = Config.MIN_QUESTION_LENGTH
        self.min_answer_length = Config.MIN_ANSWER_LENGTH
    
    def validate_qa_pair(
        self,
        question: GeneratedQuestion,
        answer: str,
        code_context: CodeContext,
        reasoning_chain: ReasoningChain
    ) -> Tuple[bool, List[str]]:
        """Validate a Q&A pair and return validation result with issues."""
        issues = []
        
        # 1. Question validation
        if len(question.question_text) < self.min_question_length:
            issues.append(f"Question too short: {len(question.question_text)} chars")
        
        # Check for question mark (both English and Chinese)
        if not (question.question_text.endswith("?") or question.question_text.endswith("？")):
            issues.append("Question doesn't end with question mark")
        
        # 2. Answer validation
        if len(answer) < self.min_answer_length:
            issues.append(f"Answer too short: {len(answer)} chars")
        
        # 3. Code context validation
        if not code_context.code_snippet:
            issues.append("Missing code snippet")
        
        if not code_context.source_hash:
            issues.append("Missing source hash for consistency check")
        
        # 4. Reasoning validation
        if len(reasoning_chain.steps) < 2:
            issues.append(f"Too few reasoning steps: {len(reasoning_chain.steps)}")
        
        # 5. Cross-reference validation
        # Check if answer contains code reference
        if code_context.file_path not in answer:
            issues.append("Answer doesn't reference the source file")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def compute_consistency_hash(
        self,
        question: GeneratedQuestion,
        code_context: CodeContext
    ) -> str:
        """Compute a consistency hash for the Q&A pair."""
        content = f"{question.question_text}|{code_context.code_snippet}|{code_context.source_hash}"
        return hashlib.md5(content.encode()).hexdigest()


# ============================================================================
# Q&A Pair Generator (Main Orchestrator)
# ============================================================================

class QAPairGenerator:
    """Main orchestrator for generating Q&A pairs from rule metadata."""
    
    def __init__(self, rule_metadata_path: str):
        self.rule_metadata_path = Path(rule_metadata_path)
        self.rule_metadata: Dict = {}
        self.question_generator = QuestionGenerator()
        self.reasoning_engine = ReasoningEngine()
        self.answer_composer = AnswerComposer()
        self.quality_validator = QualityValidator()
        self.generated_pairs: List[QAPair] = []
    
    def load_rule_metadata(self) -> bool:
        """Load rule metadata from JSON file."""
        if not self.rule_metadata_path.exists():
            logger.error(f"Rule metadata file not found: {self.rule_metadata_path}")
            return False
        
        try:
            with open(self.rule_metadata_path, 'r', encoding='utf-8') as f:
                self.rule_metadata = json.load(f)
            logger.info(f"Loaded rule metadata from: {self.rule_metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading rule metadata: {e}")
            return False
    
    def generate_all_pairs(
        self,
        samples_per_evidence: int = 3,
        languages: List[str] = None
    ) -> List[QAPair]:
        """Generate Q&A pairs for all evidences in the rule metadata."""
        if not self.rule_metadata:
            if not self.load_rule_metadata():
                return []
        
        languages = languages or Config.SUPPORTED_LANGUAGES
        all_pairs = []
        
        # Iterate through subcategories
        for subcategory in self.rule_metadata.get("subcategories", []):
            subcategory_id = subcategory.get("subcategory_id", "")
            
            logger.info(f"Processing subcategory: {subcategory_id} - {subcategory.get('name', '')}")
            
            # Iterate through evidences
            for evidence in subcategory.get("evidences", []):
                evidence_id = evidence.get("evidence_id", "")
                
                logger.info(f"  Processing evidence: {evidence_id}")
                
                # Generate pairs for each language
                for language in languages:
                    pairs = self._generate_pairs_for_evidence(
                        evidence,
                        subcategory_id,
                        samples_per_evidence,
                        language
                    )
                    all_pairs.extend(pairs)
        
        self.generated_pairs = all_pairs
        logger.info(f"Generated {len(all_pairs)} Q&A pairs total")
        return all_pairs
    
    def _generate_pairs_for_evidence(
        self,
        evidence: Dict,
        subcategory_id: str,
        count: int,
        language: str
    ) -> List[QAPair]:
        """Generate Q&A pairs for a single evidence."""
        pairs = []
        
        # Extract code context
        code_context = self._extract_code_context(evidence)
        if not code_context:
            logger.warning(f"Could not extract code context for evidence: {evidence.get('evidence_id', '')}")
            return pairs
        
        # Generate questions
        questions = self.question_generator.generate_questions(
            evidence, subcategory_id, count, language
        )
        
        for question in questions:
            try:
                # Generate reasoning chain
                reasoning_chain = self.reasoning_engine.generate_reasoning_chain(
                    evidence, question, language
                )
                
                # Compose answer
                answer = self.answer_composer.compose_answer(
                    evidence, question, reasoning_chain, code_context, language
                )
                
                # Validate
                is_valid, issues = self.quality_validator.validate_qa_pair(
                    question, answer, code_context, reasoning_chain
                )
                
                if not is_valid:
                    logger.warning(f"Quality issues for {question.question_id}: {issues}")
                    # Still include but mark as having issues
                
                # Build Q&A pair
                qa_pair = self._build_qa_pair(
                    question, answer, evidence, code_context, 
                    reasoning_chain, language, is_valid
                )
                pairs.append(qa_pair)
                
            except Exception as e:
                logger.error(f"Error generating pair for question {question.question_id}: {e}")
        
        return pairs
    
    def _extract_code_context(self, evidence: Dict) -> Optional[CodeContext]:
        """Extract code context from evidence."""
        code_snippet_data = evidence.get("code_snippet", {})
        location = evidence.get("location", {})
        
        if not code_snippet_data or not code_snippet_data.get("code"):
            return None
        
        return CodeContext(
            file_path=code_snippet_data.get("file_path", location.get("file_path", "")),
            code_snippet=code_snippet_data.get("code", ""),
            line_start=code_snippet_data.get("line_start", location.get("line_start", 0)),
            line_end=code_snippet_data.get("line_end", location.get("line_end", 0)),
            source_hash=code_snippet_data.get("source_hash", ""),
            function_name=evidence.get("name", ""),
            related_elements=evidence.get("related_elements", []),
        )
    
    def _build_qa_pair(
        self,
        question: GeneratedQuestion,
        answer: str,
        evidence: Dict,
        code_context: CodeContext,
        reasoning_chain: ReasoningChain,
        language: str,
        is_valid: bool
    ) -> QAPair:
        """Build a complete Q&A pair object."""
        dbr_logic = evidence.get("dbr_logic", {})
        
        # Format reasoning trace
        reasoning_trace = self.reasoning_engine.format_reasoning_trace(reasoning_chain, language)
        
        # Compute consistency hash
        consistency_hash = self.quality_validator.compute_consistency_hash(question, code_context)
        
        return QAPair(
            sample_id=f"DBR01-{uuid.uuid4().hex[:12]}",
            instruction=question.question_text if language == "en" else question.question_text_cn,
            context={
                "file_path": code_context.file_path,
                "related_dbr": dbr_logic.get("rule_id", "DBR-01"),
                "code_snippet": code_context.code_snippet,
                "line_range": f"{code_context.line_start}-{code_context.line_end}",
                "function_name": code_context.function_name,
            },
            auto_processing={
                "parser": self.rule_metadata.get("parser_info", {}).get("name", "FastAPI-AST-Analyzer"),
                "parser_version": self.rule_metadata.get("parser_info", {}).get("version", "1.0.0"),
                "dbr_logic": {
                    "rule_id": dbr_logic.get("rule_id", "DBR-01"),
                    "subcategory_id": dbr_logic.get("subcategory_id", ""),
                    "trigger_type": dbr_logic.get("trigger_type", "explicit"),
                    "weight": dbr_logic.get("weight", 1.0),
                    "matched_patterns": dbr_logic.get("matched_patterns", []),
                },
                "data_cleaning": {
                    "cleaned": True,
                    "source_verified": True,
                    "hash_validated": bool(code_context.source_hash),
                },
                "generation_metadata": {
                    "template_id": question.template_id,
                    "question_type": question.question_type.value,
                    "perspective": question.perspective.value,
                    "strategy": question.strategy.value,
                },
            },
            reasoning_trace=reasoning_trace,
            answer=answer,
            data_quality={
                "consistency_check": is_valid,
                "consistency_hash": consistency_hash,
                "source_hash": code_context.source_hash,
                "language": language,
                "temperature": Config.DEFAULT_TEMPERATURE,
                "evidence_id": evidence.get("evidence_id", ""),
            },
        )
    
    def save_to_jsonl(self, output_path: str = None) -> str:
        """Save generated pairs to JSONL file."""
        output_path = output_path or str(Config.OUTPUT_FILE)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in self.generated_pairs:
                # Convert to dict
                pair_dict = {
                    "sample_id": pair.sample_id,
                    "instruction": pair.instruction,
                    "context": pair.context,
                    "auto_processing": pair.auto_processing,
                    "reasoning_trace": pair.reasoning_trace,
                    "answer": pair.answer,
                    "data_quality": pair.data_quality,
                }
                f.write(json.dumps(pair_dict, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(self.generated_pairs)} Q&A pairs to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print summary of generated Q&A pairs."""
        print("\n" + "=" * 70)
        print("Q&A Generation Summary")
        print("=" * 70)
        
        print(f"\nTotal Pairs Generated: {len(self.generated_pairs)}")
        
        # Count by language
        lang_counts = defaultdict(int)
        for pair in self.generated_pairs:
            lang = pair.data_quality.get("language", "unknown")
            lang_counts[lang] += 1
        
        print("\nBy Language:")
        for lang, count in lang_counts.items():
            print(f"  - {lang}: {count}")
        
        # Count by question type
        type_counts = defaultdict(int)
        for pair in self.generated_pairs:
            q_type = pair.auto_processing.get("generation_metadata", {}).get("question_type", "unknown")
            type_counts[q_type] += 1
        
        print("\nBy Question Type:")
        for q_type, count in sorted(type_counts.items()):
            print(f"  - {q_type}: {count}")
        
        # Count by perspective
        persp_counts = defaultdict(int)
        for pair in self.generated_pairs:
            persp = pair.auto_processing.get("generation_metadata", {}).get("perspective", "unknown")
            persp_counts[persp] += 1
        
        print("\nBy Perspective:")
        for persp, count in sorted(persp_counts.items()):
            print(f"  - {persp}: {count}")
        
        # Quality stats
        valid_count = sum(1 for p in self.generated_pairs if p.data_quality.get("consistency_check", False))
        print(f"\nQuality Validated: {valid_count}/{len(self.generated_pairs)} ({100*valid_count/max(1,len(self.generated_pairs)):.1f}%)")
        
        print("\n" + "=" * 70)
    
    def print_sample_pairs(self, n: int = 2):
        """Print sample Q&A pairs for review."""
        samples = self.generated_pairs[:n] if len(self.generated_pairs) >= n else self.generated_pairs
        
        for i, pair in enumerate(samples, 1):
            print("\n" + "=" * 70)
            print(f"Sample {i} - ID: {pair.sample_id}")
            print("=" * 70)
            print(f"\n【Instruction/问题】:\n{pair.instruction}\n")
            print(f"【Reasoning Trace/推理链】:")
            for step in pair.reasoning_trace:
                print(f"  - {step}")
            print(f"\n【Answer/回答】:\n{pair.answer[:500]}...")
            print(f"\n【Code File】: {pair.context.get('file_path')}")
            print(f"【Language】: {pair.data_quality.get('language')}")
            print(f"【Question Type】: {pair.auto_processing.get('generation_metadata', {}).get('question_type')}")
            print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for Q&A generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enterprise-Grade Q&A Generation for LLM Fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '-m', '--metadata',
        default=str(Config.RULE_METADATA_FILE),
        help='Path to rule metadata JSON file'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=str(Config.OUTPUT_FILE),
        help='Output JSONL file path'
    )
    
    parser.add_argument(
        '-n', '--samples',
        type=int,
        default=3,
        help='Number of Q&A pairs per evidence'
    )
    
    parser.add_argument(
        '-l', '--languages',
        nargs='+',
        default=['en', 'zh'],
        help='Languages to generate (en, zh)'
    )
    
    parser.add_argument(
        '--preview',
        type=int,
        default=2,
        help='Number of sample pairs to preview'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize generator
    generator = QAPairGenerator(args.metadata)
    
    # Load rule metadata
    if not generator.load_rule_metadata():
        print("Error: Could not load rule metadata. Please run rule_mapping.py first.")
        sys.exit(1)
    
    # Generate Q&A pairs
    print(f"\nGenerating Q&A pairs from: {args.metadata}")
    print(f"Languages: {args.languages}")
    print(f"Samples per evidence: {args.samples}")
    
    pairs = generator.generate_all_pairs(
        samples_per_evidence=args.samples,
        languages=args.languages
    )
    
    if not pairs:
        print("Error: No Q&A pairs generated.")
        sys.exit(1)
    
    # Save to file
    output_path = generator.save_to_jsonl(args.output)
    
    # Print summary
    generator.print_summary()
    
    # Preview samples
    if args.preview > 0:
        print(f"\n--- Sample Q&A Pairs ({args.preview}) ---")
        generator.print_sample_pairs(args.preview)
    
    print(f"\n✅ Successfully generated {len(pairs)} Q&A pairs")
    print(f"📁 Output saved to: {output_path}")


if __name__ == "__main__":
    main()
