"""
Design Question and Solution Generators.

Generates design-focused questions and solutions for Scenario 2.
Reuses diversity concepts from v9.2.
"""

import re
import uuid
import random
import hashlib
import sys
import time
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    DesignScenario, DesignAspect, DesignerRole, QuestionIntent,
    GeneratedDesignQuestion, DesignSolution
)
from config import DesignConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Local copy of LLM client to avoid import conflicts."""
    
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


class DesignQuestionTemplates:
    """
    Design question templates organized by intent and aspect.
    
    Questions are from developer/designer perspective.
    """
    
    TEMPLATES = {
        "en": {
            # HOW_TO_DESIGN: How should we design/implement
            QuestionIntent.HOW_TO_DESIGN: {
                DesignAspect.VALIDATION: [
                    "How should we design the validation layer for {scenario}?",
                    "What's the recommended approach for implementing uniqueness checks in {scenario}?",
                    "How do we structure the validation pipeline for {scenario}?",
                ],
                DesignAspect.SECURITY: [
                    "How should we design the security layer for {scenario}?",
                    "What security patterns should we implement for {scenario}?",
                    "How do we ensure secure credential handling in {scenario}?",
                ],
                DesignAspect.ERROR_HANDLING: [
                    "How should we design error responses for {scenario}?",
                    "What's the best approach for error handling in {scenario}?",
                    "How do we implement user-friendly yet secure error messages?",
                ],
                DesignAspect.API_DESIGN: [
                    "How should we design the API endpoints for {scenario}?",
                    "What's the recommended REST API structure for {scenario}?",
                    "How do we design the request/response contracts?",
                ],
                DesignAspect.DATA_MODEL: [
                    "How should we design the data model for {scenario}?",
                    "What database schema best supports {scenario}?",
                    "How do we structure the user entity for this requirement?",
                ],
                DesignAspect.SESSION_MANAGEMENT: [
                    "How should we design the session management for {scenario}?",
                    "What's the best token strategy for {scenario}?",
                    "How do we implement secure session handling?",
                ],
            },
            
            # WHY_THIS_WAY: Why is this approach recommended
            QuestionIntent.WHY_THIS_WAY: {
                DesignAspect.VALIDATION: [
                    "Why should validation happen before database writes in {scenario}?",
                    "Why is sequential validation preferred over parallel in this case?",
                ],
                DesignAspect.SECURITY: [
                    "Why should we use generic error messages for authentication failures?",
                    "Why is password hashing essential before storage?",
                ],
                DesignAspect.ERROR_HANDLING: [
                    "Why shouldn't we expose which field caused the validation error?",
                    "Why is consistent error timing important for security?",
                ],
                DesignAspect.ARCHITECTURE: [
                    "Why is the repository pattern recommended for this scenario?",
                    "Why should we separate validation logic from business logic?",
                ],
            },
            
            # WHAT_PATTERN: What design pattern to use
            QuestionIntent.WHAT_PATTERN: {
                DesignAspect.ARCHITECTURE: [
                    "What design patterns are suitable for {scenario}?",
                    "Which architectural pattern best fits the authentication flow?",
                    "What patterns ensure data consistency in registration?",
                ],
                DesignAspect.VALIDATION: [
                    "What validation pattern should we use for uniqueness checks?",
                    "Which pattern prevents race conditions in registration?",
                ],
                DesignAspect.ERROR_HANDLING: [
                    "What error handling pattern prevents information leakage?",
                    "Which exception handling strategy is most secure?",
                ],
            },
            
            # TRADE_OFF: Trade-offs between approaches
            QuestionIntent.TRADE_OFF: {
                DesignAspect.ARCHITECTURE: [
                    "What are the trade-offs between synchronous and async validation?",
                    "Should we use database constraints or application-level validation?",
                ],
                DesignAspect.SECURITY: [
                    "What's the trade-off between detailed errors and security?",
                    "Should we prioritize user experience or security in error messages?",
                ],
                DesignAspect.SESSION_MANAGEMENT: [
                    "What are the trade-offs between JWT and session-based auth?",
                    "Should tokens be short-lived or long-lived?",
                ],
            },
            
            # ALTERNATIVE: Alternative approaches
            QuestionIntent.ALTERNATIVE: {
                DesignAspect.VALIDATION: [
                    "What alternatives exist for implementing uniqueness validation?",
                    "Are there other ways to handle concurrent registration?",
                ],
                DesignAspect.SECURITY: [
                    "What are alternative approaches to password hashing?",
                    "Are there other ways to prevent user enumeration?",
                ],
                DesignAspect.DATA_MODEL: [
                    "What alternative data models could support this requirement?",
                    "Could we use a different storage strategy?",
                ],
            },
            
            # BEST_PRACTICE: Industry best practices
            QuestionIntent.BEST_PRACTICE: {
                DesignAspect.SECURITY: [
                    "What are the security best practices for {scenario}?",
                    "What OWASP guidelines apply to this authentication design?",
                ],
                DesignAspect.API_DESIGN: [
                    "What are RESTful best practices for authentication endpoints?",
                    "How should we version the authentication API?",
                ],
                DesignAspect.BEST_PRACTICES: [
                    "What industry standards should we follow for {scenario}?",
                    "What are common mistakes to avoid in this design?",
                ],
            },
            
            # SECURITY_CONCERN: Security considerations
            QuestionIntent.SECURITY_CONCERN: {
                DesignAspect.SECURITY: [
                    "What security vulnerabilities should we address in {scenario}?",
                    "How do we protect against timing attacks in authentication?",
                    "What attack vectors does this design mitigate?",
                ],
                DesignAspect.VALIDATION: [
                    "How do we prevent injection attacks in the validation layer?",
                    "What security considerations exist for input validation?",
                ],
            },
        },
        
        "zh": {
            QuestionIntent.HOW_TO_DESIGN: {
                DesignAspect.VALIDATION: [
                    "我们应该如何设计{scenario}的验证层？",
                    "在{scenario}中实现唯一性检查的推荐方法是什么？",
                    "如何构建{scenario}的验证流水线？",
                ],
                DesignAspect.SECURITY: [
                    "我们应该如何设计{scenario}的安全层？",
                    "在{scenario}中应该实现哪些安全模式？",
                    "如何确保{scenario}中的凭据安全处理？",
                ],
                DesignAspect.ERROR_HANDLING: [
                    "我们应该如何设计{scenario}的错误响应？",
                    "在{scenario}中处理错误的最佳方法是什么？",
                    "如何实现用户友好但安全的错误消息？",
                ],
                DesignAspect.API_DESIGN: [
                    "我们应该如何设计{scenario}的API端点？",
                    "{scenario}的推荐REST API结构是什么？",
                    "如何设计请求/响应契约？",
                ],
                DesignAspect.DATA_MODEL: [
                    "我们应该如何设计{scenario}的数据模型？",
                    "什么样的数据库架构最能支持{scenario}？",
                    "如何为此需求构建用户实体？",
                ],
                DesignAspect.SESSION_MANAGEMENT: [
                    "我们应该如何设计{scenario}的会话管理？",
                    "{scenario}的最佳令牌策略是什么？",
                    "如何实现安全的会话处理？",
                ],
            },
            
            QuestionIntent.WHY_THIS_WAY: {
                DesignAspect.VALIDATION: [
                    "为什么在{scenario}中验证应该在数据库写入之前进行？",
                    "为什么在此情况下顺序验证优于并行验证？",
                ],
                DesignAspect.SECURITY: [
                    "为什么认证失败时应该使用通用错误消息？",
                    "为什么存储前密码哈希是必要的？",
                ],
                DesignAspect.ERROR_HANDLING: [
                    "为什么不应该暴露哪个字段导致了验证错误？",
                    "为什么一致的错误响应时间对安全性很重要？",
                ],
            },
            
            QuestionIntent.WHAT_PATTERN: {
                DesignAspect.ARCHITECTURE: [
                    "哪些设计模式适用于{scenario}？",
                    "哪种架构模式最适合认证流程？",
                    "什么模式能确保注册时的数据一致性？",
                ],
                DesignAspect.VALIDATION: [
                    "唯一性检查应该使用什么验证模式？",
                    "哪种模式可以防止注册时的竞态条件？",
                ],
            },
            
            QuestionIntent.TRADE_OFF: {
                DesignAspect.ARCHITECTURE: [
                    "同步验证和异步验证之间有什么权衡？",
                    "应该使用数据库约束还是应用层验证？",
                ],
                DesignAspect.SECURITY: [
                    "详细错误信息和安全性之间的权衡是什么？",
                    "在错误消息中应该优先考虑用户体验还是安全性？",
                ],
            },
            
            QuestionIntent.BEST_PRACTICE: {
                DesignAspect.SECURITY: [
                    "{scenario}的安全最佳实践是什么？",
                    "哪些OWASP指南适用于此认证设计？",
                ],
                DesignAspect.API_DESIGN: [
                    "认证端点的RESTful最佳实践是什么？",
                    "我们应该如何对认证API进行版本管理？",
                ],
            },
            
            QuestionIntent.SECURITY_CONCERN: {
                DesignAspect.SECURITY: [
                    "在{scenario}中我们应该解决哪些安全漏洞？",
                    "如何防止认证中的时序攻击？",
                    "这个设计能缓解哪些攻击向量？",
                ],
            },
        },
    }
    
    @classmethod
    def get_templates(
        cls,
        language: str,
        intent: QuestionIntent = None,
        aspect: DesignAspect = None
    ) -> List[str]:
        """Get templates filtered by intent and/or aspect."""
        lang_templates = cls.TEMPLATES.get(language, cls.TEMPLATES["en"])
        
        if intent and aspect:
            return lang_templates.get(intent, {}).get(aspect, [])
        
        if intent:
            all_for_intent = []
            for aspect_templates in lang_templates.get(intent, {}).values():
                all_for_intent.extend(aspect_templates)
            return all_for_intent
        
        if aspect:
            all_for_aspect = []
            for intent_dict in lang_templates.values():
                all_for_aspect.extend(intent_dict.get(aspect, []))
            return all_for_aspect
        
        all_templates = []
        for intent_dict in lang_templates.values():
            for aspect_list in intent_dict.values():
                all_templates.extend(aspect_list)
        return all_templates


class DesignQuestionGenerator:
    """
    Generates design-focused questions with diversity.
    
    Reuses semantic fingerprinting concepts from v9.2.
    """
    
    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        self.generated_fingerprints: Set[str] = set()
    
    def generate(
        self,
        scenario: DesignScenario,
        role: DesignerRole,
        count: int,
        language: str,
        used_intents: Set[QuestionIntent] = None,
        used_aspects: Set[DesignAspect] = None
    ) -> List[GeneratedDesignQuestion]:
        """
        Generate design questions for a scenario.
        
        Args:
            scenario: Design scenario
            role: Designer role
            count: Number of questions
            language: Language code
            used_intents: Already used intents
            used_aspects: Already used aspects
            
        Returns:
            List of generated questions
        """
        used_intents = used_intents or set()
        used_aspects = used_aspects or set()
        
        # Try LLM first
        if self.llm.is_available():
            questions = self._generate_with_llm(
                scenario, role, count, language, used_intents
            )
            if len(questions) >= count:
                return questions[:count]
        
        # Fallback to templates
        return self._generate_fallback(
            scenario, role, count, language, used_intents, used_aspects
        )
    
    def _generate_with_llm(
        self,
        scenario: DesignScenario,
        role: DesignerRole,
        count: int,
        language: str,
        used_intents: Set[QuestionIntent]
    ) -> List[GeneratedDesignQuestion]:
        """Generate questions using LLM."""
        # Select underrepresented intent
        all_intents = list(QuestionIntent)
        available = [i for i in all_intents if i not in used_intents]
        if not available:
            available = all_intents
        
        target_intent = random.choice(available)
        
        system = self._get_system_prompt(language, role, target_intent)
        prompt = self._get_generation_prompt(scenario, count, language, target_intent)
        
        response = self.llm.generate_with_task(prompt, "question", system=system)
        
        if response:
            return self._parse_questions(response, role, language, target_intent, scenario)
        return []
    
    def _get_system_prompt(
        self,
        language: str,
        role: DesignerRole,
        intent: QuestionIntent
    ) -> str:
        """Get system prompt for LLM."""
        if language == "zh":
            return f"""你是一位{role.value}，正在设计一个认证系统。
生成关于系统设计的专业问题。

规则：
1. 问题必须从开发设计者角度出发
2. 关注架构、安全、最佳实践
3. 每个问题必须有实际设计价值
4. 避免问代码实现细节"""
        else:
            return f"""You are a {role.value} designing an authentication system.
Generate professional questions about system design.

Rules:
1. Questions must be from developer/designer perspective
2. Focus on architecture, security, best practices
3. Each question must have practical design value
4. Avoid asking about code implementation details"""
    
    def _get_generation_prompt(
        self,
        scenario: DesignScenario,
        count: int,
        language: str,
        intent: QuestionIntent
    ) -> str:
        """Get generation prompt."""
        if language == "zh":
            return f"""设计场景：{scenario.scenario_name}
描述：{scenario.description}
需求：{'; '.join(scenario.requirements[:3])}
约束：{'; '.join(scenario.constraints[:2])}

从设计者角度生成{count}个关于此场景的问题，每行一个："""
        else:
            return f"""Design Scenario: {scenario.scenario_name}
Description: {scenario.description}
Requirements: {'; '.join(scenario.requirements[:3])}
Constraints: {'; '.join(scenario.constraints[:2])}

Generate {count} design questions about this scenario, one per line:"""
    
    def _parse_questions(
        self,
        response: str,
        role: DesignerRole,
        language: str,
        intent: QuestionIntent,
        scenario: DesignScenario
    ) -> List[GeneratedDesignQuestion]:
        """Parse LLM response into questions."""
        questions = []
        
        for line in response.strip().split('\n'):
            line = line.strip()
            line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line)
            
            if not line or len(line) < 15:
                continue
            
            if not (line.endswith('?') or line.endswith('？')):
                continue
            
            # Semantic fingerprint check
            fingerprint = self._get_fingerprint(line)
            if fingerprint in self.generated_fingerprints:
                continue
            
            self.generated_fingerprints.add(fingerprint)
            questions.append(GeneratedDesignQuestion(
                question_id=f"DES-LLM-{uuid.uuid4().hex[:8]}",
                question_text=line,
                source="llm",
                role=role.value,
                intent=intent,
                aspect=random.choice(scenario.design_aspects) if scenario.design_aspects else DesignAspect.ARCHITECTURE,
                language=language,
            ))
        
        return questions
    
    def _generate_fallback(
        self,
        scenario: DesignScenario,
        role: DesignerRole,
        count: int,
        language: str,
        used_intents: Set[QuestionIntent],
        used_aspects: Set[DesignAspect]
    ) -> List[GeneratedDesignQuestion]:
        """Generate questions from templates."""
        questions = []
        
        # Prioritize unused intents and aspects
        all_intents = list(QuestionIntent)
        all_aspects = scenario.design_aspects or list(DesignAspect)
        
        random.shuffle(all_intents)
        random.shuffle(all_aspects)
        
        # Sort by usage (unused first)
        all_intents.sort(key=lambda i: i in used_intents)
        all_aspects.sort(key=lambda a: a in used_aspects)
        
        for intent in all_intents:
            if len(questions) >= count:
                break
            
            for aspect in all_aspects:
                if len(questions) >= count:
                    break
                
                templates = DesignQuestionTemplates.get_templates(language, intent, aspect)
                if not templates:
                    continue
                
                for template in templates:
                    if len(questions) >= count:
                        break
                    
                    # Fill in scenario name
                    question_text = template.format(scenario=scenario.scenario_name)
                    
                    # Fingerprint check
                    fingerprint = self._get_fingerprint(question_text)
                    if fingerprint in self.generated_fingerprints:
                        continue
                    
                    self.generated_fingerprints.add(fingerprint)
                    questions.append(GeneratedDesignQuestion(
                        question_id=f"DES-FB-{uuid.uuid4().hex[:8]}",
                        question_text=question_text,
                        source="fallback",
                        role=role.value,
                        intent=intent,
                        aspect=aspect,
                        language=language,
                    ))
                    
                    used_intents.add(intent)
                    used_aspects.add(aspect)
                    break  # One question per intent-aspect combo
        
        return questions
    
    def _get_fingerprint(self, question: str) -> str:
        """Generate semantic fingerprint for a question."""
        # Normalize
        q = question.lower().strip()
        q = re.sub(r'[^\w\s]', '', q)
        q = re.sub(r'\s+', ' ', q)
        return hashlib.md5(q.encode()).hexdigest()[:16]


class DesignSolutionGenerator:
    """
    Generates design solutions and recommendations.
    """
    
    # Pre-defined solution patterns for each scenario
    SOLUTION_PATTERNS = {
        "DBR-01-01": {
            "en": {
                "approach": "Implement a validate-before-write pattern with sequential uniqueness checks",
                "rationale": "Sequential validation ensures deterministic behavior and clear error attribution. Writing only after all validations pass guarantees data consistency.",
                "components": [
                    "Uniqueness validator service",
                    "Error response factory",
                    "Repository layer with query methods",
                    "Conditional validation for updates",
                ],
                "patterns_used": [
                    "Guard Clause Pattern",
                    "Repository Pattern",
                    "Early Return Pattern",
                ],
                "security_considerations": [
                    "Avoid timing differences between valid/invalid responses",
                    "Use database-level unique constraints as backup",
                    "Log validation failures for monitoring",
                ],
                "trade_offs": [
                    "Sequential checks may be slower than parallel",
                    "Application-level validation adds latency",
                    "Requires careful transaction handling",
                ],
            },
            "zh": {
                "approach": "实现顺序唯一性检查的先验证后写入模式",
                "rationale": "顺序验证确保确定性行为和清晰的错误归因。仅在所有验证通过后写入保证数据一致性。",
                "components": [
                    "唯一性验证服务",
                    "错误响应工厂",
                    "带查询方法的仓储层",
                    "更新时的条件验证",
                ],
                "patterns_used": [
                    "守卫子句模式",
                    "仓储模式",
                    "提前返回模式",
                ],
                "security_considerations": [
                    "避免有效/无效响应之间的时序差异",
                    "使用数据库级唯一约束作为备份",
                    "记录验证失败以便监控",
                ],
                "trade_offs": [
                    "顺序检查可能比并行慢",
                    "应用层验证增加延迟",
                    "需要谨慎的事务处理",
                ],
            },
        },
        "DBR-01-02": {
            "en": {
                "approach": "Implement atomic account creation with secure password hashing pipeline",
                "rationale": "Atomicity prevents partial user records. Password hashing must occur before any persistence to ensure credentials are never stored in plaintext.",
                "components": [
                    "Password hashing service (bcrypt/argon2)",
                    "Atomic user creation method",
                    "Transaction management",
                    "Secure ID generation",
                ],
                "patterns_used": [
                    "Unit of Work Pattern",
                    "Factory Pattern for user creation",
                    "Service Layer Pattern",
                ],
                "security_considerations": [
                    "Use adaptive hashing algorithms",
                    "Never log password values",
                    "Ensure transaction rollback on failure",
                ],
                "trade_offs": [
                    "Hashing adds CPU overhead",
                    "Atomic operations may hold locks longer",
                    "Higher memory usage for security libraries",
                ],
            },
            "zh": {
                "approach": "实现带安全密码哈希流水线的原子账户创建",
                "rationale": "原子性防止部分用户记录。密码哈希必须在任何持久化之前进行，以确保凭据永不以明文存储。",
                "components": [
                    "密码哈希服务（bcrypt/argon2）",
                    "原子用户创建方法",
                    "事务管理",
                    "安全ID生成",
                ],
                "patterns_used": [
                    "工作单元模式",
                    "用户创建工厂模式",
                    "服务层模式",
                ],
                "security_considerations": [
                    "使用自适应哈希算法",
                    "永不记录密码值",
                    "确保失败时事务回滚",
                ],
                "trade_offs": [
                    "哈希增加CPU开销",
                    "原子操作可能持有锁更长时间",
                    "安全库的更高内存使用",
                ],
            },
        },
        "DBR-01-03": {
            "en": {
                "approach": "Implement unified error handling with constant-time comparison for authentication",
                "rationale": "Generic error messages prevent user enumeration attacks. Unified handling ensures attackers cannot distinguish between invalid email and wrong password.",
                "components": [
                    "Authentication service",
                    "Generic error response handler",
                    "Constant-time password comparison",
                    "Authentication audit logger",
                ],
                "patterns_used": [
                    "Strategy Pattern for auth methods",
                    "Null Object Pattern for non-existent users",
                    "Decorator Pattern for logging",
                ],
                "security_considerations": [
                    "Same error message for all failures",
                    "Consistent response timing",
                    "Rate limiting on authentication endpoints",
                ],
                "trade_offs": [
                    "Less helpful error messages for users",
                    "Debugging authentication issues is harder",
                    "Additional complexity for consistent timing",
                ],
            },
            "zh": {
                "approach": "实现带常量时间比较的统一错误处理认证",
                "rationale": "通用错误消息防止用户枚举攻击。统一处理确保攻击者无法区分无效邮箱和错误密码。",
                "components": [
                    "认证服务",
                    "通用错误响应处理器",
                    "常量时间密码比较",
                    "认证审计日志记录器",
                ],
                "patterns_used": [
                    "认证方法策略模式",
                    "不存在用户的空对象模式",
                    "日志装饰器模式",
                ],
                "security_considerations": [
                    "所有失败使用相同错误消息",
                    "一致的响应时间",
                    "认证端点的速率限制",
                ],
                "trade_offs": [
                    "对用户来说错误消息不够有帮助",
                    "调试认证问题更困难",
                    "一致时序增加额外复杂性",
                ],
            },
        },
        "DBR-01-04": {
            "en": {
                "approach": "Implement stateless JWT-based session management with token refresh capability",
                "rationale": "JWTs enable stateless verification while maintaining security. Refresh mechanism balances security (short-lived tokens) with user experience (no frequent re-login).",
                "components": [
                    "JWT token generator",
                    "Token verification middleware",
                    "Refresh token service",
                    "Claims management",
                ],
                "patterns_used": [
                    "Token-based Authentication Pattern",
                    "Middleware Pattern",
                    "Factory Pattern for tokens",
                ],
                "security_considerations": [
                    "Short expiration for access tokens",
                    "Secure secret key management",
                    "Token blacklist for logout",
                ],
                "trade_offs": [
                    "Cannot invalidate tokens without blacklist",
                    "Larger request payload than session cookies",
                    "Clock synchronization requirements",
                ],
            },
            "zh": {
                "approach": "实现带令牌刷新能力的无状态JWT会话管理",
                "rationale": "JWT实现无状态验证同时保持安全性。刷新机制平衡安全性（短期令牌）和用户体验（无需频繁重新登录）。",
                "components": [
                    "JWT令牌生成器",
                    "令牌验证中间件",
                    "刷新令牌服务",
                    "声明管理",
                ],
                "patterns_used": [
                    "基于令牌的认证模式",
                    "中间件模式",
                    "令牌工厂模式",
                ],
                "security_considerations": [
                    "访问令牌短过期时间",
                    "安全密钥管理",
                    "登出时的令牌黑名单",
                ],
                "trade_offs": [
                    "没有黑名单无法使令牌失效",
                    "比会话cookie更大的请求负载",
                    "时钟同步要求",
                ],
            },
        },
    }
    
    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
    
    def generate(
        self,
        scenario: DesignScenario,
        question: GeneratedDesignQuestion,
        language: str = "en"
    ) -> Tuple[DesignSolution, List[str]]:
        """
        Generate design solution and reasoning trace.
        
        Args:
            scenario: Design scenario
            question: The question being answered
            language: Language code
            
        Returns:
            Tuple of (DesignSolution, reasoning_trace)
        """
        # Try LLM enhancement
        if self.llm.is_available():
            result = self._generate_with_llm(scenario, question, language)
            if result:
                return result
        
        # Fallback to pre-defined patterns
        return self._generate_fallback(scenario, question, language)
    
    def _generate_with_llm(
        self,
        scenario: DesignScenario,
        question: GeneratedDesignQuestion,
        language: str
    ) -> Optional[Tuple[DesignSolution, List[str]]]:
        """Generate with LLM enhancement."""
        # Get base pattern
        base = self._get_base_pattern(scenario.scenario_id, language)
        
        system = self._get_system_prompt(language)
        prompt = self._get_generation_prompt(scenario, question, base, language)
        
        response = self.llm.generate_with_task(prompt, "design")
        
        if response:
            # Enhance base pattern with LLM response
            reasoning = self._extract_reasoning(response, language)
            return DesignSolution(**base), reasoning
        
        return None
    
    def _get_system_prompt(self, language: str) -> str:
        if language == "zh":
            return """你是一位资深系统架构师，正在提供认证系统设计方案。
基于提供的设计模式，给出专业的推理过程。"""
        else:
            return """You are a senior system architect providing authentication system design solutions.
Based on the provided design patterns, give professional reasoning."""
    
    def _get_generation_prompt(
        self,
        scenario: DesignScenario,
        question: GeneratedDesignQuestion,
        base: Dict,
        language: str
    ) -> str:
        if language == "zh":
            return f"""场景：{scenario.scenario_name}
问题：{question.question_text}
设计方案：{base.get('approach', '')}

请提供4-5步推理过程，解释为什么这个设计方案是合适的："""
        else:
            return f"""Scenario: {scenario.scenario_name}
Question: {question.question_text}
Design Approach: {base.get('approach', '')}

Provide 4-5 reasoning steps explaining why this design is appropriate:"""
    
    def _extract_reasoning(self, response: str, language: str) -> List[str]:
        """Extract reasoning steps from LLM response."""
        steps = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line.startswith('[') or re.match(r'^\d+\.', line) or line.startswith('-')):
                steps.append(line)
        
        if not steps:
            steps = [response[:200]]
        
        return steps[:6]
    
    def _generate_fallback(
        self,
        scenario: DesignScenario,
        question: GeneratedDesignQuestion,
        language: str
    ) -> Tuple[DesignSolution, List[str]]:
        """Generate from pre-defined patterns."""
        base = self._get_base_pattern(scenario.scenario_id, language)
        
        solution = DesignSolution(
            approach=base.get("approach", ""),
            rationale=base.get("rationale", ""),
            components=base.get("components", []),
            patterns_used=base.get("patterns_used", []),
            security_considerations=base.get("security_considerations", []),
            trade_offs=base.get("trade_offs", []),
        )
        
        # Generate deterministic reasoning
        if language == "zh":
            reasoning = [
                f"[需求分析] 场景要求：{'; '.join(scenario.requirements[:2])}",
                f"[约束识别] 设计约束：{'; '.join(scenario.constraints[:2])}",
                f"[方案选择] 推荐方案：{solution.approach}",
                f"[模式应用] 使用模式：{', '.join(solution.patterns_used[:2])}",
                f"[安全考量] 安全措施：{'; '.join(solution.security_considerations[:2])}",
            ]
        else:
            reasoning = [
                f"[REQUIREMENT] Scenario requires: {'; '.join(scenario.requirements[:2])}",
                f"[CONSTRAINT] Design constraints: {'; '.join(scenario.constraints[:2])}",
                f"[APPROACH] Recommended: {solution.approach}",
                f"[PATTERNS] Applied: {', '.join(solution.patterns_used[:2])}",
                f"[SECURITY] Considerations: {'; '.join(solution.security_considerations[:2])}",
            ]
        
        return solution, reasoning
    
    def _get_base_pattern(self, scenario_id: str, language: str) -> Dict:
        """Get base solution pattern for scenario."""
        patterns = self.SOLUTION_PATTERNS.get(scenario_id, {})
        return patterns.get(language, patterns.get("en", {}))
