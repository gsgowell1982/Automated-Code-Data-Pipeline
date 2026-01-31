"""
Enhanced LLM Enhancement Layer v2 for Q&A Generation Engine v9.

Key improvements:
1. Angle-based question templates - organized by question perspective
2. Focus-aware generation - ensures topic coverage
3. Integration with DiversityManagerV2 semantic tracking
4. More diverse and realistic question patterns
"""

import re
import uuid
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models import (
    BusinessContext, CodeContext, CodeFacts, GeneratedQuestion,
    UserRole, QuestionType
)
from utils.llm_client import OllamaClient
from utils.diversity_v2 import QuestionAngle, QuestionFocus, SemanticIntent
from layers.execution_flow import ExecutionFlowAnalyzer
from layers.consistency_validator import ConsistencyValidator


class AngleBasedTemplates:
    """
    Question templates organized by angle and focus.
    
    Each angle represents a different perspective/type of question.
    Each focus represents a specific topic area.
    """
    
    TEMPLATES = {
        # English templates
        "en": {
            # WHY_ERROR: 为什么出错/报错
            QuestionAngle.WHY_ERROR: {
                QuestionFocus.USERNAME: [
                    "Why does the system reject my new username choice?",
                    "What causes the 'username unavailable' error during registration?",
                    "Why can't I use this username even though I've never seen it before?",
                ],
                QuestionFocus.EMAIL: [
                    "Why does registration fail when I use my work email?",
                    "What triggers the email validation error?",
                    "Why is my email address being marked as already in use?",
                ],
                QuestionFocus.PASSWORD: [
                    "Why does the login fail even when I'm certain my password is correct?",
                    "What causes password validation to reject my credentials?",
                ],
                QuestionFocus.LOGIN: [
                    "Why do I get a generic error instead of knowing what's wrong with my login?",
                    "What causes the authentication to fail without specific details?",
                ],
                QuestionFocus.PROFILE_UPDATE: [
                    "Why does profile update fail when I change my email?",
                    "What triggers the rejection when I try to modify my account details?",
                ],
            },
            
            # WHAT_HAPPENS: 会发生什么
            QuestionAngle.WHAT_HAPPENS: {
                QuestionFocus.USERNAME: [
                    "What happens to my registration if the username check fails?",
                    "What's the outcome when someone else registers the same username simultaneously?",
                ],
                QuestionFocus.EMAIL: [
                    "What happens if I try to register with an email that's already in the database?",
                    "What's the result when email validation fails during signup?",
                ],
                QuestionFocus.VALIDATION: [
                    "What happens to the data if validation fails halfway through the process?",
                    "What's the state of my account if the first check passes but the second fails?",
                ],
                QuestionFocus.DATABASE: [
                    "What happens in the database when registration is interrupted?",
                    "Is any data persisted if the process terminates early?",
                ],
                QuestionFocus.PROFILE_UPDATE: [
                    "What happens to my old data if the profile update fails?",
                    "What's the outcome if I disconnect during a profile update?",
                ],
            },
            
            # HOW_IT_WORKS: 如何工作
            QuestionAngle.HOW_IT_WORKS: {
                QuestionFocus.VALIDATION: [
                    "How does the system verify that my username is unique?",
                    "What's the order of validation checks during registration?",
                    "How does the validation pipeline work step by step?",
                ],
                QuestionFocus.LOGIN: [
                    "How does the authentication process verify my identity?",
                    "What's the flow when I submit my login credentials?",
                ],
                QuestionFocus.TOKEN: [
                    "How does the system generate and manage my session token?",
                    "What's the token lifecycle after I log in?",
                ],
                QuestionFocus.ERROR_MESSAGE: [
                    "How does the system decide what error message to show me?",
                    "What determines whether I see a generic or specific error?",
                ],
            },
            
            # IS_IT_SECURE: 是否安全
            QuestionAngle.IS_IT_SECURE: {
                QuestionFocus.LOGIN: [
                    "Can an attacker determine if an account exists by trying to log in?",
                    "Does the login error message reveal whether the email is registered?",
                ],
                QuestionFocus.ERROR_MESSAGE: [
                    "Do the error messages leak information that could help attackers?",
                    "Is it safe that validation errors show which field failed?",
                ],
                QuestionFocus.SECURITY: [
                    "Is there a timing attack vulnerability in the validation process?",
                    "Could someone enumerate valid usernames through the registration endpoint?",
                ],
                QuestionFocus.PASSWORD: [
                    "How is password security handled during the registration process?",
                    "Are credentials protected throughout the authentication flow?",
                ],
            },
            
            # WHAT_IF_REMOVE: 如果去掉会怎样
            QuestionAngle.WHAT_IF_REMOVE: {
                QuestionFocus.VALIDATION: [
                    "What would happen if we removed the username uniqueness check?",
                    "What's the risk if the email validation step was bypassed?",
                ],
                QuestionFocus.ERROR_MESSAGE: [
                    "What if the system showed specific errors instead of generic ones during login?",
                ],
                QuestionFocus.SECURITY: [
                    "What vulnerabilities would exist without the pre-save validation?",
                    "What could go wrong if validation happened after saving?",
                ],
            },
            
            # CAN_I_DO: 能不能/是否可以
            QuestionAngle.CAN_I_DO: {
                QuestionFocus.USERNAME: [
                    "Can I register a username that was previously used but deleted?",
                    "Is it possible to claim a username that's temporarily unavailable?",
                ],
                QuestionFocus.EMAIL: [
                    "Can I use the same email for multiple accounts?",
                    "Is it possible to change my email to one that another user just abandoned?",
                ],
                QuestionFocus.PROFILE_UPDATE: [
                    "Can I update my username to something another user currently has?",
                    "Is it allowed to change my email during an active session?",
                ],
            },
            
            # EDGE_CASE: 边界情况
            QuestionAngle.EDGE_CASE: {
                QuestionFocus.USERNAME: [
                    "What if two users try to register the exact same username at the same millisecond?",
                    "How does the system handle usernames with special Unicode characters?",
                ],
                QuestionFocus.EMAIL: [
                    "What happens with email addresses that have plus signs or dots?",
                    "How does case sensitivity affect email uniqueness checking?",
                ],
                QuestionFocus.VALIDATION: [
                    "What if the database connection drops between validations?",
                    "How does the system behave under extreme load during validation?",
                ],
            },
            
            # DEEP_ANALYSIS: 深度分析
            QuestionAngle.DEEP_ANALYSIS: {
                QuestionFocus.SYSTEM_BEHAVIOR: [
                    "What design pattern ensures data consistency in the registration flow?",
                    "How does the validate-then-write approach guarantee atomicity?",
                ],
                QuestionFocus.SECURITY: [
                    "What's the threat model addressed by generic login error messages?",
                    "How does the current design prevent user enumeration attacks?",
                ],
                QuestionFocus.VALIDATION: [
                    "What are the trade-offs of validating before vs after database writes?",
                    "How does sequential validation affect system reliability?",
                ],
            },
        },
        
        # Chinese templates
        "zh": {
            QuestionAngle.WHY_ERROR: {
                QuestionFocus.USERNAME: [
                    "为什么系统说这个用户名已经被使用了？",
                    "注册时用户名检查失败的原因是什么？",
                    "我选的用户名明明很独特，为什么还是提示已存在？",
                ],
                QuestionFocus.EMAIL: [
                    "为什么我的邮箱注册不了，说已经被占用？",
                    "邮箱验证失败的具体原因是什么？",
                    "使用公司邮箱注册为什么会失败？",
                ],
                QuestionFocus.PASSWORD: [
                    "为什么我确定密码正确却还是登录失败？",
                    "登录密码验证出错的原因有哪些？",
                ],
                QuestionFocus.LOGIN: [
                    "登录失败为什么不告诉我具体是哪里出错了？",
                    "为什么登录错误信息这么模糊不清？",
                ],
                QuestionFocus.PROFILE_UPDATE: [
                    "修改资料时为什么会提示邮箱冲突？",
                    "更新个人信息失败的原因是什么？",
                ],
            },
            
            QuestionAngle.WHAT_HAPPENS: {
                QuestionFocus.USERNAME: [
                    "如果用户名检查失败，我的注册请求会怎样处理？",
                    "两个人同时注册同一个用户名会发生什么？",
                ],
                QuestionFocus.EMAIL: [
                    "用已存在的邮箱注册，系统会如何响应？",
                    "邮箱验证不通过时，整个注册流程会怎样？",
                ],
                QuestionFocus.VALIDATION: [
                    "如果第一项验证通过但第二项失败，数据会保存吗？",
                    "验证过程中断，之前的操作会回滚吗？",
                ],
                QuestionFocus.DATABASE: [
                    "注册中断时数据库里会留下什么记录？",
                    "验证失败会在系统里留下任何痕迹吗？",
                ],
                QuestionFocus.PROFILE_UPDATE: [
                    "资料更新失败时，原来的数据会受影响吗？",
                    "修改过程中网络断开，账户状态会怎样？",
                ],
            },
            
            QuestionAngle.HOW_IT_WORKS: {
                QuestionFocus.VALIDATION: [
                    "系统是如何确保用户名唯一性的？",
                    "注册时的验证是按什么顺序进行的？",
                    "整个验证流程的工作机制是怎样的？",
                ],
                QuestionFocus.LOGIN: [
                    "登录验证的具体流程是什么？",
                    "系统如何判断登录凭据是否正确？",
                ],
                QuestionFocus.TOKEN: [
                    "登录成功后令牌是怎么生成的？",
                    "会话令牌的管理机制是什么？",
                ],
                QuestionFocus.ERROR_MESSAGE: [
                    "系统是根据什么决定显示哪种错误信息的？",
                    "什么情况下会显示详细错误，什么情况下显示通用错误？",
                ],
            },
            
            QuestionAngle.IS_IT_SECURE: {
                QuestionFocus.LOGIN: [
                    "攻击者能否通过登录接口判断某个邮箱是否已注册？",
                    "登录错误提示会不会泄露账户是否存在的信息？",
                ],
                QuestionFocus.ERROR_MESSAGE: [
                    "错误消息是否会泄露对攻击者有用的信息？",
                    "显示具体的验证错误是否存在安全风险？",
                ],
                QuestionFocus.SECURITY: [
                    "验证过程中是否存在时序攻击的风险？",
                    "有人能通过注册接口枚举已存在的用户名吗？",
                ],
            },
            
            QuestionAngle.WHAT_IF_REMOVE: {
                QuestionFocus.VALIDATION: [
                    "如果移除用户名唯一性检查会有什么后果？",
                    "去掉邮箱验证步骤会带来什么风险？",
                ],
                QuestionFocus.ERROR_MESSAGE: [
                    "如果登录失败时显示详细错误会怎样？",
                ],
                QuestionFocus.SECURITY: [
                    "没有预保存验证会产生什么漏洞？",
                    "如果验证放在保存之后会有什么问题？",
                ],
            },
            
            QuestionAngle.CAN_I_DO: {
                QuestionFocus.USERNAME: [
                    "我能注册一个以前被删除的用户名吗？",
                    "临时不可用的用户名什么时候能再次使用？",
                ],
                QuestionFocus.EMAIL: [
                    "同一个邮箱可以注册多个账户吗？",
                    "能把邮箱改成别人刚放弃的那个吗？",
                ],
                QuestionFocus.PROFILE_UPDATE: [
                    "我能把用户名改成别人正在用的吗？",
                    "登录状态下可以修改绑定的邮箱吗？",
                ],
            },
            
            QuestionAngle.EDGE_CASE: {
                QuestionFocus.USERNAME: [
                    "如果两个人在同一毫秒注册相同的用户名会怎样？",
                    "用户名包含特殊字符时系统如何处理？",
                ],
                QuestionFocus.EMAIL: [
                    "邮箱地址有加号或点号时如何判断唯一性？",
                    "邮箱大小写对唯一性检查有影响吗？",
                ],
                QuestionFocus.VALIDATION: [
                    "两次验证之间数据库连接断开会怎样？",
                    "高并发情况下验证机制能正常工作吗？",
                ],
            },
            
            QuestionAngle.DEEP_ANALYSIS: {
                QuestionFocus.SYSTEM_BEHAVIOR: [
                    "注册流程用了什么设计模式来保证数据一致性？",
                    "先验证后写入的方式如何保证原子性？",
                ],
                QuestionFocus.SECURITY: [
                    "通用登录错误消息针对的是什么威胁模型？",
                    "当前设计如何防止用户枚举攻击？",
                ],
                QuestionFocus.VALIDATION: [
                    "写入前验证和写入后验证各有什么优缺点？",
                    "顺序验证对系统可靠性有什么影响？",
                ],
            },
        },
    }
    
    @classmethod
    def get_templates(
        cls, 
        language: str, 
        angle: QuestionAngle = None,
        focus: QuestionFocus = None
    ) -> List[str]:
        """
        Get templates for a specific angle and focus.
        
        Args:
            language: Language code
            angle: Optional specific angle
            focus: Optional specific focus
            
        Returns:
            List of template strings
        """
        lang_templates = cls.TEMPLATES.get(language, cls.TEMPLATES["en"])
        
        if angle and focus:
            return lang_templates.get(angle, {}).get(focus, [])
        
        if angle:
            all_for_angle = []
            for focus_templates in lang_templates.get(angle, {}).values():
                all_for_angle.extend(focus_templates)
            return all_for_angle
        
        if focus:
            all_for_focus = []
            for angle_templates in lang_templates.values():
                all_for_focus.extend(angle_templates.get(focus, []))
            return all_for_focus
        
        # Return all templates
        all_templates = []
        for angle_dict in lang_templates.values():
            for focus_list in angle_dict.values():
                all_templates.extend(focus_list)
        return all_templates
    
    @classmethod
    def get_diverse_templates(
        cls,
        language: str,
        used_angles: set,
        used_focuses: set,
        count: int = 5
    ) -> List[Tuple[str, QuestionAngle, QuestionFocus]]:
        """
        Get templates that maximize diversity.
        
        Prioritizes angles and focuses that haven't been used.
        
        Returns:
            List of (template, angle, focus) tuples
        """
        lang_templates = cls.TEMPLATES.get(language, cls.TEMPLATES["en"])
        candidates = []
        
        # Score and collect all templates
        for angle, focus_dict in lang_templates.items():
            for focus, templates in focus_dict.items():
                # Calculate diversity score
                angle_score = 2 if angle not in used_angles else 0
                focus_score = 1 if focus not in used_focuses else 0
                total_score = angle_score + focus_score
                
                for template in templates:
                    candidates.append((template, angle, focus, total_score))
        
        # Sort by score (descending) and add some randomness
        random.shuffle(candidates)
        candidates.sort(key=lambda x: x[3], reverse=True)
        
        # Return top candidates
        return [(c[0], c[1], c[2]) for c in candidates[:count * 2]]


class QuestionGeneratorV2:
    """
    Enhanced Question Generator with angle-based diversity.
    
    Uses semantic fingerprinting to avoid generating paraphrased questions.
    """
    
    CODE_NAME_PATTERNS = [
        r'\bcheck_\w+\b', r'\busers_repo\b', r'\buser_create\b',
        r'\buser_update\b', r'\bHTTP_\d+\b', r'\bEntityDoesNotExist\b',
        r'\bwrong_login_error\b', r'\b[a-z]+_[a-z]+_[a-z]+\b',
    ]
    
    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        self.generated_fingerprints: set = set()
    
    def generate(
        self,
        context: BusinessContext,
        code_facts: Optional[CodeFacts],
        role: UserRole,
        count: int,
        language: str,
        diversity_manager = None,
        evidence_id: str = ""
    ) -> List[GeneratedQuestion]:
        """
        Generate diverse questions using angle-based templates.
        """
        # Determine which angles/focuses have been used
        used_angles = set()
        used_focuses = set()
        
        if diversity_manager and evidence_id:
            coverage = diversity_manager.get_coverage_for_evidence(evidence_id)
            # Parse intents to angles
            for intent in coverage.get("intents", []):
                if "error" in intent:
                    used_angles.add(QuestionAngle.WHY_ERROR)
                elif "happen" in intent:
                    used_angles.add(QuestionAngle.WHAT_HAPPENS)
                elif "work" in intent:
                    used_angles.add(QuestionAngle.HOW_IT_WORKS)
                elif "secure" in intent:
                    used_angles.add(QuestionAngle.IS_IT_SECURE)
            
            # Parse focuses
            for focus_str in coverage.get("focuses", []):
                try:
                    used_focuses.add(QuestionFocus(focus_str))
                except ValueError:
                    pass
        
        # Try LLM first
        if self.llm.is_available():
            questions = self._generate_with_llm(context, role, count, language, used_angles)
            if len(questions) >= count:
                return questions[:count]
        
        # Fallback to angle-based templates
        return self._generate_fallback_v2(
            context, code_facts, role, count, language, 
            used_angles, used_focuses, evidence_id
        )
    
    def _generate_with_llm(
        self,
        context: BusinessContext,
        role: UserRole,
        count: int,
        language: str,
        used_angles: set
    ) -> List[GeneratedQuestion]:
        """Generate questions using LLM with angle guidance."""
        # Get recommended angle
        all_angles = list(QuestionAngle)
        available_angles = [a for a in all_angles if a not in used_angles]
        if not available_angles:
            available_angles = all_angles
        
        angle = random.choice(available_angles)
        
        system = self._get_system_prompt_v2(language, angle)
        prompt = self._get_generation_prompt_v2(context, role, count, language, angle)
        
        response = self.llm.generate_with_task(prompt, "question", system=system)
        
        if response:
            return self._parse_questions(response, role, language, angle)
        return []
    
    def _get_system_prompt_v2(self, language: str, angle: QuestionAngle) -> str:
        """Get system prompt with angle-specific guidance."""
        angle_guidance = {
            QuestionAngle.WHY_ERROR: ("why errors occur", "为什么会出错"),
            QuestionAngle.WHAT_HAPPENS: ("what happens in scenarios", "各种情况下会发生什么"),
            QuestionAngle.HOW_IT_WORKS: ("how the system works", "系统如何运作"),
            QuestionAngle.IS_IT_SECURE: ("security concerns", "安全问题"),
            QuestionAngle.WHAT_IF_REMOVE: ("consequences of removing features", "移除功能的后果"),
            QuestionAngle.EDGE_CASE: ("edge cases and corner scenarios", "边界情况"),
        }
        
        guidance = angle_guidance.get(angle, ("general inquiries", "一般问题"))
        
        if language == "zh":
            return f"""你是一个模拟真实用户提问的助手。当前任务是生成关于{guidance[1]}的问题。

规则：
1. 问题必须从用户角度出发，不能包含任何代码函数名
2. 问题要自然、具体、有实际意义
3. 每个问题必须表达不同的关注点
4. 避免生成意思相近的问题"""
        else:
            return f"""You simulate real users asking about {guidance[0]}.

Rules:
1. Questions must be from user perspective - NO code function names
2. Questions must be natural, specific, and meaningful
3. Each question must address a DIFFERENT concern
4. Avoid generating questions with similar meanings"""
    
    def _get_generation_prompt_v2(
        self,
        context: BusinessContext,
        role: UserRole,
        count: int,
        language: str,
        angle: QuestionAngle
    ) -> str:
        """Get prompt with angle-specific requirements."""
        from layers.user_perspective import UserPerspectiveLayer
        role_ctx = UserPerspectiveLayer.ROLE_CONTEXTS.get(role, {}).get(language, "")
        
        if language == "zh":
            return f"""场景：{context.scenario_name}
业务流程：{context.business_flow}
你的角色：{role_ctx}

请生成{count}个关于此场景的问题，每个问题必须：
- 关注不同的具体方面（如：错误原因、数据状态、安全性等）
- 使用不同的表达方式
- 有独立的实际意义

每行一个问题："""
        else:
            return f"""Scenario: {context.scenario_name}
Business Flow: {context.business_flow}
Your Role: {role_ctx}

Generate {count} questions about this scenario. Each question must:
- Focus on a DIFFERENT specific aspect (e.g., error cause, data state, security)
- Use different phrasing
- Have independent practical meaning

One question per line:"""
    
    def _parse_questions(
        self,
        response: str,
        role: UserRole,
        language: str,
        angle: QuestionAngle
    ) -> List[GeneratedQuestion]:
        """Parse LLM response with fingerprint deduplication."""
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
            
            # Semantic fingerprint check
            fingerprint = SemanticIntent.get_semantic_fingerprint(line, language)
            if fingerprint in self.generated_fingerprints:
                continue
            
            self.generated_fingerprints.add(fingerprint)
            questions.append(GeneratedQuestion(
                question_id=f"LLM-{uuid.uuid4().hex[:8]}",
                question_text=line,
                source="llm",
                role=role.value,
                language=language,
                question_type=self._classify_question(line),
            ))
        
        return questions
    
    def _generate_fallback_v2(
        self,
        context: BusinessContext,
        code_facts: Optional[CodeFacts],
        role: UserRole,
        count: int,
        language: str,
        used_angles: set,
        used_focuses: set,
        evidence_id: str
    ) -> List[GeneratedQuestion]:
        """Generate diverse fallback questions using angle-based templates."""
        # Get templates that maximize diversity
        candidates = AngleBasedTemplates.get_diverse_templates(
            language, used_angles, used_focuses, count=count * 3
        )
        
        questions = []
        
        for template, angle, focus in candidates:
            if len(questions) >= count:
                break
            
            # Check semantic fingerprint
            fingerprint = SemanticIntent.get_semantic_fingerprint(template, language)
            if fingerprint in self.generated_fingerprints:
                continue
            
            self.generated_fingerprints.add(fingerprint)
            questions.append(GeneratedQuestion(
                question_id=f"FB-{angle.value[:3]}-{uuid.uuid4().hex[:6]}",
                question_text=template,
                source="fallback",
                role=role.value,
                language=language,
                question_type=self._map_angle_to_type(angle),
            ))
            
            # Track used angles/focuses for this batch
            used_angles.add(angle)
            used_focuses.add(focus)
        
        return questions
    
    def _map_angle_to_type(self, angle: QuestionAngle) -> QuestionType:
        """Map angle to question type."""
        mapping = {
            QuestionAngle.WHY_ERROR: QuestionType.TROUBLESHOOTING,
            QuestionAngle.WHAT_HAPPENS: QuestionType.WHAT_IF,
            QuestionAngle.HOW_IT_WORKS: QuestionType.UNDERSTANDING,
            QuestionAngle.IS_IT_SECURE: QuestionType.SECURITY,
            QuestionAngle.WHAT_IF_REMOVE: QuestionType.WHAT_IF,
            QuestionAngle.CAN_I_DO: QuestionType.VALIDATION,
            QuestionAngle.EDGE_CASE: QuestionType.EDGE_CASE,
            QuestionAngle.DEEP_ANALYSIS: QuestionType.DEEP_ANALYSIS,
        }
        return mapping.get(angle, QuestionType.UNDERSTANDING)
    
    def _contains_code_names(self, text: str) -> bool:
        for pattern in self.CODE_NAME_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _classify_question(self, question: str) -> QuestionType:
        q_lower = question.lower()
        type_indicators = {
            QuestionType.TROUBLESHOOTING: ["problem", "issue", "error", "wrong", "问题", "错误"],
            QuestionType.UNDERSTANDING: ["how does", "how do", "what is", "如何", "什么是"],
            QuestionType.EDGE_CASE: ["what happens if", "what if", "如果", "会发生什么"],
            QuestionType.SECURITY: ["attack", "secure", "vulnerability", "攻击", "安全"],
            QuestionType.WHAT_IF: ["what would happen", "remove", "如果去掉"],
        }
        for qtype, indicators in type_indicators.items():
            if any(ind in q_lower for ind in indicators):
                return qtype
        return QuestionType.UNDERSTANDING


# Import unchanged components from v1
from layers.llm_enhancement import ReasoningGenerator, AnswerGenerator


class LLMEnhancementLayerV2:
    """
    Enhanced LLM Enhancement Layer with angle-based question generation.
    """
    
    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        self.flow_analyzer = ExecutionFlowAnalyzer()
        self.consistency_validator = ConsistencyValidator()
        
        # Use V2 question generator
        self.question_generator = QuestionGeneratorV2(llm_client)
        
        # Reuse V1 reasoning and answer generators
        self.reasoning_generator = ReasoningGenerator(
            llm_client, self.flow_analyzer, self.consistency_validator
        )
        self.answer_generator = AnswerGenerator(
            llm_client, self.flow_analyzer, self.consistency_validator
        )
    
    def is_llm_available(self) -> bool:
        return self.llm.is_available()
