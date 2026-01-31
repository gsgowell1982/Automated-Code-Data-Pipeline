"""
LLM Enhancement Layer for Q&A Generation Engine v9.

Contains question, reasoning, and answer generators that use LLM.
Includes code-grounded prompts from v8.
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
from layers.execution_flow import ExecutionFlowAnalyzer
from layers.consistency_validator import ConsistencyValidator


class QuestionGenerator:
    """
    Generates questions using LLM with user perspective.
    
    Questions are generated without code names, simulating
    real users who don't know internal implementation details.
    """
    
    # Patterns to detect code names (questions with these are rejected)
    CODE_NAME_PATTERNS = [
        r'\bcheck_\w+\b', r'\busers_repo\b', r'\buser_create\b',
        r'\buser_update\b', r'\bHTTP_\d+\b', r'\bEntityDoesNotExist\b',
        r'\bwrong_login_error\b', r'\b[a-z]+_[a-z]+_[a-z]+\b',
    ]
    
    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        self.generated_questions: set = set()
    
    def generate(
        self,
        context: BusinessContext,
        code_facts: Optional[CodeFacts],
        role: UserRole,
        count: int,
        language: str
    ) -> List[GeneratedQuestion]:
        """
        Generate questions for a given context and role.
        
        Args:
            context: Business context
            code_facts: Optional code facts for constraints
            role: User role to simulate
            count: Number of questions to generate
            language: Language code
            
        Returns:
            List of generated questions
        """
        # Try LLM first
        if self.llm.is_available():
            questions = self._generate_with_llm(context, role, count, language)
            if questions:
                return questions
        
        # Fallback to templates
        return self._generate_fallback(context, code_facts, role, count, language)
    
    def _generate_with_llm(
        self,
        context: BusinessContext,
        role: UserRole,
        count: int,
        language: str
    ) -> List[GeneratedQuestion]:
        """Generate questions using LLM."""
        system = self._get_system_prompt(language)
        prompt = self._get_generation_prompt(context, role, count, language)
        
        response = self.llm.generate_with_task(prompt, "question", system=system)
        
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
4. 生成多样的问题类型"""
        else:
            return """You are simulating real users asking questions about authentication systems.
Rules:
1. Generate questions real users would ask - they don't know internal code
2. Questions must be natural and conversational
3. NEVER use technical code terms
4. Generate diverse question types"""
    
    def _get_generation_prompt(
        self,
        context: BusinessContext,
        role: UserRole,
        count: int,
        language: str
    ) -> str:
        from .user_perspective import UserPerspectiveLayer
        role_ctx = UserPerspectiveLayer.ROLE_CONTEXTS.get(role, {}).get(language, "")
        
        if language == "zh":
            return f"""场景：{context.scenario_name}
用户体验：{context.user_experience}
边界情况：{', '.join(context.edge_cases[:3])}

你的角色：{role_ctx}

生成{count}个多样化问题，每行一个："""
        else:
            return f"""Scenario: {context.scenario_name}
User Experience: {context.user_experience}
Edge Cases: {', '.join(context.edge_cases[:3])}

Your Role: {role_ctx}

Generate {count} diverse questions, one per line:"""
    
    def _parse_questions(
        self,
        response: str,
        role: UserRole,
        language: str
    ) -> List[GeneratedQuestion]:
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
            
            if line.lower() in self.generated_questions:
                continue
            
            self.generated_questions.add(line.lower())
            questions.append(GeneratedQuestion(
                question_id=f"LLM-{uuid.uuid4().hex[:8]}",
                question_text=line,
                source="llm",
                role=role.value,
                language=language,
                question_type=self._classify_question(line),
            ))
        
        return questions
    
    def _generate_fallback(
        self,
        context: BusinessContext,
        code_facts: Optional[CodeFacts],
        role: UserRole,
        count: int,
        language: str
    ) -> List[GeneratedQuestion]:
        """Generate fallback questions from templates."""
        templates = self._get_templates(context, role, language)
        random.shuffle(templates)
        
        questions = []
        for template in templates[:count]:
            if template.lower() not in self.generated_questions:
                self.generated_questions.add(template.lower())
                questions.append(GeneratedQuestion(
                    question_id=f"FB-{uuid.uuid4().hex[:8]}",
                    question_text=template,
                    source="fallback",
                    role=role.value,
                    language=language,
                    question_type=self._classify_question(template),
                ))
        
        return questions
    
    def _get_templates(self, context: BusinessContext, role: UserRole, language: str) -> List[str]:
        """Get role-specific question templates."""
        templates = {
            "en": {
                UserRole.END_USER: [
                    "If I submit an email that's already registered, how does the system handle my request?",
                    "What message does the system show when I enter wrong login information?",
                    "Why doesn't the login error tell me if it's the email or password that's wrong?",
                    "If the username is taken, is my registration request completely rejected?",
                    "I updated my profile but something went wrong. Were my changes partially saved?",
                ],
                UserRole.QA_ENGINEER: [
                    "After validation fails, does the function continue executing subsequent code?",
                    "If the first validation passes but the second fails, what happens?",
                    "When validation fails, will there be any records in the database?",
                    "What is the execution flow of the function? What happens at each step?",
                ],
                UserRole.SECURITY_AUDITOR: [
                    "Can login failure messages help an attacker determine if an email exists?",
                    "Is there a timing difference that could reveal account existence?",
                    "What prevents username enumeration through the login endpoint?",
                ],
                UserRole.NEW_DEVELOPER: [
                    "What is the execution order of this code?",
                    "What role does the raise statement play here?",
                    "Why do these validations happen before writing data?",
                ],
                UserRole.PRODUCT_MANAGER: [
                    "What's the user experience if the registration process fails?",
                    "How does our session management affect users on multiple devices?",
                ],
            },
            "zh": {
                UserRole.END_USER: [
                    "如果我提交的邮箱已被注册，系统会如何处理我的请求？",
                    "当我输入错误的登录信息时，系统显示什么消息？",
                    "为什么登录错误不告诉我具体是邮箱错还是密码错？",
                    "如果用户名已被占用，我的注册请求会被完全拒绝吗？",
                ],
                UserRole.QA_ENGINEER: [
                    "验证失败后，函数是否会继续执行后续代码？",
                    "如果第一个验证通过但第二个失败，会发生什么？",
                    "当验证失败时，数据库中会有任何记录吗？",
                ],
                UserRole.SECURITY_AUDITOR: [
                    "登录失败的错误消息是否能让攻击者判断邮箱是否存在？",
                    "验证过程是否存在时序差异可能泄露信息？",
                ],
                UserRole.NEW_DEVELOPER: [
                    "这段代码的执行顺序是什么？",
                    "raise语句在这里起什么作用？",
                ],
                UserRole.PRODUCT_MANAGER: [
                    "如果注册过程中途失败，用户体验是什么？",
                ],
            }
        }
        return templates.get(language, templates["en"]).get(role, [])
    
    def _contains_code_names(self, text: str) -> bool:
        """Check if text contains code-level names."""
        for pattern in self.CODE_NAME_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _classify_question(self, question: str) -> QuestionType:
        """Classify question into a type."""
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


class ReasoningGenerator:
    """
    Generates reasoning chains constrained by code facts.
    
    Uses code-grounded prompts from v8 to ensure reasoning
    doesn't contradict deterministic code behavior.
    """
    
    def __init__(
        self,
        llm_client: OllamaClient,
        flow_analyzer: ExecutionFlowAnalyzer,
        consistency_validator: ConsistencyValidator
    ):
        self.llm = llm_client
        self.flow_analyzer = flow_analyzer
        self.validator = consistency_validator
    
    def generate(
        self,
        question: str,
        context: BusinessContext,
        code_context: CodeContext,
        code_facts: CodeFacts,
        language: str = "en"
    ) -> Tuple[List[str], bool]:
        """
        Generate reasoning and validate against code facts.
        
        Args:
            question: The question being answered
            context: Business context
            code_context: Code context
            code_facts: Deterministic code facts
            language: Language code
            
        Returns:
            Tuple of (reasoning_steps, is_valid)
        """
        if self.llm.is_available():
            prompt = self._build_prompt(question, context, code_facts, code_context, language)
            response = self.llm.generate_with_task(prompt, "reasoning")
            
            if response:
                reasoning = self._parse_reasoning(response)
                result = self.validator.validate_reasoning(reasoning, code_facts)
                
                if result.is_valid:
                    return reasoning, True
        
        # Fallback to deterministic reasoning
        return self._generate_fallback(code_facts, context, language), True
    
    def _build_prompt(
        self,
        question: str,
        context: BusinessContext,
        code_facts: CodeFacts,
        code_context: CodeContext,
        language: str
    ) -> str:
        """Build code-grounded reasoning prompt."""
        exec_order = self.flow_analyzer.format_for_prompt(code_facts, language)
        forbidden = self.validator.get_forbidden_phrases_prompt(code_facts, language)
        
        if language == "zh":
            return f"""分析问题并基于代码事实提供推理。

【问题】{question}

【代码事实】
{exec_order}
验证检查: {', '.join(code_facts.validation_checks)}
原子性: {code_facts.atomicity_type}

{forbidden}

【代码】
```python
{code_context.code_snippet[:800]}
```

提供4-5个推理步骤，格式：[步骤类型] 分析内容"""
        else:
            return f"""Analyze question based on code facts.

【QUESTION】{question}

【CODE FACTS】
{exec_order}
Validation: {', '.join(code_facts.validation_checks)}
Atomicity: {code_facts.atomicity_type}

{forbidden}

【CODE】
```python
{code_context.code_snippet[:800]}
```

Provide 4-5 reasoning steps, format: [STEP_TYPE] Analysis"""
    
    def _parse_reasoning(self, response: str) -> List[str]:
        """Parse reasoning from LLM response."""
        steps = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line.startswith('[') or re.match(r'^\d+\.', line)):
                steps.append(line)
        return steps[:6] if steps else [response[:200]]
    
    def _generate_fallback(
        self,
        code_facts: CodeFacts,
        context: BusinessContext,
        language: str
    ) -> List[str]:
        """Generate deterministic reasoning."""
        exec_order = [f"{s.order}.{s.operation}" for s in code_facts.execution_order[:4]]
        
        if language == "zh":
            return [
                f"[执行顺序] 代码按以下顺序执行: {' → '.join(exec_order)}",
                f"[验证检查] 系统执行: {', '.join(code_facts.validation_checks)}",
                f"[终止语义] 验证失败时，raise 立即终止函数",
                f"[原子性] {code_facts.atomicity_type} 执行，不可能有部分保存",
                f"[结论] 系统正确实现了验证-终止-写入的门控流程",
            ]
        else:
            return [
                f"[EXECUTION ORDER] Code executes: {' → '.join(exec_order)}",
                f"[VALIDATION] System performs: {', '.join(code_facts.validation_checks)}",
                f"[TERMINATION] On failure, raise immediately terminates",
                f"[ATOMICITY] {code_facts.atomicity_type} - partial save impossible",
                f"[CONCLUSION] System correctly implements validate-terminate-write flow",
            ]


class AnswerGenerator:
    """
    Generates answers constrained by code facts.
    
    Answers bridge user questions to code implementation while
    respecting deterministic code behavior.
    """
    
    def __init__(
        self,
        llm_client: OllamaClient,
        flow_analyzer: ExecutionFlowAnalyzer,
        consistency_validator: ConsistencyValidator
    ):
        self.llm = llm_client
        self.flow_analyzer = flow_analyzer
        self.validator = consistency_validator
    
    def generate(
        self,
        question: str,
        reasoning: List[str],
        code_context: CodeContext,
        code_facts: CodeFacts,
        language: str = "en"
    ) -> Tuple[str, bool]:
        """
        Generate answer and validate against code facts.
        
        Returns:
            Tuple of (answer, is_valid)
        """
        if self.llm.is_available():
            prompt = self._build_prompt(question, reasoning, code_facts, code_context, language)
            response = self.llm.generate_with_task(prompt, "answer")
            
            if response:
                result = self.validator.validate_answer(response, code_facts)
                if result.is_valid:
                    return self._format_answer(response, code_facts, code_context, language), True
        
        # Fallback
        return self._generate_fallback(code_facts, reasoning, code_context, language), True
    
    def _build_prompt(
        self,
        question: str,
        reasoning: List[str],
        code_facts: CodeFacts,
        code_context: CodeContext,
        language: str
    ) -> str:
        """Build code-grounded answer prompt."""
        exec_order = self.flow_analyzer.format_for_prompt(code_facts, language)
        forbidden = self.validator.get_forbidden_phrases_prompt(code_facts, language)
        
        if language == "zh":
            return f"""基于代码事实提供准确回答。

【问题】{question}

【代码事实】
{exec_order}

【推理】
{chr(10).join(reasoning)}

{forbidden}

提供准确回答，必须基于代码事实。"""
        else:
            return f"""Provide accurate answer based on code facts.

【QUESTION】{question}

【CODE FACTS】
{exec_order}

【REASONING】
{chr(10).join(reasoning)}

{forbidden}

Provide accurate answer grounded in code facts."""
    
    def _format_answer(
        self,
        response: str,
        code_facts: CodeFacts,
        code_context: CodeContext,
        language: str
    ) -> str:
        """Add code reference to answer."""
        exec_order = ' → '.join(f"{s.order}.{s.operation}" for s in code_facts.execution_order[:4])
        
        if language == "zh":
            code_section = f"""

### 代码执行事实

执行顺序: {exec_order}
原子性: {code_facts.atomicity_type}

```python
{code_context.code_snippet[:500]}
```"""
        else:
            code_section = f"""

### Code Execution Facts

Execution order: {exec_order}
Atomicity: {code_facts.atomicity_type}

```python
{code_context.code_snippet[:500]}
```"""
        
        return response + code_section
    
    def _generate_fallback(
        self,
        code_facts: CodeFacts,
        reasoning: List[str],
        code_context: CodeContext,
        language: str
    ) -> str:
        """Generate deterministic answer."""
        exec_order = ' → '.join(f"{s.order}.{s.operation}" for s in code_facts.execution_order[:4])
        
        if language == "zh":
            return f"""### 回答

基于代码的确定性分析：

**执行顺序**: {exec_order}

**关键行为**:
- 同步顺序执行
- 验证失败时 `raise` 立即终止
- 所有验证通过前不写入数据
- "部分保存"不可能发生

**推理过程**:
{chr(10).join(reasoning)}

### 代码参考

```python
{code_context.code_snippet[:500]}
```"""
        else:
            return f"""### Answer

Based on deterministic code analysis:

**Execution Order**: {exec_order}

**Key Behaviors**:
- Synchronous sequential execution
- `raise` immediately terminates on validation failure
- No data written until ALL validations pass
- "Partial save" is IMPOSSIBLE

**Reasoning**:
{chr(10).join(reasoning)}

### Code Reference

```python
{code_context.code_snippet[:500]}
```"""


class LLMEnhancementLayer:
    """
    Facade for all LLM-based generation components.
    """
    
    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        self.flow_analyzer = ExecutionFlowAnalyzer()
        self.consistency_validator = ConsistencyValidator()
        
        self.question_generator = QuestionGenerator(llm_client)
        self.reasoning_generator = ReasoningGenerator(
            llm_client, self.flow_analyzer, self.consistency_validator
        )
        self.answer_generator = AnswerGenerator(
            llm_client, self.flow_analyzer, self.consistency_validator
        )
    
    def is_llm_available(self) -> bool:
        return self.llm.is_available()
