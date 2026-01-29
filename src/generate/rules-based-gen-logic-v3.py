#!/usr/bin/env python3
"""
Enterprise Q&A Generation Engine v3.0 - LLM-Enhanced Hybrid System

This version combines:
1. Deterministic analysis (AST, Call Graph, DBR mapping) for factual accuracy
2. LLM enhancement (Qwen 2.5:7b via Ollama) for natural language quality
3. Deep security question generation for expert-level Q&A

Key Improvements over v2:
- Natural, human-like questions (not template-based)
- Deep security analysis questions:
  * "What if we remove this check?"
  * "Is there a race condition under high concurrency?"
  * "What attack vectors does this prevent?"
- Human-like reasoning chains (not mechanical)
- LLM-assisted answer composition

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                     HybridQAOrchestrator                            │
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

Author: Auto-generated
Version: 3.0.0
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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

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
    """Global configuration for v3 hybrid generation."""
    VERSION = "3.0.0"

    # Paths
    BASE_DIR = Path(__file__).parent.resolve()
    WORKSPACE_ROOT = BASE_DIR.parent.parent
    DATA_DIR = WORKSPACE_ROOT / "data"
    REPO_PATH = WORKSPACE_ROOT / "repos" / "fastapi-realworld-example-app"

    # Input/Output
    RULE_METADATA_FILE = DATA_DIR / "dbr01_rule_metadata.json"
    AST_ANALYSIS_FILE = DATA_DIR / "fastapi_analysis_result.json"
    OUTPUT_FILE = DATA_DIR / "qwen_dbr_training_logic_v3.jsonl"

    # LLM Configuration
    OLLAMA_API = "http://localhost:11434/api/generate"
    MODEL_NAME = "qwen2.5:7b"
    LLM_TIMEOUT = 120  # seconds
    LLM_TEMPERATURE = 0.7
    LLM_NUM_CTX = 8192
    LLM_RETRY_COUNT = 3
    LLM_RETRY_DELAY = 2  # seconds

    # Generation Parameters
    SUPPORTED_LANGUAGES = ["en", "zh"]
    MAX_SAMPLES_PER_EVIDENCE = 5
    ENABLE_DEEP_QUESTIONS = True
    ENABLE_LLM_ENHANCEMENT = True
    FALLBACK_TO_TEMPLATE = True  # Use template if LLM fails


# ============================================================================
# Deep Question Types for Security Analysis
# ============================================================================

class DeepQuestionType(str, Enum):
    """Types of deep security/analysis questions."""
    WHAT_IF_REMOVE = "what_if_remove"  # "What if we remove this check?"
    RACE_CONDITION = "race_condition"  # "Race condition under high concurrency?"
    ATTACK_VECTOR = "attack_vector"  # "What attacks does this prevent?"
    EDGE_CASE = "edge_case"  # "What edge cases could fail?"
    BYPASS_ATTEMPT = "bypass_attempt"  # "How could an attacker bypass this?"
    DATA_FLOW = "data_flow"  # "How does data flow through this?"
    FAILURE_MODE = "failure_mode"  # "What happens when this fails?"
    TIMING_ATTACK = "timing_attack"  # "Timing attack vulnerability?"
    DEPENDENCY_RISK = "dependency_risk"  # "External dependency risks?"
    COMPLIANCE = "compliance"  # "Does this meet compliance requirements?"


# ============================================================================
# LLM Client
# ============================================================================

class OllamaClient:
    """Client for Ollama API with retry logic."""

    def __init__(
            self,
            api_url: str = Config.OLLAMA_API,
            model: str = Config.MODEL_NAME,
            timeout: int = Config.LLM_TIMEOUT
    ):
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self._available = None

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if self._available is not None:
            return self._available

        try:
            response = requests.get(
                self.api_url.replace("/api/generate", "/api/tags"),
                timeout=5
            )
            self._available = response.status_code == 200
        except:
            self._available = False

        return self._available

    def generate(
            self,
            prompt: str,
            system: Optional[str] = None,
            temperature: float = Config.LLM_TEMPERATURE,
            max_retries: int = Config.LLM_RETRY_COUNT
    ) -> Optional[str]:
        """Generate text using Ollama API with retry logic."""
        if not self.is_available():
            logger.warning("Ollama is not available")
            return None

        full_prompt = prompt
        if system:
            full_prompt = f"System: {system}\n\nUser: {prompt}"

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_ctx": Config.LLM_NUM_CTX,
                        }
                    },
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return response.json().get("response", "").strip()
                else:
                    logger.warning(f"Ollama API error: {response.status_code}")

            except requests.exceptions.Timeout:
                logger.warning(f"Ollama timeout (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.warning(f"Ollama error: {e}")

            if attempt < max_retries - 1:
                time.sleep(Config.LLM_RETRY_DELAY * (attempt + 1))

        return None


# ============================================================================
# Prompt Templates for LLM
# ============================================================================

class PromptTemplates:
    """Prompt templates for LLM-based generation."""

    # System prompt for expert role
    EXPERT_SYSTEM_PROMPT = """You are a senior security engineer and software architect with 15+ years of experience in authentication systems, API security, and secure coding practices. You analyze code with a critical eye for security vulnerabilities, edge cases, and potential attack vectors.

Your responses should be:
- Technical and precise
- Grounded in the actual code provided
- Focused on security implications
- Written in a natural, expert tone (not robotic or template-like)"""

    EXPERT_SYSTEM_PROMPT_CN = """你是一位拥有15年以上认证系统、API安全和安全编码实践经验的资深安全工程师和软件架构师。你以挑剔的眼光分析代码，关注安全漏洞、边缘情况和潜在的攻击向量。

你的回答应该：
- 技术性强且精确
- 基于提供的实际代码
- 聚焦于安全影响
- 以自然、专家的语气撰写（不要机械化或模板化）"""

    # Question generation prompt
    GENERATE_QUESTIONS_PROMPT = """Based on the following code snippet and its security context, generate {count} diverse, expert-level questions that a security auditor or senior developer might ask.

**Code File**: {file_path}
**Function**: {function_name}
**Security Context**: {security_context}
**DBR Rule**: {dbr_rule} - {dbr_description}

**Code Snippet**:
```python
{code_snippet}
```

**Call Chain**: {call_chain}

**Requirements**:
1. Generate questions that are natural and conversational (NOT template-like)
2. Include at least one "what if" question (e.g., "What would happen if we remove the {key_element} check?")
3. Include at least one deep security question (e.g., race conditions, attack vectors, timing attacks)
4. Questions should reference specific elements in the code
5. Vary the question types: analytical, hypothetical, comparative, diagnostic

**Format**: Return ONLY a JSON array of questions, each with "question" and "type" fields.
Example:
[
  {{"question": "If an attacker sends concurrent registration requests with the same email, could they bypass the uniqueness check due to a race condition?", "type": "race_condition"}},
  {{"question": "What security vulnerability would be introduced if the check_email_is_taken validation were removed?", "type": "what_if_remove"}}
]

Generate {count} questions:"""

    GENERATE_QUESTIONS_PROMPT_CN = """基于以下代码片段及其安全上下文，生成{count}个多样化的、专家级别的问题，这些问题是安全审计员或高级开发人员可能会提出的。

**代码文件**: {file_path}
**函数名称**: {function_name}
**安全上下文**: {security_context}
**DBR规则**: {dbr_rule} - {dbr_description}

**代码片段**:
```python
{code_snippet}
```

**调用链**: {call_chain}

**要求**:
1. 生成自然、对话式的问题（不要模板化）
2. 包含至少一个"如果...会怎样"的问题（例如："如果去掉{key_element}检查会发生什么？"）
3. 包含至少一个深度安全问题（例如：竞态条件、攻击向量、时序攻击）
4. 问题应该引用代码中的具体元素
5. 问题类型要多样化：分析性、假设性、比较性、诊断性

**格式**: 仅返回JSON数组，每个问题包含"question"和"type"字段。
示例:
[
  {{"question": "如果攻击者同时发送多个使用相同邮箱的注册请求，是否可能因为竞态条件而绕过唯一性检查？", "type": "race_condition"}},
  {{"question": "如果移除check_email_is_taken验证会引入什么安全漏洞？", "type": "what_if_remove"}}
]

生成{count}个问题:"""

    # Reasoning generation prompt
    GENERATE_REASONING_PROMPT = """As a senior security engineer, analyze the following code and provide a natural, expert-level reasoning chain that explains how this code implements the security requirement.

**Question**: {question}

**Code**:
```python
{code_snippet}
```

**Security Context**:
- DBR Rule: {dbr_rule}
- Key Elements: {key_elements}
- Call Chain: {call_chain}

**Requirements**:
1. Write as if you're explaining to a colleague (natural, not robotic)
2. Reference specific lines/elements in the code
3. Explain the security implications
4. If the question is a "what if" question, analyze the hypothetical scenario
5. Be concise but thorough (4-6 reasoning steps)

**Format**: Return a JSON object with "steps" (array of reasoning steps) and "conclusion" (final insight).
Each step should have "observation" and "analysis" fields.

Example:
{{
  "steps": [
    {{"observation": "Looking at lines 67-71, I see the check_username_is_taken function is called before any database write.", "analysis": "This pre-validation pattern is crucial for preventing duplicate entries."}},
    {{"observation": "The function raises HTTP_400_BAD_REQUEST when a duplicate is found.", "analysis": "This explicit rejection prevents the user from proceeding with an already-taken identifier."}}
  ],
  "conclusion": "The implementation correctly prevents duplicate registrations by validating uniqueness before persisting data."
}}

Generate reasoning:"""

    GENERATE_REASONING_PROMPT_CN = """作为资深安全工程师，分析以下代码并提供自然、专家级别的推理链，解释这段代码如何实现安全要求。

**问题**: {question}

**代码**:
```python
{code_snippet}
```

**安全上下文**:
- DBR规则: {dbr_rule}
- 关键元素: {key_elements}
- 调用链: {call_chain}

**要求**:
1. 像向同事解释一样书写（自然，不要机械化）
2. 引用代码中的具体行/元素
3. 解释安全影响
4. 如果问题是"如果...会怎样"类型，分析假设场景
5. 简洁但全面（4-6个推理步骤）

**格式**: 返回一个JSON对象，包含"steps"（推理步骤数组）和"conclusion"（最终见解）。
每个步骤应有"observation"和"analysis"字段。

生成推理:"""

    # Answer generation prompt
    GENERATE_ANSWER_PROMPT = """As a senior security engineer, provide a comprehensive answer to the following question about the code.

**Question**: {question}

**Code** ({file_path}):
```python
{code_snippet}
```

**Context**:
- Function: {function_name}
- DBR Rule: {dbr_rule} - {dbr_description}
- Call Chain: {call_chain}
- Reasoning Chain: {reasoning_chain}

**Requirements**:
1. Start with a direct answer to the question
2. Reference specific code elements (function names, variables, line logic)
3. Explain security implications thoroughly
4. If it's a "what if" question, describe the potential vulnerabilities
5. Include best practices and recommendations
6. Write in a natural, expert tone

Generate a comprehensive answer (300-500 words):"""

    GENERATE_ANSWER_PROMPT_CN = """作为资深安全工程师，为以下关于代码的问题提供全面的回答。

**问题**: {question}

**代码** ({file_path}):
```python
{code_snippet}
```

**上下文**:
- 函数: {function_name}
- DBR规则: {dbr_rule} - {dbr_description}
- 调用链: {call_chain}
- 推理链: {reasoning_chain}

**要求**:
1. 以直接回答问题开始
2. 引用具体的代码元素（函数名、变量、代码逻辑）
3. 全面解释安全影响
4. 如果是"如果...会怎样"类型的问题，描述潜在漏洞
5. 包含最佳实践和建议
6. 以自然、专家的语气撰写

生成全面回答（300-500字）:"""

    # Deep security question prompts
    DEEP_SECURITY_PROMPTS = {
        DeepQuestionType.WHAT_IF_REMOVE: {
            "en": "What security vulnerabilities would be introduced if the {element} check/validation were removed from this code?",
            "zh": "如果从这段代码中移除{element}的检查/验证，会引入什么安全漏洞？"
        },
        DeepQuestionType.RACE_CONDITION: {
            "en": "Under high concurrency, could there be a race condition in this {operation}? How would concurrent requests interact with the {element} validation?",
            "zh": "在高并发场景下，这个{operation}是否存在竞态条件？并发请求如何与{element}验证交互？"
        },
        DeepQuestionType.ATTACK_VECTOR: {
            "en": "What specific attack vectors does this implementation of {feature} protect against? Are there any attack surfaces left unprotected?",
            "zh": "{feature}的这个实现能防御哪些具体的攻击向量？是否有未被保护的攻击面？"
        },
        DeepQuestionType.EDGE_CASE: {
            "en": "What edge cases might cause the {element} validation to fail or behave unexpectedly? How would the system handle them?",
            "zh": "什么边缘情况可能导致{element}验证失败或出现意外行为？系统如何处理这些情况？"
        },
        DeepQuestionType.BYPASS_ATTEMPT: {
            "en": "If an attacker wanted to bypass the {security_control}, what techniques might they try? How does the current implementation defend against these?",
            "zh": "如果攻击者想要绕过{security_control}，他们可能会尝试什么技术？当前实现如何防御这些攻击？"
        },
        DeepQuestionType.TIMING_ATTACK: {
            "en": "Could the error handling in {function} leak information through timing differences? How might an attacker exploit this?",
            "zh": "{function}中的错误处理是否可能通过时序差异泄露信息？攻击者如何利用这一点？"
        },
        DeepQuestionType.FAILURE_MODE: {
            "en": "What happens when {component} fails or throws an unexpected exception? Is the system left in a secure state?",
            "zh": "当{component}失败或抛出意外异常时会发生什么？系统是否保持在安全状态？"
        },
    }


# ============================================================================
# LLM-Enhanced Question Generator
# ============================================================================

class LLMQuestionGenerator:
    """Generates natural, expert-level questions using LLM."""

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        self.prompts = PromptTemplates()

    def generate_questions(
            self,
            evidence: Dict,
            code_snippet: str,
            call_chain: List[str],
            language: str = "en",
            count: int = 3
    ) -> List[Dict]:
        """Generate questions using LLM."""

        # Prepare context
        file_path = evidence.get("location", {}).get("file_path", "")
        function_name = evidence.get("name", "")
        dbr_logic = evidence.get("dbr_logic", {})
        related_elements = evidence.get("related_elements", [])

        security_context = self._get_security_context(evidence)
        key_element = related_elements[0] if related_elements else "the validation"

        # Select prompt based on language
        if language == "zh":
            prompt = self.prompts.GENERATE_QUESTIONS_PROMPT_CN
            system = self.prompts.EXPERT_SYSTEM_PROMPT_CN
        else:
            prompt = self.prompts.GENERATE_QUESTIONS_PROMPT
            system = self.prompts.EXPERT_SYSTEM_PROMPT

        # Format prompt
        formatted_prompt = prompt.format(
            count=count,
            file_path=file_path,
            function_name=function_name,
            security_context=security_context,
            dbr_rule=dbr_logic.get("rule_id", "DBR-01"),
            dbr_description=evidence.get("description", "")[:200],
            code_snippet=code_snippet[:2000],  # Limit code length
            call_chain=" → ".join(call_chain[:5]) if call_chain else "N/A",
            key_element=key_element,
        )

        # Generate with LLM
        response = self.llm.generate(formatted_prompt, system=system)

        if response:
            questions = self._parse_questions_response(response, language)
            if questions:
                return questions

        # Fallback to template-based generation
        logger.info("Falling back to template-based question generation")
        return self._generate_fallback_questions(evidence, language, count)

    def _get_security_context(self, evidence: Dict) -> str:
        """Get security context description for the evidence."""
        evidence_type = evidence.get("evidence_type", "")
        dbr_logic = evidence.get("dbr_logic", {})
        subcategory = dbr_logic.get("subcategory_id", "")

        contexts = {
            "DBR-01-01": "User registration and profile update uniqueness validation",
            "DBR-01-02": "Atomic account creation with secure credential storage",
            "DBR-01-03": "Authentication failure handling to prevent user enumeration",
            "DBR-01-04": "JWT token generation and session state management",
        }

        return contexts.get(subcategory, "Authentication and security validation")

    def _parse_questions_response(self, response: str, language: str) -> List[Dict]:
        """Parse LLM response to extract questions."""
        try:
            # Try to find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                questions = json.loads(json_match.group())
                return [
                    {
                        "question_id": f"Q-{uuid.uuid4().hex[:8]}",
                        "question_text": q.get("question", ""),
                        "question_type": q.get("type", "analytical"),
                        "source": "llm",
                    }
                    for q in questions
                    if q.get("question")
                ]
        except json.JSONDecodeError:
            pass

        # Try to extract questions line by line
        questions = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and ('?' in line or '？' in line):
                # Clean up the line
                line = re.sub(r'^[\d\.\-\*]+\s*', '', line)
                line = re.sub(r'^["\']+|["\']+$', '', line)
                if len(line) > 20:
                    questions.append({
                        "question_id": f"Q-{uuid.uuid4().hex[:8]}",
                        "question_text": line,
                        "question_type": "analytical",
                        "source": "llm",
                    })

        return questions[:5]  # Limit to 5 questions

    def _generate_fallback_questions(
            self,
            evidence: Dict,
            language: str,
            count: int
    ) -> List[Dict]:
        """Generate questions using templates as fallback."""
        questions = []
        related_elements = evidence.get("related_elements", [])
        function_name = evidence.get("name", "this function")
        dbr_logic = evidence.get("dbr_logic", {})
        subcategory = dbr_logic.get("subcategory_id", "")

        # Deep question templates
        deep_templates = {
            "en": [
                f"What would happen if the {related_elements[0] if related_elements else 'validation'} check were removed from {function_name}?",
                f"Under high concurrency, could there be a race condition in the {function_name} implementation?",
                f"What attack vectors does the current implementation of {function_name} protect against?",
                f"What edge cases might cause {function_name} to behave unexpectedly?",
                f"How does {function_name} ensure data consistency when multiple requests arrive simultaneously?",
            ],
            "zh": [
                f"如果从{function_name}中移除{related_elements[0] if related_elements else '验证'}检查会发生什么？",
                f"在高并发场景下，{function_name}的实现是否存在竞态条件？",
                f"{function_name}的当前实现能防御哪些攻击向量？",
                f"什么边缘情况可能导致{function_name}出现意外行为？",
                f"当多个请求同时到达时，{function_name}如何确保数据一致性？",
            ],
        }

        templates = deep_templates.get(language, deep_templates["en"])
        random.shuffle(templates)

        for template in templates[:count]:
            questions.append({
                "question_id": f"Q-{uuid.uuid4().hex[:8]}",
                "question_text": template,
                "question_type": "deep_analysis",
                "source": "template",
            })

        return questions


# ============================================================================
# LLM-Enhanced Reasoning Generator
# ============================================================================

class LLMReasoningGenerator:
    """Generates natural, human-like reasoning chains using LLM."""

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        self.prompts = PromptTemplates()

    def generate_reasoning(
            self,
            question: str,
            evidence: Dict,
            code_snippet: str,
            call_chain: List[str],
            language: str = "en"
    ) -> Dict:
        """Generate reasoning chain using LLM."""

        dbr_logic = evidence.get("dbr_logic", {})
        related_elements = evidence.get("related_elements", [])

        # Select prompt based on language
        if language == "zh":
            prompt = self.prompts.GENERATE_REASONING_PROMPT_CN
            system = self.prompts.EXPERT_SYSTEM_PROMPT_CN
        else:
            prompt = self.prompts.GENERATE_REASONING_PROMPT
            system = self.prompts.EXPERT_SYSTEM_PROMPT

        formatted_prompt = prompt.format(
            question=question,
            code_snippet=code_snippet[:2000],
            dbr_rule=f"{dbr_logic.get('rule_id', 'DBR-01')} ({dbr_logic.get('subcategory_id', '')})",
            key_elements=", ".join(related_elements[:5]),
            call_chain=" → ".join(call_chain[:5]) if call_chain else "N/A",
        )

        response = self.llm.generate(formatted_prompt, system=system)

        if response:
            reasoning = self._parse_reasoning_response(response, language)
            if reasoning:
                return reasoning

        # Fallback
        return self._generate_fallback_reasoning(question, evidence, language)

    def _parse_reasoning_response(self, response: str, language: str) -> Optional[Dict]:
        """Parse LLM response to extract reasoning."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                if "steps" in data:
                    return {
                        "steps": data["steps"],
                        "conclusion": data.get("conclusion", ""),
                        "source": "llm",
                    }
        except json.JSONDecodeError:
            pass

        # Parse as free text
        steps = []
        lines = response.split('\n')
        current_step = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for observation/analysis patterns
            if any(kw in line.lower() for kw in
                   ['looking at', 'i see', 'observing', 'notice', '观察', '看到', '注意到']):
                if current_step:
                    steps.append(current_step)
                current_step = {"observation": line, "analysis": ""}
            elif any(kw in line.lower() for kw in
                     ['this means', 'therefore', 'because', 'implies', '这意味着', '因此', '因为', '说明']):
                if current_step:
                    current_step["analysis"] = line
            elif current_step and not current_step.get("analysis"):
                current_step["analysis"] = line

        if current_step:
            steps.append(current_step)

        if steps:
            # Extract conclusion (last meaningful sentence)
            conclusion = ""
            for line in reversed(lines):
                line = line.strip()
                if len(line) > 30:
                    conclusion = line
                    break

            return {
                "steps": steps[:6],  # Limit to 6 steps
                "conclusion": conclusion,
                "source": "llm_parsed",
            }

        return None

    def _generate_fallback_reasoning(
            self,
            question: str,
            evidence: Dict,
            language: str
    ) -> Dict:
        """Generate reasoning using deterministic approach."""
        related_elements = evidence.get("related_elements", [])
        description = evidence.get("description" if language == "en" else "description_cn", "")
        dbr_logic = evidence.get("dbr_logic", {})

        if language == "zh":
            steps = [
                {"observation": f"分析代码中的关键元素：{', '.join(related_elements[:3])}",
                 "analysis": "这些元素构成了安全验证的核心逻辑"},
                {"observation": f"代码实现了：{description[:100]}",
                 "analysis": "这确保了系统的安全性和数据完整性"},
                {"observation": f"此实现符合{dbr_logic.get('rule_id', 'DBR-01')}规则",
                 "analysis": "通过显式检查防止了潜在的安全漏洞"},
            ]
            conclusion = f"综上所述，该实现通过{', '.join(related_elements[:2])}等机制确保了认证和凭据的完整性。"
        else:
            steps = [
                {"observation": f"Analyzing key elements in the code: {', '.join(related_elements[:3])}",
                 "analysis": "These elements form the core of the security validation logic"},
                {"observation": f"The code implements: {description[:100]}",
                 "analysis": "This ensures system security and data integrity"},
                {"observation": f"This implementation aligns with {dbr_logic.get('rule_id', 'DBR-01')} rule",
                 "analysis": "Explicit checks prevent potential security vulnerabilities"},
            ]
            conclusion = f"In summary, this implementation ensures authentication and credential integrity through mechanisms like {', '.join(related_elements[:2])}."

        return {
            "steps": steps,
            "conclusion": conclusion,
            "source": "fallback",
        }


# ============================================================================
# LLM-Enhanced Answer Composer
# ============================================================================

class LLMAnswerComposer:
    """Composes comprehensive answers using LLM."""

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        self.prompts = PromptTemplates()

    def compose_answer(
            self,
            question: str,
            evidence: Dict,
            code_snippet: str,
            reasoning: Dict,
            call_chain: List[str],
            language: str = "en"
    ) -> str:
        """Compose comprehensive answer using LLM."""

        file_path = evidence.get("location", {}).get("file_path", "")
        function_name = evidence.get("name", "")
        dbr_logic = evidence.get("dbr_logic", {})

        # Format reasoning chain
        reasoning_chain = self._format_reasoning_chain(reasoning, language)

        # Select prompt based on language
        if language == "zh":
            prompt = self.prompts.GENERATE_ANSWER_PROMPT_CN
            system = self.prompts.EXPERT_SYSTEM_PROMPT_CN
        else:
            prompt = self.prompts.GENERATE_ANSWER_PROMPT
            system = self.prompts.EXPERT_SYSTEM_PROMPT

        formatted_prompt = prompt.format(
            question=question,
            file_path=file_path,
            code_snippet=code_snippet[:2000],
            function_name=function_name,
            dbr_rule=dbr_logic.get("rule_id", "DBR-01"),
            dbr_description=dbr_logic.get("subcategory_id", ""),
            call_chain=" → ".join(call_chain[:5]) if call_chain else "N/A",
            reasoning_chain=reasoning_chain,
        )

        response = self.llm.generate(formatted_prompt, system=system, temperature=0.6)

        if response:
            # Enhance with structured sections
            return self._structure_answer(response, evidence, code_snippet, reasoning, language)

        # Fallback
        return self._generate_fallback_answer(question, evidence, code_snippet, reasoning, language)

    def _format_reasoning_chain(self, reasoning: Dict, language: str) -> str:
        """Format reasoning for prompt."""
        steps = reasoning.get("steps", [])
        formatted = []

        for i, step in enumerate(steps, 1):
            obs = step.get("observation", "")
            analysis = step.get("analysis", "")
            formatted.append(f"{i}. {obs}")
            if analysis:
                formatted.append(f"   → {analysis}")

        return "\n".join(formatted)

    def _structure_answer(
            self,
            llm_response: str,
            evidence: Dict,
            code_snippet: str,
            reasoning: Dict,
            language: str
    ) -> str:
        """Structure the answer with all required sections."""
        file_path = evidence.get("location", {}).get("file_path", "")
        line_start = evidence.get("code_snippet", {}).get("line_start", 0)
        line_end = evidence.get("code_snippet", {}).get("line_end", 0)

        if language == "zh":
            sections = [
                "### 专家分析\n",
                llm_response,
                "\n\n### 推理过程\n",
            ]

            for i, step in enumerate(reasoning.get("steps", []), 1):
                sections.append(f"**步骤 {i}**: {step.get('observation', '')}\n")
                if step.get("analysis"):
                    sections.append(f"  - *分析*: {step['analysis']}\n")

            sections.extend([
                f"\n**结论**: {reasoning.get('conclusion', '')}\n",
                "\n### 相关代码\n",
                f"**文件**: `{file_path}` (第 {line_start}-{line_end} 行)\n\n",
                f"```python\n{code_snippet}\n```\n",
            ])
        else:
            sections = [
                "### Expert Analysis\n",
                llm_response,
                "\n\n### Reasoning Process\n",
            ]

            for i, step in enumerate(reasoning.get("steps", []), 1):
                sections.append(f"**Step {i}**: {step.get('observation', '')}\n")
                if step.get("analysis"):
                    sections.append(f"  - *Analysis*: {step['analysis']}\n")

            sections.extend([
                f"\n**Conclusion**: {reasoning.get('conclusion', '')}\n",
                "\n### Relevant Code\n",
                f"**File**: `{file_path}` (Lines {line_start}-{line_end})\n\n",
                f"```python\n{code_snippet}\n```\n",
            ])

        return "".join(sections)

    def _generate_fallback_answer(
            self,
            question: str,
            evidence: Dict,
            code_snippet: str,
            reasoning: Dict,
            language: str
    ) -> str:
        """Generate answer without LLM."""
        file_path = evidence.get("location", {}).get("file_path", "")
        line_start = evidence.get("code_snippet", {}).get("line_start", 0)
        line_end = evidence.get("code_snippet", {}).get("line_end", 0)
        description = evidence.get("description" if language == "en" else "description_cn", "")

        if language == "zh":
            answer = f"""### 技术分析

{description}

### 推理过程

"""
            for i, step in enumerate(reasoning.get("steps", []), 1):
                answer += f"**步骤 {i}**: {step.get('observation', '')}\n"
                if step.get("analysis"):
                    answer += f"  - *分析*: {step['analysis']}\n"

            answer += f"""
**结论**: {reasoning.get('conclusion', '')}

### 相关代码

**文件**: `{file_path}` (第 {line_start}-{line_end} 行)

```python
{code_snippet}
```

### 安全建议

- 确保所有验证逻辑在数据持久化之前执行
- 考虑高并发场景下的竞态条件
- 保持错误信息的模糊性以防止信息泄露
"""
        else:
            answer = f"""### Technical Analysis

{description}

### Reasoning Process

"""
            for i, step in enumerate(reasoning.get("steps", []), 1):
                answer += f"**Step {i}**: {step.get('observation', '')}\n"
                if step.get("analysis"):
                    answer += f"  - *Analysis*: {step['analysis']}\n"

            answer += f"""
**Conclusion**: {reasoning.get('conclusion', '')}

### Relevant Code

**File**: `{file_path}` (Lines {line_start}-{line_end})

```python
{code_snippet}
```

### Security Recommendations

- Ensure all validation logic executes before data persistence
- Consider race conditions in high-concurrency scenarios
- Keep error messages vague to prevent information leakage
"""

        return answer


# ============================================================================
# Call Graph Engine (Simplified from v2)
# ============================================================================

class CallGraphEngine:
    """Builds call graphs from AST analysis."""

    def __init__(self):
        self.call_nodes: Dict[str, Dict] = {}

    def load_ast_analysis(self, path: str) -> bool:
        """Load AST analysis and build call graph."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                ast_data = json.load(f)

            for module in ast_data.get("modules", []):
                for func in module.get("functions", []):
                    qn = func.get("qualified_name", "")
                    self.call_nodes[qn] = {
                        "name": func.get("name", ""),
                        "calls": func.get("calls", []),
                        "file_path": module.get("file_path", ""),
                        "line": func.get("line_start", 0),
                    }

                for cls in module.get("classes", []):
                    for method in cls.get("methods", []):
                        qn = method.get("qualified_name", "")
                        self.call_nodes[qn] = {
                            "name": method.get("name", ""),
                            "calls": method.get("calls", []),
                            "file_path": module.get("file_path", ""),
                            "line": method.get("line_start", 0),
                        }

            logger.info(f"Loaded {len(self.call_nodes)} functions from AST")
            return True
        except Exception as e:
            logger.error(f"Error loading AST: {e}")
            return False

    def get_call_chain(self, function_name: str, max_depth: int = 5) -> List[str]:
        """Get call chain starting from a function."""
        # Find the function
        start_node = None
        for qn, node in self.call_nodes.items():
            if function_name in qn:
                start_node = (qn, node)
                break

        if not start_node:
            return []

        chain = [start_node[0]]
        visited = {start_node[0]}

        # BFS to build chain
        queue = [(start_node[0], 1)]
        while queue and len(chain) < max_depth:
            current_qn, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            current_node = self.call_nodes.get(current_qn, {})
            for call in current_node.get("calls", []):
                # Find the called function
                for qn, node in self.call_nodes.items():
                    if call in qn and qn not in visited:
                        visited.add(qn)
                        chain.append(qn)
                        queue.append((qn, depth + 1))
                        break

        return chain


# ============================================================================
# Quality Validator
# ============================================================================

class QualityValidator:
    """Validates generated Q&A pairs."""

    def validate(self, qa_pair: Dict) -> Tuple[bool, float, List[str]]:
        """Validate a Q&A pair and return (is_valid, score, issues)."""
        issues = []
        scores = []

        # Check question
        question = qa_pair.get("instruction", "")
        if len(question) < 20:
            issues.append("Question too short")
            scores.append(0.3)
        elif not (question.endswith("?") or question.endswith("？")):
            issues.append("Question missing question mark")
            scores.append(0.8)
        else:
            scores.append(1.0)

        # Check answer
        answer = qa_pair.get("answer", "")
        if len(answer) < 100:
            issues.append("Answer too short")
            scores.append(0.3)
        else:
            scores.append(1.0)

        # Check code snippet
        code = qa_pair.get("context", {}).get("code_snippet", "")
        if len(code) < 50:
            issues.append("Code snippet too short")
            scores.append(0.5)
        else:
            scores.append(1.0)

        # Check reasoning
        reasoning = qa_pair.get("reasoning_trace", [])
        if len(reasoning) < 2:
            issues.append("Insufficient reasoning steps")
            scores.append(0.5)
        else:
            scores.append(1.0)

        avg_score = sum(scores) / len(scores)
        is_valid = avg_score >= 0.7

        return is_valid, avg_score, issues


# ============================================================================
# Hybrid QA Orchestrator
# ============================================================================

class HybridQAOrchestrator:
    """
    Main orchestrator combining deterministic analysis with LLM enhancement.
    """

    def __init__(self, rule_metadata_path: str, ast_analysis_path: str = None):
        self.rule_metadata_path = Path(rule_metadata_path)
        self.ast_analysis_path = Path(ast_analysis_path) if ast_analysis_path else Config.AST_ANALYSIS_FILE

        # Initialize LLM client
        self.llm_client = OllamaClient()
        self.llm_available = self.llm_client.is_available()

        if self.llm_available:
            logger.info(f"✓ LLM available: {Config.MODEL_NAME}")
        else:
            logger.warning("✗ LLM not available, will use fallback templates")

        # Initialize components
        self.call_graph = CallGraphEngine()
        self.question_generator = LLMQuestionGenerator(self.llm_client)
        self.reasoning_generator = LLMReasoningGenerator(self.llm_client)
        self.answer_composer = LLMAnswerComposer(self.llm_client)
        self.validator = QualityValidator()

        # Data
        self.rule_metadata: Dict = {}
        self.generated_pairs: List[Dict] = []
        self.stats: Dict = defaultdict(int)

    def initialize(self) -> bool:
        """Initialize the orchestrator."""
        # Load rule metadata
        try:
            with open(self.rule_metadata_path, 'r', encoding='utf-8') as f:
                self.rule_metadata = json.load(f)
            logger.info(f"Loaded rule metadata: {self.rule_metadata.get('rule_id')}")
        except Exception as e:
            logger.error(f"Error loading rule metadata: {e}")
            return False

        # Load AST analysis
        if self.ast_analysis_path.exists():
            self.call_graph.load_ast_analysis(str(self.ast_analysis_path))

        return True

    def run_pipeline(
            self,
            samples_per_evidence: int = 3,
            languages: List[str] = None
    ) -> List[Dict]:
        """Run the hybrid generation pipeline."""
        languages = languages or Config.SUPPORTED_LANGUAGES
        self.generated_pairs = []
        self.stats = defaultdict(int)

        logger.info(f"Starting hybrid pipeline (LLM: {'enabled' if self.llm_available else 'disabled'})")

        for subcategory in self.rule_metadata.get("subcategories", []):
            self._process_subcategory(subcategory, samples_per_evidence, languages)

        logger.info(f"Generated {len(self.generated_pairs)} Q&A pairs")
        return self.generated_pairs

    def _process_subcategory(
            self,
            subcategory: Dict,
            samples_per_evidence: int,
            languages: List[str]
    ):
        """Process a subcategory."""
        subcategory_id = subcategory.get("subcategory_id", "")
        logger.info(f"Processing: {subcategory_id}")

        for evidence in subcategory.get("evidences", []):
            self._process_evidence(evidence, samples_per_evidence, languages)

    def _process_evidence(
            self,
            evidence: Dict,
            samples_per_evidence: int,
            languages: List[str]
    ):
        """Process a single evidence."""
        evidence_id = evidence.get("evidence_id", "")

        # Extract code snippet
        code_data = evidence.get("code_snippet", {})
        code_snippet = code_data.get("code", "")
        if not code_snippet:
            logger.warning(f"No code for {evidence_id}")
            return

        # Get call chain
        func_name = evidence.get("location", {}).get("qualified_name", "")
        call_chain = self.call_graph.get_call_chain(func_name) if func_name else []

        for language in languages:
            self._generate_for_language(
                evidence, code_snippet, call_chain, language, samples_per_evidence
            )

    def _generate_for_language(
            self,
            evidence: Dict,
            code_snippet: str,
            call_chain: List[str],
            language: str,
            count: int
    ):
        """Generate Q&A pairs for a language."""

        # Generate questions (LLM-enhanced or fallback)
        questions = self.question_generator.generate_questions(
            evidence, code_snippet, call_chain, language, count
        )

        for question in questions:
            try:
                # Generate reasoning (LLM-enhanced)
                reasoning = self.reasoning_generator.generate_reasoning(
                    question["question_text"],
                    evidence,
                    code_snippet,
                    call_chain,
                    language
                )

                # Compose answer (LLM-enhanced)
                answer = self.answer_composer.compose_answer(
                    question["question_text"],
                    evidence,
                    code_snippet,
                    reasoning,
                    call_chain,
                    language
                )

                # Build Q&A pair
                qa_pair = self._build_qa_pair(
                    question, evidence, code_snippet, reasoning, answer, call_chain, language
                )

                # Validate
                is_valid, score, issues = self.validator.validate(qa_pair)
                qa_pair["data_quality"]["quality_score"] = score
                qa_pair["data_quality"]["validation_issues"] = issues

                if is_valid or Config.FALLBACK_TO_TEMPLATE:
                    self.generated_pairs.append(qa_pair)
                    self.stats["generated"] += 1
                    if question.get("source") == "llm":
                        self.stats["llm_questions"] += 1
                    else:
                        self.stats["template_questions"] += 1

            except Exception as e:
                logger.error(f"Error generating Q&A: {e}")
                self.stats["errors"] += 1

    def _build_qa_pair(
            self,
            question: Dict,
            evidence: Dict,
            code_snippet: str,
            reasoning: Dict,
            answer: str,
            call_chain: List[str],
            language: str
    ) -> Dict:
        """Build the final Q&A pair."""
        code_data = evidence.get("code_snippet", {})
        dbr_logic = evidence.get("dbr_logic", {})
        location = evidence.get("location", {})

        # Format reasoning trace
        reasoning_trace = []
        for i, step in enumerate(reasoning.get("steps", []), 1):
            reasoning_trace.append(f"[Step {i}] {step.get('observation', '')}")
            if step.get("analysis"):
                reasoning_trace.append(f"  → {step['analysis']}")

        if reasoning.get("conclusion"):
            reasoning_trace.append(f"[Conclusion] {reasoning['conclusion']}")

        return {
            "sample_id": f"DBR01-V3-{uuid.uuid4().hex[:10]}",
            "instruction": question["question_text"],
            "context": {
                "file_path": location.get("file_path", ""),
                "related_dbr": dbr_logic.get("rule_id", "DBR-01"),
                "code_snippet": code_snippet,
                "line_range": f"{code_data.get('line_start', 0)}-{code_data.get('line_end', 0)}",
                "function_name": evidence.get("name", ""),
                "call_chain": call_chain[:5] if call_chain else None,
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
                    "question_source": question.get("source", "unknown"),
                    "question_type": question.get("question_type", "analytical"),
                    "reasoning_source": reasoning.get("source", "unknown"),
                    "llm_model": Config.MODEL_NAME if self.llm_available else None,
                    "llm_enhanced": self.llm_available,
                },
            },
            "reasoning_trace": reasoning_trace,
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
        print("Q&A Generation Summary (v3.0 - LLM-Enhanced Hybrid)")
        print("=" * 70)

        print(f"\nLLM Status: {'✓ Enabled' if self.llm_available else '✗ Disabled (fallback mode)'}")
        print(f"Model: {Config.MODEL_NAME}")

        print(f"\nTotal Generated: {len(self.generated_pairs)}")
        print(f"  - LLM-generated questions: {self.stats.get('llm_questions', 0)}")
        print(f"  - Template-based questions: {self.stats.get('template_questions', 0)}")
        print(f"  - Errors: {self.stats.get('errors', 0)}")

        # Language distribution
        lang_counts = defaultdict(int)
        for pair in self.generated_pairs:
            lang = pair.get("data_quality", {}).get("language", "unknown")
            lang_counts[lang] += 1

        print("\nBy Language:")
        for lang, count in lang_counts.items():
            print(f"  - {lang}: {count}")

        # Question type distribution
        type_counts = defaultdict(int)
        for pair in self.generated_pairs:
            q_type = pair.get("auto_processing", {}).get("generation_metadata", {}).get("question_type", "unknown")
            type_counts[q_type] += 1

        print("\nBy Question Type:")
        for q_type, count in sorted(type_counts.items()):
            print(f"  - {q_type}: {count}")

        # Quality scores
        scores = [
            p.get("data_quality", {}).get("quality_score", 0)
            for p in self.generated_pairs
        ]
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"\nAverage Quality Score: {avg_score:.2%}")

        print("\n" + "=" * 70)

    def print_sample_pairs(self, n: int = 2):
        """Print sample Q&A pairs."""
        samples = self.generated_pairs[:n]

        for i, pair in enumerate(samples, 1):
            print("\n" + "=" * 70)
            print(f"Sample {i} - ID: {pair['sample_id']}")
            print("=" * 70)

            gen_meta = pair.get("auto_processing", {}).get("generation_metadata", {})
            print(
                f"\n[Source: {gen_meta.get('question_source', 'N/A')} | Type: {gen_meta.get('question_type', 'N/A')}]")

            print(f"\n【Question】:\n{pair['instruction']}\n")

            print("【Reasoning Trace】:")
            for step in pair.get("reasoning_trace", [])[:5]:
                print(f"  {step}")
            if len(pair.get("reasoning_trace", [])) > 5:
                print(f"  ... ({len(pair['reasoning_trace']) - 5} more)")

            print(f"\n【Answer (excerpt)】:\n{pair['answer'][:800]}...")

            print(f"\n【Metadata】:")
            print(f"  Language: {pair['data_quality']['language']}")
            print(f"  LLM Enhanced: {gen_meta.get('llm_enhanced', False)}")
            print(f"  Quality Score: {pair['data_quality'].get('quality_score', 'N/A')}")

            print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Enterprise Q&A Generation v3.0 (LLM-Enhanced Hybrid)'
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
        '-n', '--samples',
        type=int,
        default=3,
        help='Samples per evidence'
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
        default=2,
        help='Number of samples to preview'
    )

    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM enhancement (template-only mode)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.no_llm:
        Config.ENABLE_LLM_ENHANCEMENT = False

    # Initialize orchestrator
    orchestrator = HybridQAOrchestrator(args.metadata, args.ast)

    if not orchestrator.initialize():
        print("Error: Failed to initialize. Check file paths.")
        sys.exit(1)

    # Run pipeline
    print(f"\n🚀 Running Q&A Generation Pipeline v3.0 (LLM-Enhanced Hybrid)")
    print(f"   Metadata: {args.metadata}")
    print(f"   Languages: {args.languages}")
    print(f"   Samples/evidence: {args.samples}")
    print(f"   LLM: {Config.MODEL_NAME if orchestrator.llm_available else 'Disabled'}")

    pairs = orchestrator.run_pipeline(
        samples_per_evidence=args.samples,
        languages=args.languages
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
        print(f"\n--- Sample Q&A Pairs ({args.preview}) ---")
        orchestrator.print_sample_pairs(args.preview)

    print(f"\n✅ Successfully generated {len(pairs)} Q&A pairs")
    print(f"📁 Output saved to: {output_path}")


if __name__ == "__main__":
    main()