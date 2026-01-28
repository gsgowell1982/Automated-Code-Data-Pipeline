#!/usr/bin/env python3
"""
Enterprise-Grade Q&A Generation Engine v2.0 for LLM Fine-tuning

Enhanced with:
1. CoT Reasoning Engine - Simulates human developer thinking:
   "Read Code → Understand Logic → Associate Rules → Draw Conclusions"
2. Call Graph Integration - Uses AST call chain for logical tracing
3. Orchestrator Class - Coordinates all components with pipeline pattern
4. Dynamic Augmentation - Vocabulary perturbation for human-like variation

Target: Qwen 2.5 series model training and fine-tuning

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                     QAOrchestrator (Main Controller)                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │
│  │ CallGraphEngine│  │ ReasoningEngine│  │ DynamicAugmenter       │ │
│  │ - Parse calls  │  │ - CoT generator│  │ - Synonym substitution │ │
│  │ - Build chain  │  │ - Think process│  │ - Phrase variation     │ │
│  │ - Trace logic  │  │ - Rule mapping │  │ - Context adaptation   │ │
│  └────────────────┘  └────────────────┘  └────────────────────────┘ │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │
│  │QuestionFactory │  │ AnswerComposer │  │ QualityGate            │ │
│  │ - Templates    │  │ - Gold standard│  │ - Multi-validation     │ │
│  │ - Augmentation │  │ - CoT + Code   │  │ - Hash verification    │ │
│  └────────────────┘  └────────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

Author: Auto-generated
Version: 2.0.0
"""

import os
import sys
import json
import uuid
import hashlib
import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Iterator
from collections import defaultdict
from itertools import chain
import copy

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
    """Global configuration for Q&A generation v2."""
    VERSION = "2.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).parent.resolve()
    WORKSPACE_ROOT = BASE_DIR.parent.parent
    DATA_DIR = WORKSPACE_ROOT / "data"
    REPO_PATH = WORKSPACE_ROOT / "repos" / "fastapi-realworld-example-app"
    
    # Input/Output
    RULE_METADATA_FILE = DATA_DIR / "dbr01_rule_metadata.json"
    AST_ANALYSIS_FILE = DATA_DIR / "fastapi_analysis_result.json"
    OUTPUT_FILE = DATA_DIR / "qwen_dbr_training_logic_v2.jsonl"
    
    # Generation parameters
    DEFAULT_TEMPERATURE = 0.7
    SUPPORTED_LANGUAGES = ["en", "zh"]
    MAX_SAMPLES_PER_EVIDENCE = 5
    MIN_QUESTION_LENGTH = 10
    MIN_ANSWER_LENGTH = 50
    AUGMENTATION_FACTOR = 3  # Number of variations per template
    ENABLE_CALL_GRAPH = True
    COT_DEPTH = 5  # Maximum depth of reasoning chain


# ============================================================================
# Enums and Data Classes
# ============================================================================

class ThinkingPhase(str, Enum):
    """Phases in the human-like thinking process."""
    READ_CODE = "read_code"           # 阅读代码
    IDENTIFY_PATTERN = "identify"      # 识别模式
    UNDERSTAND_LOGIC = "understand"    # 理解逻辑
    TRACE_CALLS = "trace_calls"        # 追踪调用
    ASSOCIATE_RULES = "associate"      # 联想规则
    EVALUATE_SECURITY = "evaluate"     # 评估安全性
    DRAW_CONCLUSION = "conclude"       # 得出结论


class QuestionType(str, Enum):
    """Types of questions to generate."""
    WHAT = "what"
    HOW = "how"
    WHY = "why"
    WHEN = "when"
    WHICH = "which"
    COMPARE = "compare"
    SCENARIO = "scenario"
    DEBUG = "debug"
    SECURITY = "security"
    TRACE = "trace"  # New: Call graph tracing questions


class QuestionPerspective(str, Enum):
    """Perspectives for question generation."""
    DEVELOPER = "developer"
    SECURITY_AUDITOR = "security_auditor"
    ARCHITECT = "architect"
    CODE_REVIEWER = "code_reviewer"
    QA_ENGINEER = "qa_engineer"
    COMPLIANCE_OFFICER = "compliance_officer"
    TECH_LEAD = "tech_lead"  # New perspective


@dataclass
class CallNode:
    """A node in the call graph."""
    function_name: str
    qualified_name: str
    file_path: str
    line_number: int
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    local_variables: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)


@dataclass
class CallChain:
    """A chain of function calls for reasoning."""
    chain_id: str
    nodes: List[CallNode]
    entry_point: str
    exit_point: str
    depth: int
    security_relevant: bool = False


@dataclass
class ThinkingStep:
    """A single step in the CoT reasoning process."""
    step_id: int
    phase: ThinkingPhase
    observation: str
    observation_cn: str
    reasoning: str
    reasoning_cn: str
    evidence_ref: Optional[str] = None
    code_ref: Optional[str] = None
    call_trace: Optional[List[str]] = None


@dataclass 
class CoTReasoning:
    """Complete Chain-of-Thought reasoning."""
    reasoning_id: str
    thinking_steps: List[ThinkingStep]
    call_chain: Optional[CallChain]
    dbr_mappings: List[str]
    final_insight: str
    final_insight_cn: str
    confidence: float = 1.0


# ============================================================================
# Dynamic Augmentation System
# ============================================================================

class SynonymLibrary:
    """Enterprise-grade synonym and phrase variation library."""
    
    # Technical action synonyms
    ACTIONS = {
        "validate": ["verify", "check", "confirm", "ensure", "assert"],
        "execute": ["perform", "run", "carry out", "invoke", "trigger"],
        "return": ["respond with", "send back", "provide", "yield", "output"],
        "prevent": ["block", "stop", "prohibit", "disallow", "reject"],
        "check": ["examine", "inspect", "verify", "validate", "assess"],
        "create": ["generate", "produce", "construct", "build", "instantiate"],
        "handle": ["manage", "process", "deal with", "address", "take care of"],
        "store": ["persist", "save", "record", "keep", "maintain"],
        "protect": ["secure", "safeguard", "shield", "defend", "guard"],
    }
    
    ACTIONS_CN = {
        "验证": ["校验", "检查", "确认", "核实", "断言"],
        "执行": ["运行", "进行", "触发", "调用", "实施"],
        "返回": ["响应", "输出", "提供", "发送", "给出"],
        "防止": ["阻止", "拦截", "禁止", "拒绝", "阻断"],
        "检查": ["检验", "审查", "核查", "验证", "评估"],
        "创建": ["生成", "构建", "产生", "建立", "实例化"],
        "处理": ["管理", "应对", "解决", "操作", "处置"],
        "存储": ["持久化", "保存", "记录", "保持", "维护"],
        "保护": ["保障", "防护", "守护", "确保安全", "维护"],
    }
    
    # Technical nouns synonyms
    NOUNS = {
        "error": ["exception", "fault", "failure", "issue", "problem"],
        "user": ["account holder", "client", "end user", "subscriber", "member"],
        "request": ["call", "query", "submission", "operation", "transaction"],
        "response": ["reply", "answer", "result", "output", "feedback"],
        "token": ["credential", "access key", "authentication token", "session key"],
        "password": ["credential", "secret", "passphrase", "authentication key"],
        "database": ["data store", "persistence layer", "repository", "storage"],
        "function": ["method", "routine", "procedure", "handler", "operation"],
    }
    
    NOUNS_CN = {
        "错误": ["异常", "故障", "失败", "问题", "缺陷"],
        "用户": ["账户持有人", "客户端", "终端用户", "订阅者", "成员"],
        "请求": ["调用", "查询", "提交", "操作", "事务"],
        "响应": ["回复", "答复", "结果", "输出", "反馈"],
        "令牌": ["凭证", "访问密钥", "认证令牌", "会话密钥"],
        "密码": ["凭据", "密钥", "口令", "认证密钥"],
        "数据库": ["数据存储", "持久层", "仓库", "存储层"],
        "函数": ["方法", "例程", "过程", "处理器", "操作"],
    }
    
    # Phrase templates for variation
    PHRASE_TEMPLATES = {
        "how_does": [
            "How does {subject} {action}?",
            "What is the mechanism by which {subject} {action}?",
            "In what way does {subject} {action}?",
            "Can you explain how {subject} {action}?",
            "What approach does {subject} use to {action}?",
        ],
        "what_happens": [
            "What happens when {condition}?",
            "What is the result when {condition}?",
            "What occurs if {condition}?",
            "Describe the outcome when {condition}.",
            "What takes place when {condition}?",
        ],
        "why_is": [
            "Why is {subject} {characteristic}?",
            "What is the reason that {subject} {characteristic}?",
            "For what purpose is {subject} {characteristic}?",
            "What justifies {subject} being {characteristic}?",
            "Can you explain why {subject} {characteristic}?",
        ],
    }
    
    PHRASE_TEMPLATES_CN = {
        "how_does": [
            "{subject}如何{action}？",
            "{subject}通过什么机制{action}？",
            "{subject}以什么方式{action}？",
            "请解释{subject}如何{action}。",
            "{subject}采用什么方法来{action}？",
        ],
        "what_happens": [
            "当{condition}时会发生什么？",
            "{condition}时的结果是什么？",
            "如果{condition}会怎样？",
            "描述{condition}时的结果。",
            "{condition}时会出现什么情况？",
        ],
        "why_is": [
            "为什么{subject}{characteristic}？",
            "{subject}{characteristic}的原因是什么？",
            "{subject}{characteristic}的目的是什么？",
            "是什么使得{subject}{characteristic}？",
            "请解释为什么{subject}{characteristic}。",
        ],
    }
    
    # Connector phrases for reasoning
    CONNECTORS = {
        "therefore": ["thus", "hence", "consequently", "as a result", "accordingly"],
        "because": ["since", "given that", "due to", "as", "owing to"],
        "first": ["initially", "to begin with", "at the outset", "primarily", "firstly"],
        "then": ["subsequently", "afterwards", "next", "following that", "thereafter"],
        "finally": ["ultimately", "in conclusion", "lastly", "to conclude", "in the end"],
    }
    
    CONNECTORS_CN = {
        "因此": ["所以", "从而", "由此", "结果是", "相应地"],
        "因为": ["由于", "鉴于", "基于", "考虑到", "源于"],
        "首先": ["起初", "一开始", "第一步", "最初", "先"],
        "然后": ["接着", "随后", "之后", "紧接着", "继而"],
        "最后": ["最终", "总之", "归结", "总结来说", "终究"],
    }


class DynamicAugmenter:
    """
    Enterprise-grade dynamic variable augmentation system.
    
    Implements vocabulary perturbation to make generated sentences
    more human-like through:
    1. Synonym substitution
    2. Phrase template variation
    3. Connector variation
    4. Context-aware adaptation
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.library = SynonymLibrary()
        self.rng = random.Random(seed)
        self._usage_counts: Dict[str, int] = defaultdict(int)
    
    def augment_variables(
        self,
        variables: Dict[str, str],
        language: str = "en",
        variation_level: float = 0.5
    ) -> Dict[str, str]:
        """
        Augment variables with synonym substitution.
        
        Args:
            variables: Original variable mappings
            language: Target language (en/zh)
            variation_level: 0.0-1.0, higher means more variation
        """
        augmented = variables.copy()
        
        synonyms = self.library.ACTIONS if language == "en" else self.library.ACTIONS_CN
        noun_synonyms = self.library.NOUNS if language == "en" else self.library.NOUNS_CN
        
        for key, value in augmented.items():
            if self.rng.random() < variation_level:
                # Try action substitution
                for base_word, alternatives in synonyms.items():
                    if base_word.lower() in value.lower():
                        replacement = self._select_weighted(alternatives)
                        augmented[key] = re.sub(
                            re.escape(base_word), 
                            replacement, 
                            value, 
                            flags=re.IGNORECASE
                        )
                        break
                
                # Try noun substitution
                for base_word, alternatives in noun_synonyms.items():
                    if base_word.lower() in value.lower():
                        replacement = self._select_weighted(alternatives)
                        augmented[key] = re.sub(
                            re.escape(base_word),
                            replacement,
                            augmented[key],
                            flags=re.IGNORECASE
                        )
                        break
        
        return augmented
    
    def generate_question_variations(
        self,
        base_question: str,
        question_type: str,
        language: str = "en",
        count: int = 3
    ) -> List[str]:
        """
        Generate multiple variations of a question.
        """
        variations = [base_question]
        
        templates = (
            self.library.PHRASE_TEMPLATES if language == "en" 
            else self.library.PHRASE_TEMPLATES_CN
        )
        
        # Extract components from base question
        components = self._extract_question_components(base_question, language)
        
        # Generate variations using different templates
        if question_type == "how" and "how_does" in templates:
            for template in self.rng.sample(templates["how_does"], min(count, len(templates["how_does"]))):
                try:
                    variation = template.format(**components)
                    if variation != base_question:
                        variations.append(variation)
                except KeyError:
                    pass
        
        elif question_type == "what" and "what_happens" in templates:
            for template in self.rng.sample(templates["what_happens"], min(count, len(templates["what_happens"]))):
                try:
                    variation = template.format(**components)
                    if variation != base_question:
                        variations.append(variation)
                except KeyError:
                    pass
        
        elif question_type == "why" and "why_is" in templates:
            for template in self.rng.sample(templates["why_is"], min(count, len(templates["why_is"]))):
                try:
                    variation = template.format(**components)
                    if variation != base_question:
                        variations.append(variation)
                except KeyError:
                    pass
        
        return variations[:count + 1]
    
    def augment_reasoning_connectors(
        self,
        reasoning_text: str,
        language: str = "en"
    ) -> str:
        """Vary connector words in reasoning text."""
        connectors = (
            self.library.CONNECTORS if language == "en"
            else self.library.CONNECTORS_CN
        )
        
        result = reasoning_text
        for base_connector, alternatives in connectors.items():
            if base_connector.lower() in result.lower():
                if self.rng.random() < 0.5:  # 50% chance to vary
                    replacement = self._select_weighted(alternatives)
                    result = re.sub(
                        r'\b' + re.escape(base_connector) + r'\b',
                        replacement,
                        result,
                        flags=re.IGNORECASE,
                        count=1
                    )
        
        return result
    
    def _select_weighted(self, alternatives: List[str]) -> str:
        """Select alternative with weighted probability based on usage."""
        # Prefer less-used alternatives for diversity
        weights = []
        for alt in alternatives:
            count = self._usage_counts[alt]
            weight = 1.0 / (1 + count)  # Inverse usage weighting
            weights.append(weight)
        
        total = sum(weights)
        weights = [w / total for w in weights]
        
        selected = self.rng.choices(alternatives, weights=weights, k=1)[0]
        self._usage_counts[selected] += 1
        return selected
    
    def _extract_question_components(
        self,
        question: str,
        language: str
    ) -> Dict[str, str]:
        """Extract subject, action, condition from question."""
        components = {
            "subject": "the system",
            "action": "perform this operation",
            "condition": "this occurs",
            "characteristic": "designed this way",
        }
        
        if language == "en":
            # Simple extraction heuristics
            if " the " in question:
                parts = question.split(" the ")
                if len(parts) > 1:
                    components["subject"] = "the " + parts[1].split()[0]
            
            for action_word in ["handle", "process", "validate", "check", "return"]:
                if action_word in question.lower():
                    components["action"] = action_word + " " + question.split(action_word)[-1].split("?")[0].strip()
                    break
        else:
            # Chinese extraction heuristics
            if "系统" in question:
                components["subject"] = "系统"
            if "如何" in question:
                components["action"] = question.split("如何")[-1].split("？")[0].strip()
        
        return components


# ============================================================================
# Call Graph Engine
# ============================================================================

class CallGraphEngine:
    """
    Builds and analyzes call graphs from AST analysis data.
    
    Enables reasoning about:
    - Function call chains
    - Data flow paths
    - Security-relevant call sequences
    """
    
    def __init__(self, ast_analysis_path: Optional[str] = None):
        self.ast_data: Dict = {}
        self.call_nodes: Dict[str, CallNode] = {}
        self.call_chains: List[CallChain] = []
        
        if ast_analysis_path:
            self.load_ast_analysis(ast_analysis_path)
    
    def load_ast_analysis(self, path: str) -> bool:
        """Load AST analysis data."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"AST analysis file not found: {path}")
            return False
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.ast_data = json.load(f)
            self._build_call_graph()
            logger.info(f"Loaded AST analysis with {len(self.call_nodes)} functions")
            return True
        except Exception as e:
            logger.error(f"Error loading AST analysis: {e}")
            return False
    
    def _build_call_graph(self):
        """Build call graph from AST data."""
        # Index all functions
        for module in self.ast_data.get("modules", []):
            file_path = module.get("file_path", "")
            
            # Process module-level functions
            for func in module.get("functions", []):
                node = self._create_call_node(func, file_path)
                self.call_nodes[node.qualified_name] = node
            
            # Process class methods
            for cls in module.get("classes", []):
                for method in cls.get("methods", []):
                    node = self._create_call_node(method, file_path)
                    self.call_nodes[node.qualified_name] = node
        
        # Build reverse call relationships
        for qualified_name, node in self.call_nodes.items():
            for call in node.calls:
                # Find the called function
                for qn, target_node in self.call_nodes.items():
                    if call in qn or qn.endswith(f".{call}"):
                        if qualified_name not in target_node.called_by:
                            target_node.called_by.append(qualified_name)
    
    def _create_call_node(self, func_data: Dict, file_path: str) -> CallNode:
        """Create a call node from function data."""
        return CallNode(
            function_name=func_data.get("name", ""),
            qualified_name=func_data.get("qualified_name", ""),
            file_path=file_path,
            line_number=func_data.get("line_start", 0),
            calls=func_data.get("calls", []),
            local_variables=func_data.get("local_variables", []),
            parameters=[p.get("name", "") for p in func_data.get("parameters", [])],
        )
    
    def get_call_chain(
        self,
        start_function: str,
        max_depth: int = 5,
        security_filter: bool = True
    ) -> Optional[CallChain]:
        """
        Build a call chain starting from a function.
        
        Args:
            start_function: Starting function name or qualified name
            max_depth: Maximum depth of call chain
            security_filter: Only include security-relevant calls
        """
        # Find the starting node
        start_node = None
        for qn, node in self.call_nodes.items():
            if start_function in qn or node.function_name == start_function:
                start_node = node
                break
        
        if not start_node:
            return None
        
        # Build chain through BFS
        chain_nodes = [start_node]
        visited = {start_node.qualified_name}
        queue = [(start_node, 1)]
        
        security_keywords = {
            "password", "token", "auth", "login", "user", "check",
            "validate", "hash", "encrypt", "session", "credential"
        }
        
        while queue and len(chain_nodes) < max_depth:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            
            for call in current.calls:
                # Find the called function
                for qn, target_node in self.call_nodes.items():
                    if call in qn or qn.endswith(f".{call}"):
                        if qn not in visited:
                            # Apply security filter
                            if security_filter:
                                is_security_relevant = any(
                                    kw in qn.lower() or kw in call.lower()
                                    for kw in security_keywords
                                )
                                if not is_security_relevant:
                                    continue
                            
                            visited.add(qn)
                            chain_nodes.append(target_node)
                            queue.append((target_node, depth + 1))
        
        if len(chain_nodes) <= 1:
            return None
        
        return CallChain(
            chain_id=f"CC-{uuid.uuid4().hex[:8]}",
            nodes=chain_nodes,
            entry_point=chain_nodes[0].qualified_name,
            exit_point=chain_nodes[-1].qualified_name,
            depth=len(chain_nodes),
            security_relevant=True,
        )
    
    def get_callers(self, function_name: str) -> List[CallNode]:
        """Get all functions that call the given function."""
        callers = []
        for qn, node in self.call_nodes.items():
            if function_name in qn or node.function_name == function_name:
                for caller_qn in node.called_by:
                    if caller_qn in self.call_nodes:
                        callers.append(self.call_nodes[caller_qn])
        return callers
    
    def get_callees(self, function_name: str) -> List[str]:
        """Get all functions called by the given function."""
        for qn, node in self.call_nodes.items():
            if function_name in qn or node.function_name == function_name:
                return node.calls
        return []
    
    def format_call_chain_for_reasoning(
        self,
        chain: CallChain,
        language: str = "en"
    ) -> List[str]:
        """Format call chain as reasoning steps."""
        steps = []
        
        for i, node in enumerate(chain.nodes):
            if language == "en":
                if i == 0:
                    steps.append(f"Entry point: {node.function_name} in {node.file_path}")
                else:
                    steps.append(f"  → Calls: {node.function_name} (line {node.line_number})")
            else:
                if i == 0:
                    steps.append(f"入口点：{node.function_name}，位于 {node.file_path}")
                else:
                    steps.append(f"  → 调用：{node.function_name}（第 {node.line_number} 行）")
        
        return steps


# ============================================================================
# Enhanced Reasoning Engine (CoT Generator)
# ============================================================================

class CoTReasoningEngine:
    """
    Chain-of-Thought Reasoning Engine.
    
    Simulates human developer thinking process:
    1. Read Code - Observe the code structure
    2. Identify Pattern - Recognize design patterns
    3. Understand Logic - Comprehend the business logic
    4. Trace Calls - Follow the execution path
    5. Associate Rules - Map to design rules
    6. Evaluate Security - Assess security implications
    7. Draw Conclusion - Synthesize insights
    """
    
    # Phase descriptions for generating reasoning
    PHASE_DESCRIPTIONS = {
        ThinkingPhase.READ_CODE: {
            "en": "Reading and observing the code structure",
            "zh": "阅读和观察代码结构",
            "action_en": "I observe that",
            "action_zh": "我观察到",
        },
        ThinkingPhase.IDENTIFY_PATTERN: {
            "en": "Identifying the design pattern used",
            "zh": "识别所使用的设计模式",
            "action_en": "I identify",
            "action_zh": "我识别出",
        },
        ThinkingPhase.UNDERSTAND_LOGIC: {
            "en": "Understanding the business logic",
            "zh": "理解业务逻辑",
            "action_en": "I understand that",
            "action_zh": "我理解",
        },
        ThinkingPhase.TRACE_CALLS: {
            "en": "Tracing the function call chain",
            "zh": "追踪函数调用链",
            "action_en": "Following the call chain, I trace",
            "action_zh": "沿着调用链追踪，我发现",
        },
        ThinkingPhase.ASSOCIATE_RULES: {
            "en": "Associating with design business rules",
            "zh": "关联设计业务规则",
            "action_en": "This maps to DBR rule",
            "action_zh": "这对应到DBR规则",
        },
        ThinkingPhase.EVALUATE_SECURITY: {
            "en": "Evaluating security implications",
            "zh": "评估安全影响",
            "action_en": "From a security perspective",
            "action_zh": "从安全角度来看",
        },
        ThinkingPhase.DRAW_CONCLUSION: {
            "en": "Drawing conclusions",
            "zh": "得出结论",
            "action_en": "Therefore, I conclude that",
            "action_zh": "因此，我得出结论",
        },
    }
    
    def __init__(self, call_graph_engine: Optional[CallGraphEngine] = None):
        self.call_graph = call_graph_engine
        self.augmenter = DynamicAugmenter()
    
    def generate_cot_reasoning(
        self,
        evidence: Dict,
        call_chain: Optional[CallChain] = None,
        language: str = "en",
        depth: int = 5
    ) -> CoTReasoning:
        """
        Generate Chain-of-Thought reasoning for an evidence.
        
        Simulates: Read Code → Understand → Trace → Associate → Conclude
        """
        thinking_steps = []
        dbr_mappings = []
        
        # Get evidence details
        evidence_type = evidence.get("evidence_type", "pattern")
        evidence_name = evidence.get("name", "")
        description = evidence.get("description", "") if language == "en" else evidence.get("description_cn", "")
        related_elements = evidence.get("related_elements", [])
        code_snippet = evidence.get("code_snippet", {})
        dbr_logic = evidence.get("dbr_logic", {})
        
        step_id = 0
        
        # Phase 1: READ_CODE
        step_id += 1
        thinking_steps.append(self._create_read_code_step(
            step_id, evidence_name, code_snippet, evidence_type, language
        ))
        
        # Phase 2: IDENTIFY_PATTERN
        step_id += 1
        thinking_steps.append(self._create_identify_pattern_step(
            step_id, evidence_type, related_elements, language
        ))
        
        # Phase 3: UNDERSTAND_LOGIC
        step_id += 1
        thinking_steps.append(self._create_understand_logic_step(
            step_id, description, evidence_name, language
        ))
        
        # Phase 4: TRACE_CALLS (if call chain available)
        if call_chain and len(call_chain.nodes) > 1:
            step_id += 1
            thinking_steps.append(self._create_trace_calls_step(
                step_id, call_chain, language
            ))
        
        # Phase 5: ASSOCIATE_RULES
        step_id += 1
        dbr_step = self._create_associate_rules_step(
            step_id, dbr_logic, language
        )
        thinking_steps.append(dbr_step)
        if dbr_logic.get("rule_id"):
            dbr_mappings.append(dbr_logic["rule_id"])
        if dbr_logic.get("subcategory_id"):
            dbr_mappings.append(dbr_logic["subcategory_id"])
        
        # Phase 6: EVALUATE_SECURITY (for security-relevant evidence)
        if self._is_security_relevant(evidence):
            step_id += 1
            thinking_steps.append(self._create_evaluate_security_step(
                step_id, evidence, language
            ))
        
        # Phase 7: DRAW_CONCLUSION
        step_id += 1
        thinking_steps.append(self._create_conclusion_step(
            step_id, evidence, dbr_logic, language
        ))
        
        # Generate final insight
        final_insight, final_insight_cn = self._generate_final_insight(
            evidence, dbr_logic, thinking_steps
        )
        
        return CoTReasoning(
            reasoning_id=f"COT-{uuid.uuid4().hex[:8]}",
            thinking_steps=thinking_steps,
            call_chain=call_chain,
            dbr_mappings=list(set(dbr_mappings)),
            final_insight=final_insight,
            final_insight_cn=final_insight_cn,
            confidence=self._calculate_confidence(thinking_steps),
        )
    
    def _create_read_code_step(
        self,
        step_id: int,
        evidence_name: str,
        code_snippet: Dict,
        evidence_type: str,
        language: str
    ) -> ThinkingStep:
        """Create the READ_CODE thinking step."""
        phase_info = self.PHASE_DESCRIPTIONS[ThinkingPhase.READ_CODE]
        
        file_path = code_snippet.get("file_path", "")
        line_start = code_snippet.get("line_start", 0)
        line_end = code_snippet.get("line_end", 0)
        
        if language == "en":
            observation = f"{phase_info['action_en']} the {evidence_type} '{evidence_name}' in {file_path} (lines {line_start}-{line_end})"
            reasoning = f"This code block contains {evidence_type} implementation that needs analysis"
        else:
            observation = f"{phase_info['action_zh']} {file_path} 中的 {evidence_type} '{evidence_name}'（第 {line_start}-{line_end} 行）"
            reasoning = f"这段代码块包含需要分析的 {evidence_type} 实现"
        
        return ThinkingStep(
            step_id=step_id,
            phase=ThinkingPhase.READ_CODE,
            observation=observation if language == "en" else "",
            observation_cn=observation if language == "zh" else "",
            reasoning=reasoning if language == "en" else "",
            reasoning_cn=reasoning if language == "zh" else "",
            code_ref=f"{file_path}:{line_start}-{line_end}",
        )
    
    def _create_identify_pattern_step(
        self,
        step_id: int,
        evidence_type: str,
        related_elements: List[str],
        language: str
    ) -> ThinkingStep:
        """Create the IDENTIFY_PATTERN thinking step."""
        phase_info = self.PHASE_DESCRIPTIONS[ThinkingPhase.IDENTIFY_PATTERN]
        
        elements_str = ", ".join(related_elements[:5]) if related_elements else "key elements"
        
        pattern_mapping = {
            "pattern": ("a validation/check pattern", "验证/检查模式"),
            "exception_handling": ("an exception handling pattern", "异常处理模式"),
            "call": ("a function call pattern", "函数调用模式"),
            "function": ("a utility function pattern", "工具函数模式"),
        }
        
        pattern_desc = pattern_mapping.get(evidence_type, ("a code pattern", "代码模式"))
        
        if language == "en":
            observation = f"{phase_info['action_en']} {pattern_desc[0]} involving: {elements_str}"
            reasoning = f"This pattern indicates a specific design decision for handling {evidence_type}"
        else:
            observation = f"{phase_info['action_zh']} {pattern_desc[1]}，涉及：{elements_str}"
            reasoning = f"这种模式表明了处理 {evidence_type} 的特定设计决策"
        
        return ThinkingStep(
            step_id=step_id,
            phase=ThinkingPhase.IDENTIFY_PATTERN,
            observation=observation if language == "en" else "",
            observation_cn=observation if language == "zh" else "",
            reasoning=reasoning if language == "en" else "",
            reasoning_cn=reasoning if language == "zh" else "",
            evidence_ref=evidence_type,
        )
    
    def _create_understand_logic_step(
        self,
        step_id: int,
        description: str,
        evidence_name: str,
        language: str
    ) -> ThinkingStep:
        """Create the UNDERSTAND_LOGIC thinking step."""
        phase_info = self.PHASE_DESCRIPTIONS[ThinkingPhase.UNDERSTAND_LOGIC]
        
        # Truncate description if too long
        desc_short = description[:200] + "..." if len(description) > 200 else description
        
        if language == "en":
            observation = f"{phase_info['action_en']} {desc_short}"
            reasoning = f"The '{evidence_name}' serves a specific business purpose in the authentication flow"
        else:
            observation = f"{phase_info['action_zh']} {desc_short}"
            reasoning = f"'{evidence_name}' 在认证流程中服务于特定的业务目的"
        
        return ThinkingStep(
            step_id=step_id,
            phase=ThinkingPhase.UNDERSTAND_LOGIC,
            observation=observation if language == "en" else "",
            observation_cn=observation if language == "zh" else "",
            reasoning=reasoning if language == "en" else "",
            reasoning_cn=reasoning if language == "zh" else "",
        )
    
    def _create_trace_calls_step(
        self,
        step_id: int,
        call_chain: CallChain,
        language: str
    ) -> ThinkingStep:
        """Create the TRACE_CALLS thinking step."""
        phase_info = self.PHASE_DESCRIPTIONS[ThinkingPhase.TRACE_CALLS]
        
        # Format call chain
        chain_str = " → ".join([n.function_name for n in call_chain.nodes])
        
        if language == "en":
            observation = f"{phase_info['action_en']} the execution path: {chain_str}"
            reasoning = f"This call chain shows how data flows through {len(call_chain.nodes)} functions"
        else:
            observation = f"{phase_info['action_zh']} 执行路径：{chain_str}"
            reasoning = f"这条调用链显示数据如何流经 {len(call_chain.nodes)} 个函数"
        
        return ThinkingStep(
            step_id=step_id,
            phase=ThinkingPhase.TRACE_CALLS,
            observation=observation if language == "en" else "",
            observation_cn=observation if language == "zh" else "",
            reasoning=reasoning if language == "en" else "",
            reasoning_cn=reasoning if language == "zh" else "",
            call_trace=[n.qualified_name for n in call_chain.nodes],
        )
    
    def _create_associate_rules_step(
        self,
        step_id: int,
        dbr_logic: Dict,
        language: str
    ) -> ThinkingStep:
        """Create the ASSOCIATE_RULES thinking step."""
        phase_info = self.PHASE_DESCRIPTIONS[ThinkingPhase.ASSOCIATE_RULES]
        
        rule_id = dbr_logic.get("rule_id", "DBR-01")
        subcategory_id = dbr_logic.get("subcategory_id", "")
        weight = dbr_logic.get("weight", 1.0)
        trigger_type = dbr_logic.get("trigger_type", "explicit")
        
        if language == "en":
            observation = f"{phase_info['action_en']} {rule_id} ({subcategory_id}) with confidence {weight:.0%}"
            reasoning = f"The {trigger_type} trigger matches the DBR criteria for authentication integrity"
        else:
            observation = f"{phase_info['action_zh']} {rule_id}（{subcategory_id}），置信度 {weight:.0%}"
            reasoning = f"{trigger_type} 触发器符合认证完整性的 DBR 标准"
        
        return ThinkingStep(
            step_id=step_id,
            phase=ThinkingPhase.ASSOCIATE_RULES,
            observation=observation if language == "en" else "",
            observation_cn=observation if language == "zh" else "",
            reasoning=reasoning if language == "en" else "",
            reasoning_cn=reasoning if language == "zh" else "",
        )
    
    def _create_evaluate_security_step(
        self,
        step_id: int,
        evidence: Dict,
        language: str
    ) -> ThinkingStep:
        """Create the EVALUATE_SECURITY thinking step."""
        phase_info = self.PHASE_DESCRIPTIONS[ThinkingPhase.EVALUATE_SECURITY]
        
        evidence_type = evidence.get("evidence_type", "pattern")
        
        security_implications = {
            "exception_handling": (
                "this prevents information leakage through error messages",
                "这防止了通过错误消息泄露信息"
            ),
            "pattern": (
                "this enforces security controls at the application layer",
                "这在应用层强制执行安全控制"
            ),
            "call": (
                "this ensures secure session management",
                "这确保了安全的会话管理"
            ),
        }
        
        implication = security_implications.get(
            evidence_type, 
            ("this has security implications", "这具有安全影响")
        )
        
        if language == "en":
            observation = f"{phase_info['action_en']}, {implication[0]}"
            reasoning = "Security-conscious design reduces attack surface and protects user data"
        else:
            observation = f"{phase_info['action_zh']}，{implication[1]}"
            reasoning = "注重安全的设计减少了攻击面并保护用户数据"
        
        return ThinkingStep(
            step_id=step_id,
            phase=ThinkingPhase.EVALUATE_SECURITY,
            observation=observation if language == "en" else "",
            observation_cn=observation if language == "zh" else "",
            reasoning=reasoning if language == "en" else "",
            reasoning_cn=reasoning if language == "zh" else "",
        )
    
    def _create_conclusion_step(
        self,
        step_id: int,
        evidence: Dict,
        dbr_logic: Dict,
        language: str
    ) -> ThinkingStep:
        """Create the DRAW_CONCLUSION thinking step."""
        phase_info = self.PHASE_DESCRIPTIONS[ThinkingPhase.DRAW_CONCLUSION]
        
        evidence_name = evidence.get("name", "")
        rule_id = dbr_logic.get("rule_id", "DBR-01")
        
        if language == "en":
            observation = f"{phase_info['action_en']} '{evidence_name}' correctly implements {rule_id} requirements"
            reasoning = "The implementation follows security best practices and design rule compliance"
        else:
            observation = f"{phase_info['action_zh']} '{evidence_name}' 正确实现了 {rule_id} 的要求"
            reasoning = "该实现遵循安全最佳实践和设计规则合规性"
        
        return ThinkingStep(
            step_id=step_id,
            phase=ThinkingPhase.DRAW_CONCLUSION,
            observation=observation if language == "en" else "",
            observation_cn=observation if language == "zh" else "",
            reasoning=reasoning if language == "en" else "",
            reasoning_cn=reasoning if language == "zh" else "",
        )
    
    def _is_security_relevant(self, evidence: Dict) -> bool:
        """Check if evidence is security-relevant."""
        security_keywords = {
            "password", "token", "auth", "login", "security",
            "credential", "hash", "encrypt", "validate", "check"
        }
        
        name = evidence.get("name", "").lower()
        desc = evidence.get("description", "").lower()
        
        return any(kw in name or kw in desc for kw in security_keywords)
    
    def _generate_final_insight(
        self,
        evidence: Dict,
        dbr_logic: Dict,
        steps: List[ThinkingStep]
    ) -> Tuple[str, str]:
        """Generate final insight from reasoning steps."""
        evidence_name = evidence.get("name", "")
        rule_id = dbr_logic.get("rule_id", "DBR-01")
        subcategory = dbr_logic.get("subcategory_id", "")
        
        insight_en = (
            f"Through systematic code analysis and rule mapping, we conclude that "
            f"'{evidence_name}' is a critical implementation of {rule_id} ({subcategory}), "
            f"ensuring authentication integrity and security compliance."
        )
        
        insight_cn = (
            f"通过系统性的代码分析和规则映射，我们得出结论：'{evidence_name}' "
            f"是 {rule_id}（{subcategory}）的关键实现，确保了认证完整性和安全合规性。"
        )
        
        return insight_en, insight_cn
    
    def _calculate_confidence(self, steps: List[ThinkingStep]) -> float:
        """Calculate confidence score based on reasoning completeness."""
        # Base confidence
        confidence = 0.7
        
        # Add for each phase covered
        phases_covered = set(s.phase for s in steps)
        phase_bonus = len(phases_covered) * 0.05
        
        # Add for call trace presence
        has_call_trace = any(s.call_trace for s in steps)
        call_bonus = 0.1 if has_call_trace else 0
        
        return min(1.0, confidence + phase_bonus + call_bonus)
    
    def format_cot_as_reasoning_trace(
        self,
        cot: CoTReasoning,
        language: str = "en"
    ) -> List[str]:
        """Format CoT reasoning as a list of strings for output."""
        trace = []
        
        for step in cot.thinking_steps:
            phase_name = step.phase.value.upper()
            
            if language == "zh":
                obs = step.observation_cn or step.observation
                reason = step.reasoning_cn or step.reasoning
            else:
                obs = step.observation or step.observation_cn
                reason = step.reasoning or step.reasoning_cn
            
            trace.append(f"[{phase_name}] {obs}")
            if reason:
                trace.append(f"  → {reason}")
            
            if step.call_trace:
                trace.append(f"  Call chain: {' → '.join(step.call_trace)}")
        
        return trace


# ============================================================================
# Question Factory with Dynamic Augmentation
# ============================================================================

class QuestionFactory:
    """
    Factory for generating diverse questions with dynamic augmentation.
    """
    
    # Extended template library
    TEMPLATES = {
        "WHAT": [
            {
                "id": "WHAT-001",
                "en": "What mechanism does the system use to {action} in the {component}?",
                "zh": "系统在{component_cn}中使用什么机制来{action_cn}？",
                "perspective": QuestionPerspective.DEVELOPER,
            },
            {
                "id": "WHAT-002", 
                "en": "What security measure prevents {threat} during {operation}?",
                "zh": "在{operation_cn}期间，什么安全措施防止{threat_cn}？",
                "perspective": QuestionPerspective.SECURITY_AUDITOR,
            },
            {
                "id": "WHAT-003",
                "en": "What validation is performed before {operation} to ensure {goal}?",
                "zh": "在{operation_cn}之前执行什么验证以确保{goal_cn}？",
                "perspective": QuestionPerspective.CODE_REVIEWER,
            },
            {
                "id": "WHAT-004",
                "en": "What is the purpose of {element} in the {context}?",
                "zh": "{element}在{context_cn}中的目的是什么？",
                "perspective": QuestionPerspective.TECH_LEAD,
            },
        ],
        "HOW": [
            {
                "id": "HOW-001",
                "en": "How does the {component} handle {scenario} to maintain {goal}?",
                "zh": "{component_cn}如何处理{scenario_cn}以维护{goal_cn}？",
                "perspective": QuestionPerspective.ARCHITECT,
            },
            {
                "id": "HOW-002",
                "en": "How is {resource} protected from {attack_type} in this implementation?",
                "zh": "在此实现中，{resource_cn}如何防止{attack_type_cn}？",
                "perspective": QuestionPerspective.SECURITY_AUDITOR,
            },
            {
                "id": "HOW-003",
                "en": "How does the system ensure atomicity when {operation}?",
                "zh": "系统在{operation_cn}时如何确保原子性？",
                "perspective": QuestionPerspective.DEVELOPER,
            },
            {
                "id": "HOW-004",
                "en": "How does the call chain from {entry} to {exit} implement {feature}?",
                "zh": "从{entry}到{exit}的调用链如何实现{feature_cn}？",
                "perspective": QuestionPerspective.ARCHITECT,
            },
        ],
        "WHY": [
            {
                "id": "WHY-001",
                "en": "Why does the system return {response_type} instead of {alternative} when {condition}?",
                "zh": "为什么系统在{condition_cn}时返回{response_type_cn}而不是{alternative_cn}？",
                "perspective": QuestionPerspective.SECURITY_AUDITOR,
            },
            {
                "id": "WHY-002",
                "en": "Why is {check} performed before {action} in the {flow}?",
                "zh": "为什么在{flow_cn}中的{action_cn}之前要执行{check_cn}？",
                "perspective": QuestionPerspective.CODE_REVIEWER,
            },
            {
                "id": "WHY-003",
                "en": "Why is {design_decision} important for {domain} security?",
                "zh": "为什么{design_decision_cn}对{domain_cn}安全很重要？",
                "perspective": QuestionPerspective.COMPLIANCE_OFFICER,
            },
        ],
        "TRACE": [
            {
                "id": "TRACE-001",
                "en": "Trace the execution path when a user attempts to {user_action}.",
                "zh": "追踪用户尝试{user_action_cn}时的执行路径。",
                "perspective": QuestionPerspective.DEVELOPER,
            },
            {
                "id": "TRACE-002",
                "en": "What functions are invoked in sequence when {trigger_event} occurs?",
                "zh": "当{trigger_event_cn}发生时，哪些函数按顺序被调用？",
                "perspective": QuestionPerspective.ARCHITECT,
            },
        ],
        "SCENARIO": [
            {
                "id": "SCENARIO-001",
                "en": "A user attempts to {user_action}. Walk through the validation steps.",
                "zh": "用户尝试{user_action_cn}。请逐步说明验证步骤。",
                "perspective": QuestionPerspective.QA_ENGINEER,
            },
            {
                "id": "SCENARIO-002",
                "en": "When {error_condition} occurs during {operation}, describe the system's response.",
                "zh": "当{operation_cn}期间发生{error_condition_cn}时，描述系统的响应。",
                "perspective": QuestionPerspective.DEVELOPER,
            },
            {
                "id": "SCENARIO-003",
                "en": "Consider a scenario where {scenario_desc}. What security controls are activated?",
                "zh": "考虑这样一个场景：{scenario_desc_cn}。哪些安全控制被激活？",
                "perspective": QuestionPerspective.SECURITY_AUDITOR,
            },
        ],
        "SECURITY": [
            {
                "id": "SEC-001",
                "en": "What vulnerability does the {mechanism} mitigate in the {context}?",
                "zh": "{mechanism_cn}在{context_cn}中缓解了什么漏洞？",
                "perspective": QuestionPerspective.SECURITY_AUDITOR,
            },
            {
                "id": "SEC-002",
                "en": "How does the {feature} implementation prevent {attack_type}?",
                "zh": "{feature_cn}的实现如何防止{attack_type_cn}？",
                "perspective": QuestionPerspective.SECURITY_AUDITOR,
            },
        ],
    }
    
    def __init__(self, augmenter: DynamicAugmenter):
        self.augmenter = augmenter
        self._template_usage: Dict[str, int] = defaultdict(int)
    
    def generate_questions(
        self,
        evidence: Dict,
        variables: Dict[str, str],
        call_chain: Optional[CallChain],
        language: str = "en",
        count: int = 3
    ) -> List[Dict]:
        """Generate diverse questions for an evidence."""
        questions = []
        evidence_type = evidence.get("evidence_type", "pattern")
        
        # Select appropriate question types based on evidence
        question_types = self._select_question_types(evidence_type, call_chain)
        
        for q_type in question_types[:count]:
            templates = self.TEMPLATES.get(q_type, [])
            if not templates:
                continue
            
            # Select least-used template for diversity
            template = self._select_template(templates)
            
            # Augment variables
            aug_variables = self.augmenter.augment_variables(
                variables, language, variation_level=0.4
            )
            
            # Add call chain variables if available
            if call_chain:
                aug_variables["entry"] = call_chain.nodes[0].function_name
                aug_variables["exit"] = call_chain.nodes[-1].function_name
            
            # Generate question
            question = self._generate_from_template(template, aug_variables, language)
            if question:
                questions.append({
                    "question_id": f"Q-{uuid.uuid4().hex[:8]}",
                    "question_text": question,
                    "question_type": q_type.lower(),
                    "template_id": template["id"],
                    "perspective": template["perspective"].value,
                    "variables": aug_variables,
                })
        
        # Generate additional variations
        if questions and len(questions) < count:
            base_q = questions[0]["question_text"]
            variations = self.augmenter.generate_question_variations(
                base_q,
                questions[0]["question_type"],
                language,
                count - len(questions)
            )
            for i, var in enumerate(variations[1:], len(questions)):
                questions.append({
                    "question_id": f"Q-{uuid.uuid4().hex[:8]}",
                    "question_text": var,
                    "question_type": questions[0]["question_type"],
                    "template_id": f"{questions[0]['template_id']}-VAR{i}",
                    "perspective": questions[0]["perspective"],
                    "variables": questions[0]["variables"],
                })
        
        return questions[:count]
    
    def _select_question_types(
        self,
        evidence_type: str,
        call_chain: Optional[CallChain]
    ) -> List[str]:
        """Select appropriate question types based on evidence."""
        types = ["WHAT", "HOW"]  # Always include basic types
        
        if evidence_type == "exception_handling":
            types.extend(["WHY", "SECURITY", "SCENARIO"])
        elif evidence_type == "pattern":
            types.extend(["WHY", "SCENARIO"])
        elif evidence_type == "call":
            types.extend(["TRACE", "WHEN"])
        elif evidence_type == "function":
            types.extend(["WHAT", "TRACE"])
        
        # Add TRACE if call chain available
        if call_chain and "TRACE" not in types:
            types.append("TRACE")
        
        random.shuffle(types)
        return types
    
    def _select_template(self, templates: List[Dict]) -> Dict:
        """Select template with preference for less-used ones."""
        # Weight by inverse usage
        weights = []
        for t in templates:
            usage = self._template_usage[t["id"]]
            weights.append(1.0 / (1 + usage))
        
        total = sum(weights)
        weights = [w / total for w in weights]
        
        selected = random.choices(templates, weights=weights, k=1)[0]
        self._template_usage[selected["id"]] += 1
        return selected
    
    def _generate_from_template(
        self,
        template: Dict,
        variables: Dict[str, str],
        language: str
    ) -> Optional[str]:
        """Generate question from template with variables."""
        template_text = template.get("zh" if language == "zh" else "en", "")
        
        try:
            # Fill in variables
            question = template_text.format(**variables)
            
            # Ensure question mark
            if not (question.endswith("?") or question.endswith("？") or question.endswith(".")):
                question += "？" if language == "zh" else "?"
            
            return question
        except KeyError as e:
            logger.debug(f"Missing variable in template: {e}")
            # Try with default values
            try:
                defaults = {k: f"[{k}]" for k in re.findall(r'\{(\w+)\}', template_text)}
                defaults.update(variables)
                return template_text.format(**defaults)
            except:
                return None


# ============================================================================
# Answer Composer with CoT Integration
# ============================================================================

class AnswerComposer:
    """Composes gold-standard answers with CoT reasoning integration."""
    
    def __init__(self, augmenter: DynamicAugmenter):
        self.augmenter = augmenter
    
    def compose_answer(
        self,
        evidence: Dict,
        question: Dict,
        cot_reasoning: CoTReasoning,
        code_context: Dict,
        language: str = "en"
    ) -> str:
        """Compose a comprehensive answer with CoT reasoning."""
        parts = []
        
        # Section 1: Chain-of-Thought Reasoning
        parts.append(self._format_cot_section(cot_reasoning, language))
        
        # Section 2: Technical Explanation
        parts.append(self._format_explanation_section(evidence, language))
        
        # Section 3: Source Code
        parts.append(self._format_code_section(code_context, language))
        
        # Section 4: Call Chain (if available)
        if cot_reasoning.call_chain:
            parts.append(self._format_call_chain_section(cot_reasoning.call_chain, language))
        
        # Section 5: Security Analysis (for security questions)
        if question.get("question_type") in ["security", "why"]:
            parts.append(self._format_security_section(evidence, language))
        
        # Section 6: Key Insights
        parts.append(self._format_insights_section(cot_reasoning, language))
        
        return "\n".join(parts)
    
    def _format_cot_section(self, cot: CoTReasoning, language: str) -> str:
        """Format CoT reasoning section."""
        if language == "zh":
            header = "### 思维链推理过程\n"
        else:
            header = "### Chain-of-Thought Reasoning\n"
        
        lines = [header]
        
        for step in cot.thinking_steps:
            phase_label = step.phase.value.replace("_", " ").title()
            
            if language == "zh":
                obs = step.observation_cn or step.observation
                reason = step.reasoning_cn or step.reasoning
                lines.append(f"**{phase_label}**: {obs}")
                if reason:
                    lines.append(f"  - *推理*: {reason}")
            else:
                obs = step.observation or step.observation_cn
                reason = step.reasoning or step.reasoning_cn
                lines.append(f"**{phase_label}**: {obs}")
                if reason:
                    lines.append(f"  - *Reasoning*: {reason}")
            
            if step.call_trace:
                trace_str = " → ".join(step.call_trace)
                lines.append(f"  - *Call trace*: `{trace_str}`")
        
        lines.append("")
        return "\n".join(lines)
    
    def _format_explanation_section(self, evidence: Dict, language: str) -> str:
        """Format technical explanation section."""
        if language == "zh":
            header = "### 技术解释\n"
            desc = evidence.get("description_cn", evidence.get("description", ""))
        else:
            header = "### Technical Explanation\n"
            desc = evidence.get("description", "")
        
        # Augment connectors for variety
        desc = self.augmenter.augment_reasoning_connectors(desc, language)
        
        return f"{header}{desc}\n"
    
    def _format_code_section(self, code_context: Dict, language: str) -> str:
        """Format source code section."""
        if language == "zh":
            header = "### 相关源代码\n"
        else:
            header = "### Relevant Source Code\n"
        
        file_path = code_context.get("file_path", "")
        line_start = code_context.get("line_start", 0)
        line_end = code_context.get("line_end", 0)
        code = code_context.get("code", "")
        
        return f"""{header}**File:** `{file_path}` (Lines {line_start}-{line_end})

```python
{code}
```
"""
    
    def _format_call_chain_section(self, call_chain: CallChain, language: str) -> str:
        """Format call chain section."""
        if language == "zh":
            header = "### 调用链分析\n"
            intro = f"执行路径涉及 {call_chain.depth} 个函数：\n"
        else:
            header = "### Call Chain Analysis\n"
            intro = f"The execution path involves {call_chain.depth} functions:\n"
        
        lines = [header, intro]
        
        for i, node in enumerate(call_chain.nodes):
            prefix = "├──" if i < len(call_chain.nodes) - 1 else "└──"
            lines.append(f"{prefix} `{node.function_name}` ({node.file_path}:{node.line_number})")
        
        lines.append("")
        return "\n".join(lines)
    
    def _format_security_section(self, evidence: Dict, language: str) -> str:
        """Format security analysis section."""
        dbr_logic = evidence.get("dbr_logic", {})
        subcategory = dbr_logic.get("subcategory_id", "")
        
        security_analysis = {
            "DBR-01-01": {
                "en": "**Security Benefit**: Prevents duplicate account creation which could lead to data integrity issues and enable account takeover attacks.",
                "zh": "**安全收益**：防止重复账户创建，避免数据完整性问题和账户接管攻击。",
            },
            "DBR-01-02": {
                "en": "**Security Benefit**: Atomic transactions prevent partial state corruption. Password hashing protects credentials from exposure in case of database breach.",
                "zh": "**安全收益**：原子事务防止部分状态损坏。密码哈希保护凭据在数据库泄露时不被暴露。",
            },
            "DBR-01-03": {
                "en": "**Security Benefit**: Vague error messages prevent user enumeration attacks where attackers probe for valid usernames through different error responses.",
                "zh": "**安全收益**：模糊错误信息防止用户枚举攻击，攻击者无法通过不同错误响应探测有效用户名。",
            },
            "DBR-01-04": {
                "en": "**Security Benefit**: Token refresh limits the window of opportunity for token theft attacks and ensures session state consistency.",
                "zh": "**安全收益**：令牌刷新限制了令牌盗窃攻击的时间窗口，确保会话状态一致性。",
            },
        }
        
        if language == "zh":
            header = "### 安全性分析\n"
        else:
            header = "### Security Analysis\n"
        
        analysis = security_analysis.get(subcategory, {}).get(
            "zh" if language == "zh" else "en",
            "Security implications require further analysis."
        )
        
        return f"{header}{analysis}\n"
    
    def _format_insights_section(self, cot: CoTReasoning, language: str) -> str:
        """Format key insights section."""
        if language == "zh":
            header = "### 关键洞察\n"
            insight = cot.final_insight_cn or cot.final_insight
            confidence_label = f"置信度: {cot.confidence:.0%}"
        else:
            header = "### Key Insights\n"
            insight = cot.final_insight or cot.final_insight_cn
            confidence_label = f"Confidence: {cot.confidence:.0%}"
        
        rules_str = ", ".join(cot.dbr_mappings) if cot.dbr_mappings else "N/A"
        
        return f"""{header}{insight}

**DBR Rules**: {rules_str}
**{confidence_label}**
"""


# ============================================================================
# Quality Gate
# ============================================================================

class QualityGate:
    """Multi-point quality validation gate."""
    
    def __init__(self):
        self.validators = [
            self._validate_question_length,
            self._validate_answer_length,
            self._validate_code_presence,
            self._validate_reasoning_depth,
            self._validate_hash_consistency,
            self._validate_language_consistency,
        ]
    
    def validate(
        self,
        qa_pair: Dict,
        strict: bool = False
    ) -> Tuple[bool, List[str], float]:
        """
        Validate a Q&A pair.
        
        Returns:
            Tuple of (is_valid, issues_list, quality_score)
        """
        issues = []
        scores = []
        
        for validator in self.validators:
            passed, issue, score = validator(qa_pair)
            if not passed:
                issues.append(issue)
            scores.append(score)
        
        quality_score = sum(scores) / len(scores) if scores else 0.0
        
        if strict:
            is_valid = len(issues) == 0
        else:
            is_valid = quality_score >= 0.6  # Allow minor issues
        
        return is_valid, issues, quality_score
    
    def _validate_question_length(self, qa: Dict) -> Tuple[bool, str, float]:
        q_len = len(qa.get("instruction", ""))
        if q_len < Config.MIN_QUESTION_LENGTH:
            return False, f"Question too short ({q_len} chars)", 0.3
        return True, "", 1.0
    
    def _validate_answer_length(self, qa: Dict) -> Tuple[bool, str, float]:
        a_len = len(qa.get("answer", ""))
        if a_len < Config.MIN_ANSWER_LENGTH:
            return False, f"Answer too short ({a_len} chars)", 0.3
        return True, "", 1.0
    
    def _validate_code_presence(self, qa: Dict) -> Tuple[bool, str, float]:
        code = qa.get("context", {}).get("code_snippet", "")
        if not code or len(code) < 10:
            return False, "Missing or invalid code snippet", 0.0
        return True, "", 1.0
    
    def _validate_reasoning_depth(self, qa: Dict) -> Tuple[bool, str, float]:
        trace = qa.get("reasoning_trace", [])
        if len(trace) < 3:
            return False, f"Insufficient reasoning depth ({len(trace)} steps)", 0.5
        return True, "", 1.0
    
    def _validate_hash_consistency(self, qa: Dict) -> Tuple[bool, str, float]:
        source_hash = qa.get("data_quality", {}).get("source_hash", "")
        if not source_hash:
            return False, "Missing source hash", 0.7
        return True, "", 1.0
    
    def _validate_language_consistency(self, qa: Dict) -> Tuple[bool, str, float]:
        lang = qa.get("data_quality", {}).get("language", "")
        instruction = qa.get("instruction", "")
        
        # Check if language matches content
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in instruction)
        
        if lang == "zh" and not has_chinese:
            return False, "Language mismatch (expected Chinese)", 0.5
        if lang == "en" and has_chinese:
            return False, "Language mismatch (expected English)", 0.5
        
        return True, "", 1.0


# ============================================================================
# QA Orchestrator (Main Controller)
# ============================================================================

class QAOrchestrator:
    """
    Main orchestrator that coordinates all components.
    
    Pipeline:
    1. Load rule metadata and AST analysis
    2. Build call graph
    3. For each evidence:
       a. Generate call chain
       b. Generate CoT reasoning
       c. Generate diverse questions
       d. Compose answers
       e. Validate quality
    4. Output results
    """
    
    def __init__(
        self,
        rule_metadata_path: str,
        ast_analysis_path: Optional[str] = None
    ):
        self.rule_metadata_path = Path(rule_metadata_path)
        self.ast_analysis_path = Path(ast_analysis_path) if ast_analysis_path else None
        
        # Initialize components
        self.call_graph = CallGraphEngine()
        self.augmenter = DynamicAugmenter(seed=42)  # Reproducible randomness
        self.cot_engine = CoTReasoningEngine(self.call_graph)
        self.question_factory = QuestionFactory(self.augmenter)
        self.answer_composer = AnswerComposer(self.augmenter)
        self.quality_gate = QualityGate()
        
        # Data
        self.rule_metadata: Dict = {}
        self.generated_pairs: List[Dict] = []
        self.stats: Dict[str, int] = defaultdict(int)
    
    def initialize(self) -> bool:
        """Initialize the orchestrator by loading all data."""
        # Load rule metadata
        if not self._load_rule_metadata():
            return False
        
        # Load AST analysis for call graph
        if self.ast_analysis_path:
            self.call_graph.load_ast_analysis(str(self.ast_analysis_path))
        elif Config.AST_ANALYSIS_FILE.exists():
            self.call_graph.load_ast_analysis(str(Config.AST_ANALYSIS_FILE))
        
        logger.info("Orchestrator initialized successfully")
        return True
    
    def _load_rule_metadata(self) -> bool:
        """Load rule metadata."""
        if not self.rule_metadata_path.exists():
            logger.error(f"Rule metadata not found: {self.rule_metadata_path}")
            return False
        
        try:
            with open(self.rule_metadata_path, 'r', encoding='utf-8') as f:
                self.rule_metadata = json.load(f)
            logger.info(f"Loaded rule metadata: {self.rule_metadata.get('rule_id', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Error loading rule metadata: {e}")
            return False
    
    def run_pipeline(
        self,
        samples_per_evidence: int = 3,
        languages: List[str] = None,
        strict_validation: bool = False
    ) -> List[Dict]:
        """
        Run the complete Q&A generation pipeline.
        """
        languages = languages or Config.SUPPORTED_LANGUAGES
        self.generated_pairs = []
        self.stats = defaultdict(int)
        
        logger.info(f"Starting pipeline: {samples_per_evidence} samples/evidence, languages: {languages}")
        
        # Process each subcategory
        for subcategory in self.rule_metadata.get("subcategories", []):
            self._process_subcategory(
                subcategory,
                samples_per_evidence,
                languages,
                strict_validation
            )
        
        logger.info(f"Pipeline complete: {len(self.generated_pairs)} Q&A pairs generated")
        return self.generated_pairs
    
    def _process_subcategory(
        self,
        subcategory: Dict,
        samples_per_evidence: int,
        languages: List[str],
        strict_validation: bool
    ):
        """Process a single subcategory."""
        subcategory_id = subcategory.get("subcategory_id", "")
        logger.info(f"Processing subcategory: {subcategory_id}")
        
        for evidence in subcategory.get("evidences", []):
            self._process_evidence(
                evidence,
                subcategory_id,
                samples_per_evidence,
                languages,
                strict_validation
            )
    
    def _process_evidence(
        self,
        evidence: Dict,
        subcategory_id: str,
        samples_per_evidence: int,
        languages: List[str],
        strict_validation: bool
    ):
        """Process a single evidence."""
        evidence_id = evidence.get("evidence_id", "")
        logger.debug(f"  Processing evidence: {evidence_id}")
        
        # Get code context
        code_context = self._extract_code_context(evidence)
        if not code_context:
            logger.warning(f"    Skipping {evidence_id}: no code context")
            return
        
        # Build call chain
        call_chain = None
        if Config.ENABLE_CALL_GRAPH:
            func_name = evidence.get("location", {}).get("qualified_name", "")
            if func_name:
                call_chain = self.call_graph.get_call_chain(
                    func_name,
                    max_depth=Config.COT_DEPTH
                )
        
        # Get variables
        variables = self._get_variables(subcategory_id, evidence)
        
        # Generate for each language
        for language in languages:
            self._generate_pairs_for_language(
                evidence,
                subcategory_id,
                code_context,
                call_chain,
                variables,
                language,
                samples_per_evidence,
                strict_validation
            )
    
    def _generate_pairs_for_language(
        self,
        evidence: Dict,
        subcategory_id: str,
        code_context: Dict,
        call_chain: Optional[CallChain],
        variables: Dict[str, str],
        language: str,
        count: int,
        strict_validation: bool
    ):
        """Generate Q&A pairs for a specific language."""
        
        # Generate CoT reasoning
        cot_reasoning = self.cot_engine.generate_cot_reasoning(
            evidence,
            call_chain,
            language,
            Config.COT_DEPTH
        )
        
        # Generate questions
        questions = self.question_factory.generate_questions(
            evidence,
            variables,
            call_chain,
            language,
            count
        )
        
        for question in questions:
            # Compose answer
            answer = self.answer_composer.compose_answer(
                evidence,
                question,
                cot_reasoning,
                code_context,
                language
            )
            
            # Build Q&A pair
            qa_pair = self._build_qa_pair(
                evidence,
                question,
                cot_reasoning,
                code_context,
                answer,
                language
            )
            
            # Validate
            is_valid, issues, quality_score = self.quality_gate.validate(
                qa_pair, strict_validation
            )
            
            qa_pair["data_quality"]["quality_score"] = quality_score
            qa_pair["data_quality"]["validation_issues"] = issues
            qa_pair["data_quality"]["is_valid"] = is_valid
            
            if is_valid:
                self.generated_pairs.append(qa_pair)
                self.stats["valid"] += 1
            else:
                self.stats["invalid"] += 1
                if not strict_validation:
                    # Include with warning
                    self.generated_pairs.append(qa_pair)
                logger.debug(f"    Quality issues: {issues}")
    
    def _extract_code_context(self, evidence: Dict) -> Optional[Dict]:
        """Extract code context from evidence."""
        code_snippet = evidence.get("code_snippet", {})
        if not code_snippet or not code_snippet.get("code"):
            return None
        
        return {
            "file_path": code_snippet.get("file_path", ""),
            "code": code_snippet.get("code", ""),
            "line_start": code_snippet.get("line_start", 0),
            "line_end": code_snippet.get("line_end", 0),
            "source_hash": code_snippet.get("source_hash", ""),
        }
    
    def _get_variables(self, subcategory_id: str, evidence: Dict) -> Dict[str, str]:
        """Get variable mapping for evidence."""
        # Import the variable mapper
        try:
            # Base variables from subcategory mapping
            base_vars = {
                "action": "perform the operation",
                "action_cn": "执行操作",
                "component": "system component",
                "component_cn": "系统组件",
                "goal": "intended outcome",
                "goal_cn": "预期结果",
                "scenario": "specific situation",
                "scenario_cn": "特定情况",
                "operation": "the operation",
                "operation_cn": "操作",
                "threat": "potential threat",
                "threat_cn": "潜在威胁",
                "resource": "protected resource",
                "resource_cn": "受保护资源",
                "attack_type": "attack vector",
                "attack_type_cn": "攻击向量",
                "mechanism": "security mechanism",
                "mechanism_cn": "安全机制",
                "context": "application context",
                "context_cn": "应用上下文",
                "feature": "implemented feature",
                "feature_cn": "实现的功能",
                "check": "validation check",
                "check_cn": "验证检查",
                "flow": "process flow",
                "flow_cn": "处理流程",
                "user_action": "user action",
                "user_action_cn": "用户操作",
                "error_condition": "error condition",
                "error_condition_cn": "错误条件",
                "response_type": "response type",
                "response_type_cn": "响应类型",
                "alternative": "alternative approach",
                "alternative_cn": "替代方案",
                "condition": "trigger condition",
                "condition_cn": "触发条件",
                "domain": "domain area",
                "domain_cn": "领域",
                "design_decision": "design decision",
                "design_decision_cn": "设计决策",
                "element": evidence.get("name", "element"),
                "trigger_event": "trigger event",
                "trigger_event_cn": "触发事件",
                "scenario_desc": "specific scenario",
                "scenario_desc_cn": "特定场景",
                "problem": "identified problem",
                "problem_cn": "识别的问题",
                "responsibility": "functional responsibility",
                "responsibility_cn": "功能职责",
                "process": "business process",
                "process_cn": "业务流程",
            }
            
            # Override with subcategory-specific variables
            subcategory_vars = self._get_subcategory_variables(subcategory_id)
            base_vars.update(subcategory_vars)
            
            # Add evidence-specific variables
            base_vars["evidence_name"] = evidence.get("name", "")
            base_vars["evidence_type"] = evidence.get("evidence_type", "")
            
            return base_vars
            
        except Exception as e:
            logger.error(f"Error getting variables: {e}")
            return {}
    
    def _get_subcategory_variables(self, subcategory_id: str) -> Dict[str, str]:
        """Get subcategory-specific variables."""
        mappings = {
            "DBR-01-01": {
                "action": "enforce uniqueness validation",
                "action_cn": "执行唯一性验证",
                "component": "user registration flow",
                "component_cn": "用户注册流程",
                "goal": "data integrity",
                "goal_cn": "数据完整性",
                "threat": "duplicate account creation",
                "threat_cn": "重复账户创建",
                "operation": "creating or updating user accounts",
                "operation_cn": "创建或更新用户账户",
                "mechanism": "pre-validation check",
                "mechanism_cn": "预验证检查",
                "user_action": "register with an existing email",
                "user_action_cn": "使用已存在的邮箱注册",
            },
            "DBR-01-02": {
                "action": "ensure transaction atomicity",
                "action_cn": "确保事务原子性",
                "component": "user repository",
                "component_cn": "用户仓库",
                "goal": "data consistency",
                "goal_cn": "数据一致性",
                "threat": "partial data corruption",
                "threat_cn": "部分数据损坏",
                "operation": "persisting user credentials",
                "operation_cn": "持久化用户凭据",
                "mechanism": "transaction-based storage",
                "mechanism_cn": "基于事务的存储",
            },
            "DBR-01-03": {
                "action": "return vague error messages",
                "action_cn": "返回模糊错误信息",
                "component": "login handler",
                "component_cn": "登录处理器",
                "goal": "prevent information leakage",
                "goal_cn": "防止信息泄露",
                "threat": "user enumeration attacks",
                "threat_cn": "用户枚举攻击",
                "operation": "user authentication",
                "operation_cn": "用户认证",
                "mechanism": "unified error handling",
                "mechanism_cn": "统一错误处理",
                "attack_type": "credential enumeration",
                "attack_type_cn": "凭据枚举",
                "response_type": "generic error message",
                "response_type_cn": "通用错误信息",
                "alternative": "specific error details",
                "alternative_cn": "具体错误详情",
                "condition": "authentication failure",
                "condition_cn": "认证失败",
                "scenario_desc": "an attacker probes for valid usernames",
                "scenario_desc_cn": "攻击者探测有效用户名",
            },
            "DBR-01-04": {
                "action": "regenerate JWT tokens",
                "action_cn": "重新生成JWT令牌",
                "component": "session management",
                "component_cn": "会话管理",
                "goal": "session state consistency",
                "goal_cn": "会话状态一致性",
                "threat": "session hijacking",
                "threat_cn": "会话劫持",
                "operation": "completing user operations",
                "operation_cn": "完成用户操作",
                "mechanism": "automatic token refresh",
                "mechanism_cn": "自动令牌刷新",
                "process": "authentication flow",
                "process_cn": "认证流程",
            },
        }
        return mappings.get(subcategory_id, {})
    
    def _build_qa_pair(
        self,
        evidence: Dict,
        question: Dict,
        cot_reasoning: CoTReasoning,
        code_context: Dict,
        answer: str,
        language: str
    ) -> Dict:
        """Build a complete Q&A pair."""
        dbr_logic = evidence.get("dbr_logic", {})
        
        # Format reasoning trace
        reasoning_trace = self.cot_engine.format_cot_as_reasoning_trace(
            cot_reasoning, language
        )
        
        return {
            "sample_id": f"DBR01-{uuid.uuid4().hex[:12]}",
            "instruction": question["question_text"],
            "context": {
                "file_path": code_context["file_path"],
                "related_dbr": dbr_logic.get("rule_id", "DBR-01"),
                "code_snippet": code_context["code"],
                "line_range": f"{code_context['line_start']}-{code_context['line_end']}",
                "function_name": evidence.get("name", ""),
                "call_chain": (
                    [n.qualified_name for n in cot_reasoning.call_chain.nodes]
                    if cot_reasoning.call_chain else None
                ),
            },
            "auto_processing": {
                "parser": self.rule_metadata.get("parser_info", {}).get("name", "FastAPI-AST-Analyzer"),
                "parser_version": self.rule_metadata.get("parser_info", {}).get("version", "2.0.0"),
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
                    "hash_validated": bool(code_context.get("source_hash")),
                },
                "generation_metadata": {
                    "template_id": question["template_id"],
                    "question_type": question["question_type"],
                    "perspective": question["perspective"],
                    "cot_reasoning_id": cot_reasoning.reasoning_id,
                    "cot_confidence": cot_reasoning.confidence,
                    "augmentation_applied": True,
                },
            },
            "reasoning_trace": reasoning_trace,
            "answer": answer,
            "data_quality": {
                "consistency_check": True,
                "source_hash": code_context.get("source_hash", ""),
                "language": language,
                "temperature": Config.DEFAULT_TEMPERATURE,
                "evidence_id": evidence.get("evidence_id", ""),
            },
        }
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """Save generated Q&A pairs to JSONL file."""
        output_path = Path(output_path) if output_path else Config.OUTPUT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in self.generated_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(self.generated_pairs)} Q&A pairs to: {output_path}")
        return str(output_path)
    
    def print_summary(self):
        """Print generation summary."""
        print("\n" + "=" * 70)
        print("Q&A Generation Summary (v2.0 - CoT Enhanced)")
        print("=" * 70)
        
        print(f"\nTotal Pairs Generated: {len(self.generated_pairs)}")
        print(f"Valid: {self.stats['valid']}, Invalid: {self.stats['invalid']}")
        
        # Stats by language
        lang_counts = defaultdict(int)
        for pair in self.generated_pairs:
            lang = pair.get("data_quality", {}).get("language", "unknown")
            lang_counts[lang] += 1
        
        print("\nBy Language:")
        for lang, count in lang_counts.items():
            print(f"  - {lang}: {count}")
        
        # Stats by question type
        type_counts = defaultdict(int)
        for pair in self.generated_pairs:
            q_type = pair.get("auto_processing", {}).get("generation_metadata", {}).get("question_type", "unknown")
            type_counts[q_type] += 1
        
        print("\nBy Question Type:")
        for q_type, count in sorted(type_counts.items()):
            print(f"  - {q_type}: {count}")
        
        # Stats by perspective
        persp_counts = defaultdict(int)
        for pair in self.generated_pairs:
            persp = pair.get("auto_processing", {}).get("generation_metadata", {}).get("perspective", "unknown")
            persp_counts[persp] += 1
        
        print("\nBy Perspective:")
        for persp, count in sorted(persp_counts.items()):
            print(f"  - {persp}: {count}")
        
        # CoT stats
        cot_with_chain = sum(
            1 for p in self.generated_pairs 
            if p.get("context", {}).get("call_chain")
        )
        print(f"\nCoT with Call Chain: {cot_with_chain}/{len(self.generated_pairs)}")
        
        # Quality stats
        avg_quality = sum(
            p.get("data_quality", {}).get("quality_score", 0)
            for p in self.generated_pairs
        ) / max(1, len(self.generated_pairs))
        print(f"Average Quality Score: {avg_quality:.2%}")
        
        print("\n" + "=" * 70)
    
    def print_sample_pairs(self, n: int = 2):
        """Print sample Q&A pairs."""
        samples = self.generated_pairs[:n]
        
        for i, pair in enumerate(samples, 1):
            print("\n" + "=" * 70)
            print(f"Sample {i} - ID: {pair['sample_id']}")
            print("=" * 70)
            
            print(f"\n【Instruction】:\n{pair['instruction']}\n")
            
            print("【Reasoning Trace (CoT)】:")
            for step in pair.get("reasoning_trace", [])[:6]:
                print(f"  {step}")
            if len(pair.get("reasoning_trace", [])) > 6:
                print(f"  ... ({len(pair['reasoning_trace']) - 6} more steps)")
            
            print(f"\n【Answer (excerpt)】:\n{pair['answer'][:600]}...")
            
            print(f"\n【Metadata】:")
            print(f"  Language: {pair['data_quality']['language']}")
            print(f"  Question Type: {pair['auto_processing']['generation_metadata']['question_type']}")
            print(f"  Perspective: {pair['auto_processing']['generation_metadata']['perspective']}")
            print(f"  CoT Confidence: {pair['auto_processing']['generation_metadata']['cot_confidence']:.0%}")
            print(f"  Quality Score: {pair['data_quality'].get('quality_score', 'N/A')}")
            
            if pair['context'].get('call_chain'):
                print(f"  Call Chain: {' → '.join(pair['context']['call_chain'][:3])}...")
            
            print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enterprise Q&A Generation Engine v2.0 (CoT Enhanced)',
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
        '--strict',
        action='store_true',
        help='Enable strict validation'
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
    orchestrator = QAOrchestrator(
        args.metadata,
        args.ast
    )
    
    if not orchestrator.initialize():
        print("Error: Failed to initialize. Please check file paths.")
        sys.exit(1)
    
    # Run pipeline
    print(f"\n🚀 Running Q&A Generation Pipeline v2.0")
    print(f"   Metadata: {args.metadata}")
    print(f"   Languages: {args.languages}")
    print(f"   Samples/evidence: {args.samples}")
    print(f"   Call Graph: {'Enabled' if Config.ENABLE_CALL_GRAPH else 'Disabled'}")
    
    pairs = orchestrator.run_pipeline(
        samples_per_evidence=args.samples,
        languages=args.languages,
        strict_validation=args.strict
    )
    
    if not pairs:
        print("Error: No Q&A pairs generated.")
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
