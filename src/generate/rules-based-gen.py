#!/usr/bin/env python3
"""
Scenario 1: Rules-Based Training Data Generation

This script generates training data by analyzing source code and generating
Q&A pairs with reasoning traces based on Domain Business Rules (DBR).

The generated data is suitable for fine-tuning LLMs to understand
code-based business logic and provide answers grounded in source code.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Any

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config, get_config, get_lang_config
from core.llm_client import OllamaClient
from core.dbr_rules import DBRRegistry, DBR_01
from parser.ast_extractor import load_code_map
from parser.response_parser import ResponseParser
from validator.schema_validator import (
    SchemaValidator,
    ScenarioType,
    create_rules_sample
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================
# Generation Scenarios
# ==========================================
GENERATION_SCENARIOS = [
    {
        "topic": "身份认证异常模糊化",
        "role": "安全审计员",
        "intent_desc": "防止通过报错进行账户枚举探测",
        "forbidden": ["注册", "register", "唯一性"]
    },
    {
        "topic": "新账户准入唯一性预检",
        "role": "首席架构师",
        "intent_desc": "注册环节的身份标识冲突拦截",
        "forbidden": ["登录", "login", "模糊反馈"]
    },
    {
        "topic": "存量数据更新合规",
        "role": "合规官",
        "intent_desc": "修改资料时的唯一性一致性检查",
        "forbidden": ["初次", "注册", "register"]
    }
]


def build_system_prompt(scenario: Dict[str, Any]) -> str:
    """
    Build the system prompt for generation.
    
    Args:
        scenario: The current generation scenario
        
    Returns:
        System prompt string
    """
    return (
        f"Role: {scenario['role']}. Topic: {scenario['topic']}.\n"
        "【严格约束】：\n"
        "1. [[QUESTION]]: 纯业务提问，严禁出现函数名、文件名或变量名。\n"
        "2. [[REASONING]]: 结构化推理步骤。禁止使用'步骤1'等占位符，直接描述分析逻辑。\n"
        "3. [[CODE]]: 仅提取核心逻辑行，移除装饰器。\n"
        "4. [[ANSWER]]: 详细解答，必须包含业务价值说明。\n"
        "【输出格式】：\n"
        "[[QUESTION]]: 内容\n"
        "[[REASONING]]: 推理步骤内容\n"
        "[[CODE]]: 代码片段\n"
        "[[ANSWER]]: 解答内容"
    )


def build_full_prompt(
    system_prompt: str,
    dbr_content: str,
    evidence_guide: str,
    code_context: str
) -> str:
    """
    Build the complete prompt for the LLM.
    
    Args:
        system_prompt: The system instructions
        dbr_content: The DBR rule content
        evidence_guide: Guide for code extraction
        code_context: The source code context
        
    Returns:
        Complete prompt string
    """
    return (
        f"{system_prompt}\n\n"
        f"[DBR]:\n{dbr_content}\n\n"
        f"[Extraction Guide]:\n{evidence_guide}\n\n"
        f"[Source]:\n{code_context}"
    )


def generate_qa_sample(
    llm_client: OllamaClient,
    parser: ResponseParser,
    code_map: Dict[str, str],
    scenario_index: int,
    config: Config
) -> Optional[Dict[str, Any]]:
    """
    Generate a single Q&A sample.
    
    Args:
        llm_client: The LLM client instance
        parser: The response parser
        code_map: Dictionary of file paths to source code
        scenario_index: Index for selecting generation scenario
        config: Configuration instance
        
    Returns:
        Generated sample dictionary or None if generation fails
    """
    # Select scenario
    scenario = GENERATION_SCENARIOS[scenario_index % len(GENERATION_SCENARIOS)]
    
    # Build prompts
    system_prompt = build_system_prompt(scenario)
    code_context = "\n".join([
        f"--- File: {path} ---\n{code}"
        for path, code in code_map.items()
    ])
    
    full_prompt = build_full_prompt(
        system_prompt=system_prompt,
        dbr_content=DBR_01.get_content(),
        evidence_guide=DBR_01.evidence_guide,
        code_context=code_context
    )
    
    # Generate response
    response = llm_client.generate(
        prompt=full_prompt,
        temperature=config.generation_temp
    )
    
    if not response.success:
        logger.error(f"Generation failed: {response.error}")
        return None
    
    # Parse the response
    parsed = parser.parse_qa_response(response.text)
    
    # Validate parsed response
    if not parser.validate_qa_response(parsed):
        logger.warning("Parsed response validation failed")
        return None
    
    return {
        "instruction": parsed["question"],
        "reasoning_steps": parsed["reasoning_steps"],
        "relevant_code": parsed["code_snippet"],
        "answer_text": parsed["answer"],
        "intent_desc": scenario["intent_desc"]
    }


def filter_sample(sample: Dict[str, Any], forbidden_words: List[str]) -> bool:
    """
    Filter out samples that contain forbidden words.
    
    Args:
        sample: The generated sample
        forbidden_words: List of words that shouldn't appear
        
    Returns:
        True if sample passes filter, False otherwise
    """
    instruction = sample.get("instruction", "").lower()
    return not any(word.lower() in instruction for word in forbidden_words)


def print_sample_preview(sample_id: str, instruction: str, answer: str) -> None:
    """Print a preview of the generated sample to terminal."""
    print("\n" + "=" * 80)
    print(f" [写入成功] ID: {sample_id}")
    print(f"【问题 (Instruction)】: {instruction}")
    print("-" * 40)
    print(f"【回答 (Answer)】:\n{answer}")
    print("=" * 80 + "\n")


def main(n: int = 2, output_file: Optional[str] = None):
    """
    Main execution function for rules-based data generation.
    
    Args:
        n: Number of samples to generate
        output_file: Optional custom output file path
    """
    # Initialize configuration
    config = get_config()
    config.ensure_data_dir()
    
    # Set output file
    if output_file is None:
        output_file = config.get_output_path("qwen_dbr_training_final_v3.jsonl")
    
    # Initialize components
    llm_client = OllamaClient(
        api_url=config.ollama_api,
        model_name=config.model_name,
        default_temperature=config.generation_temp,
        context_window=config.context_window
    )
    parser = ResponseParser()
    validator = SchemaValidator(scenario=ScenarioType.RULES_BASED)
    
    # Load source code
    code_map = load_code_map(
        repo_path=config.repo_path,
        target_files=config.target_files,
        max_chars=4000
    )
    
    if not code_map:
        logger.error("No source code loaded. Check target files configuration.")
        return
    
    logger.info(f"开始生成符合设计文档的语料 (目标: {n} 条)...")
    
    # Forbidden words for filtering
    forbidden_words = ["login", "register", "update_current_user", "函数"]
    
    success_count = 0
    attempt_count = 0
    max_attempts = n * 5  # Prevent infinite loops
    
    with open(output_file, 'a', encoding='utf-8') as f:
        while success_count < n and attempt_count < max_attempts:
            attempt_count += 1
            current_rel_path = config.target_files[success_count % len(config.target_files)]
            
            # Generate sample
            raw = generate_qa_sample(
                llm_client=llm_client,
                parser=parser,
                code_map=code_map,
                scenario_index=success_count,
                config=config
            )
            
            if raw is None:
                continue
            
            # Filter sample
            if not filter_sample(raw, forbidden_words):
                logger.debug("Sample filtered due to forbidden words")
                continue
            
            # Build final sample using factory function
            final_entry = create_rules_sample(
                instruction=raw["instruction"],
                reasoning_steps=raw["reasoning_steps"],
                code_snippet=raw["relevant_code"],
                answer=raw["answer_text"],
                file_path=current_rel_path,
                dbr_id="DBR-01",
                intent_desc=raw["intent_desc"],
                language="zh-cn",
                temperature=config.generation_temp
            )
            
            # Validate sample
            validation_result = validator.validate_sample(final_entry)
            if not validation_result.valid:
                logger.warning(f"Validation failed: {validation_result.errors}")
                continue
            
            # Print preview and save
            print_sample_preview(
                sample_id=final_entry["sample_id"],
                instruction=final_entry["instruction"],
                answer=final_entry["answer"]
            )
            
            f.write(json.dumps(final_entry, ensure_ascii=False) + "\n")
            success_count += 1
            
            # Rate limiting
            time.sleep(1)
    
    logger.info(f"任务结束。成功生成 {success_count} 条样本。文件已保存至: {output_file}")


if __name__ == "__main__":
    import argparse
    
    arg_parser = argparse.ArgumentParser(
        description="Generate rules-based training data (Scenario 1)"
    )
    arg_parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=2,
        help="Number of samples to generate (default: 2)"
    )
    arg_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: data/qwen_dbr_training_final_v3.jsonl)"
    )
    
    args = arg_parser.parse_args()
    main(n=args.num_samples, output_file=args.output)
