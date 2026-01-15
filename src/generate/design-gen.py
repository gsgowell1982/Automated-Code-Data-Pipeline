#!/usr/bin/env python3
"""
Scenario 2: Design-Based Training Data Generation

This script generates training data for architectural design schemes
based on Domain Business Rules (DBR). The generated samples include
design solutions with reasoning traces that explain the design decisions.

The generated data is suitable for fine-tuning LLMs to produce
architecture-aware design recommendations.
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
from parser.response_parser import ResponseParser
from validator.schema_validator import (
    SchemaValidator,
    ScenarioType,
    create_design_sample
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================
# Design Requirements
# ==========================================
DESIGN_REQUIREMENTS = [
    "实现用户注册接口，需包含身份校验逻辑。",
    "设计安全的登录 API，要求抵御账户枚举攻击。",
    "设计用户个人资料更新接口，特别是针对邮箱变更的安全性设计。",
    "设计基于 JWT 的全流程认证颁发方案。"
]


def build_design_system_prompt(role: str, lang_instruction: str) -> str:
    """
    Build the system prompt for design generation.
    
    Args:
        role: The role persona for generation
        lang_instruction: Language-specific instructions
        
    Returns:
        System prompt string
    """
    return (
        f"你现在是一位【{role}】。请针对以下需求，基于【DBR 规则】给出顶层设计方案。\n"
        f"语言：{lang_instruction}\n"
        "【方案结构要求】：\n"
        "1. [[REASONING]]: 简述设计思路。说明如果不遵守 DBR 规则将面临的安全风险。\n"
        "2. [[DESIGN_SOLUTION]]: 包含步骤说明和 Markdown 伪代码块。\n"
        "注意：严禁开场白，直接输出 [[TAG]] 内容。"
    )


def build_design_prompt(
    system_prompt: str,
    requirement: str,
    dbr_content: str
) -> str:
    """
    Build the complete prompt for design generation.
    
    Args:
        system_prompt: The system instructions
        requirement: The design requirement
        dbr_content: The DBR rule content
        
    Returns:
        Complete prompt string
    """
    return (
        f"{system_prompt}\n\n"
        f"【用户需求】: {requirement}\n\n"
        f"【DBR 规则】:\n{dbr_content}"
    )


def generate_design_sample(
    llm_client: OllamaClient,
    parser: ResponseParser,
    requirement_index: int,
    config: Config,
    lang: str = "zh-cn"
) -> Optional[Dict[str, Any]]:
    """
    Generate a single design sample.
    
    Args:
        llm_client: The LLM client instance
        parser: The response parser
        requirement_index: Index for selecting design requirement
        config: Configuration instance
        lang: Language code
        
    Returns:
        Generated sample dictionary or None if generation fails
    """
    # Get language config
    lang_config = get_lang_config(lang)
    
    # Select requirement
    requirement = DESIGN_REQUIREMENTS[requirement_index % len(DESIGN_REQUIREMENTS)]
    role = "高级架构师"
    
    # Build prompts
    system_prompt = build_design_system_prompt(
        role=role,
        lang_instruction=lang_config["instruction"]
    )
    
    full_prompt = build_design_prompt(
        system_prompt=system_prompt,
        requirement=requirement,
        dbr_content=DBR_01.get_content()
    )
    
    # Generate response
    response = llm_client.generate(
        prompt=full_prompt,
        temperature=config.generation_temp,
        num_predict=config.max_predict
    )
    
    if not response.success:
        logger.error(f"Generation failed: {response.error}")
        return None
    
    # Parse the response
    parsed = parser.parse_design_response(response.text)
    
    # Validate parsed response
    if not parser.validate_design_response(parsed):
        logger.warning("Parsed design response validation failed")
        return None
    
    return {
        "instruction": requirement,
        "reasoning_steps": parsed["reasoning_steps"],
        "answer_text": parsed["design_solution"]
    }


def print_design_sample(index: int, raw_data: Dict[str, Any]) -> None:
    """Print a preview of the generated design sample to terminal."""
    separator = "=" * 70
    sub_separator = "-" * 50
    
    print(f"\n{separator}")
    print(f"  [ 样本生成监控 ] 序号: {index + 1}")
    print(f"{separator}")
    print(f"【用户需求】:\n{raw_data['instruction']}")
    print(f"\n{sub_separator}")
    print(f"【推理过程 (Reasoning Trace)】:")
    
    for i, step in enumerate(raw_data['reasoning_steps']):
        print(f"  {i + 1}. {step}")
    
    print(f"{sub_separator}")
    print(f"【设计方案 (Design Answer)】:\n{raw_data['answer_text']}")
    print(f"{separator}\n")


def main(n: int = 2, lang: str = "zh-cn", output_file: Optional[str] = None):
    """
    Main execution function for design-based data generation.
    
    Args:
        n: Number of samples to generate
        lang: Language code for generation
        output_file: Optional custom output file path
    """
    # Initialize configuration
    config = get_config()
    config.ensure_data_dir()
    
    # Set output file
    if output_file is None:
        output_file = config.get_output_path("qwen_dbr_design_data_v5.jsonl")
    
    # Initialize components
    llm_client = OllamaClient(
        api_url=config.ollama_api,
        model_name=config.model_name,
        default_temperature=config.generation_temp,
        context_window=config.context_window,
        max_predict=config.max_predict
    )
    parser = ResponseParser()
    validator = SchemaValidator(scenario=ScenarioType.DESIGN_BASED)
    
    logger.info(f"开始生成场景 2 设计方案, 目标: {n} 条...")
    
    success_count = 0
    attempt_count = 0
    max_attempts = n * 5  # Prevent infinite loops
    
    with open(output_file, 'a', encoding='utf-8') as f:
        while success_count < n and attempt_count < max_attempts:
            attempt_count += 1
            
            # Generate sample
            raw_data = generate_design_sample(
                llm_client=llm_client,
                parser=parser,
                requirement_index=success_count,
                config=config,
                lang=lang
            )
            
            if raw_data is None:
                continue
            
            # Print terminal preview
            print_design_sample(success_count, raw_data)
            
            # Build final sample using factory function
            final_entry = create_design_sample(
                instruction=raw_data["instruction"],
                reasoning_steps=raw_data["reasoning_steps"],
                design_solution=raw_data["answer_text"],
                dbr_id="DBR-01",
                architecture_context="FastAPI + SQLAlchemy + Repository Pattern",
                design_standard="Security-First API Design",
                language=lang,
                temperature=config.generation_temp
            )
            
            # Validate sample
            validation_result = validator.validate_sample(final_entry)
            if not validation_result.valid:
                logger.warning(f"Validation failed: {validation_result.errors}")
                continue
            
            # Log any warnings
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.debug(f"Validation warning: {warning}")
            
            # Save sample
            f.write(json.dumps(final_entry, ensure_ascii=False) + "\n")
            success_count += 1
            logger.info(f"已成功持久化第 {success_count} 条样本。")
    
    logger.info(f"任务圆满完成。成功生成 {success_count} 条样本。文件已保存至: {output_file}")


if __name__ == "__main__":
    import argparse
    
    arg_parser = argparse.ArgumentParser(
        description="Generate design-based training data (Scenario 2)"
    )
    arg_parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=2,
        help="Number of samples to generate (default: 2)"
    )
    arg_parser.add_argument(
        "-l", "--lang",
        type=str,
        default="zh-cn",
        choices=["zh-cn", "en"],
        help="Language for generation (default: zh-cn)"
    )
    arg_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: data/qwen_dbr_design_data_v5.jsonl)"
    )
    
    args = arg_parser.parse_args()
    main(n=args.num_samples, lang=args.lang, output_file=args.output)
