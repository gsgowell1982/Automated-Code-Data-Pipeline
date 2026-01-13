import os
import json
import uuid
import requests
import logging
import re
import time
from typing import List, Dict, Any

# --- 1. 基础配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data"))
OUTPUT_FILE = os.path.join(DATA_DIR, "qwen_dbr_design_data_v5.jsonl")

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

# --- 2. 多语言扩展配置 ---
LANG_CONFIG = {
    "zh-cn": {
        "instruction": "请使用中文进行方案设计，并确保包含逻辑伪代码。",
        "schema_lang": "zh-cn"
    },
    "en": {
        "instruction": "Please provide the design solution in English, including logical pseudocode.",
        "schema_lang": "en"
    }
}

# --- 3. DBR 规则 ---
DBR_01_CONTENT = """
DBR-01：身份准入与账户凭据完整性（Authentication & Credential Integrity）
1. 唯一性拦截：注册/更新必须检查用户名/邮箱唯一性。
2. 存储安全：密码必须进行哈希加密存储。
3. 模糊反馈：认证失败统一返回 INCORRECT_LOGIN_INPUT，防止账户枚举。
4. 会话管理：操作成功必须生成并返回新的 JWT Token。
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# --- 4. 辅助打印函数 (终端监控) ---
def print_sample(index: int, raw_data: Dict):
    """在终端美化打印生成的问答对"""
    separator = "=" * 70
    sub_separator = "-" * 50
    print(f"\n{separator}")
    print(f"  [ 样本生成监控 ] 序号: {index + 1}")
    print(f"{separator}")
    print(f"【用户需求】:\n{raw_data['instruction']}")
    print(f"\n{sub_separator}")
    print(f"【推理过程 (Reasoning Trace)】:")
    # 此处 i+1 会提供干净的序号
    for i, step in enumerate(raw_data['reasoning_steps']):
        print(f"  {i + 1}. {step}")
    print(f"{sub_separator}")
    print(f"【设计方案 (Design Answer)】:\n{raw_data['answer_text']}")
    print(f"{separator}\n")


# --- 5. 核心生成逻辑 (含正则序号清洗) ---
def generate_design_qa_v5(dbr_text: str, index: int, lang: str = "zh-cn", max_retries: int = 3) -> Dict:
    l_cfg = LANG_CONFIG.get(lang, LANG_CONFIG["zh-cn"])

    requirements = [
        "实现用户注册接口，需包含身份校验逻辑。",
        "设计安全的登录 API，要求抵御账户枚举攻击。",
        "设计用户个人资料更新接口，特别是针对邮箱变更的安全性设计。",
        "设计基于 JWT 的全流程认证颁发方案。"
    ]

    current_req = requirements[index % len(requirements)]
    current_role = "高级架构师"

    system_prompt = (
        f"你现在是一位【{current_role}】。请针对以下需求，基于【DBR 规则】给出顶层设计方案。\n"
        f"语言：{l_cfg['instruction']}\n"
        "【方案结构要求】：\n"
        "1. [[REASONING]]: 简述设计思路。说明如果不遵守 DBR 规则将面临的安全风险。\n"
        "2. [[DESIGN_SOLUTION]]: 包含步骤说明和 Markdown 伪代码块。\n"
        "注意：严禁开场白，直接输出 [[TAG]] 内容。"
    )

    prompt = f"{system_prompt}\n\n【用户需求】: {current_req}\n\n【DBR 规则】:\n{dbr_text}"

    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_API, json={
                "model": MODEL_NAME, "prompt": prompt, "stream": False,
                "temperature": 0.7, "options": {"num_ctx": 4096, "num_predict": 1200}
            }, timeout=600)

            raw_text = response.json().get("response", "")

            def extract(tag):
                pattern = rf"\[\[{tag}\]\]:\s*(.*?)(?=\[\[|$)"
                match = re.search(pattern, raw_text, re.DOTALL)
                return match.group(1).strip() if match else ""

            # --- 核心修复：清洗推理步骤中的冗余序号 ---
            reasoning_raw = extract("REASONING")
            # 使用正则删掉行首的 "1. ", "1) ", "步骤一：" 等干扰字符
            reasoning_clean = re.sub(r'^\s*(\d+[\.\s、\)]+|步骤\w：)', '', reasoning_raw, flags=re.MULTILINE)
            # 兼容分号和换行进行切分
            steps = [s.strip() for s in re.split(r'[;\n]', reasoning_clean) if s.strip()]

            data = {
                "instruction": current_req,
                "reasoning_steps": steps,
                "answer_text": extract("DESIGN_SOLUTION")
            }
            if data["answer_text"]: return data
        except Exception as e:
            logging.warning(f"尝试 {attempt + 1} 失败: {e}")
            time.sleep(2)

    return None


def build_design_schema(raw_item: Dict, lang: str) -> Dict:
    return {
        "sample_id": str(uuid.uuid4()),
        "instruction": raw_item["instruction"],
        "context": {
            "related_dbr": "DBR-01",
            "architecture_context": "FastAPI + SQLAlchemy + Repository Pattern",
            "design_standard": "Security-First API Design"
        },
        "auto_processing": {
            "parser": "design_logic_generator_v5",
            "feature": "Regex-based-formatting & Logic-pumping"
        },
        "reasoning_trace": raw_item["reasoning_steps"],
        "answer": raw_item["answer_text"],
        "data_quality": {"consistency_check": True, "language": lang, "temperature": 0.7}
    }


# --- 6. 主程序 ---
def main(n=2, lang="zh-cn"):
    ensure_dir(DATA_DIR)
    logging.info(f"开始生成场景 2 设计方案, 目标: {n} 条...")
    success_count = 0

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        while success_count < n:
            raw_data = generate_design_qa_v5(DBR_01_CONTENT, success_count, lang)
            if raw_data:
                # 终端实时展示清洗后的结果
                print_sample(success_count, raw_data)

                # 持久化存储
                final_entry = build_design_schema(raw_data, lang)
                f.write(json.dumps(final_entry, ensure_ascii=False) + "\n")

                success_count += 1
                logging.info(f"已成功持久化第 {success_count} 条样本。")

    logging.info("任务圆满完成。")


if __name__ == "__main__":

    main(n=2, lang="zh-cn")
