import os
import json
import uuid
import requests
import logging
import random
import re
from typing import List, Dict, Any

# ==========================================
# 1. 基础配置 (已修复变量定义顺序)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data"))
# 请确保此路径指向您的真实仓库
REPO_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "repos", "fastapi-realworld-example-app"))
OUTPUT_FILE = os.path.join(DATA_DIR, "qwen_dbr_train_data_v23.jsonl")

# 必须在 main 调用前定义的常量
TARGET_FILES = [
    os.path.join("app", "api", "routes", "authentication.py"),
    os.path.join("app", "api", "routes", "users.py")
]

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

# ==========================================
# 2. 多语言扩展接口配置
# ==========================================
LANG_CONFIG = {
    "zh-cn": {
        "desc": "Chinese (Simplified)",
        "instruction": "请使用中文进行提问和回答。",
        "tags": {
            "QUESTION": "问题",
            "REASONING": "推理步骤",
            "CODE": "代码片段",
            "ANSWER": "详细解答",
            "FILE": "文件路径"
        }
    },
    "en": {
        "desc": "English",
        "instruction": "Please generate the question and answer in English.",
        "tags": {
            "QUESTION": "Question",
            "REASONING": "Reasoning Steps",
            "CODE": "Code Snippet",
            "ANSWER": "Detailed Answer",
            "FILE": "File Path"
        }
    }
}

# ==========================================
# 3. DBR 规则与代码依据
# ==========================================
DBR_01_CONTENT = """
DBR-01：身份准入与账户凭据完整性（Authentication & Credential Integrity）
1. 多场景唯一性拦截：注册/更新时强制检查用户名/邮箱唯一性。
2. 存储原子性与哈希：密码必须哈希处理。
3. 登录安全反馈：登录失败统一返回模糊错误信息。
4. 动态会话：成功后返回新 JWT 令牌。
"""

DBR_EVIDENCE_GUIDE = """
代码精准抽取指令：
1. 统一登录异常：定位 login 函数，仅抽取 try-except 捕获 EntityDoesNotExist 并抛出 wrong_login_error 的逻辑块。
2. 注册预检：定位 register 函数，仅抽取校验 check_username_is_taken 和 check_email_is_taken 的 if 块。
3. 条件式更新：定位 update_current_user，仅抽取对比新旧值 (user_update.email != current_user.email) 的校验逻辑。
4. 令牌刷新：识别各函数末尾调用 create_access_token_for_user 的行。
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==========================================
# 4. 工具函数
# ==========================================
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_code_files(repo_path: str, relative_paths: List[str]) -> Dict[str, str]:
    code_context = {}
    for rel_path in relative_paths:
        full_path = os.path.join(repo_path, rel_path)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                code_context[rel_path] = f.read()[:4000]
        else:
            logging.warning(f"文件不存在: {full_path}")
    return code_context


# ==========================================
# 5. 核心生成逻辑 (多语言与精准抽取)
# ==========================================
def generate_single_qa_precise(dbr_text: str, evidence_guide: str, code_map: Dict[str, str], index: int,
                               lang: str = "zh-cn") -> Dict:
    l_cfg = LANG_CONFIG.get(lang, LANG_CONFIG["zh-cn"])

    scenarios = [
        {"topic": "登录异常处理", "focus": "login 函数中的异常捕获与模糊反馈", "role": "安全专家"},
        {"topic": "注册唯一性检查", "focus": "register 函数中的冲突拦截逻辑", "role": "后端开发"},
        {"topic": "用户信息更新", "focus": "update_current_user 的条件查重性能", "role": "架构师"}
    ]

    current = scenarios[index % len(scenarios)]
    code_context_str = "\n".join([f"--- File: {p} ---\n{c}" for p, c in code_map.items()])

    system_prompt = (
        f"Role: {current['role']}. Focus: {current['topic']}.\n"
        f"Language Constraint: {l_cfg['instruction']}\n"
        f"Requirement: Keep [[CODE]] extremely minimal (No decorators, no Depends).\n"
        "Strict Format:\n"
        "[[QUESTION]]: content\n"
        "[[REASONING]]: step1; step2\n"
        "[[CODE]]: snippet\n"
        "[[ANSWER]]: content\n"
        "[[FILE_PATH]]: path"
    )

    prompt = f"{system_prompt}\n\n[DBR Rules]:\n{dbr_text}\n{evidence_guide}\n\n[Source Code]:\n{code_context_str}"

    try:
        response = requests.post(OLLAMA_API, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,
            "num_predict": 1000
        }, timeout=200)

        raw_text = response.json().get("response", "")

        def extract(tag):
            pattern = rf"\[\[{tag}\]\]:\s*(.*?)(?=\[\[|$)"
            match = re.search(pattern, raw_text, re.DOTALL)
            return match.group(1).strip() if match else ""

        return {
            "instruction": extract("QUESTION"),
            "reasoning_steps": [s.strip() for s in extract("REASONING").split(";") if s.strip()],
            "relevant_code": extract("CODE"),
            "answer_text": extract("ANSWER"),
            "file_source": extract("FILE_PATH")
        }
    except Exception as e:
        logging.error(f"Generate error: {e}")
        return None


def build_schema_entry(raw_item: Dict, lang: str) -> Dict:
    raw_code = raw_item["relevant_code"].replace("```python", "").replace("```", "").strip()
    clean_lines = [
        line for line in raw_code.split('\n')
        if not any(x in line for x in ['@router', 'status_code', 'response_model', 'Depends('])
    ]
    clean_code = "\n".join(clean_lines).strip()

    return {
        "sample_id": str(uuid.uuid4()),
        "instruction": raw_item["instruction"],
        "context": {
            "file_path": raw_item["file_source"] or "app/api/routes/authentication.py",
            "related_dbr": "DBR-01",
            "code_snippet": clean_code
        },
        "auto_processing": {
            "parser": "multilingual_evidence_aligned_parser",
            "dbr_logic": "Authentication & Credential Integrity",
            "data_cleaning": f"Language: {lang}, Logic-block extraction"
        },
        "reasoning_trace": raw_item["reasoning_steps"],
        "answer": f"{raw_item['answer_text']}\n\n### Business Logic:\n```python\n{clean_code}\n```",
        "data_quality": {
            "consistency_check": True,
            "language": lang,
            "temperature": 0.3
        }
    }


def main(n=3, lang="zh-cn"):
    ensure_dir(OUTPUT_FILE)
    logging.info(f"Target Language: {lang}. Reading code...")

    # 修复：确保 read_code_files 接收正确的参数
    code_map = read_code_files(REPO_PATH, TARGET_FILES)

    if not code_map:
        logging.error("未能读取代码文件，请检查 REPO_PATH 或 TARGET_FILES。")
        return

    logging.info(f"开始生成 {n} 条数据...")
    success_count = 0
    attempt = 0

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        while success_count < n and attempt < n * 3:
            attempt += 1
            logging.info(f"正在生成第 {success_count + 1} 条 (尝试: {attempt})...")

            raw_data = generate_single_qa_precise(DBR_01_CONTENT, DBR_EVIDENCE_GUIDE, code_map, success_count, lang)

            if raw_data and raw_data["instruction"]:
                final_entry = build_schema_entry(raw_data, lang)
                f.write(json.dumps(final_entry, ensure_ascii=False) + "\n")
                success_count += 1
                logging.info(f"成功保存: {raw_data['instruction'][:20]}...")

    logging.info(f"任务结束。文件路径: {OUTPUT_FILE}")


# ==========================================
# 6. 执行入口
# ==========================================
if __name__ == "__main__":
    # 可以通过修改参数切换语言，例如 main(n=3, lang="en")
    main(n=3, lang="zh-cn")