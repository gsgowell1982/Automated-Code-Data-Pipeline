import os
import json
import uuid
import requests
import logging
import re
import time
from typing import List, Dict, Any

# ==========================================
# 1. 基础配置与环境
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data"))
REPO_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "repos", "fastapi-realworld-example-app"))
OUTPUT_FILE = os.path.join(DATA_DIR, "qwen_dbr_training_final_v3.jsonl")

TARGET_FILES = [
    os.path.join("app", "api", "routes", "authentication.py"),
    os.path.join("app", "api", "routes", "users.py")
]

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
GEN_TEMP = 0.7

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 2. DBR 知识基座
# ==========================================
DBR_01_CONTENT = """
DBR-01：身份准入与账户凭据完整性
1. 唯一性拦截：注册/更新时强制检查用户名/邮箱唯一性。
2. 存储安全：密码必须哈希处理。
3. 登录安全反馈：登录失败统一返回模糊错误信息。
4. 会话管理：成功后返回新 JWT 令牌。
"""

DBR_EVIDENCE_GUIDE = """
代码精准抽取指令：
1. 登录异常块：定位 login 函数，仅抽取 try-except 捕获 EntityDoesNotExist 的块。
2. 注册检查块：定位 register 函数，仅抽取校验 username/email 占用的 if 块。
3. 更新校验块：定位 update_current_user，仅抽取对比新旧值并校验唯一性的逻辑。
4. 令牌生成行：识别调用 create_access_token_for_user 的行。
"""


# ==========================================
# 3. 核心生成引擎
# ==========================================
def generate_precise_intent_qa(index: int, code_map: Dict[str, str]) -> Dict:
    scenarios = [
        {
            "topic": "身份认证异常模糊化",
            "role": "安全审计员",
            "intent_desc": "防止通过报错进行账户枚举探测",
            "forbidden": "注册, register, 唯一性"
        },
        {
            "topic": "新账户准入唯一性预检",
            "role": "首席架构师",
            "intent_desc": "注册环节的身份标识冲突拦截",
            "forbidden": "登录, login, 模糊反馈"
        },
        {
            "topic": "存量数据更新合规",
            "role": "合规官",
            "intent_desc": "修改资料时的唯一性一致性检查",
            "forbidden": "初次, 注册, register"
        }
    ]

    current = scenarios[index % len(scenarios)]
    code_context_str = "\n".join([f"--- File: {p} ---\n{c}" for p, c in code_map.items()])

    system_prompt = (
        f"Role: {current['role']}. Topic: {current['topic']}.\n"
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

    prompt = f"{system_prompt}\n\n[DBR]:\n{DBR_01_CONTENT}\n\n[Extraction Guide]:\n{DBR_EVIDENCE_GUIDE}\n\n[Source]:\n{code_context_str}"

    try:
        response = requests.post(OLLAMA_API, json={
            "model": MODEL_NAME, "prompt": prompt, "stream": False,
            "temperature": GEN_TEMP, "options": {"num_ctx": 4096}
        }, timeout=300)
        raw_text = response.json().get("response", "")

        def extract(tag):
            pattern = rf"\[\[{tag}\]\]:\s*(.*?)(?=\[\[|$)"
            match = re.search(pattern, raw_text, re.DOTALL)
            return match.group(1).strip() if match else ""

        # 清洗推理链中的冗余序号和占位符
        reasoning_raw = extract("REASONING")
        reasoning_clean = re.sub(r'(\d+[\.\s、\)]+|步骤\w：|步骤\d)', '', reasoning_raw).strip()
        steps = [s.strip() for s in re.split(r'[;\n]', reasoning_clean) if len(s.strip()) > 5]

        return {
            "instruction": extract("QUESTION"),
            "reasoning_steps": steps,
            "relevant_code": extract("CODE").replace("```python", "").replace("```", "").strip(),
            "answer_text": extract("ANSWER"),
            "intent_desc": current["intent_desc"]
        }
    except Exception as e:
        logging.error(f"Generate Error: {e}")
        return None


# ==========================================
# 4. Schema 封装 (严格对齐设计文档)
# ==========================================
def build_schema_entry(raw_item: Dict, file_path: str) -> Dict:
    # 构造金标准回答 (Answer)
    combined_answer = (
            "### 💡 推理链与合规逻辑\n"
            + "\n".join([f"- {step}" for step in raw_item["reasoning_steps"]])
            + "\n\n### 📝 业务方案解答\n"
            + raw_item["answer_text"]
            + "\n\n### 💻 核心源代码实现\n"
            + f"```python\n{raw_item['relevant_code']}\n```"
    )

    return {
        "sample_id": str(uuid.uuid4()),
        "instruction": raw_item["instruction"],
        "context": {
            "file_path": file_path,
            "related_dbr": "DBR-01",
            "code_snippet": raw_item["relevant_code"]
        },
        "auto_processing": {
            "parser": "multilingual_evidence_aligned_parser",
            "dbr_logic": f"DBR-01 Trigger: {raw_item['intent_desc']}",
            "data_cleaning": "Step-placeholder removal, Markdown code normalization"
        },
        "reasoning_trace": raw_item["reasoning_steps"],
        "answer": combined_answer,
        "data_quality": {
            "consistency_check": True,
            "language": "zh-cn",
            "temperature": GEN_TEMP
        }
    }


# ==========================================
# 5. 执行流程
# ==========================================
def main(n=2):
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

    code_map = {}
    for rel_path in TARGET_FILES:
        full_path = os.path.join(REPO_PATH, rel_path)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                code_map[rel_path] = f.read()[:4000]

    logging.info(f"开始生成符合设计文档的语料 (目标: {n} 条)...")
    success_count = 0

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        while success_count < n:
            current_rel_path = TARGET_FILES[success_count % len(TARGET_FILES)]

            raw = generate_precise_intent_qa(success_count, code_map)

            # 基础过滤逻辑
            if raw and len(raw["instruction"]) > 5 and len(raw["relevant_code"]) > 5:
                # 排除提问中包含函数名的样本
                forbidden_words = ["login", "register", "update_current_user", "函数"]
                if any(w in raw["instruction"].lower() for w in forbidden_words):
                    continue

                final_entry = build_schema_entry(raw, current_rel_path)

                # --- 终端全量预览 (满足用户看到回答的要求) ---
                print("\n" + "=" * 80)
                print(f" [写入成功] ID: {final_entry['sample_id']}")
                print(f"【问题 (Instruction)】: {final_entry['instruction']}")
                print("-" * 40)
                print(f"【回答 (Answer)】:\n{final_entry['answer']}")
                print("=" * 80 + "\n")

                f.write(json.dumps(final_entry, ensure_ascii=False) + "\n")
                success_count += 1
                time.sleep(1)

    logging.info(f"任务结束。文件已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main(n=2)