"""
Answer Composer v2 for Design Q&A Generation.

Generates targeted answers that directly address the question intent,
then provide design solutions with pseudo-code illustrations.

Fixes the "template-based answer" problem where answers don't
directly respond to the specific question.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    DesignScenario, DesignSolution, GeneratedDesignQuestion,
    QuestionIntent, DesignAspect
)


class IntentBasedAnswerComposer:
    """
    Composes answers that directly address the question intent.
    
    Structure:
    1. Direct answer to the specific question (intent-based)
    2. Design rationale and approach
    3. Pseudo-code illustration
    4. Additional considerations
    """
    
    # Direct answers for "WHY" questions by aspect
    WHY_ANSWERS = {
        "en": {
            # Why shouldn't we expose which field failed
            (QuestionIntent.WHY_THIS_WAY, DesignAspect.ERROR_HANDLING): {
                "direct_answer": """**Direct Answer**: Exposing which specific field (username or email) caused a validation error enables **user enumeration attacks**. 

An attacker can systematically probe your registration endpoint:
- If error says "email already exists" → attacker learns this email is registered
- If error says "username taken" → attacker confirms valid usernames

This information leakage allows attackers to:
1. Build a list of valid user accounts for targeted phishing
2. Attempt credential stuffing with known-valid emails
3. Identify high-value targets (e.g., admin@company.com exists)

**Security Principle**: Authentication-related errors should be **intentionally vague** to provide the same response regardless of which validation failed.""",
            },
            
            # Why validate before write
            (QuestionIntent.WHY_THIS_WAY, DesignAspect.VALIDATION): {
                "direct_answer": """**Direct Answer**: Validating before database writes prevents **partial data corruption** and ensures **atomicity**.

If validation happens after or during writes:
- Failed validation mid-write leaves orphan/incomplete records
- Rollback complexity increases significantly
- Race conditions become harder to handle
- Database constraints may throw less user-friendly errors

**Design Principle**: The "Guard Clause" pattern ensures that if ANY validation fails, the function exits immediately with NO side effects.""",
            },
            
            # Why use generic error messages for auth
            (QuestionIntent.WHY_THIS_WAY, DesignAspect.SECURITY): {
                "direct_answer": """**Direct Answer**: Generic authentication error messages prevent **user enumeration** - a critical OWASP Top 10 vulnerability.

Specific errors like "Invalid password" vs "User not found" reveal:
- Whether an account exists (valuable for attackers)
- Which credential component failed (reduces brute-force space)

**Security Requirement**: Both "wrong password" and "user not found" must return identical responses with identical timing to prevent:
1. Error message-based enumeration
2. Timing-based enumeration (response time differences)""",
            },
            
            # Why consistent timing matters
            (QuestionIntent.WHY_THIS_WAY, DesignAspect.BEST_PRACTICES): {
                "direct_answer": """**Direct Answer**: Consistent response timing prevents **timing attacks** where attackers measure response delays to infer information.

Example vulnerability:
- Checking if user exists: 5ms (fast database lookup)
- Verifying password hash: 100ms (slow by design)

An attacker measuring response times can determine which step failed, enabling enumeration without seeing the error message.

**Mitigation**: Always perform password verification (even with dummy hash) regardless of whether user exists.""",
            },
        },
        "zh": {
            (QuestionIntent.WHY_THIS_WAY, DesignAspect.ERROR_HANDLING): {
                "direct_answer": """**直接回答**：暴露具体哪个字段（用户名或邮箱）导致验证错误会使系统面临**用户枚举攻击**。

攻击者可以系统性地探测注册端点：
- 如果错误显示"邮箱已存在" → 攻击者得知该邮箱已注册
- 如果错误显示"用户名已占用" → 攻击者确认有效用户名

这种信息泄露使攻击者能够：
1. 建立有效用户账户列表用于定向钓鱼
2. 使用已知有效邮箱进行凭据填充攻击
3. 识别高价值目标（例如 admin@company.com 存在）

**安全原则**：与认证相关的错误应该**故意模糊**，无论哪个验证失败都返回相同响应。""",
            },
            
            (QuestionIntent.WHY_THIS_WAY, DesignAspect.VALIDATION): {
                "direct_answer": """**直接回答**：在数据库写入之前进行验证可以防止**部分数据损坏**并确保**原子性**。

如果验证发生在写入之后或期间：
- 写入中途的验证失败会留下孤立/不完整的记录
- 回滚复杂性显著增加
- 竞态条件变得更难处理
- 数据库约束可能抛出不够友好的错误

**设计原则**："守卫子句"模式确保如果任何验证失败，函数立即退出且没有任何副作用。""",
            },
            
            (QuestionIntent.WHY_THIS_WAY, DesignAspect.SECURITY): {
                "direct_answer": """**直接回答**：通用的认证错误消息可以防止**用户枚举**——这是OWASP Top 10中的关键漏洞。

具体错误如"密码无效"与"用户不存在"会泄露：
- 账户是否存在（对攻击者有价值）
- 哪个凭据组件失败（减少暴力破解空间）

**安全要求**："密码错误"和"用户不存在"必须返回完全相同的响应和完全相同的时间，以防止：
1. 基于错误消息的枚举
2. 基于时序的枚举（响应时间差异）""",
            },
        },
    }
    
    # Pseudo-code templates by scenario
    PSEUDO_CODE_TEMPLATES = {
        "DBR-01-01": {
            "en": '''```
# Pseudo-code: Secure Registration with Validation

def register_user(username, email, password):
    # Guard Clause 1: Username uniqueness
    if is_username_taken(username):
        raise ValidationError("Registration failed")  # Generic message
    
    # Guard Clause 2: Email uniqueness  
    if is_email_taken(email):
        raise ValidationError("Registration failed")  # Same generic message
    
    # Only reach here if ALL validations pass
    hashed_password = secure_hash(password)
    user = create_user(username, email, hashed_password)
    
    return generate_token(user)
```''',
            "zh": '''```
# 伪代码：带验证的安全注册

def register_user(username, email, password):
    # 守卫子句 1：用户名唯一性
    if is_username_taken(username):
        raise ValidationError("注册失败")  # 通用消息
    
    # 守卫子句 2：邮箱唯一性
    if is_email_taken(email):
        raise ValidationError("注册失败")  # 相同的通用消息
    
    # 仅在所有验证通过后才执行到这里
    hashed_password = secure_hash(password)
    user = create_user(username, email, hashed_password)
    
    return generate_token(user)
```''',
        },
        "DBR-01-02": {
            "en": '''```
# Pseudo-code: Atomic Account Creation

def create_account(user_data):
    # Step 1: Hash password BEFORE any database operation
    hashed = bcrypt.hash(user_data.password, cost=12)
    
    # Step 2: Atomic transaction
    with database.transaction():
        user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=hashed  # Never store plaintext
        )
        database.insert(user)
        # If ANY step fails, entire transaction rolls back
    
    return user
```''',
            "zh": '''```
# 伪代码：原子账户创建

def create_account(user_data):
    # 步骤 1：在任何数据库操作之前哈希密码
    hashed = bcrypt.hash(user_data.password, cost=12)
    
    # 步骤 2：原子事务
    with database.transaction():
        user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=hashed  # 永不存储明文
        )
        database.insert(user)
        # 如果任何步骤失败，整个事务回滚
    
    return user
```''',
        },
        "DBR-01-03": {
            "en": '''```
# Pseudo-code: Secure Authentication

def authenticate(email, password):
    # CRITICAL: Define generic error BEFORE any logic
    auth_error = AuthError("Invalid credentials")  # Same for all failures
    
    try:
        user = find_user_by_email(email)
    except UserNotFound:
        # IMPORTANT: Still verify a dummy hash to prevent timing attacks
        bcrypt.verify(password, DUMMY_HASH)
        raise auth_error
    
    if not bcrypt.verify(password, user.password_hash):
        raise auth_error  # Same error as user-not-found
    
    return generate_session(user)
```''',
            "zh": '''```
# 伪代码：安全认证

def authenticate(email, password):
    # 关键：在任何逻辑之前定义通用错误
    auth_error = AuthError("凭据无效")  # 所有失败使用相同消息
    
    try:
        user = find_user_by_email(email)
    except UserNotFound:
        # 重要：仍然验证虚拟哈希以防止时序攻击
        bcrypt.verify(password, DUMMY_HASH)
        raise auth_error
    
    if not bcrypt.verify(password, user.password_hash):
        raise auth_error  # 与用户不存在相同的错误
    
    return generate_session(user)
```''',
        },
        "DBR-01-04": {
            "en": '''```
# Pseudo-code: JWT Token Management

def generate_token(user):
    claims = {
        "sub": user.id,
        "email": user.email,
        "exp": now() + ACCESS_TOKEN_LIFETIME,  # Short-lived
        "iat": now()
    }
    return jwt.encode(claims, SECRET_KEY, algorithm="HS256")

def verify_token(token):
    try:
        claims = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return claims
    except jwt.ExpiredSignatureError:
        raise AuthError("Token expired")
    except jwt.InvalidTokenError:
        raise AuthError("Invalid token")
```''',
            "zh": '''```
# 伪代码：JWT 令牌管理

def generate_token(user):
    claims = {
        "sub": user.id,
        "email": user.email,
        "exp": now() + ACCESS_TOKEN_LIFETIME,  # 短期有效
        "iat": now()
    }
    return jwt.encode(claims, SECRET_KEY, algorithm="HS256")

def verify_token(token):
    try:
        claims = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return claims
    except jwt.ExpiredSignatureError:
        raise AuthError("令牌已过期")
    except jwt.InvalidTokenError:
        raise AuthError("无效令牌")
```''',
        },
    }
    
    def compose(
        self,
        question: GeneratedDesignQuestion,
        scenario: DesignScenario,
        solution: DesignSolution,
        reasoning: List[str],
        language: str = "en"
    ) -> str:
        """
        Compose a targeted answer that directly addresses the question.
        
        Structure:
        1. Direct answer (intent-specific)
        2. Design approach
        3. Pseudo-code illustration
        4. Key considerations
        """
        parts = []
        
        # 1. Get intent-specific direct answer
        direct_answer = self._get_direct_answer(question, language)
        if direct_answer:
            parts.append(direct_answer)
        
        # 2. Design approach (shorter, more focused)
        approach_section = self._compose_approach_section(solution, language)
        parts.append(approach_section)
        
        # 3. Pseudo-code illustration
        pseudo_code = self._get_pseudo_code(scenario.scenario_id, language)
        if pseudo_code:
            if language == "zh":
                parts.append(f"\n### 设计实现示例\n\n{pseudo_code}")
            else:
                parts.append(f"\n### Design Implementation Example\n\n{pseudo_code}")
        
        # 4. Key considerations (condensed)
        considerations = self._compose_considerations(solution, question, language)
        parts.append(considerations)
        
        return "\n".join(parts)
    
    def _get_direct_answer(
        self,
        question: GeneratedDesignQuestion,
        language: str
    ) -> Optional[str]:
        """Get intent-specific direct answer."""
        lang_answers = self.WHY_ANSWERS.get(language, self.WHY_ANSWERS.get("en", {}))
        
        # Try exact match
        key = (question.intent, question.aspect)
        if key in lang_answers:
            return lang_answers[key].get("direct_answer", "")
        
        # Try intent-only match
        for (intent, aspect), data in lang_answers.items():
            if intent == question.intent:
                return data.get("direct_answer", "")
        
        # Generate generic direct answer based on intent
        return self._generate_generic_direct_answer(question, language)
    
    def _generate_generic_direct_answer(
        self,
        question: GeneratedDesignQuestion,
        language: str
    ) -> str:
        """Generate a generic but targeted direct answer."""
        intent = question.intent
        
        if language == "zh":
            intros = {
                QuestionIntent.HOW_TO_DESIGN: "**设计方法**：",
                QuestionIntent.WHY_THIS_WAY: "**原因分析**：这种设计选择基于以下关键考量：",
                QuestionIntent.WHAT_PATTERN: "**推荐模式**：针对此场景，推荐使用以下设计模式：",
                QuestionIntent.TRADE_OFF: "**权衡分析**：",
                QuestionIntent.ALTERNATIVE: "**替代方案**：除了推荐方案外，还可以考虑：",
                QuestionIntent.BEST_PRACTICE: "**最佳实践**：行业标准建议：",
                QuestionIntent.SECURITY_CONCERN: "**安全考量**：此设计需要防范以下安全风险：",
            }
        else:
            intros = {
                QuestionIntent.HOW_TO_DESIGN: "**Design Approach**:",
                QuestionIntent.WHY_THIS_WAY: "**Rationale**: This design choice is based on the following key considerations:",
                QuestionIntent.WHAT_PATTERN: "**Recommended Patterns**: For this scenario, the following design patterns are recommended:",
                QuestionIntent.TRADE_OFF: "**Trade-off Analysis**:",
                QuestionIntent.ALTERNATIVE: "**Alternatives**: Besides the recommended approach, consider:",
                QuestionIntent.BEST_PRACTICE: "**Best Practices**: Industry standards recommend:",
                QuestionIntent.SECURITY_CONCERN: "**Security Considerations**: This design must protect against:",
            }
        
        return intros.get(intent, "")
    
    def _compose_approach_section(self, solution: DesignSolution, language: str) -> str:
        """Compose the design approach section (condensed)."""
        if language == "zh":
            return f"""
### 推荐设计方案

**方法**: {solution.approach}

**核心原理**: {solution.rationale}

**关键组件**: {', '.join(solution.components[:3])}

**设计模式**: {', '.join(solution.patterns_used)}"""
        else:
            return f"""
### Recommended Design Approach

**Method**: {solution.approach}

**Core Rationale**: {solution.rationale}

**Key Components**: {', '.join(solution.components[:3])}

**Design Patterns**: {', '.join(solution.patterns_used)}"""
    
    def _get_pseudo_code(self, scenario_id: str, language: str) -> Optional[str]:
        """Get pseudo-code for the scenario."""
        templates = self.PSEUDO_CODE_TEMPLATES.get(scenario_id, {})
        return templates.get(language, templates.get("en"))
    
    def _compose_considerations(
        self,
        solution: DesignSolution,
        question: GeneratedDesignQuestion,
        language: str
    ) -> str:
        """Compose considerations section based on question focus."""
        # Prioritize security for security-related questions
        if question.intent == QuestionIntent.SECURITY_CONCERN or question.aspect == DesignAspect.SECURITY:
            primary = solution.security_considerations
            secondary = solution.trade_offs[:2]
        else:
            primary = solution.trade_offs
            secondary = solution.security_considerations[:2]
        
        if language == "zh":
            return f"""
### 关键考量

**主要关注点**:
{chr(10).join(f'- {item}' for item in primary)}

**其他注意事项**:
{chr(10).join(f'- {item}' for item in secondary)}"""
        else:
            return f"""
### Key Considerations

**Primary Concerns**:
{chr(10).join(f'- {item}' for item in primary)}

**Additional Notes**:
{chr(10).join(f'- {item}' for item in secondary)}"""
