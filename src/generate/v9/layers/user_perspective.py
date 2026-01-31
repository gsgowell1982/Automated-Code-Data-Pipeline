"""
User Perspective Layer for Q&A Generation Engine v9.

Transforms code-level evidence into business-friendly context
that doesn't expose internal function names or variables.

From v6: Fixes "god's eye view" problem.
From v8: Filters edge_cases based on actual code behavior.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import BusinessContext, CodeFacts, UserRole


class UserPerspectiveLayer:
    """
    Transforms code evidence into user-perspective business context.
    
    The LLM receives business scenarios, NOT function names.
    This prevents "hallucination dependency" where model only
    identifies issues when seeing specific code names.
    """
    
    # Scenario contexts for each subcategory
    SCENARIO_CONTEXTS = {
        "DBR-01-01": {
            "en": {
                "scenario_name": "User Registration & Profile Uniqueness",
                "business_flow": "Validate uniqueness before creating/updating user",
                "user_experience": "User submits registration; system checks uniqueness; rejects or accepts",
                "edge_cases": [
                    "Email already registered by another user",
                    "Username already taken",
                    "Same user updating to taken email",
                ],
                "security_concerns": [
                    "Account enumeration through error messages",
                    "Timing attacks on validation",
                ],
            },
            "zh": {
                "scenario_name": "用户注册与资料唯一性验证",
                "business_flow": "在创建/更新用户前验证唯一性",
                "user_experience": "用户提交注册；系统检查唯一性；拒绝或接受",
                "edge_cases": [
                    "邮箱已被其他用户注册",
                    "用户名已被占用",
                    "用户更新为已被占用的邮箱",
                ],
                "security_concerns": [
                    "通过错误消息进行账户枚举",
                    "验证时序攻击",
                ],
            }
        },
        "DBR-01-02": {
            "en": {
                "scenario_name": "Secure Account Creation",
                "business_flow": "Hash password and create account atomically",
                "user_experience": "User provides credentials; system securely stores",
                "edge_cases": [
                    "Password at maximum allowed length",
                    "Special characters in password",
                ],
                "security_concerns": [
                    "Password hashing security",
                    "Credential storage",
                ],
            },
            "zh": {
                "scenario_name": "安全账户创建",
                "business_flow": "哈希密码并原子性创建账户",
                "user_experience": "用户提供凭据；系统安全存储",
                "edge_cases": [
                    "密码达到最大允许长度",
                    "密码中包含特殊字符",
                ],
                "security_concerns": [
                    "密码哈希安全",
                    "凭据存储",
                ],
            }
        },
        "DBR-01-03": {
            "en": {
                "scenario_name": "Login Authentication",
                "business_flow": "Verify credentials and return generic error on failure",
                "user_experience": "User enters credentials; system validates; generic error if wrong",
                "edge_cases": [
                    "Non-existent email",
                    "Wrong password for existing account",
                ],
                "security_concerns": [
                    "User enumeration prevention",
                    "Consistent error responses",
                ],
            },
            "zh": {
                "scenario_name": "登录认证",
                "business_flow": "验证凭据，失败时返回通用错误",
                "user_experience": "用户输入凭据；系统验证；错误时显示通用错误",
                "edge_cases": [
                    "不存在的邮箱",
                    "现有账户的错误密码",
                ],
                "security_concerns": [
                    "防止用户枚举",
                    "一致的错误响应",
                ],
            }
        },
        "DBR-01-04": {
            "en": {
                "scenario_name": "Token Management",
                "business_flow": "Generate and return authentication token",
                "user_experience": "After successful auth, user receives token",
                "edge_cases": [
                    "Token generation after login",
                    "Token refresh after profile update",
                ],
                "security_concerns": [
                    "Token security",
                    "Session management",
                ],
            },
            "zh": {
                "scenario_name": "令牌管理",
                "business_flow": "生成并返回认证令牌",
                "user_experience": "认证成功后，用户收到令牌",
                "edge_cases": [
                    "登录后生成令牌",
                    "资料更新后刷新令牌",
                ],
                "security_concerns": [
                    "令牌安全",
                    "会话管理",
                ],
            }
        },
    }
    
    # Role contexts
    ROLE_CONTEXTS = {
        UserRole.END_USER: {
            "en": "a regular user who encountered an issue or has questions about how things work",
            "zh": "一个遇到问题或对事物如何工作有疑问的普通用户",
        },
        UserRole.PRODUCT_MANAGER: {
            "en": "a product manager who needs to understand user flows and business logic",
            "zh": "一个需要了解用户流程和业务逻辑的产品经理",
        },
        UserRole.QA_ENGINEER: {
            "en": "a QA engineer testing edge cases and potential failure scenarios",
            "zh": "一个测试边界情况和潜在失败场景的QA工程师",
        },
        UserRole.SECURITY_AUDITOR: {
            "en": "a security auditor examining potential vulnerabilities",
            "zh": "一个检查潜在漏洞的安全审计员",
        },
        UserRole.NEW_DEVELOPER: {
            "en": "a new developer trying to understand how the system works",
            "zh": "一个试图理解系统如何工作的新开发者",
        },
    }
    
    # Code name patterns to abstract
    ABSTRACTION_PATTERNS = {
        r'\bcheck_username_is_taken\b': ('username availability check', '用户名可用性检查'),
        r'\bcheck_email_is_taken\b': ('email uniqueness verification', '邮箱唯一性验证'),
        r'\busers_repo\.create_user\b': ('account creation process', '账户创建流程'),
        r'\busers_repo\.update_user\b': ('profile update process', '资料更新流程'),
        r'\bcreate_access_token_for_user\b': ('session token generation', '会话令牌生成'),
        r'\buser\.check_password\b': ('password verification', '密码验证'),
        r'\bwrong_login_error\b': ('authentication error response', '认证错误响应'),
        r'\buser_create\b': ('registration data', '注册数据'),
        r'\buser_update\b': ('update data', '更新数据'),
        r'\bHTTP_400_BAD_REQUEST\b': ('validation error', '验证错误'),
        r'\bHTTP_401_UNAUTHORIZED\b': ('authentication failure', '认证失败'),
        r'\bEntityDoesNotExist\b': ('resource not found', '资源不存在'),
        r'\busers_repo\b': ('user data service', '用户数据服务'),
    }
    
    def build_context(
        self,
        evidence: Dict,
        subcategory_id: str,
        code_facts: Optional[CodeFacts],
        language: str = "en"
    ) -> BusinessContext:
        """
        Build business context from evidence.
        
        Args:
            evidence: Evidence dictionary from rule metadata
            subcategory_id: The subcategory ID
            code_facts: Optional code facts for filtering edge cases
            language: Language code
            
        Returns:
            BusinessContext with user-friendly descriptions
        """
        base = self._get_base_context(subcategory_id, language)
        
        # Filter edge cases based on code facts (v8)
        if code_facts:
            edge_cases = self._filter_edge_cases(base.get("edge_cases", []), code_facts)
            security_concerns = self._filter_security_concerns(
                base.get("security_concerns", []), code_facts
            )
            code_behavior = self._describe_code_behavior(code_facts, language)
        else:
            edge_cases = base.get("edge_cases", [])
            security_concerns = base.get("security_concerns", [])
            code_behavior = ""
        
        return BusinessContext(
            scenario_name=base.get("scenario_name", "Authentication Process"),
            business_flow=base.get("business_flow", ""),
            user_experience=base.get("user_experience", ""),
            edge_cases=edge_cases,
            security_concerns=security_concerns,
            code_behavior=code_behavior,
            language=language,
        )
    
    def _get_base_context(self, subcategory_id: str, language: str) -> Dict:
        """Get base context without filtering."""
        context = self.SCENARIO_CONTEXTS.get(subcategory_id, {})
        return context.get(language, context.get("en", {}))
    
    def _filter_edge_cases(self, edge_cases: List[str], code_facts: CodeFacts) -> List[str]:
        """Remove edge cases that are impossible given the code."""
        if not code_facts.is_synchronous:
            return edge_cases
        
        invalid_patterns = [
            r'partial', r'部分',
            r'network\s+timeout', r'网络超时',
            r'concurrent', r'并发',
            r'half[\s-]?', r'一半',
        ]
        
        return [ec for ec in edge_cases 
                if not any(re.search(p, ec, re.IGNORECASE) for p in invalid_patterns)]
    
    def _filter_security_concerns(self, concerns: List[str], code_facts: CodeFacts) -> List[str]:
        """Remove security concerns that don't apply."""
        if not code_facts.is_synchronous:
            return concerns
        
        invalid_patterns = [r'race\s+condition', r'竞态条件']
        return [c for c in concerns 
                if not any(re.search(p, c, re.IGNORECASE) for p in invalid_patterns)]
    
    def _describe_code_behavior(self, code_facts: CodeFacts, language: str) -> str:
        """Generate human-readable description of code behavior."""
        if language == "zh":
            parts = []
            if code_facts.is_synchronous:
                parts.append("同步顺序执行")
            if code_facts.has_early_exit:
                parts.append("验证失败立即终止")
            if code_facts.atomicity_type == "gated_sequential":
                parts.append("门控原子性（验证通过才写入）")
            return "；".join(parts) if parts else "标准执行"
        else:
            parts = []
            if code_facts.is_synchronous:
                parts.append("Synchronous sequential execution")
            if code_facts.has_early_exit:
                parts.append("Immediate termination on validation failure")
            if code_facts.atomicity_type == "gated_sequential":
                parts.append("Gated atomicity (write only after all validations pass)")
            return "; ".join(parts) if parts else "Standard execution"
    
    def abstract_code_names(self, text: str, language: str = "en") -> str:
        """
        Remove code-level names from text, replacing with business terms.
        
        Args:
            text: Text potentially containing code names
            language: Language code
            
        Returns:
            Text with code names replaced by business descriptions
        """
        result = text
        lang_idx = 0 if language == "en" else 1
        
        for pattern, replacements in self.ABSTRACTION_PATTERNS.items():
            result = re.sub(pattern, replacements[lang_idx], result, flags=re.IGNORECASE)
        
        return result
    
    def get_role_context(self, role: UserRole, language: str = "en") -> str:
        """Get role description for prompts."""
        return self.ROLE_CONTEXTS.get(role, {}).get(language, "")
