"""
Design Context Builder for Design Q&A Generation.

Transforms DBR rule metadata into design scenarios that
developers/architects would need to address.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DesignScenario, DesignAspect, DesignerRole


class DesignContextBuilder:
    """
    Builds design scenarios from DBR rule metadata.
    
    Each DBR subcategory is transformed into a design scenario
    that a developer would need to implement.
    """
    
    # Design scenarios for each DBR-01 subcategory
    DESIGN_SCENARIOS = {
        "DBR-01-01": {
            "en": {
                "scenario_name": "User Registration & Profile Uniqueness System",
                "description": "Design a system that ensures username and email uniqueness during user registration and profile updates",
                "requirements": [
                    "Validate username uniqueness before account creation",
                    "Validate email uniqueness before account creation",
                    "Check uniqueness only when values change during updates",
                    "Return appropriate error messages for conflicts",
                    "Prevent data inconsistency from race conditions",
                ],
                "constraints": [
                    "Must use existing database infrastructure",
                    "Must maintain backward compatibility with API",
                    "Response time under 200ms for validation",
                    "Must handle concurrent registration attempts",
                ],
                "design_aspects": [
                    DesignAspect.VALIDATION,
                    DesignAspect.DATA_MODEL,
                    DesignAspect.ERROR_HANDLING,
                    DesignAspect.API_DESIGN,
                ],
            },
            "zh": {
                "scenario_name": "用户注册与资料唯一性验证系统",
                "description": "设计一个在用户注册和资料更新时确保用户名和邮箱唯一性的系统",
                "requirements": [
                    "在创建账户前验证用户名唯一性",
                    "在创建账户前验证邮箱唯一性",
                    "更新时仅在值变化时检查唯一性",
                    "为冲突返回适当的错误消息",
                    "防止竞态条件导致的数据不一致",
                ],
                "constraints": [
                    "必须使用现有数据库基础设施",
                    "必须保持API向后兼容",
                    "验证响应时间在200ms以内",
                    "必须处理并发注册请求",
                ],
                "design_aspects": [
                    DesignAspect.VALIDATION,
                    DesignAspect.DATA_MODEL,
                    DesignAspect.ERROR_HANDLING,
                    DesignAspect.API_DESIGN,
                ],
            },
        },
        "DBR-01-02": {
            "en": {
                "scenario_name": "Secure Account Creation Pipeline",
                "description": "Design a secure account creation system with proper password handling and atomic operations",
                "requirements": [
                    "Hash passwords before storage using secure algorithm",
                    "Ensure atomic account creation (all or nothing)",
                    "Generate secure user identifiers",
                    "Implement proper error handling during creation",
                    "Support transaction rollback on failure",
                ],
                "constraints": [
                    "Password must never be stored in plaintext",
                    "Must use industry-standard hashing (bcrypt/argon2)",
                    "Creation must be atomic - no partial records",
                    "Must comply with security audit requirements",
                ],
                "design_aspects": [
                    DesignAspect.SECURITY,
                    DesignAspect.DATA_MODEL,
                    DesignAspect.ARCHITECTURE,
                    DesignAspect.BEST_PRACTICES,
                ],
            },
            "zh": {
                "scenario_name": "安全账户创建流水线",
                "description": "设计一个具有正确密码处理和原子操作的安全账户创建系统",
                "requirements": [
                    "使用安全算法在存储前对密码进行哈希",
                    "确保账户创建的原子性（全有或全无）",
                    "生成安全的用户标识符",
                    "实现创建过程中的正确错误处理",
                    "支持失败时的事务回滚",
                ],
                "constraints": [
                    "密码绝不能以明文存储",
                    "必须使用行业标准哈希（bcrypt/argon2）",
                    "创建必须是原子的 - 不能有部分记录",
                    "必须符合安全审计要求",
                ],
                "design_aspects": [
                    DesignAspect.SECURITY,
                    DesignAspect.DATA_MODEL,
                    DesignAspect.ARCHITECTURE,
                    DesignAspect.BEST_PRACTICES,
                ],
            },
        },
        "DBR-01-03": {
            "en": {
                "scenario_name": "Secure Authentication System",
                "description": "Design a login authentication system that prevents user enumeration and provides secure error handling",
                "requirements": [
                    "Verify user credentials securely",
                    "Return generic error messages to prevent enumeration",
                    "Implement consistent response timing",
                    "Handle non-existent users and wrong passwords uniformly",
                    "Log authentication attempts for security monitoring",
                ],
                "constraints": [
                    "Error messages must not reveal if email exists",
                    "Response time should be consistent regardless of failure type",
                    "Must support future MFA integration",
                    "Must comply with OWASP authentication guidelines",
                ],
                "design_aspects": [
                    DesignAspect.SECURITY,
                    DesignAspect.ERROR_HANDLING,
                    DesignAspect.API_DESIGN,
                    DesignAspect.BEST_PRACTICES,
                ],
            },
            "zh": {
                "scenario_name": "安全认证系统",
                "description": "设计一个防止用户枚举并提供安全错误处理的登录认证系统",
                "requirements": [
                    "安全地验证用户凭据",
                    "返回通用错误消息以防止枚举",
                    "实现一致的响应时间",
                    "统一处理不存在的用户和错误密码",
                    "记录认证尝试以进行安全监控",
                ],
                "constraints": [
                    "错误消息不得透露邮箱是否存在",
                    "响应时间应与失败类型无关保持一致",
                    "必须支持未来的MFA集成",
                    "必须符合OWASP认证指南",
                ],
                "design_aspects": [
                    DesignAspect.SECURITY,
                    DesignAspect.ERROR_HANDLING,
                    DesignAspect.API_DESIGN,
                    DesignAspect.BEST_PRACTICES,
                ],
            },
        },
        "DBR-01-04": {
            "en": {
                "scenario_name": "JWT Token Management System",
                "description": "Design a JWT-based session management system with proper token lifecycle",
                "requirements": [
                    "Generate secure JWT tokens after authentication",
                    "Include appropriate claims in tokens",
                    "Implement token refresh mechanism",
                    "Maintain session state consistency",
                    "Support token invalidation when needed",
                ],
                "constraints": [
                    "Tokens must be cryptographically secure",
                    "Must support stateless verification",
                    "Token expiration must be configurable",
                    "Must handle token across multiple endpoints",
                ],
                "design_aspects": [
                    DesignAspect.SESSION_MANAGEMENT,
                    DesignAspect.SECURITY,
                    DesignAspect.ARCHITECTURE,
                    DesignAspect.API_DESIGN,
                ],
            },
            "zh": {
                "scenario_name": "JWT令牌管理系统",
                "description": "设计一个具有正确令牌生命周期的基于JWT的会话管理系统",
                "requirements": [
                    "认证后生成安全的JWT令牌",
                    "在令牌中包含适当的声明",
                    "实现令牌刷新机制",
                    "维护会话状态一致性",
                    "在需要时支持令牌失效",
                ],
                "constraints": [
                    "令牌必须是加密安全的",
                    "必须支持无状态验证",
                    "令牌过期必须可配置",
                    "必须处理跨多个端点的令牌",
                ],
                "design_aspects": [
                    DesignAspect.SESSION_MANAGEMENT,
                    DesignAspect.SECURITY,
                    DesignAspect.ARCHITECTURE,
                    DesignAspect.API_DESIGN,
                ],
            },
        },
    }
    
    # Role-specific design perspectives
    ROLE_PERSPECTIVES = {
        DesignerRole.BACKEND_DEVELOPER: {
            "en": "implementing the actual code and APIs",
            "zh": "实现实际代码和API",
        },
        DesignerRole.SYSTEM_ARCHITECT: {
            "en": "overall system structure and component interactions",
            "zh": "整体系统结构和组件交互",
        },
        DesignerRole.SECURITY_ENGINEER: {
            "en": "security vulnerabilities and protective measures",
            "zh": "安全漏洞和防护措施",
        },
        DesignerRole.TECH_LEAD: {
            "en": "technical decisions and trade-offs",
            "zh": "技术决策和权衡",
        },
        DesignerRole.API_DESIGNER: {
            "en": "API contracts and client experience",
            "zh": "API契约和客户端体验",
        },
    }
    
    def __init__(self, rule_metadata: Dict = None):
        self.rule_metadata = rule_metadata or {}
    
    def build_scenario(
        self,
        subcategory_id: str,
        language: str = "en"
    ) -> Optional[DesignScenario]:
        """
        Build a design scenario from subcategory.
        
        Args:
            subcategory_id: The DBR subcategory ID
            language: Language code
            
        Returns:
            DesignScenario object
        """
        scenario_data = self.DESIGN_SCENARIOS.get(subcategory_id, {})
        lang_data = scenario_data.get(language, scenario_data.get("en", {}))
        
        if not lang_data:
            return None
        
        return DesignScenario(
            scenario_id=subcategory_id,
            scenario_name=lang_data.get("scenario_name", ""),
            description=lang_data.get("description", ""),
            requirements=lang_data.get("requirements", []),
            constraints=lang_data.get("constraints", []),
            related_dbr="DBR-01",
            design_aspects=lang_data.get("design_aspects", []),
        )
    
    def get_role_perspective(
        self,
        role: DesignerRole,
        language: str = "en"
    ) -> str:
        """Get the perspective description for a role."""
        perspectives = self.ROLE_PERSPECTIVES.get(role, {})
        return perspectives.get(language, perspectives.get("en", ""))
    
    def get_all_subcategory_ids(self) -> List[str]:
        """Get all available subcategory IDs."""
        return list(self.DESIGN_SCENARIOS.keys())
    
    def get_design_aspects(self, subcategory_id: str) -> List[DesignAspect]:
        """Get design aspects for a subcategory."""
        scenario_data = self.DESIGN_SCENARIOS.get(subcategory_id, {})
        # Get from either language, they should be the same
        for lang_data in scenario_data.values():
            if "design_aspects" in lang_data:
                return lang_data["design_aspects"]
        return []
