"""
Domain Business Rules (DBR) module.

Defines the business rules extracted from the target codebase that serve as
the "ground truth" for training data generation. These rules ensure generated
samples are grounded in actual code behavior and business logic.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class DBRCategory(Enum):
    """Categories for Domain Business Rules."""
    AUTHENTICATION = "authentication"
    SOCIAL = "social"
    CONTENT = "content"
    ROUTING = "routing"
    ERROR_HANDLING = "error_handling"
    PERSISTENCE = "persistence"


@dataclass
class DomainBusinessRule:
    """
    Represents a Domain Business Rule (DBR).
    
    Attributes:
        id: Unique identifier (e.g., "DBR-01")
        name: Short name of the rule
        category: Category the rule belongs to
        description: Full description of the business logic
        evidence_guide: Guide for extracting code evidence
        code_links: List of related source files
        keywords: Keywords for matching code patterns
    """
    id: str
    name: str
    category: DBRCategory
    description: str
    evidence_guide: str = ""
    code_links: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def get_content(self) -> str:
        """Get the full rule content for prompts."""
        return f"{self.id}：{self.name}\n{self.description}"


# ============================================================
# DBR-01: Identity Access & Credential Integrity
# ============================================================
DBR_01 = DomainBusinessRule(
    id="DBR-01",
    name="身份准入与账户凭据完整性 (Authentication & Credential Integrity)",
    category=DBRCategory.AUTHENTICATION,
    description="""
1. 唯一性拦截：注册/更新时强制检查用户名/邮箱唯一性。若标识符已被占用，系统通过 400 Bad Request 显式拦截。
2. 存储安全：密码必须哈希处理后持久化。账户创建过程通过 UsersRepository 确保原子性。
3. 登录安全反馈：登录失败统一返回模糊错误信息 INCORRECT_LOGIN_INPUT，防止用户枚举。
4. 会话管理：操作成功后重新生成并返回 JWT 令牌，维持客户端会话状态一致性。
""",
    evidence_guide="""
代码精准抽取指令：
1. 登录异常块：定位 login 函数，仅抽取 try-except 捕获 EntityDoesNotExist 的块。
2. 注册检查块：定位 register 函数，仅抽取校验 username/email 占用的 if 块。
3. 更新校验块：定位 update_current_user，仅抽取对比新旧值并校验唯一性的逻辑。
4. 令牌生成行：识别调用 create_access_token_for_user 的行。
""",
    code_links=[
        "app/api/routes/authentication.py",
        "app/api/routes/users.py"
    ],
    keywords=[
        "login", "register", "authentication", "password", "token",
        "JWT", "check_username", "check_email", "EntityDoesNotExist"
    ]
)


# ============================================================
# DBR-02: Social Relations & Profiles
# ============================================================
DBR_02 = DomainBusinessRule(
    id="DBR-02",
    name="社交关系约束与个人资料访问 (Social Relations & Profiles)",
    category=DBRCategory.SOCIAL,
    description="""
1. 资源存在性预检 (隐式 404)：所有个人资料操作均严格依赖于路径参数。若目标用户不存在，由依赖层直接抛出 404 Not Found。
2. 身份一致性拦截 (显式 400)：系统禁止"自操作"社交行为。若试图操作自身，系统将显式抛出 400 Bad Request。
3. 社交状态幂等性约束 (显式 400)：禁止重复关注已关注用户，或取消关注未关注用户，违规请求均返回 400 Bad Request。
4. 公开资料访问 (GET)：支持通过用户名检索公开 Profile，支持匿名访问。
""",
    evidence_guide="""
代码精准抽取指令：
1. 404 预检机制：定位 Depends(get_profile_by_username_from_path) 依赖注入。
2. 身份一致性拦截：定位 if user.username == profile.username 比较逻辑。
3. 社交状态幂等性：定位 profile.following 状态检查逻辑。
4. 数据持久化：定位 profiles_repo 的 add_user_into_followers / remove_user_from_followers 调用。
""",
    code_links=[
        "app/api/routes/profiles.py"
    ],
    keywords=[
        "profile", "follow", "unfollow", "following", "followers",
        "get_profile_by_username_from_path"
    ]
)


# ============================================================
# DBR-03: Article Lifecycle & Persistent Identifier Constraints
# ============================================================
DBR_03 = DomainBusinessRule(
    id="DBR-03",
    name="文章实体生命周期与持久化标识约束 (Article Lifecycle & Persistent Identifier Constraints)",
    category=DBRCategory.CONTENT,
    description="""
1. 唯一标识符自动生成 (Slugification)：系统基于文章标题 title 自动生成 slug。创建文章时强制校验 slug 唯一性。
2. 资源存在性预检 (隐式 404)：通过路径参数 {slug} 进行操作时，系统利用前置依赖项预检资源。
3. 权属审计与修改保护 (显式 403)：更新和删除仅限文章作者。非作者身份请求时返回 403 Forbidden。
4. 灵活更新机制：更新时若 article_update.title 发生变动，系统将同步重新生成 slug。
""",
    evidence_guide="""
代码精准抽取指令：
1. Slug 查重：定位 create_new_article 中 get_slug_for_article 和 check_article_exists 调用。
2. 404 拦截：定位 Depends(get_article_by_slug_from_path) 依赖注入。
3. 403 权限控制：定位 Depends(check_article_modification_permissions) 依赖注入。
4. 持久化操作：定位 articles_repo 的 create_article / update_article / delete_article 调用。
""",
    code_links=[
        "app/api/routes/articles/articles_resource.py"
    ],
    keywords=[
        "article", "slug", "create_article", "update_article", "delete_article",
        "check_article_modification_permissions"
    ]
)


# ============================================================
# DBR-04: Social Interaction & Comments
# ============================================================
DBR_04 = DomainBusinessRule(
    id="DBR-04",
    name="内容社交交互与评论生命周期 (Social Interaction & Comments)",
    category=DBRCategory.CONTENT,
    description="""
1. 个性化馈送机制 (Feed)：系统通过 get_articles_for_user_feed 接口提供动态流，严格要求身份认证。
2. 交互状态幂等性 (显式 400)：禁止对已点赞的文章重复点赞，或对未点赞的文章执行取消操作。
3. 跨模块资源关联 (POST/GET)：评论生命周期通过 article 变量严格绑定于特定文章。
4. 评论权属与销毁保护 (前置 403)：仅评论原始作者拥有销毁权限。
""",
    evidence_guide="""
代码精准抽取指令：
1. 点赞逻辑拦截：定位 mark_article_as_favorite 中 article.favorited 检查。
2. 评论创建与关联：定位 create_comment_for_article 中 article 变量绑定。
3. 评论删除权限：定位 Depends(check_comment_modification_permissions) 依赖注入。
4. 资源存在性审计：定位 Depends(get_article_by_slug_from_path) 和 Depends(get_comment_by_id_from_path)。
""",
    code_links=[
        "app/api/routes/articles/articles_common.py",
        "app/api/routes/comments.py"
    ],
    keywords=[
        "comment", "favorite", "feed", "favorited", "check_comment_modification_permissions"
    ]
)


# ============================================================
# DBR-05: System Routing & Integration
# ============================================================
DBR_05 = DomainBusinessRule(
    id="DBR-05",
    name="系统路由命名空间与集成规范 (System Routing & Integration)",
    category=DBRCategory.ROUTING,
    description="""
1. 全局路由总线制 (Centralized Routing)：系统采用分层挂载机制，实现业务域的逻辑隔离与统一管理。
2. RESTful 资源嵌套约束：评论模块逻辑前缀绑定在文章路径 {slug} 之下。
3. 统一接入点与版本控制：应用通过全局前缀配置实现版本管理入口的统一。
4. 标准化错误响应注入：系统统一挂载 HTTPException 和 RequestValidationError 异常处理器。
""",
    evidence_guide="""
代码精准抽取指令：
1. 路由模块集成：定位 api.py 中 router.include_router 调用。
2. 嵌套路径实现：定位 comments.router 的 prefix 配置。
3. 全局应用配置：定位 main.py 中 application.include_router 调用。
4. 异常审计挂载：定位 application.add_exception_handler 调用。
""",
    code_links=[
        "app/api/routes/api.py",
        "app/main.py"
    ],
    keywords=[
        "router", "include_router", "api_prefix", "exception_handler", "middleware"
    ]
)


# ============================================================
# DBR-06: Global Exception & Response Contract
# ============================================================
DBR_06 = DomainBusinessRule(
    id="DBR-06",
    name="全局异常规约与响应一致性 (Global Exception & Response Contract)",
    category=DBRCategory.ERROR_HANDLING,
    description="""
1. 标准化错误出口：系统通过中央错误处理器接管所有业务逻辑中抛出的异常。
2. 错误详情解耦：系统将底层的 HTTPException 细节字段自动映射至标准化的 errors 数组中。
3. 统一 JSON 响应格式：无论错误源自身份验证、权限拦截还是资源不存在，均强制执行统一的 JSON 响应格式。
""",
    evidence_guide="""
代码精准抽取指令：
1. 异常捕获函数：定位 http_error.py 中 http_error_handler 异步函数。
2. 响应体构造：定位 JSONResponse 返回结构 {"errors": [exc.detail]}。
3. 全局挂载点：定位 main.py 中 add_exception_handler(HTTPException, http_error_handler)。
""",
    code_links=[
        "app/api/errors/http_error.py"
    ],
    keywords=[
        "http_error_handler", "JSONResponse", "HTTPException", "errors"
    ]
)


# ============================================================
# DBR-07: Repository Pattern & Persistence
# ============================================================
DBR_07 = DomainBusinessRule(
    id="DBR-07",
    name="仓储模式与数据持久化抽象 (Repository Pattern & Persistence)",
    category=DBRCategory.PERSISTENCE,
    description="""
1. 关注点分离 (SoC)：系统采用仓储模式隔离业务逻辑层与物理数据库层。所有数据库 CRUD 操作均封装在特定的 Repository 类中。
2. 连接生命周期管理：通过基类抽象，确保每个 Repository 实例在处理请求时持有合法的、受生命周期管理的数据库连接。
3. 接口封装：通过 @property 装饰器公开 connection 只读属性，确保底层连接实例的不可变性和线程安全性。
""",
    evidence_guide="""
代码精准抽取指令：
1. 基类定义：定位 base.py 中 BaseRepository 类定义。
2. 连接注入：定位构造函数 __init__ 中 Connection 类型参数 conn。
3. 接口封装：定位 @property 装饰器的 connection 属性。
""",
    code_links=[
        "app/db/repositories/base.py"
    ],
    keywords=[
        "BaseRepository", "Repository", "connection", "conn", "_conn"
    ]
)


class DBRRegistry:
    """
    Registry for Domain Business Rules.
    
    Provides access to all defined DBRs and utility methods for
    looking up rules by ID or category.
    """
    
    _rules: Dict[str, DomainBusinessRule] = {
        "DBR-01": DBR_01,
        "DBR-02": DBR_02,
        "DBR-03": DBR_03,
        "DBR-04": DBR_04,
        "DBR-05": DBR_05,
        "DBR-06": DBR_06,
        "DBR-07": DBR_07,
    }
    
    @classmethod
    def get(cls, dbr_id: str) -> Optional[DomainBusinessRule]:
        """Get a DBR by its ID."""
        return cls._rules.get(dbr_id)
    
    @classmethod
    def get_all(cls) -> List[DomainBusinessRule]:
        """Get all registered DBRs."""
        return list(cls._rules.values())
    
    @classmethod
    def get_by_category(cls, category: DBRCategory) -> List[DomainBusinessRule]:
        """Get all DBRs in a specific category."""
        return [dbr for dbr in cls._rules.values() if dbr.category == category]
    
    @classmethod
    def get_ids(cls) -> List[str]:
        """Get all DBR IDs."""
        return list(cls._rules.keys())
    
    @classmethod
    def get_code_links(cls, dbr_id: str) -> List[str]:
        """Get the code links for a specific DBR."""
        dbr = cls.get(dbr_id)
        return dbr.code_links if dbr else []
