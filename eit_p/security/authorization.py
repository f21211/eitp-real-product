"""
授权管理模块
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from ..utils import get_global_logger


class Permission(Enum):
    """权限枚举"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class Role:
    """角色定义"""
    name: str
    permissions: Set[Permission]
    description: str = ""


class AuthorizationManager:
    """授权管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = get_global_logger()
        self.roles: Dict[str, Role] = {}
        self._initialize_default_roles()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'default_role': 'user',
            'admin_role': 'admin'
        }
    
    def _initialize_default_roles(self):
        """初始化默认角色"""
        # 用户角色
        user_role = Role(
            name="user",
            permissions={Permission.READ},
            description="普通用户"
        )
        self.roles["user"] = user_role
        
        # 研究员角色
        researcher_role = Role(
            name="researcher",
            permissions={Permission.READ, Permission.WRITE},
            description="研究员"
        )
        self.roles["researcher"] = researcher_role
        
        # 管理员角色
        admin_role = Role(
            name="admin",
            permissions={Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN},
            description="管理员"
        )
        self.roles["admin"] = admin_role
    
    def check_permission(self, user_roles: List[str], required_permission: Permission) -> bool:
        """检查用户权限"""
        for role_name in user_roles:
            if role_name in self.roles:
                role = self.roles[role_name]
                if required_permission in role.permissions:
                    return True
        return False
    
    def add_role(self, name: str, permissions: Set[Permission], description: str = ""):
        """添加角色"""
        role = Role(name=name, permissions=permissions, description=description)
        self.roles[name] = role
        self.logger.info(f"添加角色: {name}")
    
    def remove_role(self, name: str) -> bool:
        """删除角色"""
        if name in self.roles:
            del self.roles[name]
            self.logger.info(f"删除角色: {name}")
            return True
        return False
    
    def get_role(self, name: str) -> Optional[Role]:
        """获取角色"""
        return self.roles.get(name)
    
    def list_roles(self) -> List[Dict[str, Any]]:
        """列出所有角色"""
        return [
            {
                'name': role.name,
                'permissions': [p.value for p in role.permissions],
                'description': role.description
            }
            for role in self.roles.values()
        ]


class RoleManager:
    """角色管理器"""
    
    def __init__(self, auth_manager: AuthorizationManager):
        self.auth_manager = auth_manager
        self.logger = get_global_logger()
    
    def assign_role(self, username: str, role_name: str) -> bool:
        """分配角色给用户"""
        # 这里需要与认证管理器集成
        self.logger.info(f"分配角色: {username} -> {role_name}")
        return True
    
    def revoke_role(self, username: str, role_name: str) -> bool:
        """撤销用户角色"""
        self.logger.info(f"撤销角色: {username} -> {role_name}")
        return True
    
    def get_user_roles(self, username: str) -> List[str]:
        """获取用户角色"""
        # 这里需要与认证管理器集成
        return ["user"]  # 默认返回用户角色
