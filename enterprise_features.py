#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P Enterprise Features
企业级功能：多租户、权限管理、审计日志
"""

import sys
import os
sys.path.append('/mnt/sda1/myproject/datainall/AGI')

import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from advanced_features_manager import AdvancedFeaturesManager

class UserRole(Enum):
    """用户角色"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    DEVELOPER = "developer"

class Permission(Enum):
    """权限"""
    READ_MODELS = "read_models"
    WRITE_MODELS = "write_models"
    DELETE_MODELS = "delete_models"
    CREATE_TESTS = "create_tests"
    VIEW_TESTS = "view_tests"
    MANAGE_USERS = "manage_users"
    VIEW_LOGS = "view_logs"
    SYSTEM_ADMIN = "system_admin"

@dataclass
class User:
    """用户信息"""
    user_id: str
    username: str
    email: str
    role: UserRole
    tenant_id: str
    created_at: str
    last_login: str
    is_active: bool

@dataclass
class Tenant:
    """租户信息"""
    tenant_id: str
    name: str
    description: str
    created_at: str
    is_active: bool
    max_models: int = 100
    max_tests: int = 50

@dataclass
class AuditLog:
    """审计日志"""
    log_id: str
    user_id: str
    tenant_id: str
    action: str
    resource: str
    details: Dict
    timestamp: str
    ip_address: str

class EnterpriseFeaturesManager:
    """企业级功能管理器"""
    
    def __init__(self):
        self.advanced_manager = AdvancedFeaturesManager()
        self.users_file = Path("enterprise_users.json")
        self.tenants_file = Path("enterprise_tenants.json")
        self.audit_logs_file = Path("enterprise_audit_logs.jsonl")
        
        self.users = self.load_users()
        self.tenants = self.load_tenants()
        
        # 角色权限映射
        self.role_permissions = {
            UserRole.ADMIN: {p for p in Permission},
            UserRole.USER: {
                Permission.READ_MODELS,
                Permission.WRITE_MODELS,
                Permission.CREATE_TESTS,
                Permission.VIEW_TESTS
            },
            UserRole.VIEWER: {
                Permission.READ_MODELS,
                Permission.VIEW_TESTS
            },
            UserRole.DEVELOPER: {
                Permission.READ_MODELS,
                Permission.WRITE_MODELS,
                Permission.CREATE_TESTS,
                Permission.VIEW_TESTS,
                Permission.VIEW_LOGS
            }
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("EnterpriseFeaturesManager")
    
    def load_users(self) -> Dict[str, User]:
        """加载用户数据"""
        if self.users_file.exists():
            with open(self.users_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {k: User(**v) for k, v in data.items()}
        return {}
    
    def save_users(self):
        """保存用户数据"""
        data = {}
        for k, v in self.users.items():
            user_dict = asdict(v)
            user_dict['role'] = v.role.value  # 转换枚举为字符串
            data[k] = user_dict
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_tenants(self) -> Dict[str, Tenant]:
        """加载租户数据"""
        if self.tenants_file.exists():
            with open(self.tenants_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {k: Tenant(**v) for k, v in data.items()}
        return {}
    
    def save_tenants(self):
        """保存租户数据"""
        data = {k: asdict(v) for k, v in self.tenants.items()}
        with open(self.tenants_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def create_tenant(self, tenant_id: str, name: str, description: str = "") -> str:
        """创建租户"""
        if tenant_id in self.tenants:
            raise ValueError(f"租户 {tenant_id} 已存在")
        
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
            is_active=True
        )
        
        self.tenants[tenant_id] = tenant
        self.save_tenants()
        
        self.log_audit("system", "system", "create_tenant", f"tenant:{tenant_id}", {
            'tenant_name': name,
            'description': description
        })
        
        self.logger.info(f"创建租户: {tenant_id}")
        return tenant_id
    
    def create_user(self, user_id: str, username: str, email: str, 
                   role: UserRole, tenant_id: str) -> str:
        """创建用户"""
        if user_id in self.users:
            raise ValueError(f"用户 {user_id} 已存在")
        
        if tenant_id not in self.tenants:
            raise ValueError(f"租户 {tenant_id} 不存在")
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            tenant_id=tenant_id,
            created_at=datetime.now().isoformat(),
            last_login=datetime.now().isoformat(),
            is_active=True
        )
        
        self.users[user_id] = user
        self.save_users()
        
        self.log_audit("system", "system", "create_user", f"user:{user_id}", {
            'username': username,
            'email': email,
            'role': role.value,
            'tenant_id': tenant_id
        })
        
        self.logger.info(f"创建用户: {user_id}")
        return user_id
    
    def authenticate_user(self, user_id: str) -> Optional[User]:
        """用户认证"""
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        if not user.is_active:
            return None
        
        # 更新最后登录时间
        user.last_login = datetime.now().isoformat()
        self.save_users()
        
        return user
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """检查用户权限"""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        if not user.is_active:
            return False
        
        user_permissions = self.role_permissions.get(user.role, set())
        return permission in user_permissions
    
    def log_audit(self, user_id: str, tenant_id: str, action: str, 
                  resource: str, details: Dict, ip_address: str = "127.0.0.1"):
        """记录审计日志"""
        log_id = hashlib.md5(f"{user_id}{action}{resource}{time.time()}".encode()).hexdigest()[:16]
        
        audit_log = AuditLog(
            log_id=log_id,
            user_id=user_id,
            tenant_id=tenant_id,
            action=action,
            resource=resource,
            details=details,
            timestamp=datetime.now().isoformat(),
            ip_address=ip_address
        )
        
        # 保存到文件
        with open(self.audit_logs_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(audit_log), ensure_ascii=False) + '\n')
    
    def get_audit_logs(self, user_id: str = None, tenant_id: str = None, 
                      action: str = None, limit: int = 100) -> List[Dict]:
        """获取审计日志"""
        logs = []
        
        if not self.audit_logs_file.exists():
            return logs
        
        with open(self.audit_logs_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log = json.loads(line.strip())
                    
                    # 过滤条件
                    if user_id and log['user_id'] != user_id:
                        continue
                    if tenant_id and log['tenant_id'] != tenant_id:
                        continue
                    if action and log['action'] != action:
                        continue
                    
                    logs.append(log)
                    
                    if len(logs) >= limit:
                        break
                        
                except json.JSONDecodeError:
                    continue
        
        return logs
    
    def get_tenant_models(self, tenant_id: str) -> List[Dict]:
        """获取租户的模型版本"""
        if not self.check_permission("system", Permission.READ_MODELS):
            return []
        
        all_models = self.advanced_manager.list_model_versions()
        
        # 过滤租户相关的模型（这里简化处理）
        tenant_models = []
        for model in all_models:
            if tenant_id in model.get('tags', []):
                tenant_models.append(model)
        
        return tenant_models
    
    def get_tenant_tests(self, tenant_id: str) -> List[Dict]:
        """获取租户的A/B测试"""
        if not self.check_permission("system", Permission.VIEW_TESTS):
            return []
        
        all_tests = self.advanced_manager.list_ab_tests()
        
        # 过滤租户相关的测试（这里简化处理）
        tenant_tests = []
        for test in all_tests:
            if tenant_id in test.get('test_id', ''):
                tenant_tests.append(test)
        
        return tenant_tests
    
    def create_tenant_model(self, tenant_id: str, user_id: str, 
                           model_data: Dict) -> Optional[str]:
        """为租户创建模型"""
        if not self.check_permission(user_id, Permission.WRITE_MODELS):
            self.log_audit(user_id, tenant_id, "create_model_denied", "model", {
                'reason': 'insufficient_permissions'
            })
            return None
        
        # 检查租户限制
        tenant_models = self.get_tenant_models(tenant_id)
        if len(tenant_models) >= self.tenants[tenant_id].max_models:
            self.log_audit(user_id, tenant_id, "create_model_denied", "model", {
                'reason': 'tenant_model_limit_exceeded'
            })
            return None
        
        try:
            # 创建模型版本
            from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters
            
            model = EnhancedCEPEITP(
                input_dim=784,
                hidden_dims=[512, 256, 128],
                output_dim=10,
                cep_params=CEPParameters()
            )
            
            version = self.advanced_manager.version_manager.create_version(
                model,
                model_data.get('performance_metrics', {}),
                description=f"租户 {tenant_id} 模型",
                tags=[tenant_id, 'enterprise']
            )
            
            self.log_audit(user_id, tenant_id, "create_model", f"model:{version}", {
                'model_version': version,
                'performance_metrics': model_data.get('performance_metrics', {})
            })
            
            return version
            
        except Exception as e:
            self.log_audit(user_id, tenant_id, "create_model_failed", "model", {
                'error': str(e)
            })
            return None
    
    def get_system_statistics(self) -> Dict:
        """获取系统统计信息"""
        return {
            'total_users': len(self.users),
            'active_users': len([u for u in self.users.values() if u.is_active]),
            'total_tenants': len(self.tenants),
            'active_tenants': len([t for t in self.tenants.values() if t.is_active]),
            'total_models': len(self.advanced_manager.version_manager.versions),
            'total_tests': len(self.advanced_manager.ab_test_manager.tests),
            'audit_logs_count': self._count_audit_logs(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _count_audit_logs(self) -> int:
        """统计审计日志数量"""
        if not self.audit_logs_file.exists():
            return 0
        
        count = 0
        with open(self.audit_logs_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        
        return count

def main():
    """主函数"""
    print("🏢 Enhanced CEP-EIT-P Enterprise Features")
    print("=" * 50)
    
    # 创建企业级功能管理器
    enterprise_manager = EnterpriseFeaturesManager()
    
    # 创建测试租户
    tenant_id = enterprise_manager.create_tenant(
        "tenant_001", 
        "测试租户", 
        "用于测试的企业租户"
    )
    print(f"✅ 创建租户: {tenant_id}")
    
    # 创建测试用户
    user_id = enterprise_manager.create_user(
        "user_001",
        "testuser",
        "test@example.com",
        UserRole.USER,
        tenant_id
    )
    print(f"✅ 创建用户: {user_id}")
    
    # 认证用户
    user = enterprise_manager.authenticate_user(user_id)
    if user:
        print(f"✅ 用户认证成功: {user.username}")
    
    # 检查权限
    can_read = enterprise_manager.check_permission(user_id, Permission.READ_MODELS)
    can_write = enterprise_manager.check_permission(user_id, Permission.WRITE_MODELS)
    can_admin = enterprise_manager.check_permission(user_id, Permission.SYSTEM_ADMIN)
    
    print(f"📋 权限检查:")
    print(f"  读取模型: {can_read}")
    print(f"  写入模型: {can_write}")
    print(f"  系统管理: {can_admin}")
    
    # 创建租户模型
    model_data = {
        'performance_metrics': {
            'consciousness_level': 2,
            'accuracy': 0.9,
            'inference_time': 0.001
        }
    }
    
    model_version = enterprise_manager.create_tenant_model(tenant_id, user_id, model_data)
    if model_version:
        print(f"✅ 创建租户模型: {model_version}")
    
    # 获取系统统计
    stats = enterprise_manager.get_system_statistics()
    print(f"\n📊 系统统计:")
    print(f"  总用户数: {stats['total_users']}")
    print(f"  活跃用户: {stats['active_users']}")
    print(f"  总租户数: {stats['total_tenants']}")
    print(f"  总模型数: {stats['total_models']}")
    print(f"  审计日志: {stats['audit_logs_count']}")
    
    # 获取审计日志
    audit_logs = enterprise_manager.get_audit_logs(limit=5)
    print(f"\n📝 最近审计日志 ({len(audit_logs)} 条):")
    for log in audit_logs:
        print(f"  {log['timestamp']}: {log['action']} - {log['resource']}")
    
    print("🎉 企业级功能测试完成!")

if __name__ == "__main__":
    main()
