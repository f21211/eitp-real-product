"""
EIT-P 安全管理器 - 简化实现
基于IEM理论的安全管理
"""

import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class SecurityManager:
    """安全管理器 - 基于IEM理论的安全管理"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.users = {}
        self.sessions = {}
        self.logger = logging.getLogger(__name__)
        self.logger.info("安全管理器初始化完成")
    
    def hash_password(self, password: str) -> str:
        """哈希密码"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """验证密码"""
        return self.hash_password(password) == hashed
    
    def create_user(self, username: str, password: str, role: str = 'user') -> bool:
        """创建用户"""
        try:
            if username in self.users:
                return False
            
            self.users[username] = {
                'username': username,
                'password_hash': self.hash_password(password),
                'role': role,
                'created_at': datetime.now().isoformat(),
                'active': True
            }
            
            self.logger.info(f"用户创建成功: {username}")
            return True
            
        except Exception as e:
            self.logger.error(f"创建用户失败: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """认证用户"""
        try:
            if username not in self.users:
                return False
            
            user = self.users[username]
            if not user['active']:
                return False
            
            return self.verify_password(password, user['password_hash'])
            
        except Exception as e:
            self.logger.error(f"用户认证失败: {e}")
            return False
    
    def generate_token(self, username: str) -> str:
        """生成JWT令牌"""
        try:
            payload = {
                'username': username,
                'role': self.users[username]['role'],
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(hours=24)
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm='HS256')
            return token
            
        except Exception as e:
            self.logger.error(f"生成令牌失败: {e}")
            return None
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("令牌已过期")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("无效令牌")
            return None
        except Exception as e:
            self.logger.error(f"验证令牌失败: {e}")
            return None
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        return self.users.get(username)
    
    def update_user_role(self, username: str, role: str) -> bool:
        """更新用户角色"""
        try:
            if username not in self.users:
                return False
            
            self.users[username]['role'] = role
            self.logger.info(f"用户角色更新成功: {username} -> {role}")
            return True
            
        except Exception as e:
            self.logger.error(f"更新用户角色失败: {e}")
            return False
    
    def deactivate_user(self, username: str) -> bool:
        """停用用户"""
        try:
            if username not in self.users:
                return False
            
            self.users[username]['active'] = False
            self.logger.info(f"用户停用成功: {username}")
            return True
            
        except Exception as e:
            self.logger.error(f"停用用户失败: {e}")
            return False
