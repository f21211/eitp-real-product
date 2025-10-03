"""
认证管理模块
提供用户认证、JWT令牌管理等功能
"""

import jwt
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import bcrypt

from ..utils import get_global_logger


@dataclass
class User:
    """用户信息"""
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: List[str]
    is_active: bool
    created_at: str
    last_login: Optional[str] = None
    failed_login_attempts: int = 0
    locked_until: Optional[str] = None


@dataclass
class TokenInfo:
    """令牌信息"""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    scope: List[str] = None


class AuthenticationManager:
    """认证管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = get_global_logger()
        
        # 用户存储
        self.users: Dict[str, User] = {}
        
        # 活跃令牌
        self.active_tokens: Dict[str, Dict[str, Any]] = {}
        
        # 登录尝试记录
        self.login_attempts: Dict[str, List[float]] = {}
        
        # 加载用户数据
        self._load_users()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'jwt_secret': secrets.token_urlsafe(32),
            'jwt_algorithm': 'HS256',
            'access_token_expiry': 3600,  # 1小时
            'refresh_token_expiry': 86400 * 7,  # 7天
            'max_login_attempts': 5,
            'lockout_duration': 300,  # 5分钟
            'password_min_length': 8,
            'password_require_special': True,
            'password_require_numbers': True,
            'password_require_uppercase': True,
            'users_file': './security/users.json'
        }
    
    def register_user(self, username: str, email: str, password: str, 
                     roles: List[str] = None) -> Tuple[bool, str]:
        """注册用户"""
        try:
            # 验证输入
            if not self._validate_username(username):
                return False, "用户名格式无效"
            
            if not self._validate_email(email):
                return False, "邮箱格式无效"
            
            if not self._validate_password(password):
                return False, "密码不符合要求"
            
            # 检查用户是否已存在
            if username in self.users:
                return False, "用户名已存在"
            
            if any(user.email == email for user in self.users.values()):
                return False, "邮箱已被使用"
            
            # 创建用户
            user_id = self._generate_user_id()
            password_hash = self._hash_password(password)
            
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                roles=roles or ['user'],
                is_active=True,
                created_at=datetime.now().isoformat()
            )
            
            # 保存用户
            self.users[username] = user
            self._save_users()
            
            self.logger.info(f"用户注册成功: {username}")
            return True, "注册成功"
            
        except Exception as e:
            self.logger.error(f"用户注册失败: {e}")
            return False, f"注册失败: {str(e)}"
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[TokenInfo], str]:
        """用户认证"""
        try:
            # 检查用户是否存在
            if username not in self.users:
                return False, None, "用户不存在"
            
            user = self.users[username]
            
            # 检查用户是否被锁定
            if self._is_user_locked(user):
                return False, None, "账户已被锁定"
            
            # 检查用户是否激活
            if not user.is_active:
                return False, None, "账户未激活"
            
            # 验证密码
            if not self._verify_password(password, user.password_hash):
                self._record_failed_login(username)
                return False, None, "密码错误"
            
            # 清除失败登录记录
            self._clear_failed_logins(username)
            
            # 更新最后登录时间
            user.last_login = datetime.now().isoformat()
            
            # 生成令牌
            token_info = self._generate_tokens(user)
            
            # 保存活跃令牌
            self.active_tokens[token_info.access_token] = {
                'user_id': user.user_id,
                'username': username,
                'roles': user.roles,
                'issued_at': time.time(),
                'expires_at': time.time() + token_info.expires_in
            }
            
            self.logger.info(f"用户认证成功: {username}")
            return True, token_info, "认证成功"
            
        except Exception as e:
            self.logger.error(f"用户认证失败: {e}")
            return False, None, f"认证失败: {str(e)}"
    
    def refresh_token(self, refresh_token: str) -> Tuple[bool, Optional[TokenInfo], str]:
        """刷新令牌"""
        try:
            # 验证刷新令牌
            payload = jwt.decode(refresh_token, self.config['jwt_secret'], 
                               algorithms=[self.config['jwt_algorithm']])
            
            username = payload.get('username')
            if not username or username not in self.users:
                return False, None, "无效的刷新令牌"
            
            user = self.users[username]
            
            # 生成新令牌
            token_info = self._generate_tokens(user)
            
            # 更新活跃令牌
            self.active_tokens[token_info.access_token] = {
                'user_id': user.user_id,
                'username': username,
                'roles': user.roles,
                'issued_at': time.time(),
                'expires_at': time.time() + token_info.expires_in
            }
            
            self.logger.info(f"令牌刷新成功: {username}")
            return True, token_info, "令牌刷新成功"
            
        except jwt.ExpiredSignatureError:
            return False, None, "刷新令牌已过期"
        except jwt.InvalidTokenError:
            return False, None, "无效的刷新令牌"
        except Exception as e:
            self.logger.error(f"令牌刷新失败: {e}")
            return False, None, f"令牌刷新失败: {str(e)}"
    
    def validate_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """验证令牌"""
        try:
            # 检查令牌是否在活跃列表中
            if token not in self.active_tokens:
                return False, None, "令牌不存在"
            
            token_info = self.active_tokens[token]
            
            # 检查令牌是否过期
            if time.time() > token_info['expires_at']:
                del self.active_tokens[token]
                return False, None, "令牌已过期"
            
            # 验证JWT令牌
            payload = jwt.decode(token, self.config['jwt_secret'], 
                               algorithms=[self.config['jwt_algorithm']])
            
            # 返回用户信息
            user_info = {
                'user_id': token_info['user_id'],
                'username': token_info['username'],
                'roles': token_info['roles']
            }
            
            return True, user_info, "令牌有效"
            
        except jwt.ExpiredSignatureError:
            return False, None, "令牌已过期"
        except jwt.InvalidTokenError:
            return False, None, "无效的令牌"
        except Exception as e:
            self.logger.error(f"令牌验证失败: {e}")
            return False, None, f"令牌验证失败: {str(e)}"
    
    def revoke_token(self, token: str) -> bool:
        """撤销令牌"""
        try:
            if token in self.active_tokens:
                del self.active_tokens[token]
                self.logger.info("令牌已撤销")
                return True
            return False
        except Exception as e:
            self.logger.error(f"撤销令牌失败: {e}")
            return False
    
    def revoke_all_user_tokens(self, username: str) -> int:
        """撤销用户所有令牌"""
        try:
            revoked_count = 0
            tokens_to_remove = []
            
            for token, token_info in self.active_tokens.items():
                if token_info['username'] == username:
                    tokens_to_remove.append(token)
            
            for token in tokens_to_remove:
                del self.active_tokens[token]
                revoked_count += 1
            
            self.logger.info(f"撤销用户令牌: {username}, 数量: {revoked_count}")
            return revoked_count
            
        except Exception as e:
            self.logger.error(f"撤销用户令牌失败: {e}")
            return 0
    
    def _validate_username(self, username: str) -> bool:
        """验证用户名"""
        if not username or len(username) < 3:
            return False
        return username.isalnum() or '_' in username
    
    def _validate_email(self, email: str) -> bool:
        """验证邮箱"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_password(self, password: str) -> bool:
        """验证密码"""
        if len(password) < self.config['password_min_length']:
            return False
        
        if self.config['password_require_special']:
            if not any(c in password for c in '!@#$%^&*()_+-=[]{}|;:,.<>?'):
                return False
        
        if self.config['password_require_numbers']:
            if not any(c.isdigit() for c in password):
                return False
        
        if self.config['password_require_uppercase']:
            if not any(c.isupper() for c in password):
                return False
        
        return True
    
    def _hash_password(self, password: str) -> str:
        """哈希密码"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _generate_user_id(self) -> str:
        """生成用户ID"""
        return hashlib.sha256(f"{time.time()}{secrets.token_urlsafe(16)}".encode()).hexdigest()[:16]
    
    def _generate_tokens(self, user: User) -> TokenInfo:
        """生成令牌"""
        now = time.time()
        
        # 访问令牌载荷
        access_payload = {
            'user_id': user.user_id,
            'username': user.username,
            'roles': user.roles,
            'iat': now,
            'exp': now + self.config['access_token_expiry'],
            'type': 'access'
        }
        
        # 刷新令牌载荷
        refresh_payload = {
            'user_id': user.user_id,
            'username': user.username,
            'iat': now,
            'exp': now + self.config['refresh_token_expiry'],
            'type': 'refresh'
        }
        
        # 生成令牌
        access_token = jwt.encode(access_payload, self.config['jwt_secret'], 
                                algorithm=self.config['jwt_algorithm'])
        refresh_token = jwt.encode(refresh_payload, self.config['jwt_secret'], 
                                 algorithm=self.config['jwt_algorithm'])
        
        return TokenInfo(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.config['access_token_expiry']
        )
    
    def _is_user_locked(self, user: User) -> bool:
        """检查用户是否被锁定"""
        if user.locked_until:
            locked_until = datetime.fromisoformat(user.locked_until)
            if datetime.now() < locked_until:
                return True
            else:
                # 解锁用户
                user.locked_until = None
                user.failed_login_attempts = 0
        return False
    
    def _record_failed_login(self, username: str):
        """记录失败登录"""
        now = time.time()
        
        if username not in self.login_attempts:
            self.login_attempts[username] = []
        
        self.login_attempts[username].append(now)
        
        # 清理过期的失败记录
        cutoff_time = now - self.config['lockout_duration']
        self.login_attempts[username] = [
            attempt for attempt in self.login_attempts[username] 
            if attempt > cutoff_time
        ]
        
        # 检查是否需要锁定用户
        if len(self.login_attempts[username]) >= self.config['max_login_attempts']:
            user = self.users[username]
            user.locked_until = (datetime.now() + 
                               timedelta(seconds=self.config['lockout_duration'])).isoformat()
            self.logger.warning(f"用户被锁定: {username}")
    
    def _clear_failed_logins(self, username: str):
        """清除失败登录记录"""
        if username in self.login_attempts:
            del self.login_attempts[username]
    
    def _load_users(self):
        """加载用户数据"""
        try:
            users_file = Path(self.config['users_file'])
            if users_file.exists():
                with open(users_file, 'r') as f:
                    users_data = json.load(f)
                
                for username, user_data in users_data.items():
                    self.users[username] = User(**user_data)
                
                self.logger.info(f"加载用户数据: {len(self.users)} 个用户")
        except Exception as e:
            self.logger.error(f"加载用户数据失败: {e}")
    
    def _save_users(self):
        """保存用户数据"""
        try:
            users_file = Path(self.config['users_file'])
            users_file.parent.mkdir(parents=True, exist_ok=True)
            
            users_data = {
                username: asdict(user) for username, user in self.users.items()
            }
            
            with open(users_file, 'w') as f:
                json.dump(users_data, f, indent=2)
            
            self.logger.info("用户数据已保存")
        except Exception as e:
            self.logger.error(f"保存用户数据失败: {e}")
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        if username in self.users:
            user = self.users[username]
            return {
                'user_id': user.user_id,
                'username': user.username,
                'email': user.email,
                'roles': user.roles,
                'is_active': user.is_active,
                'created_at': user.created_at,
                'last_login': user.last_login
            }
        return None
    
    def update_user_roles(self, username: str, roles: List[str]) -> bool:
        """更新用户角色"""
        try:
            if username in self.users:
                self.users[username].roles = roles
                self._save_users()
                self.logger.info(f"更新用户角色: {username} -> {roles}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"更新用户角色失败: {e}")
            return False
    
    def deactivate_user(self, username: str) -> bool:
        """停用用户"""
        try:
            if username in self.users:
                self.users[username].is_active = False
                self._save_users()
                self.revoke_all_user_tokens(username)
                self.logger.info(f"停用用户: {username}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"停用用户失败: {e}")
            return False
    
    def get_active_tokens_count(self) -> int:
        """获取活跃令牌数量"""
        return len(self.active_tokens)
    
    def cleanup_expired_tokens(self) -> int:
        """清理过期令牌"""
        now = time.time()
        expired_tokens = []
        
        for token, token_info in self.active_tokens.items():
            if now > token_info['expires_at']:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self.active_tokens[token]
        
        if expired_tokens:
            self.logger.info(f"清理过期令牌: {len(expired_tokens)} 个")
        
        return len(expired_tokens)


class JWTManager:
    """JWT令牌管理器"""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.logger = get_global_logger()
    
    def create_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """创建JWT令牌"""
        try:
            now = time.time()
            payload.update({
                'iat': now,
                'exp': now + expires_in
            })
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            return token
        except Exception as e:
            self.logger.error(f"创建JWT令牌失败: {e}")
            raise
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """解码JWT令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("令牌已过期")
        except jwt.InvalidTokenError:
            raise ValueError("无效的令牌")
        except Exception as e:
            self.logger.error(f"解码JWT令牌失败: {e}")
            raise
    
    def is_token_valid(self, token: str) -> bool:
        """检查令牌是否有效"""
        try:
            self.decode_token(token)
            return True
        except:
            return False
    
    def get_token_expiry(self, token: str) -> Optional[datetime]:
        """获取令牌过期时间"""
        try:
            payload = self.decode_token(token)
            return datetime.fromtimestamp(payload['exp'])
        except:
            return None
