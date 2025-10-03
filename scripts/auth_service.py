#!/usr/bin/env python3
"""
EIT-P 用户认证和权限管理服务
提供JWT认证、用户管理、权限控制和审计功能
"""

import os
import sys
import json
import time
import hashlib
import jwt
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import wraps

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from flask import Flask, request, jsonify, g
    from flask_cors import CORS
    import sqlite3
    from werkzeug.security import generate_password_hash, check_password_hash
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: pip install flask flask-cors pyjwt pyyaml")
    sys.exit(1)

from eit_p.utils import get_global_logger


class UserManager:
    """用户管理器"""
    
    def __init__(self, db_path: str = "data/auth.db"):
        self.db_path = db_path
        self.logger = get_global_logger()
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                permissions TEXT NOT NULL DEFAULT '[]',
                is_active BOOLEAN NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # 创建会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 创建审计日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                resource TEXT,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # 创建默认管理员用户
        self._create_default_admin()
    
    def _create_default_admin(self):
        """创建默认管理员用户"""
        if not self.get_user_by_username('admin'):
            self.create_user(
                username='admin',
                email='admin@eitp.com',
                password='admin123',
                role='admin',
                permissions=['read', 'write', 'admin', 'delete']
            )
            self.logger.info("默认管理员用户已创建: admin/admin123")
    
    def create_user(self, username: str, email: str, password: str, 
                   role: str = 'user', permissions: List[str] = None):
        """创建用户"""
        if permissions is None:
            permissions = ['read', 'write'] if role == 'user' else ['read']
        
        password_hash = generate_password_hash(password)
        permissions_json = json.dumps(permissions)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role, permissions)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, password_hash, role, permissions_json))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            self.log_audit(user_id, 'user_created', 'user', {
                'username': username,
                'email': email,
                'role': role
            })
            
            return user_id
        except sqlite3.IntegrityError as e:
            if 'UNIQUE constraint failed' in str(e):
                raise ValueError("用户名或邮箱已存在")
            raise
        finally:
            conn.close()
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """根据用户名获取用户"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, password_hash, role, permissions, 
                   is_active, created_at, updated_at, last_login
            FROM users WHERE username = ?
        ''', (username,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'username': row[1],
                'email': row[2],
                'password_hash': row[3],
                'role': row[4],
                'permissions': json.loads(row[5]),
                'is_active': bool(row[6]),
                'created_at': row[7],
                'updated_at': row[8],
                'last_login': row[9]
            }
        return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """根据ID获取用户"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, password_hash, role, permissions, 
                   is_active, created_at, updated_at, last_login
            FROM users WHERE id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'username': row[1],
                'email': row[2],
                'password_hash': row[3],
                'role': row[4],
                'permissions': json.loads(row[5]),
                'is_active': bool(row[6]),
                'created_at': row[7],
                'updated_at': row[8],
                'last_login': row[9]
            }
        return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """验证用户"""
        user = self.get_user_by_username(username)
        if not user or not user['is_active']:
            return None
        
        if check_password_hash(user['password_hash'], password):
            # 更新最后登录时间
            self.update_last_login(user['id'])
            return user
        return None
    
    def update_last_login(self, user_id: int):
        """更新最后登录时间"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
    
    def update_user(self, user_id: int, **kwargs):
        """更新用户信息"""
        allowed_fields = ['email', 'role', 'permissions', 'is_active']
        updates = []
        values = []
        
        for field, value in kwargs.items():
            if field in allowed_fields:
                if field == 'permissions':
                    value = json.dumps(value)
                updates.append(f"{field} = ?")
                values.append(value)
        
        if not updates:
            return False
        
        updates.append("updated_at = CURRENT_TIMESTAMP")
        values.append(user_id)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f'''
            UPDATE users SET {', '.join(updates)} WHERE id = ?
        ''', values)
        
        conn.commit()
        conn.close()
        
        self.log_audit(user_id, 'user_updated', 'user', kwargs)
        return True
    
    def delete_user(self, user_id: int):
        """删除用户"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 软删除：设置为非活跃状态
        cursor.execute('''
            UPDATE users SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
        
        self.log_audit(user_id, 'user_deleted', 'user', {})
        return True
    
    def list_users(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """列出用户"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, role, permissions, is_active, 
                   created_at, updated_at, last_login
            FROM users 
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        users = []
        for row in rows:
            users.append({
                'id': row[0],
                'username': row[1],
                'email': row[2],
                'role': row[3],
                'permissions': json.loads(row[4]),
                'is_active': bool(row[5]),
                'created_at': row[6],
                'updated_at': row[7],
                'last_login': row[8]
            })
        
        return users
    
    def log_audit(self, user_id: Optional[int], action: str, resource: str, 
                  details: Dict[str, Any], ip_address: str = None, user_agent: str = None):
        """记录审计日志"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_logs (user_id, action, resource, details, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, action, resource, json.dumps(details), ip_address, user_agent))
        
        conn.commit()
        conn.close()
    
    def get_audit_logs(self, user_id: Optional[int] = None, limit: int = 100, 
                      offset: int = 0) -> List[Dict[str, Any]]:
        """获取审计日志"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT id, user_id, action, resource, details, ip_address, 
                       user_agent, created_at
                FROM audit_logs 
                WHERE user_id = ?
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            ''', (user_id, limit, offset))
        else:
            cursor.execute('''
                SELECT id, user_id, action, resource, details, ip_address, 
                       user_agent, created_at
                FROM audit_logs 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        logs = []
        for row in rows:
            logs.append({
                'id': row[0],
                'user_id': row[1],
                'action': row[2],
                'resource': row[3],
                'details': json.loads(row[4]) if row[4] else {},
                'ip_address': row[5],
                'user_agent': row[6],
                'created_at': row[7]
            })
        
        return logs


class AuthService:
    """认证服务"""
    
    def __init__(self, host='0.0.0.0', port=8087, config_path=None):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.host = host
        self.port = port
        self.logger = get_global_logger()
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化用户管理器
        self.user_manager = UserManager(self.config.get('database_path', 'data/auth.db'))
        
        # JWT配置
        self.jwt_secret = self.config.get('jwt_secret', 'eitp_jwt_secret_2025')
        self.jwt_expiry = self.config.get('jwt_expiry_hours', 24)
        
        # 设置路由
        self._setup_routes()
    
    def _load_config(self, config_path):
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return {
                'jwt_secret': 'eitp_jwt_secret_2025',
                'jwt_expiry_hours': 24,
                'database_path': 'data/auth.db'
            }
    
    def _generate_token(self, user: Dict[str, Any]) -> str:
        """生成JWT令牌"""
        payload = {
            'user_id': user['id'],
            'username': user['username'],
            'role': user['role'],
            'permissions': user['permissions'],
            'exp': datetime.utcnow() + timedelta(hours=self.jwt_expiry)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def _require_auth(self, f):
        """认证装饰器"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'status': 'error', 'message': '缺少认证令牌'}), 401
            
            if token.startswith('Bearer '):
                token = token[7:]
            
            payload = self._verify_token(token)
            if not payload:
                return jsonify({'status': 'error', 'message': '无效或过期的令牌'}), 401
            
            g.current_user = payload
            return f(*args, **kwargs)
        return decorated_function
    
    def _require_permission(self, permission):
        """权限检查装饰器"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if not hasattr(g, 'current_user'):
                    return jsonify({'status': 'error', 'message': '需要认证'}), 401
                
                user_permissions = g.current_user.get('permissions', [])
                if permission not in user_permissions:
                    return jsonify({'status': 'error', 'message': '权限不足'}), 403
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'service': 'EIT-P Auth Service',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/auth/login', methods=['POST'])
        def login():
            """用户登录"""
            try:
                data = request.get_json()
                username = data.get('username')
                password = data.get('password')
                
                if not username or not password:
                    return jsonify({'status': 'error', 'message': '用户名和密码不能为空'}), 400
                
                # 验证用户
                user = self.user_manager.authenticate_user(username, password)
                if not user:
                    self.user_manager.log_audit(
                        None, 'login_failed', 'auth', 
                        {'username': username},
                        request.remote_addr, request.headers.get('User-Agent')
                    )
                    return jsonify({'status': 'error', 'message': '用户名或密码错误'}), 401
                
                # 生成令牌
                token = self._generate_token(user)
                
                # 记录登录日志
                self.user_manager.log_audit(
                    user['id'], 'login_success', 'auth', 
                    {'username': username},
                    request.remote_addr, request.headers.get('User-Agent')
                )
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'token': token,
                        'user': {
                            'id': user['id'],
                            'username': user['username'],
                            'email': user['email'],
                            'role': user['role'],
                            'permissions': user['permissions']
                        }
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/auth/logout', methods=['POST'])
        @self._require_auth
        def logout():
            """用户登出"""
            try:
                # 记录登出日志
                self.user_manager.log_audit(
                    g.current_user['user_id'], 'logout', 'auth', {},
                    request.remote_addr, request.headers.get('User-Agent')
                )
                
                return jsonify({'status': 'success', 'message': '登出成功'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/auth/profile', methods=['GET'])
        @self._require_auth
        def get_profile():
            """获取用户信息"""
            try:
                user = self.user_manager.get_user_by_id(g.current_user['user_id'])
                if not user:
                    return jsonify({'status': 'error', 'message': '用户不存在'}), 404
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'id': user['id'],
                        'username': user['username'],
                        'email': user['email'],
                        'role': user['role'],
                        'permissions': user['permissions'],
                        'is_active': user['is_active'],
                        'created_at': user['created_at'],
                        'last_login': user['last_login']
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/auth/register', methods=['POST'])
        def register():
            """用户注册"""
            try:
                data = request.get_json()
                username = data.get('username')
                email = data.get('email')
                password = data.get('password')
                role = data.get('role', 'user')
                
                if not all([username, email, password]):
                    return jsonify({'status': 'error', 'message': '缺少必需字段'}), 400
                
                # 创建用户
                user_id = self.user_manager.create_user(username, email, password, role)
                
                return jsonify({
                    'status': 'success',
                    'message': '用户注册成功',
                    'data': {'user_id': user_id}
                })
            except ValueError as e:
                return jsonify({'status': 'error', 'message': str(e)}), 400
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/users', methods=['GET'])
        @self._require_auth
        @self._require_permission('admin')
        def list_users():
            """列出用户"""
            try:
                limit = request.args.get('limit', 100, type=int)
                offset = request.args.get('offset', 0, type=int)
                
                users = self.user_manager.list_users(limit, offset)
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'users': users,
                        'total_count': len(users)
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/users/<int:user_id>', methods=['GET'])
        @self._require_auth
        @self._require_permission('admin')
        def get_user(user_id):
            """获取用户详情"""
            try:
                user = self.user_manager.get_user_by_id(user_id)
                if not user:
                    return jsonify({'status': 'error', 'message': '用户不存在'}), 404
                
                return jsonify({
                    'status': 'success',
                    'data': user
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/users/<int:user_id>', methods=['PUT'])
        @self._require_auth
        @self._require_permission('admin')
        def update_user(user_id):
            """更新用户"""
            try:
                data = request.get_json()
                
                success = self.user_manager.update_user(user_id, **data)
                if success:
                    return jsonify({'status': 'success', 'message': '用户更新成功'})
                else:
                    return jsonify({'status': 'error', 'message': '用户更新失败'}), 400
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/users/<int:user_id>', methods=['DELETE'])
        @self._require_auth
        @self._require_permission('admin')
        def delete_user(user_id):
            """删除用户"""
            try:
                success = self.user_manager.delete_user(user_id)
                if success:
                    return jsonify({'status': 'success', 'message': '用户删除成功'})
                else:
                    return jsonify({'status': 'error', 'message': '用户删除失败'}), 400
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/audit/logs', methods=['GET'])
        @self._require_auth
        @self._require_permission('admin')
        def get_audit_logs():
            """获取审计日志"""
            try:
                user_id = request.args.get('user_id', type=int)
                limit = request.args.get('limit', 100, type=int)
                offset = request.args.get('offset', 0, type=int)
                
                logs = self.user_manager.get_audit_logs(user_id, limit, offset)
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'logs': logs,
                        'total_count': len(logs)
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def run(self, debug=False):
        """运行认证服务"""
        self.logger.info(f"启动EIT-P认证服务: http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)


def main():
    """主函数"""
    print("🚀 启动EIT-P认证服务...")
    
    # 检查配置文件
    config_path = project_root / "config" / "production.yaml"
    if not config_path.exists():
        print(f"警告: 配置文件不存在 {config_path}，使用默认配置")
        config_path = None
    
    # 启动认证服务
    service = AuthService(config_path=str(config_path) if config_path else None)
    service.run(debug=False)


if __name__ == "__main__":
    main()