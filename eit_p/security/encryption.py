"""
加密管理模块
"""

import hashlib
import secrets
from typing import Dict, Any, Optional, Union, List
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from ..utils import get_global_logger


class EncryptionManager:
    """加密管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = get_global_logger()
        self._initialize_encryption()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'encryption_key': secrets.token_urlsafe(32),
            'salt': secrets.token_urlsafe(16),
            'algorithm': 'AES-256-GCM'
        }
    
    def _initialize_encryption(self):
        """初始化加密"""
        try:
            # 生成加密密钥
            password = self.config['encryption_key'].encode()
            salt = self.config['salt'].encode()
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self.cipher = Fernet(key)
            
            self.logger.info("加密管理器初始化完成")
        except Exception as e:
            self.logger.error(f"加密管理器初始化失败: {e}")
            raise
    
    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """加密数据"""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = self.cipher.encrypt(data)
            return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
        except Exception as e:
            self.logger.error(f"数据加密失败: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """解密数据"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self.cipher.decrypt(encrypted_bytes)
            return decrypted_data.decode('utf-8')
        except Exception as e:
            self.logger.error(f"数据解密失败: {e}")
            raise
    
    def hash_data(self, data: str, algorithm: str = 'sha256') -> str:
        """哈希数据"""
        try:
            if algorithm == 'sha256':
                return hashlib.sha256(data.encode('utf-8')).hexdigest()
            elif algorithm == 'md5':
                return hashlib.md5(data.encode('utf-8')).hexdigest()
            else:
                raise ValueError(f"不支持的哈希算法: {algorithm}")
        except Exception as e:
            self.logger.error(f"数据哈希失败: {e}")
            raise


class DataEncryption:
    """数据加密类"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.logger = get_global_logger()
    
    def encrypt_sensitive_data(self, data: Dict[str, Any], 
                             sensitive_fields: List[str]) -> Dict[str, Any]:
        """加密敏感数据"""
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data:
                encrypted_data[field] = self.encryption_manager.encrypt_data(
                    str(encrypted_data[field])
                )
        
        return encrypted_data
    
    def decrypt_sensitive_data(self, data: Dict[str, Any], 
                             sensitive_fields: List[str]) -> Dict[str, Any]:
        """解密敏感数据"""
        decrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in decrypted_data:
                decrypted_data[field] = self.encryption_manager.decrypt_data(
                    decrypted_data[field]
                )
        
        return decrypted_data
