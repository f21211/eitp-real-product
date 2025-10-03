"""
速率限制模块
提供API和服务的速率限制功能
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum


class RateLimitType(Enum):
    """速率限制类型"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    window_size: int = 60  # 秒
    rate_limit_type: RateLimitType = RateLimitType.SLIDING_WINDOW


class RateLimiter:
    """基础速率限制器"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests = defaultdict(deque)
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """
        检查请求是否被允许
        
        Args:
            identifier: 请求标识符（如用户ID、IP地址等）
            
        Returns:
            是否允许请求
        """
        with self.lock:
            now = time.time()
            request_queue = self.requests[identifier]
            
            # 清理过期请求
            while request_queue and now - request_queue[0] > self.config.window_size:
                request_queue.popleft()
            
            # 检查是否超过限制
            if len(request_queue) >= self.config.requests_per_minute:
                return False
            
            # 记录新请求
            request_queue.append(now)
            return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """
        获取剩余请求数
        
        Args:
            identifier: 请求标识符
            
        Returns:
            剩余请求数
        """
        with self.lock:
            now = time.time()
            request_queue = self.requests[identifier]
            
            # 清理过期请求
            while request_queue and now - request_queue[0] > self.config.window_size:
                request_queue.popleft()
            
            return max(0, self.config.requests_per_minute - len(request_queue))
    
    def get_reset_time(self, identifier: str) -> float:
        """
        获取限制重置时间
        
        Args:
            identifier: 请求标识符
            
        Returns:
            重置时间戳
        """
        with self.lock:
            request_queue = self.requests[identifier]
            if not request_queue:
                return time.time()
            
            return request_queue[0] + self.config.window_size


class TokenBucketRateLimiter(RateLimiter):
    """令牌桶速率限制器"""
    
    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.tokens = defaultdict(lambda: config.burst_limit)
        self.last_refill = defaultdict(time.time)
    
    def is_allowed(self, identifier: str) -> bool:
        """检查请求是否被允许（令牌桶算法）"""
        with self.lock:
            now = time.time()
            time_passed = now - self.last_refill[identifier]
            
            # 补充令牌
            tokens_to_add = time_passed * (self.config.requests_per_minute / 60)
            self.tokens[identifier] = min(
                self.config.burst_limit,
                self.tokens[identifier] + tokens_to_add
            )
            self.last_refill[identifier] = now
            
            # 检查是否有可用令牌
            if self.tokens[identifier] >= 1:
                self.tokens[identifier] -= 1
                return True
            
            return False


class LeakyBucketRateLimiter(RateLimiter):
    """漏桶速率限制器"""
    
    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.bucket_level = defaultdict(float)
        self.last_leak = defaultdict(time.time)
    
    def is_allowed(self, identifier: str) -> bool:
        """检查请求是否被允许（漏桶算法）"""
        with self.lock:
            now = time.time()
            time_passed = now - self.last_leak[identifier]
            
            # 漏水（处理请求）
            leak_rate = self.config.requests_per_minute / 60
            self.bucket_level[identifier] = max(
                0,
                self.bucket_level[identifier] - time_passed * leak_rate
            )
            self.last_leak[identifier] = now
            
            # 检查桶是否已满
            if self.bucket_level[identifier] < self.config.burst_limit:
                self.bucket_level[identifier] += 1
                return True
            
            return False


class APIRateLimiter:
    """API速率限制器"""
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        
        # 根据类型创建限制器
        if self.config.rate_limit_type == RateLimitType.TOKEN_BUCKET:
            self.limiter = TokenBucketRateLimiter(self.config)
        elif self.config.rate_limit_type == RateLimitType.LEAKY_BUCKET:
            self.limiter = LeakyBucketRateLimiter(self.config)
        else:
            self.limiter = RateLimiter(self.config)
    
    def check_rate_limit(self, identifier: str) -> Dict[str, Any]:
        """
        检查速率限制
        
        Args:
            identifier: 请求标识符
            
        Returns:
            限制检查结果
        """
        is_allowed = self.limiter.is_allowed(identifier)
        remaining = self.limiter.get_remaining_requests(identifier)
        reset_time = self.limiter.get_reset_time(identifier)
        
        return {
            "allowed": is_allowed,
            "remaining_requests": remaining,
            "reset_time": reset_time,
            "limit": self.config.requests_per_minute
        }
    
    def apply_rate_limit(self, identifier: str, 
                        on_rate_limited: Optional[Callable] = None) -> bool:
        """
        应用速率限制
        
        Args:
            identifier: 请求标识符
            on_rate_limited: 被限制时的回调函数
            
        Returns:
            是否允许请求
        """
        result = self.check_rate_limit(identifier)
        
        if not result["allowed"]:
            if on_rate_limited:
                on_rate_limited(identifier, result)
            return False
        
        return True


class GlobalRateLimiter:
    """全局速率限制器"""
    
    def __init__(self):
        self.limiters: Dict[str, APIRateLimiter] = {}
        self.default_config = RateLimitConfig()
    
    def get_limiter(self, service_name: str, 
                   config: Optional[RateLimitConfig] = None) -> APIRateLimiter:
        """
        获取服务限制器
        
        Args:
            service_name: 服务名称
            config: 限制配置
            
        Returns:
            速率限制器
        """
        if service_name not in self.limiters:
            self.limiters[service_name] = APIRateLimiter(config or self.default_config)
        
        return self.limiters[service_name]
    
    def check_service_rate_limit(self, service_name: str, 
                               identifier: str) -> Dict[str, Any]:
        """
        检查服务速率限制
        
        Args:
            service_name: 服务名称
            identifier: 请求标识符
            
        Returns:
            限制检查结果
        """
        limiter = self.get_limiter(service_name)
        return limiter.check_rate_limit(identifier)
    
    def apply_service_rate_limit(self, service_name: str, identifier: str,
                               on_rate_limited: Optional[Callable] = None) -> bool:
        """
        应用服务速率限制
        
        Args:
            service_name: 服务名称
            identifier: 请求标识符
            on_rate_limited: 被限制时的回调函数
            
        Returns:
            是否允许请求
        """
        limiter = self.get_limiter(service_name)
        return limiter.apply_rate_limit(identifier, on_rate_limited)


# 全局速率限制器实例
_global_rate_limiter = GlobalRateLimiter()


def get_global_rate_limiter() -> GlobalRateLimiter:
    """获取全局速率限制器"""
    return _global_rate_limiter


def create_rate_limiter(config: RateLimitConfig) -> APIRateLimiter:
    """创建速率限制器"""
    return APIRateLimiter(config)


def rate_limit(service_name: str, identifier: str = None):
    """
    速率限制装饰器
    
    Args:
        service_name: 服务名称
        identifier: 请求标识符（可选，默认使用函数名）
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成标识符
            if identifier is None:
                func_id = f"{func.__module__}.{func.__name__}"
            else:
                func_id = identifier
            
            # 检查速率限制
            limiter = _global_rate_limiter.get_limiter(service_name)
            if not limiter.apply_rate_limit(func_id):
                raise Exception(f"Rate limit exceeded for {service_name}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
