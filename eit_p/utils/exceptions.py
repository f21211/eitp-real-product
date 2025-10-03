"""
EIT-P 异常处理模块
定义项目特定的异常类型和错误处理机制
"""

import traceback
from typing import Optional, Dict, Any
import torch


class EITPException(Exception):
    """EIT-P基础异常类"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.traceback = traceback.format_exc()
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details,
            'traceback': self.traceback
        }


class MemoryOverflowError(EITPException):
    """内存溢出异常"""
    
    def __init__(self, memory_type: str, current_usage: float, max_usage: float, **kwargs):
        message = f"{memory_type}内存溢出: 当前使用 {current_usage:.2f}GB, 最大限制 {max_usage:.2f}GB"
        super().__init__(message, error_code="MEMORY_OVERFLOW", **kwargs)
        self.memory_type = memory_type
        self.current_usage = current_usage
        self.max_usage = max_usage


class ConvergenceError(EITPException):
    """收敛异常"""
    
    def __init__(self, loss_value: float, threshold: float, **kwargs):
        message = f"损失值异常: {loss_value:.4f} 超过阈值 {threshold:.4f}"
        super().__init__(message, error_code="CONVERGENCE_ERROR", **kwargs)
        self.loss_value = loss_value
        self.threshold = threshold


class ConfigurationError(EITPException):
    """配置错误"""
    
    def __init__(self, config_key: str, expected_type: str, actual_value: Any, **kwargs):
        message = f"配置错误: {config_key} 期望类型 {expected_type}, 实际值 {actual_value}"
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key
        self.expected_type = expected_type
        self.actual_value = actual_value


class ModelError(EITPException):
    """模型相关错误"""
    
    def __init__(self, model_name: str, operation: str, details: str, **kwargs):
        message = f"模型错误: {model_name} 在 {operation} 操作中失败 - {details}"
        super().__init__(message, error_code="MODEL_ERROR", **kwargs)
        self.model_name = model_name
        self.operation = operation


class TrainingError(EITPException):
    """训练相关错误"""
    
    def __init__(self, step: int, loss_value: float, details: str, **kwargs):
        message = f"训练错误: 第 {step} 步, 损失值 {loss_value:.4f} - {details}"
        super().__init__(message, error_code="TRAINING_ERROR", **kwargs)
        self.step = step
        self.loss_value = loss_value


class EvaluationError(EITPException):
    """评估相关错误"""
    
    def __init__(self, metric_name: str, details: str, **kwargs):
        message = f"评估错误: {metric_name} 计算失败 - {details}"
        super().__init__(message, error_code="EVALUATION_ERROR", **kwargs)
        self.metric_name = metric_name


class HardwareError(EITPException):
    """硬件相关错误"""
    
    def __init__(self, device: str, details: str, **kwargs):
        message = f"硬件错误: {device} 设备问题 - {details}"
        super().__init__(message, error_code="HARDWARE_ERROR", **kwargs)
        self.device = device


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.error_count = {}
        self.max_retries = 3
    
    def handle_error(self, exception: Exception, context: Optional[str] = None) -> None:
        """
        处理异常（handle_exception的别名）
        
        Args:
            exception: 异常对象
            context: 上下文信息
        """
        self.handle_exception(exception, context)
    
    def handle_exception(self, exception: Exception, context: Optional[str] = None) -> None:
        """
        处理异常
        
        Args:
            exception: 异常对象
            context: 上下文信息
        """
        error_type = type(exception).__name__
        
        # 记录错误计数
        if error_type not in self.error_count:
            self.error_count[error_type] = 0
        self.error_count[error_type] += 1
        
        # 记录错误
        if self.logger:
            self.logger.error(f"异常处理: {error_type} - {str(exception)}")
            if context:
                self.logger.error(f"上下文: {context}")
            self.logger.error(f"堆栈跟踪: {traceback.format_exc()}")
        
        # 特殊处理
        if isinstance(exception, MemoryOverflowError):
            self._handle_memory_overflow(exception)
        elif isinstance(exception, ConvergenceError):
            self._handle_convergence_error(exception)
        elif isinstance(exception, HardwareError):
            self._handle_hardware_error(exception)
    
    def _handle_memory_overflow(self, exception: MemoryOverflowError) -> None:
        """处理内存溢出"""
        if self.logger:
            self.logger.warning("检测到内存溢出，执行清理...")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        import gc
        gc.collect()
    
    def _handle_convergence_error(self, exception: ConvergenceError) -> None:
        """处理收敛错误"""
        if self.logger:
            self.logger.warning(f"检测到收敛问题: 损失值 {exception.loss_value:.4f}")
    
    def _handle_hardware_error(self, exception: HardwareError) -> None:
        """处理硬件错误"""
        if self.logger:
            self.logger.error(f"硬件错误: {exception.device}")
    
    def get_error_summary(self) -> Dict[str, int]:
        """获取错误统计摘要"""
        return self.error_count.copy()
    
    def reset_error_count(self) -> None:
        """重置错误计数"""
        self.error_count.clear()


def safe_execute(func, *args, error_handler: Optional[ErrorHandler] = None, **kwargs):
    """
    安全执行函数，自动处理异常
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        error_handler: 错误处理器
        **kwargs: 函数关键字参数
        
    Returns:
        函数执行结果或None（如果出错）
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if error_handler:
            error_handler.handle_exception(e, f"执行函数 {func.__name__}")
        return None


def check_gpu_memory(threshold: float = 0.9) -> bool:
    """
    检查GPU内存使用率
    
    Args:
        threshold: 内存使用率阈值
        
    Returns:
        是否超过阈值
    """
    if not torch.cuda.is_available():
        return False
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    usage_ratio = allocated / total
    return usage_ratio > threshold


def check_convergence(loss_value: float, threshold: float = 1000.0) -> bool:
    """
    检查收敛状态
    
    Args:
        loss_value: 当前损失值
        threshold: 损失阈值
        
    Returns:
        是否收敛异常
    """
    return loss_value > threshold
