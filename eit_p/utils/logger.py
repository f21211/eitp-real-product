"""
EIT-P 日志系统
提供统一的日志管理和监控功能
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime
import torch


class EITPLogger:
    """EIT-P专用日志器"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, log_file: Optional[str] = None):
        self.name = name
        self.config = config or {}
        if log_file:
            self.config['file'] = log_file
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志器"""
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 设置日志级别
        level = self.config.get('level', 'INFO')
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 创建格式化器
        formatter = logging.Formatter(
            self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        log_file = self.config.get('file', 'eitp_training.log')
        if log_file:
            # 确保日志目录存在
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建轮转文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.get('max_file_size', 10 * 1024 * 1024),  # 10MB
                backupCount=self.config.get('backup_count', 5)
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """记录错误日志"""
        self.logger.error(self._format_message(message, **kwargs))
    
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """记录严重错误日志"""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """格式化消息"""
        if kwargs:
            return f"{message} | {json.dumps(kwargs, ensure_ascii=False)}"
        return message


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, logger: EITPLogger):
        self.logger = logger
        self.start_time = time.time()
        self.step_count = 0
        self.loss_history = []
        self.memory_history = []
        
    def log_step(self, step: int, loss: float, **metrics):
        """记录训练步骤"""
        self.step_count = step
        
        # 记录损失
        self.loss_history.append(loss)
        if len(self.loss_history) > 1000:  # 保持最近1000步的历史
            self.loss_history.pop(0)
        
        # 记录内存使用
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.memory_history.append({
                'step': step,
                'allocated': allocated,
                'reserved': reserved,
                'timestamp': time.time()
            })
        
        # 记录日志
        self.logger.info(
            f"训练步骤 {step}",
            loss=loss,
            **metrics
        )
        
        # 每100步记录详细统计
        if step % 100 == 0:
            self._log_detailed_stats(step)
    
    def _log_detailed_stats(self, step: int):
        """记录详细统计信息"""
        if not self.loss_history:
            return
        
        # 损失统计
        recent_losses = self.loss_history[-100:]  # 最近100步
        avg_loss = sum(recent_losses) / len(recent_losses)
        min_loss = min(recent_losses)
        max_loss = max(recent_losses)
        
        # 内存统计
        memory_stats = {}
        if self.memory_history:
            recent_memory = self.memory_history[-10:]  # 最近10步
            avg_allocated = sum(m['allocated'] for m in recent_memory) / len(recent_memory)
            avg_reserved = sum(m['reserved'] for m in recent_memory) / len(recent_memory)
            memory_stats = {
                'avg_allocated_gb': avg_allocated,
                'avg_reserved_gb': avg_reserved
            }
        
        # 时间统计
        elapsed_time = time.time() - self.start_time
        steps_per_second = step / elapsed_time if elapsed_time > 0 else 0
        
        self.logger.info(
            f"训练统计 - 步骤 {step}",
            avg_loss=avg_loss,
            min_loss=min_loss,
            max_loss=max_loss,
            elapsed_time=elapsed_time,
            steps_per_second=steps_per_second,
            **memory_stats
        )
    
    def log_memory_cleanup(self, before_gb: float, after_gb: float):
        """记录内存清理"""
        freed = before_gb - after_gb
        self.logger.info(
            "内存清理完成",
            before_gb=before_gb,
            after_gb=after_gb,
            freed_gb=freed
        )
    
    def log_error(self, error: Exception, context: str = ""):
        """记录错误"""
        self.logger.error(
            f"训练错误: {type(error).__name__}",
            error_message=str(error),
            context=context,
            step=self.step_count
        )


def setup_logging(config: Optional[Dict[str, Any]] = None) -> EITPLogger:
    """
    设置全局日志系统
    
    Args:
        config: 日志配置
        
    Returns:
        配置好的日志器
    """
    if config is None:
        config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'eitp_training.log',
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5
        }
    
    return EITPLogger('eitp', config)


def get_logger(name: str) -> EITPLogger:
    """
    获取指定名称的日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        日志器实例
    """
    return EITPLogger(name)


# 全局日志器实例
_global_logger = None

def get_global_logger() -> EITPLogger:
    """获取全局日志器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logging()
    return _global_logger
