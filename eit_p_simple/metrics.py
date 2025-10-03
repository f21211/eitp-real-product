"""
EIT-P 指标跟踪器 - 简化实现
基于IEM理论的指标管理
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MetricsTracker:
    """指标跟踪器 - 基于IEM理论的指标管理"""
    
    def __init__(self, max_history=1000):
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        self.current_metrics = {}
        self.logger = logging.getLogger(__name__)
        self.logger.info("指标跟踪器初始化完成")
    
    def record_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """记录指标"""
        try:
            if timestamp is None:
                timestamp = time.time()
            
            metric_data = {
                'value': value,
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).isoformat()
            }
            
            self.metrics_history[name].append(metric_data)
            self.current_metrics[name] = value
            
            self.logger.debug(f"指标记录成功: {name} = {value}")
            
        except Exception as e:
            self.logger.error(f"记录指标失败: {e}")
    
    def record_inference_metrics(self, metrics: Dict[str, Any]):
        """记录推理指标"""
        try:
            timestamp = time.time()
            
            for key, value in metrics.items():
                self.record_metric(f"inference_{key}", value, timestamp)
            
            # 计算吞吐量
            if 'inference_time' in metrics and 'response_length' in metrics:
                throughput = metrics['response_length'] / metrics['inference_time']
                self.record_metric('inference_throughput', throughput, timestamp)
            
            self.logger.info(f"推理指标记录成功: {len(metrics)} 个指标")
            
        except Exception as e:
            self.logger.error(f"记录推理指标失败: {e}")
    
    def record_training_metrics(self, metrics: Dict[str, Any]):
        """记录训练指标"""
        try:
            timestamp = time.time()
            
            for key, value in metrics.items():
                self.record_metric(f"training_{key}", value, timestamp)
            
            self.logger.info(f"训练指标记录成功: {len(metrics)} 个指标")
            
        except Exception as e:
            self.logger.error(f"记录训练指标失败: {e}")
    
    def get_metric(self, name: str) -> Optional[float]:
        """获取当前指标值"""
        return self.current_metrics.get(name)
    
    def get_metric_history(self, name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """获取指标历史"""
        try:
            if name not in self.metrics_history:
                return []
            
            cutoff_time = time.time() - (hours * 3600)
            history = []
            
            for metric_data in self.metrics_history[name]:
                if metric_data['timestamp'] >= cutoff_time:
                    history.append(metric_data)
            
            return history
            
        except Exception as e:
            self.logger.error(f"获取指标历史失败: {e}")
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        try:
            metrics = {
                'current': dict(self.current_metrics),
                'summary': self._calculate_summary(),
                'timestamp': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"获取指标失败: {e}")
            return {}
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """计算指标摘要"""
        try:
            summary = {}
            
            for name, history in self.metrics_history.items():
                if not history:
                    continue
                
                values = [h['value'] for h in history]
                summary[name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'latest': values[-1] if values else None
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"计算指标摘要失败: {e}")
            return {}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3)
            }
            
            return system_metrics
            
        except Exception as e:
            self.logger.error(f"获取系统指标失败: {e}")
            return {}
