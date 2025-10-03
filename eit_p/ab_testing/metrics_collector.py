"""
A/B测试指标收集器
收集和分析A/B测试的指标数据
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import statistics


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"  # 计数器
    GAUGE = "gauge"      # 仪表盘
    HISTOGRAM = "histogram"  # 直方图
    TIMER = "timer"      # 计时器


@dataclass
class MetricData:
    """指标数据"""
    metric_name: str
    metric_type: MetricType
    value: Union[float, int]
    timestamp: datetime
    experiment_id: str
    variant: str
    user_id: Optional[str] = None
    tags: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['metric_type'] = self.metric_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, storage_file: str = "ab_test_metrics.json"):
        self.storage_file = storage_file
        self.logger = logging.getLogger("metrics_collector")
        self.metrics: List[MetricData] = []
        self._load_metrics()
    
    def _load_metrics(self):
        """从文件加载指标数据"""
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    metric = MetricData(
                        metric_name=item['metric_name'],
                        metric_type=MetricType(item['metric_type']),
                        value=item['value'],
                        timestamp=datetime.fromisoformat(item['timestamp']),
                        experiment_id=item['experiment_id'],
                        variant=item['variant'],
                        user_id=item.get('user_id'),
                        tags=item.get('tags', {})
                    )
                    self.metrics.append(metric)
            self.logger.info(f"Loaded {len(self.metrics)} metrics from {self.storage_file}")
        except FileNotFoundError:
            self.logger.info(f"No existing metrics file found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading metrics: {e}")
    
    def _save_metrics(self):
        """保存指标数据到文件"""
        try:
            data = [metric.to_dict() for metric in self.metrics]
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
    
    def record_metric(self, metric_name: str, value: Union[float, int],
                     experiment_id: str, variant: str,
                     metric_type: MetricType = MetricType.GAUGE,
                     user_id: Optional[str] = None,
                     tags: Dict[str, str] = None) -> str:
        """记录指标"""
        metric = MetricData(
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            experiment_id=experiment_id,
            variant=variant,
            user_id=user_id,
            tags=tags or {}
        )
        
        self.metrics.append(metric)
        self._save_metrics()
        
        self.logger.info(f"Recorded metric: {metric_name}={value} for experiment {experiment_id} variant {variant}")
        return metric.metric_name
    
    def increment_counter(self, metric_name: str, experiment_id: str, variant: str,
                        user_id: Optional[str] = None, tags: Dict[str, str] = None):
        """增加计数器"""
        self.record_metric(metric_name, 1, experiment_id, variant, 
                          MetricType.COUNTER, user_id, tags)
    
    def record_gauge(self, metric_name: str, value: Union[float, int],
                    experiment_id: str, variant: str,
                    user_id: Optional[str] = None, tags: Dict[str, str] = None):
        """记录仪表盘值"""
        self.record_metric(metric_name, value, experiment_id, variant,
                          MetricType.GAUGE, user_id, tags)
    
    def record_timer(self, metric_name: str, duration_seconds: float,
                    experiment_id: str, variant: str,
                    user_id: Optional[str] = None, tags: Dict[str, str] = None):
        """记录计时器"""
        self.record_metric(metric_name, duration_seconds, experiment_id, variant,
                          MetricType.TIMER, user_id, tags)
    
    def get_metrics(self, experiment_id: Optional[str] = None,
                   variant: Optional[str] = None,
                   metric_name: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[MetricData]:
        """获取指标数据"""
        filtered_metrics = self.metrics
        
        if experiment_id:
            filtered_metrics = [m for m in filtered_metrics if m.experiment_id == experiment_id]
        
        if variant:
            filtered_metrics = [m for m in filtered_metrics if m.variant == variant]
        
        if metric_name:
            filtered_metrics = [m for m in filtered_metrics if m.metric_name == metric_name]
        
        if start_time:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= start_time]
        
        if end_time:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp <= end_time]
        
        return sorted(filtered_metrics, key=lambda x: x.timestamp)
    
    def get_metric_summary(self, experiment_id: str, metric_name: str) -> Dict[str, Any]:
        """获取指标摘要统计"""
        metrics = self.get_metrics(experiment_id=experiment_id, metric_name=metric_name)
        
        if not metrics:
            return {"error": "No metrics found"}
        
        # 按变体分组
        variant_data = {}
        for metric in metrics:
            if metric.variant not in variant_data:
                variant_data[metric.variant] = []
            variant_data[metric.variant].append(metric.value)
        
        summary = {}
        for variant, values in variant_data.items():
            summary[variant] = {
                "count": len(values),
                "sum": sum(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0
            }
        
        return summary
    
    def get_experiment_comparison(self, experiment_id: str, metric_name: str) -> Dict[str, Any]:
        """获取实验对比结果"""
        summary = self.get_metric_summary(experiment_id, metric_name)
        
        if "error" in summary:
            return summary
        
        variants = list(summary.keys())
        if len(variants) < 2:
            return {"error": "Need at least 2 variants for comparison"}
        
        # 计算统计显著性（简化版本）
        control_variant = variants[0]  # 假设第一个是控制组
        treatment_variants = variants[1:]
        
        control_data = summary[control_variant]
        results = {
            "control_variant": control_variant,
            "control_stats": control_data,
            "treatment_variants": {}
        }
        
        for variant in treatment_variants:
            treatment_data = summary[variant]
            
            # 计算提升百分比
            improvement = ((treatment_data["mean"] - control_data["mean"]) / 
                          control_data["mean"] * 100) if control_data["mean"] != 0 else 0
            
            # 简化的统计显著性检验
            control_std = control_data["std_dev"]
            treatment_std = treatment_data["std_dev"]
            pooled_std = ((control_std ** 2 + treatment_std ** 2) / 2) ** 0.5
            
            if pooled_std > 0:
                z_score = (treatment_data["mean"] - control_data["mean"]) / pooled_std
                # 简化的p值计算（实际应用中应使用更严格的统计检验）
                p_value = 2 * (1 - self._normal_cdf(abs(z_score)))
            else:
                p_value = 1.0
            
            results["treatment_variants"][variant] = {
                "stats": treatment_data,
                "improvement_percent": round(improvement, 2),
                "z_score": round(z_score, 4),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05
            }
        
        return results
    
    def _normal_cdf(self, x: float) -> float:
        """标准正态分布累积分布函数的近似"""
        # 使用误差函数的近似
        import math
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def export_metrics(self, filename: str, experiment_id: Optional[str] = None):
        """导出指标数据"""
        metrics = self.get_metrics(experiment_id=experiment_id)
        
        export_data = [metric.to_dict() for metric in metrics]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported {len(export_data)} metrics to {filename}")


# 全局指标收集器实例
_global_collector = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器实例"""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def record_metric(metric_name: str, value: Union[float, int],
                 experiment_id: str, variant: str, **kwargs) -> str:
    """记录指标的便捷函数"""
    collector = get_metrics_collector()
    return collector.record_metric(metric_name, value, experiment_id, variant, **kwargs)