"""
A/B测试管理器
提供模型版本A/B测试功能
"""

import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import defaultdict

from ..utils import get_global_logger
from ..experiments import ModelRegistry, MetricsTracker


@dataclass
class ABTest:
    """A/B测试配置"""
    test_id: str
    name: str
    description: str
    control_model_id: str
    treatment_model_id: str
    traffic_split: float  # 0.0-1.0, 控制组流量比例
    start_time: str
    end_time: Optional[str] = None
    status: str = "draft"  # draft, running, completed, paused, cancelled
    metrics: List[str] = None
    min_sample_size: int = 1000
    significance_level: float = 0.05
    power: float = 0.8
    created_by: str = "system"
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.metrics is None:
            self.metrics = ["accuracy", "latency", "throughput"]


@dataclass
class TestResult:
    """测试结果"""
    test_id: str
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]
    statistical_significance: Dict[str, bool]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    effect_sizes: Dict[str, float]
    recommendation: str
    analysis_time: str


class ABTestManager:
    """A/B测试管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = get_global_logger()
        
        # 测试存储
        self.tests: Dict[str, ABTest] = {}
        
        # 测试结果
        self.test_results: Dict[str, TestResult] = {}
        
        # 流量分配器
        self.traffic_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> test_id -> variant
        
        # 指标收集器
        self.metrics_collectors: Dict[str, MetricsCollector] = {}
        
        # 模型注册表
        self.model_registry = ModelRegistry()
        
        # 测试状态锁
        self.test_lock = threading.Lock()
        
        # 加载测试数据
        self._load_tests()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'tests_file': './ab_testing/tests.json',
            'results_file': './ab_testing/results.json',
            'max_concurrent_tests': 10,
            'min_test_duration_hours': 24,
            'max_test_duration_days': 30,
            'auto_stop_on_significance': True,
            'default_metrics': ['accuracy', 'latency', 'throughput'],
            'statistical_engine': 'scipy'  # scipy, statsmodels
        }
    
    def create_test(self, 
                   name: str,
                   description: str,
                   control_model_id: str,
                   treatment_model_id: str,
                   traffic_split: float = 0.5,
                   metrics: List[str] = None,
                   min_sample_size: int = 1000,
                   significance_level: float = 0.05,
                   created_by: str = "system") -> str:
        """创建A/B测试"""
        try:
            with self.test_lock:
                # 验证输入
                if not self._validate_test_creation(control_model_id, treatment_model_id, traffic_split):
                    raise ValueError("测试创建参数无效")
                
                # 生成测试ID
                test_id = self._generate_test_id(name)
                
                # 检查测试是否已存在
                if test_id in self.tests:
                    raise ValueError("测试ID已存在")
                
                # 创建测试
                test = ABTest(
                    test_id=test_id,
                    name=name,
                    description=description,
                    control_model_id=control_model_id,
                    treatment_model_id=treatment_model_id,
                    traffic_split=traffic_split,
                    start_time=datetime.now().isoformat(),
                    metrics=metrics or self.config['default_metrics'],
                    min_sample_size=min_sample_size,
                    significance_level=significance_level,
                    created_by=created_by
                )
                
                # 保存测试
                self.tests[test_id] = test
                self._save_tests()
                
                # 初始化指标收集器
                self.metrics_collectors[test_id] = MetricsCollector(test_id)
                
                self.logger.info(f"A/B测试创建成功: {test_id} - {name}")
                return test_id
                
        except Exception as e:
            self.logger.error(f"创建A/B测试失败: {e}")
            raise
    
    def start_test(self, test_id: str) -> bool:
        """启动A/B测试"""
        try:
            with self.test_lock:
                if test_id not in self.tests:
                    raise ValueError("测试不存在")
                
                test = self.tests[test_id]
                
                if test.status != "draft":
                    raise ValueError("测试状态不允许启动")
                
                # 检查并发测试数量
                running_tests = len([t for t in self.tests.values() if t.status == "running"])
                if running_tests >= self.config['max_concurrent_tests']:
                    raise ValueError("并发测试数量已达上限")
                
                # 更新测试状态
                test.status = "running"
                test.start_time = datetime.now().isoformat()
                
                # 保存测试
                self._save_tests()
                
                self.logger.info(f"A/B测试已启动: {test_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"启动A/B测试失败: {e}")
            return False
    
    def stop_test(self, test_id: str, reason: str = "manual") -> bool:
        """停止A/B测试"""
        try:
            with self.test_lock:
                if test_id not in self.tests:
                    raise ValueError("测试不存在")
                
                test = self.tests[test_id]
                
                if test.status != "running":
                    raise ValueError("测试未在运行")
                
                # 更新测试状态
                test.status = "completed"
                test.end_time = datetime.now().isoformat()
                
                # 分析测试结果
                self._analyze_test_results(test_id)
                
                # 保存测试
                self._save_tests()
                
                self.logger.info(f"A/B测试已停止: {test_id} - {reason}")
                return True
                
        except Exception as e:
            self.logger.error(f"停止A/B测试失败: {e}")
            return False
    
    def pause_test(self, test_id: str) -> bool:
        """暂停A/B测试"""
        try:
            with self.test_lock:
                if test_id not in self.tests:
                    raise ValueError("测试不存在")
                
                test = self.tests[test_id]
                
                if test.status != "running":
                    raise ValueError("测试未在运行")
                
                # 更新测试状态
                test.status = "paused"
                
                # 保存测试
                self._save_tests()
                
                self.logger.info(f"A/B测试已暂停: {test_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"暂停A/B测试失败: {e}")
            return False
    
    def resume_test(self, test_id: str) -> bool:
        """恢复A/B测试"""
        try:
            with self.test_lock:
                if test_id not in self.tests:
                    raise ValueError("测试不存在")
                
                test = self.tests[test_id]
                
                if test.status != "paused":
                    raise ValueError("测试未暂停")
                
                # 更新测试状态
                test.status = "running"
                
                # 保存测试
                self._save_tests()
                
                self.logger.info(f"A/B测试已恢复: {test_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"恢复A/B测试失败: {e}")
            return False
    
    def assign_user_to_variant(self, user_id: str, test_id: str) -> str:
        """为用户分配测试变体"""
        try:
            if test_id not in self.tests:
                raise ValueError("测试不存在")
            
            test = self.tests[test_id]
            
            if test.status != "running":
                return "control"  # 非运行状态默认返回控制组
            
            # 检查是否已分配
            if user_id in self.traffic_assignments and test_id in self.traffic_assignments[user_id]:
                return self.traffic_assignments[user_id][test_id]
            
            # 基于用户ID和测试ID生成一致的分配
            hash_input = f"{user_id}_{test_id}_{test.start_time}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            assignment_ratio = hash_value % 10000 / 10000.0
            
            # 分配变体
            if assignment_ratio < test.traffic_split:
                variant = "control"
            else:
                variant = "treatment"
            
            # 记录分配
            if user_id not in self.traffic_assignments:
                self.traffic_assignments[user_id] = {}
            self.traffic_assignments[user_id][test_id] = variant
            
            return variant
            
        except Exception as e:
            self.logger.error(f"分配用户变体失败: {e}")
            return "control"
    
    def record_metric(self, test_id: str, user_id: str, metric_name: str, 
                     value: float, timestamp: Optional[str] = None):
        """记录测试指标"""
        try:
            if test_id not in self.metrics_collectors:
                return
            
            # 获取用户变体
            variant = self.assign_user_to_variant(user_id, test_id)
            
            # 记录指标
            self.metrics_collectors[test_id].record_metric(
                user_id=user_id,
                variant=variant,
                metric_name=metric_name,
                value=value,
                timestamp=timestamp
            )
            
            # 检查是否需要自动停止
            if self.config['auto_stop_on_significance']:
                self._check_auto_stop(test_id)
                
        except Exception as e:
            self.logger.error(f"记录测试指标失败: {e}")
    
    def get_test_results(self, test_id: str) -> Optional[TestResult]:
        """获取测试结果"""
        return self.test_results.get(test_id)
    
    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """获取测试状态"""
        if test_id not in self.tests:
            return None
        
        test = self.tests[test_id]
        
        # 获取指标统计
        metrics_stats = {}
        if test_id in self.metrics_collectors:
            metrics_stats = self.metrics_collectors[test_id].get_metrics_summary()
        
        return {
            'test_id': test_id,
            'name': test.name,
            'status': test.status,
            'start_time': test.start_time,
            'end_time': test.end_time,
            'traffic_split': test.traffic_split,
            'metrics_stats': metrics_stats,
            'has_results': test_id in self.test_results
        }
    
    def list_tests(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出测试"""
        tests = []
        
        for test in self.tests.values():
            if status is None or test.status == status:
                tests.append({
                    'test_id': test.test_id,
                    'name': test.name,
                    'status': test.status,
                    'start_time': test.start_time,
                    'end_time': test.end_time,
                    'traffic_split': test.traffic_split,
                    'created_by': test.created_by
                })
        
        return sorted(tests, key=lambda x: x['start_time'], reverse=True)
    
    def _validate_test_creation(self, control_model_id: str, treatment_model_id: str, 
                              traffic_split: float) -> bool:
        """验证测试创建参数"""
        # 检查模型是否存在
        if not self.model_registry.get_model_metadata(control_model_id):
            return False
        if not self.model_registry.get_model_metadata(treatment_model_id):
            return False
        
        # 检查流量分配
        if not 0.0 <= traffic_split <= 1.0:
            return False
        
        return True
    
    def _generate_test_id(self, name: str) -> str:
        """生成测试ID"""
        timestamp = str(int(time.time()))
        hash_input = f"{name}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _analyze_test_results(self, test_id: str):
        """分析测试结果"""
        try:
            if test_id not in self.metrics_collectors:
                return
            
            collector = self.metrics_collectors[test_id]
            test = self.tests[test_id]
            
            # 获取指标数据
            control_data = collector.get_variant_data("control")
            treatment_data = collector.get_variant_data("treatment")
            
            # 计算统计指标
            control_metrics = {}
            treatment_metrics = {}
            statistical_significance = {}
            p_values = {}
            confidence_intervals = {}
            effect_sizes = {}
            
            for metric in test.metrics:
                # 控制组指标
                control_values = [point['value'] for point in control_data.get(metric, [])]
                control_metrics[metric] = self._calculate_metric_stats(control_values)
                
                # 治疗组指标
                treatment_values = [point['value'] for point in treatment_data.get(metric, [])]
                treatment_metrics[metric] = self._calculate_metric_stats(treatment_values)
                
                # 统计显著性
                if len(control_values) > 0 and len(treatment_values) > 0:
                    p_value = self._calculate_p_value(control_values, treatment_values)
                    p_values[metric] = p_value
                    statistical_significance[metric] = p_value < test.significance_level
                    
                    # 置信区间
                    ci = self._calculate_confidence_interval(control_values, treatment_values)
                    confidence_intervals[metric] = ci
                    
                    # 效应大小
                    effect_size = self._calculate_effect_size(control_values, treatment_values)
                    effect_sizes[metric] = effect_size
                else:
                    p_values[metric] = 1.0
                    statistical_significance[metric] = False
                    confidence_intervals[metric] = (0.0, 0.0)
                    effect_sizes[metric] = 0.0
            
            # 生成推荐
            recommendation = self._generate_recommendation(
                statistical_significance, effect_sizes, test.metrics
            )
            
            # 创建测试结果
            result = TestResult(
                test_id=test_id,
                control_metrics=control_metrics,
                treatment_metrics=treatment_metrics,
                statistical_significance=statistical_significance,
                p_values=p_values,
                confidence_intervals=confidence_intervals,
                effect_sizes=effect_sizes,
                recommendation=recommendation,
                analysis_time=datetime.now().isoformat()
            )
            
            # 保存结果
            self.test_results[test_id] = result
            self._save_test_results()
            
            self.logger.info(f"测试结果分析完成: {test_id}")
            
        except Exception as e:
            self.logger.error(f"分析测试结果失败: {e}")
    
    def _calculate_metric_stats(self, values: List[float]) -> Dict[str, float]:
        """计算指标统计"""
        if not values:
            return {'mean': 0.0, 'std': 0.0, 'count': 0}
        
        import numpy as np
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'count': len(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    
    def _calculate_p_value(self, control_values: List[float], treatment_values: List[float]) -> float:
        """计算p值"""
        try:
            from scipy import stats
            _, p_value = stats.ttest_ind(control_values, treatment_values)
            return p_value
        except ImportError:
            # 简单的t检验实现
            return self._simple_ttest(control_values, treatment_values)
        except Exception:
            return 1.0
    
    def _simple_ttest(self, control_values: List[float], treatment_values: List[float]) -> float:
        """简单的t检验实现"""
        import math
        
        n1, n2 = len(control_values), len(treatment_values)
        if n1 < 2 or n2 < 2:
            return 1.0
        
        mean1, mean2 = sum(control_values) / n1, sum(treatment_values) / n2
        
        var1 = sum((x - mean1) ** 2 for x in control_values) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in treatment_values) / (n2 - 1)
        
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        se = math.sqrt(pooled_var * (1/n1 + 1/n2))
        
        if se == 0:
            return 1.0
        
        t_stat = (mean1 - mean2) / se
        df = n1 + n2 - 2
        
        # 简化的p值计算
        if abs(t_stat) > 2.0:
            return 0.05
        elif abs(t_stat) > 1.96:
            return 0.1
        else:
            return 0.5
    
    def _calculate_confidence_interval(self, control_values: List[float], 
                                     treatment_values: List[float]) -> Tuple[float, float]:
        """计算置信区间"""
        if not control_values or not treatment_values:
            return (0.0, 0.0)
        
        import math
        
        mean1 = sum(control_values) / len(control_values)
        mean2 = sum(treatment_values) / len(treatment_values)
        
        diff = mean2 - mean1
        
        # 简化的置信区间计算
        se = math.sqrt(sum((x - mean1) ** 2 for x in control_values) / len(control_values) / len(control_values) +
                       sum((x - mean2) ** 2 for x in treatment_values) / len(treatment_values) / len(treatment_values))
        
        margin = 1.96 * se  # 95%置信区间
        
        return (diff - margin, diff + margin)
    
    def _calculate_effect_size(self, control_values: List[float], treatment_values: List[float]) -> float:
        """计算效应大小（Cohen's d）"""
        if not control_values or not treatment_values:
            return 0.0
        
        import math
        
        mean1 = sum(control_values) / len(control_values)
        mean2 = sum(treatment_values) / len(treatment_values)
        
        var1 = sum((x - mean1) ** 2 for x in control_values) / len(control_values)
        var2 = sum((x - mean2) ** 2 for x in treatment_values) / len(treatment_values)
        
        pooled_std = math.sqrt((var1 + var2) / 2)
        
        if pooled_std == 0:
            return 0.0
        
        return (mean2 - mean1) / pooled_std
    
    def _generate_recommendation(self, statistical_significance: Dict[str, bool],
                               effect_sizes: Dict[str, float], metrics: List[str]) -> str:
        """生成推荐"""
        significant_metrics = [m for m, sig in statistical_significance.items() if sig]
        
        if not significant_metrics:
            return "继续收集数据，当前样本量不足以得出统计显著结论"
        
        # 分析效应大小
        large_effects = [m for m, effect in effect_sizes.items() if abs(effect) > 0.8]
        medium_effects = [m for m, effect in effect_sizes.items() if 0.5 < abs(effect) <= 0.8]
        
        if large_effects:
            return f"强烈推荐采用治疗组，在{', '.join(large_effects)}指标上有大效应"
        elif medium_effects:
            return f"推荐采用治疗组，在{', '.join(medium_effects)}指标上有中等效应"
        else:
            return f"可以考虑采用治疗组，在{', '.join(significant_metrics)}指标上有统计显著差异"
    
    def _check_auto_stop(self, test_id: str):
        """检查是否需要自动停止"""
        try:
            test = self.tests[test_id]
            if test.status != "running":
                return
            
            # 检查最小样本量
            collector = self.metrics_collectors[test_id]
            control_data = collector.get_variant_data("control")
            treatment_data = collector.get_variant_data("treatment")
            
            total_samples = sum(len(data) for data in control_data.values()) + \
                          sum(len(data) for data in treatment_data.values())
            
            if total_samples < test.min_sample_size:
                return
            
            # 检查统计显著性
            for metric in test.metrics:
                control_values = [point['value'] for point in control_data.get(metric, [])]
                treatment_values = [point['value'] for point in treatment_data.get(metric, [])]
                
                if len(control_values) > 0 and len(treatment_values) > 0:
                    p_value = self._calculate_p_value(control_values, treatment_values)
                    if p_value < test.significance_level:
                        self.stop_test(test_id, "statistical_significance")
                        return
                        
        except Exception as e:
            self.logger.error(f"检查自动停止失败: {e}")
    
    def _load_tests(self):
        """加载测试数据"""
        try:
            tests_file = Path(self.config['tests_file'])
            if tests_file.exists():
                with open(tests_file, 'r') as f:
                    tests_data = json.load(f)
                
                for test_id, test_data in tests_data.items():
                    self.tests[test_id] = ABTest(**test_data)
                
                self.logger.info(f"加载A/B测试数据: {len(self.tests)} 个测试")
        except Exception as e:
            self.logger.error(f"加载A/B测试数据失败: {e}")
    
    def _save_tests(self):
        """保存测试数据"""
        try:
            tests_file = Path(self.config['tests_file'])
            tests_file.parent.mkdir(parents=True, exist_ok=True)
            
            tests_data = {
                test_id: asdict(test) for test_id, test in self.tests.items()
            }
            
            with open(tests_file, 'w') as f:
                json.dump(tests_data, f, indent=2)
            
            self.logger.info("A/B测试数据已保存")
        except Exception as e:
            self.logger.error(f"保存A/B测试数据失败: {e}")
    
    def _save_test_results(self):
        """保存测试结果"""
        try:
            results_file = Path(self.config['results_file'])
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            results_data = {
                test_id: asdict(result) for test_id, result in self.test_results.items()
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            self.logger.info("A/B测试结果已保存")
        except Exception as e:
            self.logger.error(f"保存A/B测试结果失败: {e}")
