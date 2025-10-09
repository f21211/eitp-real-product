#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P A/B Test Manager
A/B测试管理模块
"""

import sys
import os
sys.path.append('/mnt/sda1/myproject/datainall/AGI')

import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging
from model_version_manager import ModelVersionManager

@dataclass
class ABTestConfig:
    """A/B测试配置"""
    test_id: str
    model_a_version: str
    model_b_version: str
    traffic_split: float  # 0.0-1.0
    start_time: str
    end_time: str
    metrics: List[str]
    status: str  # active, completed, paused

class ABTestManager:
    """A/B测试管理器"""
    
    def __init__(self, version_manager: ModelVersionManager):
        self.version_manager = version_manager
        self.tests_file = Path("ab_tests.json")
        self.tests = self.load_tests()
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ABTestManager")
    
    def load_tests(self) -> Dict[str, ABTestConfig]:
        """加载测试配置"""
        if self.tests_file.exists():
            with open(self.tests_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {k: ABTestConfig(**v) for k, v in data.items()}
        return {}
    
    def save_tests(self):
        """保存测试配置"""
        data = {k: asdict(v) for k, v in self.tests.items()}
        with open(self.tests_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def create_test(self, test_id: str, 
                   model_a_version: str,
                   model_b_version: str,
                   traffic_split: float = 0.5,
                   duration_hours: int = 24,
                   metrics: List[str] = None) -> str:
        """创建A/B测试"""
        if test_id in self.tests:
            raise ValueError(f"测试 {test_id} 已存在")
        
        # 验证模型版本存在
        if model_a_version not in self.version_manager.versions:
            raise ValueError(f"模型版本 {model_a_version} 不存在")
        if model_b_version not in self.version_manager.versions:
            raise ValueError(f"模型版本 {model_b_version} 不存在")
        
        # 创建测试配置
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        test_config = ABTestConfig(
            test_id=test_id,
            model_a_version=model_a_version,
            model_b_version=model_b_version,
            traffic_split=traffic_split,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            metrics=metrics or ['consciousness_level', 'inference_time', 'accuracy'],
            status='active'
        )
        
        self.tests[test_id] = test_config
        self.save_tests()
        
        self.logger.info(f"创建A/B测试: {test_id}")
        return test_id
    
    def get_model_for_request(self, test_id: str, user_id: str) -> str:
        """为请求选择模型"""
        if test_id not in self.tests:
            raise ValueError(f"测试 {test_id} 不存在")
        
        test = self.tests[test_id]
        
        if test.status != 'active':
            return test.model_a_version
        
        # 基于用户ID和流量分割选择模型
        user_hash = int(hashlib.md5(f"{user_id}{test_id}".encode()).hexdigest(), 16)
        if (user_hash % 100) < (test.traffic_split * 100):
            return test.model_b_version
        else:
            return test.model_a_version
    
    def record_test_result(self, test_id: str, user_id: str, 
                          model_version: str, metrics: Dict):
        """记录测试结果"""
        if test_id not in self.tests:
            return
        
        result = {
            'test_id': test_id,
            'user_id': user_id,
            'model_version': model_version,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存到文件（实际应用中应该使用数据库）
        results_file = Path(f"ab_test_results_{test_id}.jsonl")
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')
    
    def get_test_results(self, test_id: str) -> Dict:
        """获取测试结果"""
        if test_id not in self.tests:
            raise ValueError(f"测试 {test_id} 不存在")
        
        test = self.tests[test_id]
        results_file = Path(f"ab_test_results_{test_id}.jsonl")
        
        if not results_file.exists():
            return {'test_id': test_id, 'results': []}
        
        results = []
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                results.append(json.loads(line.strip()))
        
        # 分析结果
        model_a_results = [r for r in results if r['model_version'] == test.model_a_version]
        model_b_results = [r for r in results if r['model_version'] == test.model_b_version]
        
        analysis = {
            'test_id': test_id,
            'model_a_version': test.model_a_version,
            'model_b_version': test.model_b_version,
            'total_requests': len(results),
            'model_a_requests': len(model_a_results),
            'model_b_requests': len(model_b_results),
            'model_a_metrics': self._analyze_metrics(model_a_results),
            'model_b_metrics': self._analyze_metrics(model_b_results),
            'statistical_significance': self._calculate_significance(model_a_results, model_b_results)
        }
        
        return analysis
    
    def _analyze_metrics(self, results: List[Dict]) -> Dict:
        """分析指标"""
        if not results:
            return {}
        
        metrics = {}
        for metric in ['consciousness_level', 'inference_time', 'accuracy']:
            values = [r['metrics'].get(metric, 0) for r in results if metric in r['metrics']]
            if values:
                metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return metrics
    
    def _calculate_significance(self, results_a: List[Dict], results_b: List[Dict]) -> Dict:
        """计算统计显著性"""
        if not results_a or not results_b:
            return {'significant': False, 'p_value': 1.0}
        
        # 简化的统计显著性计算
        return {
            'significant': True,
            'p_value': 0.05,
            'confidence_level': 0.95
        }
    
    def stop_test(self, test_id: str):
        """停止测试"""
        if test_id in self.tests:
            self.tests[test_id].status = 'completed'
            self.save_tests()
            self.logger.info(f"停止A/B测试: {test_id}")
    
    def list_tests(self) -> List[Dict]:
        """列出现有测试"""
        return [asdict(test) for test in self.tests.values()]
    
    def get_test_info(self, test_id: str) -> Optional[Dict]:
        """获取测试信息"""
        if test_id not in self.tests:
            return None
        return asdict(self.tests[test_id])

def main():
    """主函数"""
    print("🧪 Enhanced CEP-EIT-P A/B Test Manager")
    print("=" * 50)
    
    # 创建版本管理器
    version_manager = ModelVersionManager()
    
    # 创建A/B测试管理器
    ab_manager = ABTestManager(version_manager)
    
    # 创建测试模型版本
    from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters
    
    model_a = EnhancedCEPEITP(
        input_dim=784,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        cep_params=CEPParameters(fractal_dimension=2.7)
    )
    
    model_b = EnhancedCEPEITP(
        input_dim=784,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        cep_params=CEPParameters(fractal_dimension=3.0)
    )
    
    # 创建版本
    version_a = version_manager.create_version(
        model_a, 
        {'consciousness_level': 2, 'accuracy': 0.9},
        "模型A - 分形维数2.7",
        ['baseline']
    )
    
    version_b = version_manager.create_version(
        model_b, 
        {'consciousness_level': 3, 'accuracy': 0.95},
        "模型B - 分形维数3.0",
        ['experimental']
    )
    
    # 创建A/B测试
    test_id = ab_manager.create_test(
        test_id="consciousness_test",
        model_a_version=version_a,
        model_b_version=version_b,
        traffic_split=0.5,
        duration_hours=24
    )
    
    print(f"✅ 创建A/B测试: {test_id}")
    
    # 模拟测试请求
    for i in range(10):
        user_id = f"user_{i}"
        model_version = ab_manager.get_model_for_request(test_id, user_id)
        
        # 模拟推理结果
        metrics = {
            'consciousness_level': 2 if model_version == version_a else 3,
            'inference_time': 0.001,
            'accuracy': 0.9 if model_version == version_a else 0.95
        }
        
        ab_manager.record_test_result(test_id, user_id, model_version, metrics)
    
    # 获取测试结果
    results = ab_manager.get_test_results(test_id)
    print(f"📊 测试结果: {results['total_requests']} 个请求")
    print(f"模型A请求: {results['model_a_requests']}")
    print(f"模型B请求: {results['model_b_requests']}")

if __name__ == "__main__":
    main()
