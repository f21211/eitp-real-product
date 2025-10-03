"""
性能基准测试
测试EIT-P各组件的性能表现
"""

import unittest
import time
import torch
import numpy as np
from eit_p.utils.config_manager import ConfigManager
from eit_p.utils.exceptions import check_gpu_memory, check_convergence


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self):
        self.results = {}
    
    def time_function(self, func, *args, **kwargs):
        """测量函数执行时间"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        return result, execution_time
    
    def memory_usage(self, func, *args, **kwargs):
        """测量函数内存使用"""
        if not torch.cuda.is_available():
            return None, None
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 记录初始内存
        initial_memory = torch.cuda.memory_allocated()
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 记录峰值内存
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        
        # 计算内存使用
        memory_used = (peak_memory - initial_memory) / 1024**3  # GB
        
        return result, memory_used
    
    def benchmark_config_loading(self):
        """基准测试配置加载性能"""
        def load_config():
            return ConfigManager()
        
        result, time_taken = self.time_function(load_config)
        self.results['config_loading'] = {
            'time': time_taken,
            'success': result is not None
        }
        
        return time_taken
    
    def benchmark_tensor_operations(self, size=(1000, 1000)):
        """基准测试张量操作性能"""
        def tensor_ops():
            # 创建张量
            a = torch.randn(size, device='cuda' if torch.cuda.is_available() else 'cpu')
            b = torch.randn(size, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # 矩阵乘法
            c = torch.matmul(a, b)
            
            # 其他操作
            d = torch.sum(c, dim=1)
            e = torch.mean(d)
            
            return e.item()
        
        result, time_taken = self.time_function(tensor_ops)
        result, memory_used = self.memory_usage(tensor_ops)
        
        self.results['tensor_operations'] = {
            'time': time_taken,
            'memory_gb': memory_used,
            'result': result
        }
        
        return time_taken, memory_used
    
    def benchmark_memory_checks(self, iterations=1000):
        """基准测试内存检查性能"""
        def memory_checks():
            for _ in range(iterations):
                check_gpu_memory()
                check_convergence(500.0, 1000.0)
        
        result, time_taken = self.time_function(memory_checks)
        
        self.results['memory_checks'] = {
            'time': time_taken,
            'iterations': iterations,
            'time_per_check': time_taken / iterations
        }
        
        return time_taken
    
    def benchmark_config_operations(self, iterations=1000):
        """基准测试配置操作性能"""
        config = ConfigManager()
        
        def config_ops():
            for i in range(iterations):
                # 设置配置
                config.set(f'test.key_{i}', f'value_{i}')
                
                # 获取配置
                value = config.get(f'test.key_{i}')
                
                # 验证
                assert value == f'value_{i}'
        
        result, time_taken = self.time_function(config_ops)
        
        self.results['config_operations'] = {
            'time': time_taken,
            'iterations': iterations,
            'time_per_operation': time_taken / iterations
        }
        
        return time_taken
    
    def run_all_benchmarks(self):
        """运行所有基准测试"""
        print("开始性能基准测试...")
        
        # 配置加载测试
        print("测试配置加载性能...")
        self.benchmark_config_loading()
        
        # 张量操作测试
        print("测试张量操作性能...")
        self.benchmark_tensor_operations()
        
        # 内存检查测试
        print("测试内存检查性能...")
        self.benchmark_memory_checks()
        
        # 配置操作测试
        print("测试配置操作性能...")
        self.benchmark_config_operations()
        
        print("性能基准测试完成!")
        return self.results
    
    def print_results(self):
        """打印基准测试结果"""
        print("\n=== 性能基准测试结果 ===")
        
        for test_name, results in self.results.items():
            print(f"\n{test_name}:")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    def get_performance_summary(self):
        """获取性能摘要"""
        summary = {
            'total_tests': len(self.results),
            'successful_tests': sum(1 for r in self.results.values() if r.get('success', True)),
            'average_time': np.mean([r.get('time', 0) for r in self.results.values()]),
            'max_memory_usage': max([r.get('memory_gb', 0) for r in self.results.values()])
        }
        return summary


class TestPerformance(unittest.TestCase):
    """性能测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.benchmark = PerformanceBenchmark()
    
    def test_config_loading_performance(self):
        """测试配置加载性能"""
        time_taken = self.benchmark.benchmark_config_loading()
        
        # 配置加载应该在合理时间内完成
        self.assertLess(time_taken, 1.0, "配置加载时间过长")
        
        # 验证结果
        self.assertIn('config_loading', self.benchmark.results)
        self.assertTrue(self.benchmark.results['config_loading']['success'])
    
    def test_tensor_operations_performance(self):
        """测试张量操作性能"""
        time_taken, memory_used = self.benchmark.benchmark_tensor_operations()
        
        # 张量操作应该在合理时间内完成
        self.assertLess(time_taken, 5.0, "张量操作时间过长")
        
        # 如果有GPU，检查内存使用
        if torch.cuda.is_available() and memory_used is not None:
            self.assertLess(memory_used, 2.0, "张量操作内存使用过多")
    
    def test_memory_checks_performance(self):
        """测试内存检查性能"""
        time_taken = self.benchmark.benchmark_memory_checks()
        
        # 内存检查应该很快
        self.assertLess(time_taken, 1.0, "内存检查时间过长")
        
        # 验证每次检查的时间
        time_per_check = self.benchmark.results['memory_checks']['time_per_check']
        self.assertLess(time_per_check, 0.001, "单次内存检查时间过长")
    
    def test_config_operations_performance(self):
        """测试配置操作性能"""
        time_taken = self.benchmark.benchmark_config_operations()
        
        # 配置操作应该很快
        self.assertLess(time_taken, 2.0, "配置操作时间过长")
        
        # 验证每次操作的时间
        time_per_operation = self.benchmark.results['config_operations']['time_per_operation']
        self.assertLess(time_per_operation, 0.001, "单次配置操作时间过长")
    
    def test_overall_performance(self):
        """测试整体性能"""
        # 运行所有基准测试
        results = self.benchmark.run_all_benchmarks()
        
        # 验证所有测试都成功
        self.assertEqual(len(results), 4, "应该运行4个基准测试")
        
        # 获取性能摘要
        summary = self.benchmark.get_performance_summary()
        
        # 验证性能摘要
        self.assertEqual(summary['total_tests'], 4)
        self.assertEqual(summary['successful_tests'], 4)
        self.assertGreater(summary['average_time'], 0)
        
        # 打印结果
        self.benchmark.print_results()
        print(f"\n性能摘要: {summary}")


if __name__ == '__main__':
    # 运行性能测试
    unittest.main(verbosity=2)
