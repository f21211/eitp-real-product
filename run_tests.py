#!/usr/bin/env python3
"""
EIT-P 测试运行脚本
运行所有测试套件并生成测试报告
"""

import unittest
import sys
import os
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_unit_tests():
    """运行单元测试"""
    print("=" * 60)
    print("运行单元测试...")
    print("=" * 60)
    
    # 发现并运行单元测试
    loader = unittest.TestLoader()
    start_dir = project_root / 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful(), result.testsRun, len(result.failures), len(result.errors)

def run_performance_tests():
    """运行性能测试"""
    print("\n" + "=" * 60)
    print("运行性能测试...")
    print("=" * 60)
    
    try:
        from tests.test_performance import TestPerformance
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformance)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful(), result.testsRun, len(result.failures), len(result.errors)
    except ImportError as e:
        print(f"性能测试导入失败: {e}")
        return False, 0, 0, 1

def run_integration_tests():
    """运行集成测试"""
    print("\n" + "=" * 60)
    print("运行集成测试...")
    print("=" * 60)
    
    try:
        from tests.test_integration import TestIntegration
        suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful(), result.testsRun, len(result.failures), len(result.errors)
    except ImportError as e:
        print(f"集成测试导入失败: {e}")
        return False, 0, 0, 1

def generate_test_report(unit_success, unit_tests, unit_failures, unit_errors,
                        perf_success, perf_tests, perf_failures, perf_errors,
                        int_success, int_tests, int_failures, int_errors,
                        total_time):
    """生成测试报告"""
    print("\n" + "=" * 60)
    print("测试报告")
    print("=" * 60)
    
    # 计算总体统计
    total_tests = unit_tests + perf_tests + int_tests
    total_failures = unit_failures + perf_failures + int_failures
    total_errors = unit_errors + perf_errors + int_errors
    total_success = total_tests - total_failures - total_errors
    
    # 打印详细统计
    print(f"单元测试:     {unit_tests:3d} 测试, {unit_failures:2d} 失败, {unit_errors:2d} 错误 - {'通过' if unit_success else '失败'}")
    print(f"性能测试:     {perf_tests:3d} 测试, {perf_failures:2d} 失败, {perf_errors:2d} 错误 - {'通过' if perf_success else '失败'}")
    print(f"集成测试:     {int_tests:3d} 测试, {int_failures:2d} 失败, {int_errors:2d} 错误 - {'通过' if int_success else '失败'}")
    print("-" * 60)
    print(f"总计:         {total_tests:3d} 测试, {total_failures:2d} 失败, {total_errors:2d} 错误")
    print(f"成功率:       {(total_success/total_tests*100):.1f}%")
    print(f"总时间:       {total_time:.2f} 秒")
    
    # 整体结果
    overall_success = unit_success and perf_success and int_success
    print(f"\n整体结果:     {'通过' if overall_success else '失败'}")
    
    return overall_success

def main():
    """主函数"""
    print("EIT-P 测试套件")
    print("=" * 60)
    
    start_time = time.time()
    
    # 运行单元测试
    unit_success, unit_tests, unit_failures, unit_errors = run_unit_tests()
    
    # 运行性能测试
    perf_success, perf_tests, perf_failures, perf_errors = run_performance_tests()
    
    # 运行集成测试
    int_success, int_tests, int_failures, int_errors = run_integration_tests()
    
    # 计算总时间
    total_time = time.time() - start_time
    
    # 生成测试报告
    overall_success = generate_test_report(
        unit_success, unit_tests, unit_failures, unit_errors,
        perf_success, perf_tests, perf_failures, perf_errors,
        int_success, int_tests, int_failures, int_errors,
        total_time
    )
    
    # 返回退出码
    sys.exit(0 if overall_success else 1)

if __name__ == '__main__':
    main()
