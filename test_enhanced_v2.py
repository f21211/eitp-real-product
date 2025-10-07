#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P API Server V2 测试脚本
测试所有API端点的功能
"""

import requests
import json
import time
import numpy as np
from datetime import datetime

class EnhancedAPIV2Tester:
    """增强版API V2测试器"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
    
    def test_endpoint(self, method, endpoint, data=None, expected_status=200):
        """测试单个端点"""
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == 'GET':
                response = requests.get(url, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, json=data, timeout=10)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            success = response.status_code == expected_status
            result = {
                'endpoint': endpoint,
                'method': method,
                'status_code': response.status_code,
                'expected_status': expected_status,
                'success': success,
                'response_time': response.elapsed.total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
            if success:
                try:
                    result['data'] = response.json()
                except:
                    result['data'] = response.text
            else:
                result['error'] = response.text
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            result = {
                'endpoint': endpoint,
                'method': method,
                'status_code': 0,
                'expected_status': expected_status,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.test_results.append(result)
            return result
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 Enhanced CEP-EIT-P API Server V2 测试套件")
        print("=" * 60)
        
        # 1. 健康检查
        print("\n1. 测试健康检查...")
        result = self.test_endpoint('GET', '/api/health')
        if result['success']:
            print(f"   ✅ 健康检查: {result['response_time']:.3f}s")
        else:
            print(f"   ❌ 健康检查失败: {result.get('error', 'Unknown error')}")
        
        # 2. 模型信息
        print("\n2. 测试模型信息...")
        result = self.test_endpoint('GET', '/api/model_info')
        if result['success']:
            model_info = result['data']
            print(f"   ✅ 模型信息: {model_info['model_name']}")
            print(f"   📊 参数数量: {model_info['total_parameters']:,}")
        else:
            print(f"   ❌ 模型信息失败: {result.get('error', 'Unknown error')}")
        
        # 3. 推理服务
        print("\n3. 测试推理服务...")
        test_input = np.random.randn(784).tolist()
        result = self.test_endpoint('POST', '/api/inference', {'input': test_input})
        if result['success']:
            data = result['data']
            print(f"   ✅ 推理成功: {data['inference_time']:.3f}s")
            print(f"   🧠 意识水平: {data['consciousness_metrics']['level']}/4")
            print(f"   📐 分形维数: {data['consciousness_metrics']['fractal_dimension']:.3f}")
        else:
            print(f"   ❌ 推理失败: {result.get('error', 'Unknown error')}")
        
        # 4. 批量推理
        print("\n4. 测试批量推理...")
        batch_inputs = [np.random.randn(784).tolist() for _ in range(5)]
        result = self.test_endpoint('POST', '/api/batch_inference', {'inputs': batch_inputs})
        if result['success']:
            data = result['data']
            print(f"   ✅ 批量推理成功: {data['inference_time']:.3f}s")
            print(f"   📦 批次大小: {data['batch_size']}")
            print(f"   🧠 平均意识水平: {data['consciousness_metrics']['avg_level']:.2f}")
        else:
            print(f"   ❌ 批量推理失败: {result.get('error', 'Unknown error')}")
        
        # 5. 意识分析
        print("\n5. 测试意识分析...")
        result = self.test_endpoint('GET', '/api/consciousness')
        if result['success']:
            data = result['data']
            analysis = data['analysis']
            print(f"   ✅ 意识分析成功")
            print(f"   📊 平均意识水平: {analysis['avg_consciousness_level']:.2f}")
            print(f"   📈 样本数量: {analysis['samples_count']}")
        else:
            print(f"   ❌ 意识分析失败: {result.get('error', 'Unknown error')}")
        
        # 6. 能量分析
        print("\n6. 测试能量分析...")
        test_input = np.random.randn(784).tolist()
        result = self.test_endpoint('POST', '/api/energy_analysis', {'input': test_input})
        if result['success']:
            data = result['data']
            energy = data['energy_analysis']
            print(f"   ✅ 能量分析成功")
            print(f"   ⚡ 总能量: {energy['cep_energies']['total_energy']:.6f}")
            print(f"   🔋 效率: {energy['efficiency']:.6f}")
        else:
            print(f"   ❌ 能量分析失败: {result.get('error', 'Unknown error')}")
        
        # 7. 性能指标
        print("\n7. 测试性能指标...")
        result = self.test_endpoint('GET', '/api/performance')
        if result['success']:
            data = result['data']
            perf = data['performance']
            print(f"   ✅ 性能指标成功")
            print(f"   📊 总请求数: {perf['total_requests']}")
            print(f"   ⚡ 平均推理时间: {perf['avg_inference_time']:.3f}s")
            print(f"   🚀 请求速率: {perf['requests_per_second']:.2f} req/s")
        else:
            print(f"   ❌ 性能指标失败: {result.get('error', 'Unknown error')}")
        
        # 8. 模型优化
        print("\n8. 测试模型优化...")
        result = self.test_endpoint('POST', '/api/optimize', {'epochs': 5})
        if result['success']:
            data = result['data']
            print(f"   ✅ 模型优化成功: {data['optimization_time']:.3f}s")
            print(f"   🔧 更新参数: {data['updated_params']}")
        else:
            print(f"   ❌ 模型优化失败: {result.get('error', 'Unknown error')}")
        
        # 9. 历史数据查询
        print("\n9. 测试历史数据查询...")
        result = self.test_endpoint('GET', '/api/history?type=consciousness&limit=10')
        if result['success']:
            data = result['data']
            print(f"   ✅ 历史数据查询成功")
            print(f"   📊 数据类型: {data['data_type']}")
            print(f"   📈 数据数量: {data['count']}")
        else:
            print(f"   ❌ 历史数据查询失败: {result.get('error', 'Unknown error')}")
        
        # 10. 统计分析
        print("\n10. 测试统计分析...")
        result = self.test_endpoint('GET', '/api/statistics')
        if result['success']:
            data = result['data']
            stats = data['statistics']
            print(f"   ✅ 统计分析成功")
            print(f"   📊 总样本数: {stats['total_samples']}")
            print(f"   🧠 意识水平统计: 均值={stats['consciousness_level']['mean']:.2f}")
        else:
            print(f"   ❌ 统计分析失败: {result.get('error', 'Unknown error')}")
        
        # 11. 重置指标
        print("\n11. 测试重置指标...")
        result = self.test_endpoint('POST', '/api/reset_metrics')
        if result['success']:
            data = result['data']
            print(f"   ✅ 重置指标成功: {data['message']}")
        else:
            print(f"   ❌ 重置指标失败: {result.get('error', 'Unknown error')}")
        
        # 12. 欢迎页面
        print("\n12. 测试欢迎页面...")
        result = self.test_endpoint('GET', '/')
        if result['success']:
            data = result['data']
            print(f"   ✅ 欢迎页面成功")
            print(f"   🎉 消息: {data['message']}")
            print(f"   📋 功能数量: {len(data['features'])}")
        else:
            print(f"   ❌ 欢迎页面失败: {result.get('error', 'Unknown error')}")
        
        # 生成测试报告
        self.generate_report()
    
    def generate_report(self):
        """生成测试报告"""
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r['success']])
        failed_tests = total_tests - successful_tests
        
        print("\n" + "=" * 60)
        print("📊 测试报告")
        print("=" * 60)
        print(f"总测试数: {total_tests}")
        print(f"成功: {successful_tests} ✅")
        print(f"失败: {failed_tests} ❌")
        print(f"成功率: {successful_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ 失败的测试:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - {result['method']} {result['endpoint']}: {result.get('error', 'Unknown error')}")
        
        # 计算平均响应时间
        response_times = [r['response_time'] for r in self.test_results if 'response_time' in r]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            print(f"\n⏱️ 平均响应时间: {avg_response_time:.3f}s")
        
        # 保存详细报告
        report_data = {
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': successful_tests/total_tests*100,
                'avg_response_time': avg_response_time if response_times else 0
            },
            'test_results': self.test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('enhanced_api_v2_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 详细报告已保存: enhanced_api_v2_test_report.json")

def main():
    """主函数"""
    print("🚀 开始测试 Enhanced CEP-EIT-P API Server V2...")
    
    # 等待服务启动
    print("⏳ 等待服务启动...")
    time.sleep(2)
    
    # 创建测试器
    tester = EnhancedAPIV2Tester()
    
    # 运行测试
    tester.run_all_tests()
    
    print("\n🎉 测试完成!")

if __name__ == "__main__":
    main()
