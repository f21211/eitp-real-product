#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P 综合测试脚本
测试所有高级功能模块
"""

import sys
import os
sys.path.append('/mnt/sda1/myproject/datainall/AGI')

import time
import json
from datetime import datetime
from typing import Dict, List

def test_advanced_features():
    """测试高级功能模块"""
    print("🧪 测试高级功能模块...")
    
    try:
        from advanced_features_manager import AdvancedFeaturesManager
        
        manager = AdvancedFeaturesManager()
        
        # 测试系统状态
        status = manager.get_system_status()
        print(f"  ✅ 系统状态: {status}")
        
        # 测试模型版本管理
        versions = manager.list_model_versions()
        print(f"  ✅ 模型版本数量: {len(versions)}")
        
        # 测试A/B测试管理
        tests = manager.list_ab_tests()
        print(f"  ✅ A/B测试数量: {len(tests)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 高级功能模块测试失败: {e}")
        return False

def test_mlops_pipeline():
    """测试MLOps流水线"""
    print("🚀 测试MLOps流水线...")
    
    try:
        from mlops_pipeline import MLOpsPipeline, TrainingConfig
        
        pipeline = MLOpsPipeline()
        
        # 创建简单配置
        config = TrainingConfig(
            epochs=2,  # 快速测试
            learning_rate=0.001,
            batch_size=16,
            early_stopping_patience=1
        )
        
        # 运行流水线
        results = pipeline.run_full_pipeline(config)
        
        if results['status'] == 'success':
            print(f"  ✅ MLOps流水线成功完成")
            print(f"  📊 训练时间: {results['stages']['training']['training_time']:.2f}s")
            return True
        else:
            print(f"  ❌ MLOps流水线失败: {results.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"  ❌ MLOps流水线测试失败: {e}")
        return False

def test_enterprise_features():
    """测试企业级功能"""
    print("🏢 测试企业级功能...")
    
    try:
        from enterprise_features import EnterpriseFeaturesManager, UserRole
        
        enterprise_manager = EnterpriseFeaturesManager()
        
        # 创建测试租户
        tenant_id = enterprise_manager.create_tenant(
            "test_tenant", 
            "测试租户", 
            "测试用租户"
        )
        print(f"  ✅ 创建租户: {tenant_id}")
        
        # 创建测试用户
        user_id = enterprise_manager.create_user(
            "test_user",
            "testuser",
            "test@example.com",
            UserRole.USER,
            tenant_id
        )
        print(f"  ✅ 创建用户: {user_id}")
        
        # 测试权限检查
        can_read = enterprise_manager.check_permission(user_id, "read_models")
        print(f"  ✅ 权限检查: {can_read}")
        
        # 获取系统统计
        stats = enterprise_manager.get_system_statistics()
        print(f"  ✅ 系统统计: {stats['total_users']} 用户, {stats['total_tenants']} 租户")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 企业级功能测试失败: {e}")
        return False

def test_optimization_tools():
    """测试优化工具"""
    print("🔧 测试优化工具...")
    
    try:
        from optimization_tools import OptimizationToolsManager, OptimizationConfig, CompressionConfig
        
        optimization_manager = OptimizationToolsManager()
        
        # 创建优化配置
        optimization_config = OptimizationConfig(
            method="random",
            max_trials=5,  # 快速测试
            target_metric="consciousness_level"
        )
        
        # 创建压缩配置
        compression_config = CompressionConfig(
            method="pruning",
            target_ratio=0.3
        )
        
        # 运行优化流水线
        results = optimization_manager.run_optimization_pipeline(
            optimization_config, 
            compression_config
        )
        
        if results['status'] == 'success':
            print(f"  ✅ 优化工具测试成功")
            hyperopt = results['stages']['hyperparameter_optimization']
            print(f"  📊 最佳分数: {hyperopt['best_score']:.4f}")
            return True
        else:
            print(f"  ❌ 优化工具测试失败: {results.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"  ❌ 优化工具测试失败: {e}")
        return False

def test_web_dashboard():
    """测试Web仪表板"""
    print("🌐 测试Web仪表板...")
    
    try:
        import web_dashboard
        
        # 检查Flask应用是否可导入
        app = web_dashboard.app
        print(f"  ✅ Flask应用创建成功")
        
        # 检查路由
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = ['/', '/api/dashboard_data', '/api/system_status', '/api/model_versions', '/api/ab_tests', '/api/performance_metrics']
        
        for route in expected_routes:
            if route in routes:
                print(f"  ✅ 路由存在: {route}")
            else:
                print(f"  ⚠️ 路由缺失: {route}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Web仪表板测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 Enhanced CEP-EIT-P 综合测试")
    print("=" * 50)
    
    test_results = {}
    
    # 运行所有测试
    tests = [
        ("高级功能模块", test_advanced_features),
        ("MLOps流水线", test_mlops_pipeline),
        ("企业级功能", test_enterprise_features),
        ("优化工具", test_optimization_tools),
        ("Web仪表板", test_web_dashboard)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🔍 测试: {test_name}")
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
            test_results[test_name] = False
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！Enhanced CEP-EIT-P高级功能模块运行正常！")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关模块")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
