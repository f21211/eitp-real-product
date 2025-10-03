#!/usr/bin/env python3
"""
EIT-P 简化演示脚本
展示核心功能，避免复杂的依赖问题
"""

import os
import sys
import logging
import time
import json
from datetime import datetime

# 设置环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """打印横幅"""
    print("=" * 80)
    print("🚀 EIT-P (Emergent Intelligence Theory - PyTorch) 简化演示")
    print("=" * 80)
    print("基于涌现智能理论的深度学习框架")
    print("集成企业级功能：配置管理、错误处理、日志系统、实验管理")
    print("=" * 80)

def demo_config_manager():
    """演示配置管理"""
    print("\n📋 1. 配置管理演示")
    print("-" * 40)
    
    try:
        from eit_p.utils import ConfigManager
        
        # 创建配置管理器
        config_manager = ConfigManager()
        
        # 加载配置
        if os.path.exists('config.yaml'):
            config_manager.load_config('config.yaml')
            print("✅ 成功加载配置文件")
        else:
            print("⚠️  配置文件不存在，使用默认配置")
        
        # 显示配置信息
        print(f"训练批次大小: {config_manager.get('training.batch_size', '未设置')}")
        print(f"学习率: {config_manager.get('training.learning_rate', '未设置')}")
        print(f"模型隐藏维度: {config_manager.get('model.hidden_dim', '未设置')}")
        
        # 更新配置
        config_manager.set('demo.timestamp', datetime.now().isoformat())
        print("✅ 配置更新完成")
        
    except Exception as e:
        print(f"❌ 配置管理演示失败: {e}")

def demo_error_handling():
    """演示错误处理"""
    print("\n🛡️  2. 错误处理演示")
    print("-" * 40)
    
    try:
        from eit_p.utils import ErrorHandler, EITPException, MemoryOverflowError
        
        # 创建错误处理器
        error_handler = ErrorHandler()
        
        # 模拟不同类型的错误
        print("测试自定义异常...")
        try:
            raise EITPException("这是一个自定义异常")
        except EITPException as e:
            error_handler.handle_error(e)
            print("✅ 自定义异常处理成功")
        
        print("测试内存溢出异常...")
        try:
            raise MemoryOverflowError("模拟内存溢出")
        except MemoryOverflowError as e:
            error_handler.handle_error(e)
            print("✅ 内存溢出异常处理成功")
        
        print("测试通用异常...")
        try:
            raise ValueError("这是一个通用异常")
        except Exception as e:
            error_handler.handle_error(e)
            print("✅ 通用异常处理成功")
        
    except Exception as e:
        print(f"❌ 错误处理演示失败: {e}")

def demo_logging_system():
    """演示日志系统"""
    print("\n📝 3. 日志系统演示")
    print("-" * 40)
    
    try:
        from eit_p.utils import get_global_logger, EITPLogger
        
        # 获取全局日志器
        logger = get_global_logger()
        
        # 记录不同级别的日志
        logger.info("这是一条信息日志")
        logger.warning("这是一条警告日志")
        logger.error("这是一条错误日志")
        
        # 创建专门的日志器
        demo_logger = EITPLogger("demo_logger", log_file="demo.log")
        demo_logger.info("演示日志记录")
        demo_logger.warning("演示警告")
        
        print("✅ 日志系统演示完成")
        print("📁 日志文件已保存到: demo.log")
        
    except Exception as e:
        print(f"❌ 日志系统演示失败: {e}")

def demo_experiment_management():
    """演示实验管理"""
    print("\n🧪 4. 实验管理演示")
    print("-" * 40)
    
    try:
        from eit_p.experiments import ExperimentManager, ModelRegistry, MetricsTracker
        
        # 创建实验管理器
        exp_manager = ExperimentManager()
        
        # 创建新实验
        experiment = exp_manager.create_experiment(
            name="EIT-P演示实验",
            description="展示EIT-P框架的实验管理功能",
            config={"learning_rate": 0.001, "batch_size": 32}
        )
        print(f"✅ 创建实验: {experiment['experiment_id']}")
        
        # 创建模型注册表
        model_registry = ModelRegistry()
        
        # 注册模型版本
        model_version = model_registry.register_model(
            experiment_id=experiment['experiment_id'],
            model_path="demo_model.pth",
            metrics={"accuracy": 0.95, "loss": 0.05},
            metadata={"epochs": 10, "optimizer": "Adam"}
        )
        print(f"✅ 注册模型版本: {model_version['version_id']}")
        
        # 创建指标跟踪器
        metrics_tracker = MetricsTracker(experiment['experiment_id'])
        
        # 记录指标
        for epoch in range(1, 6):
            metrics_tracker.log_metric("loss", 1.0 / epoch, epoch)
            metrics_tracker.log_metric("accuracy", 0.8 + epoch * 0.04, epoch)
        
        print("✅ 指标跟踪完成")
        
        # 生成报告
        report = metrics_tracker.generate_report()
        print("✅ 实验报告生成完成")
        
    except Exception as e:
        print(f"❌ 实验管理演示失败: {e}")

def demo_ab_testing():
    """演示A/B测试"""
    print("\n🔬 5. A/B测试演示")
    print("-" * 40)
    
    try:
        from eit_p.ab_testing import ABTestManager, MetricsCollector, TrafficSplitter
        
        # 创建A/B测试管理器
        ab_manager = ABTestManager()
        
        # 创建流量分割器
        traffic_splitter = TrafficSplitter()
        
        # 创建变体
        from eit_p.ab_testing.traffic_splitter import Variant
        variants = [
            Variant(name="control", weight=0.5, description="控制组"),
            Variant(name="treatment", weight=0.5, description="实验组")
        ]
        
        # 创建实验
        experiment = traffic_splitter.create_experiment(
            experiment_id="demo_ab_test",
            name="演示A/B测试",
            description="展示A/B测试功能",
            variants=variants
        )
        print(f"✅ 创建A/B测试实验: {experiment.experiment_id}")
        
        # 模拟用户分配
        users = [f"user_{i}" for i in range(10)]
        assignments = {}
        
        for user in users:
            variant = traffic_splitter.get_variant_for_user(user, experiment.experiment_id)
            assignments[user] = variant
            print(f"用户 {user} 分配到变体: {variant}")
        
        # 创建指标收集器
        metrics_collector = MetricsCollector()
        
        # 记录指标
        for user, variant in assignments.items():
            # 模拟用户行为指标
            conversion_rate = 0.3 if variant == "control" else 0.4
            metrics_collector.record_metric(
                "conversion_rate", conversion_rate, 
                experiment.experiment_id, variant, user_id=user
            )
        
        print("✅ A/B测试指标收集完成")
        
        # 获取实验统计
        stats = traffic_splitter.get_experiment_stats(experiment.experiment_id)
        print(f"✅ 实验统计: {stats['total_users']} 用户参与")
        
    except Exception as e:
        print(f"❌ A/B测试演示失败: {e}")

def demo_security():
    """演示安全功能"""
    print("\n🔒 6. 安全功能演示")
    print("-" * 40)
    
    try:
        from eit_p.security import SecurityAuditor, AuthenticationManager
        
        # 创建安全审计器
        auditor = SecurityAuditor()
        
        # 记录安全事件
        event_id = auditor.log_event(
            event_type="authentication",
            user_id="demo_user",
            resource="model_access",
            action="login",
            result="success",
            details={"ip": "192.168.1.1", "user_agent": "Demo Browser"}
        )
        print(f"✅ 记录安全事件: {event_id}")
        
        # 创建认证管理器
        auth_manager = AuthenticationManager()
        
        # 注册用户
        user_id = auth_manager.register_user("demo_user", "demo_password")
        print(f"✅ 注册用户: {user_id}")
        
        # 验证用户
        is_valid = auth_manager.verify_user("demo_user", "demo_password")
        print(f"✅ 用户验证: {'成功' if is_valid else '失败'}")
        
        # 生成安全报告
        report = auditor.generate_report()
        print(f"✅ 安全报告生成完成，共 {report['summary']['total_events']} 个事件")
        
    except Exception as e:
        print(f"❌ 安全功能演示失败: {e}")

def demo_compression():
    """演示模型压缩"""
    print("\n🗜️  7. 模型压缩演示")
    print("-" * 40)
    
    try:
        from eit_p.compression import QuantizationManager, PruningManager
        
        # 创建量化管理器
        quant_manager = QuantizationManager()
        print("✅ 量化管理器创建成功")
        
        # 创建剪枝管理器
        prune_manager = PruningManager()
        print("✅ 剪枝管理器创建成功")
        
        # 模拟压缩配置
        compression_config = {
            "quantization": {
                "method": "dynamic",
                "bits": 8
            },
            "pruning": {
                "method": "magnitude",
                "sparsity": 0.5
            }
        }
        
        print(f"✅ 压缩配置: {json.dumps(compression_config, indent=2)}")
        
    except Exception as e:
        print(f"❌ 模型压缩演示失败: {e}")

def demo_optimization():
    """演示超参数优化"""
    print("\n⚡ 8. 超参数优化演示")
    print("-" * 40)
    
    try:
        from eit_p.optimization import BayesianOptimizer, GridSearchOptimizer
        
        # 创建贝叶斯优化器
        bayesian_opt = BayesianOptimizer()
        print("✅ 贝叶斯优化器创建成功")
        
        # 创建网格搜索优化器
        grid_opt = GridSearchOptimizer()
        print("✅ 网格搜索优化器创建成功")
        
        # 定义搜索空间
        search_space = {
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": [16, 32, 64],
            "hidden_dim": [64, 128, 256]
        }
        
        print(f"✅ 搜索空间定义: {len(search_space)} 个参数")
        
    except Exception as e:
        print(f"❌ 超参数优化演示失败: {e}")

def main():
    """主函数"""
    print_banner()
    
    print("\n🎯 开始EIT-P框架功能演示...")
    print("注意：这是一个简化版本，专注于核心功能展示")
    
    # 运行各个演示
    demo_config_manager()
    demo_error_handling()
    demo_logging_system()
    demo_experiment_management()
    demo_ab_testing()
    demo_security()
    demo_compression()
    demo_optimization()
    
    print("\n" + "=" * 80)
    print("🎉 EIT-P 简化演示完成！")
    print("=" * 80)
    print("✨ 主要功能展示：")
    print("  • 配置管理 - 集中化配置管理")
    print("  • 错误处理 - 企业级异常处理")
    print("  • 日志系统 - 结构化日志记录")
    print("  • 实验管理 - 完整的实验生命周期")
    print("  • A/B测试 - 流量分割和指标收集")
    print("  • 安全功能 - 认证和审计")
    print("  • 模型压缩 - 量化和剪枝")
    print("  • 超参数优化 - 自动调优")
    print("=" * 80)
    print("🚀 EIT-P框架已准备就绪，可以开始生产级AI模型训练！")

if __name__ == "__main__":
    main()
