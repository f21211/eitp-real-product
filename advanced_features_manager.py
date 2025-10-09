#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P Advanced Features Manager
高级功能管理器 - 整合所有高级功能
"""

import sys
import os
sys.path.append('/mnt/sda1/myproject/datainall/AGI')

import torch
import logging
from datetime import datetime
from typing import Dict, List, Optional
from model_version_manager import ModelVersionManager
from ab_test_manager import ABTestManager
from distributed_training import run_distributed_training

class AdvancedFeaturesManager:
    """高级功能管理器"""
    
    def __init__(self):
        self.version_manager = ModelVersionManager()
        self.ab_test_manager = ABTestManager(self.version_manager)
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AdvancedFeaturesManager")
    
    def run_distributed_training(self, world_size: int = 2, 
                                epochs: int = 100) -> Dict:
        """运行分布式训练"""
        self.logger.info(f"开始分布式训练 (World Size: {world_size})")
        
        # 运行分布式训练
        results = run_distributed_training(world_size, epochs)
        
        # 保存训练结果
        if 'final_loss' in results:
            from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters
            
            model = EnhancedCEPEITP(
                input_dim=784,
                hidden_dims=[512, 256, 128],
                output_dim=10,
                cep_params=CEPParameters()
            )
            
            performance_metrics = {
                'final_loss': results['final_loss'],
                'consciousness_level': results['final_consciousness_level'],
                'training_time': results['training_time']
            }
            
            version = self.version_manager.create_version(
                model, performance_metrics, 
                description=f"分布式训练结果 (World Size: {world_size})",
                tags=['distributed', 'training']
            )
            
            self.logger.info(f"分布式训练完成，模型版本: {version}")
            results['model_version'] = version
        
        return results
    
    def create_ab_test(self, model_a_version: str, model_b_version: str,
                      traffic_split: float = 0.5) -> str:
        """创建A/B测试"""
        test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return self.ab_test_manager.create_test(
            test_id=test_id,
            model_a_version=model_a_version,
            model_b_version=model_b_version,
            traffic_split=traffic_split
        )
    
    def get_model_for_inference(self, test_id: str, user_id: str) -> str:
        """为推理获取模型版本"""
        return self.ab_test_manager.get_model_for_request(test_id, user_id)
    
    def record_inference_result(self, test_id: str, user_id: str, 
                               model_version: str, metrics: Dict):
        """记录推理结果"""
        self.ab_test_manager.record_test_result(test_id, user_id, model_version, metrics)
    
    def get_test_analysis(self, test_id: str) -> Dict:
        """获取测试分析"""
        return self.ab_test_manager.get_test_results(test_id)
    
    def list_model_versions(self) -> List[Dict]:
        """列出现有模型版本"""
        return self.version_manager.list_versions()
    
    def get_latest_model_version(self) -> Optional[str]:
        """获取最新模型版本"""
        return self.version_manager.get_latest_version()
    
    def load_model_version(self, version: str):
        """加载指定版本的模型"""
        return self.version_manager.load_model(version)
    
    def list_ab_tests(self) -> List[Dict]:
        """列出现有A/B测试"""
        return self.ab_test_manager.list_tests()
    
    def stop_ab_test(self, test_id: str):
        """停止A/B测试"""
        self.ab_test_manager.stop_test(test_id)
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            'model_versions': len(self.version_manager.versions),
            'active_tests': len([t for t in self.ab_test_manager.tests.values() 
                                if t.status == 'active']),
            'total_tests': len(self.ab_test_manager.tests),
            'latest_version': self.version_manager.get_latest_version(),
            'timestamp': datetime.now().isoformat()
        }
    
    def create_model_comparison(self, versions: List[str]) -> Dict:
        """创建模型对比分析"""
        comparison = {
            'versions': versions,
            'comparison_data': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for version in versions:
            try:
                model, metrics = self.version_manager.load_model(version)
                comparison['comparison_data'][version] = {
                    'performance_metrics': metrics,
                    'cep_params': self.version_manager.get_version_info(version)['cep_params']
                }
            except Exception as e:
                self.logger.error(f"加载版本 {version} 失败: {e}")
                comparison['comparison_data'][version] = {'error': str(e)}
        
        return comparison
    
    def export_model_version(self, version: str, export_path: str) -> bool:
        """导出模型版本"""
        try:
            model, metrics = self.version_manager.load_model(version)
            
            export_data = {
                'version': version,
                'model_state_dict': model.state_dict(),
                'cep_params': model.cep_params.__dict__,
                'performance_metrics': metrics,
                'architecture': {
                    'input_dim': model.input_dim,
                    'hidden_dims': model.hidden_dims,
                    'output_dim': model.output_dim
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            torch.save(export_data, export_path)
            self.logger.info(f"模型版本 {version} 已导出到 {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出模型版本 {version} 失败: {e}")
            return False
    
    def import_model_version(self, import_path: str, description: str = "") -> Optional[str]:
        """导入模型版本"""
        try:
            import_data = torch.load(import_path, map_location='cpu')
            
            from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters
            
            # 创建模型
            cep_params = CEPParameters(**import_data['cep_params'])
            model = EnhancedCEPEITP(
                input_dim=import_data['architecture']['input_dim'],
                hidden_dims=import_data['architecture']['hidden_dims'],
                output_dim=import_data['architecture']['output_dim'],
                cep_params=cep_params
            )
            
            # 加载权重
            model.load_state_dict(import_data['model_state_dict'])
            
            # 创建版本
            version = self.version_manager.create_version(
                model,
                import_data['performance_metrics'],
                description=description or f"导入的模型版本",
                tags=['imported']
            )
            
            self.logger.info(f"模型版本已导入: {version}")
            return version
            
        except Exception as e:
            self.logger.error(f"导入模型版本失败: {e}")
            return None

def main():
    """主函数"""
    print("🚀 Enhanced CEP-EIT-P Advanced Features Manager")
    print("=" * 60)
    
    # 创建高级功能管理器
    manager = AdvancedFeaturesManager()
    
    # 显示系统状态
    status = manager.get_system_status()
    print(f"📊 系统状态:")
    print(f"  模型版本: {status['model_versions']}")
    print(f"  活跃测试: {status['active_tests']}")
    print(f"  总测试数: {status['total_tests']}")
    print(f"  最新版本: {status['latest_version']}")
    
    # 创建测试模型版本
    from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters
    
    model = EnhancedCEPEITP(
        input_dim=784,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        cep_params=CEPParameters()
    )
    
    # 创建版本
    version = manager.version_manager.create_version(
        model,
        {'consciousness_level': 2, 'accuracy': 0.9, 'inference_time': 0.001},
        "测试模型版本",
        ['test', 'demo']
    )
    
    print(f"✅ 创建模型版本: {version}")
    
    # 创建A/B测试
    test_id = manager.create_ab_test(version, version, 0.5)
    print(f"✅ 创建A/B测试: {test_id}")
    
    # 模拟测试
    for i in range(5):
        user_id = f"user_{i}"
        model_version = manager.get_model_for_inference(test_id, user_id)
        metrics = {'consciousness_level': 2, 'inference_time': 0.001, 'accuracy': 0.9}
        manager.record_inference_result(test_id, user_id, model_version, metrics)
    
    # 获取测试分析
    analysis = manager.get_test_analysis(test_id)
    print(f"📊 测试分析: {analysis['total_requests']} 个请求")
    
    print("🎉 高级功能管理器测试完成!")

if __name__ == "__main__":
    main()
