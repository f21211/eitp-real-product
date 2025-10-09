#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P Optimization Tools
优化工具：超参数调优、模型压缩、量化
"""

import sys
import os
sys.path.append('/mnt/sda1/myproject/datainall/AGI')

import torch
import numpy as np
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters
from advanced_features_manager import AdvancedFeaturesManager

@dataclass
class OptimizationConfig:
    """优化配置"""
    method: str = "bayesian"  # bayesian, grid, random
    max_trials: int = 100
    timeout: int = 3600  # seconds
    target_metric: str = "consciousness_level"
    target_value: float = 3.0

@dataclass
class CompressionConfig:
    """压缩配置"""
    method: str = "pruning"  # pruning, quantization, distillation
    target_ratio: float = 0.5  # 压缩比例
    preserve_accuracy: bool = True

class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self):
        self.setup_logging()
        self.trial_history = []
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("HyperparameterOptimizer")
    
    def bayesian_optimization(self, config: OptimizationConfig) -> Dict:
        """贝叶斯优化"""
        self.logger.info("开始贝叶斯优化")
        
        best_params = None
        best_score = -float('inf')
        
        for trial in range(config.max_trials):
            # 生成候选参数
            candidate_params = self._generate_candidate_params()
            
            # 评估参数
            score = self._evaluate_params(candidate_params, config)
            
            # 记录试验
            trial_result = {
                'trial': trial,
                'params': candidate_params,
                'score': score,
                'timestamp': datetime.now().isoformat()
            }
            self.trial_history.append(trial_result)
            
            # 更新最佳参数
            if score > best_score:
                best_score = score
                best_params = candidate_params
                self.logger.info(f"试验 {trial}: 新最佳分数 {score:.4f}")
            
            if trial % 10 == 0:
                self.logger.info(f"完成 {trial}/{config.max_trials} 试验")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'total_trials': len(self.trial_history),
            'optimization_method': 'bayesian'
        }
    
    def grid_search(self, config: OptimizationConfig) -> Dict:
        """网格搜索"""
        self.logger.info("开始网格搜索")
        
        # 定义参数网格
        param_grid = {
            'fractal_dimension': [2.0, 2.5, 3.0, 3.5],
            'complexity_coefficient': [0.5, 0.7, 0.9, 1.0],
            'critical_temperature': [0.5, 1.0, 1.5, 2.0],
            'field_strength': [0.5, 1.0, 1.5, 2.0],
            'entropy_balance': [-0.5, 0.0, 0.5, 1.0]
        }
        
        best_params = None
        best_score = -float('inf')
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        
        trial = 0
        for fd in param_grid['fractal_dimension']:
            for cc in param_grid['complexity_coefficient']:
                for ct in param_grid['critical_temperature']:
                    for fs in param_grid['field_strength']:
                        for eb in param_grid['entropy_balance']:
                            candidate_params = {
                                'fractal_dimension': fd,
                                'complexity_coefficient': cc,
                                'critical_temperature': ct,
                                'field_strength': fs,
                                'entropy_balance': eb
                            }
                            
                            score = self._evaluate_params(candidate_params, config)
                            
                            trial_result = {
                                'trial': trial,
                                'params': candidate_params,
                                'score': score,
                                'timestamp': datetime.now().isoformat()
                            }
                            self.trial_history.append(trial_result)
                            
                            if score > best_score:
                                best_score = score
                                best_params = candidate_params
                                self.logger.info(f"试验 {trial}: 新最佳分数 {score:.4f}")
                            
                            trial += 1
                            
                            if trial % 50 == 0:
                                self.logger.info(f"完成 {trial}/{total_combinations} 试验")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'total_trials': len(self.trial_history),
            'optimization_method': 'grid_search'
        }
    
    def random_search(self, config: OptimizationConfig) -> Dict:
        """随机搜索"""
        self.logger.info("开始随机搜索")
        
        best_params = None
        best_score = -float('inf')
        
        for trial in range(config.max_trials):
            candidate_params = self._generate_candidate_params()
            score = self._evaluate_params(candidate_params, config)
            
            trial_result = {
                'trial': trial,
                'params': candidate_params,
                'score': score,
                'timestamp': datetime.now().isoformat()
            }
            self.trial_history.append(trial_result)
            
            if score > best_score:
                best_score = score
                best_params = candidate_params
                self.logger.info(f"试验 {trial}: 新最佳分数 {score:.4f}")
            
            if trial % 20 == 0:
                self.logger.info(f"完成 {trial}/{config.max_trials} 试验")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'total_trials': len(self.trial_history),
            'optimization_method': 'random_search'
        }
    
    def _generate_candidate_params(self) -> Dict:
        """生成候选参数"""
        return {
            'fractal_dimension': np.random.uniform(2.0, 4.0),
            'complexity_coefficient': np.random.uniform(0.1, 1.0),
            'critical_temperature': np.random.uniform(0.1, 3.0),
            'field_strength': np.random.uniform(0.1, 3.0),
            'entropy_balance': np.random.uniform(-1.0, 1.0)
        }
    
    def _evaluate_params(self, params: Dict, config: OptimizationConfig) -> float:
        """评估参数"""
        try:
            # 创建模型
            cep_params = CEPParameters(**params)
            model = EnhancedCEPEITP(
                input_dim=784,
                hidden_dims=[512, 256, 128],
                output_dim=10,
                cep_params=cep_params
            )
            
            # 生成测试数据
            test_input = torch.randn(10, 784)
            
            # 前向传播
            with torch.no_grad():
                output, metrics = model(test_input)
                consciousness_level = metrics['consciousness_metrics'].consciousness_level
            
            # 计算分数
            if config.target_metric == "consciousness_level":
                score = consciousness_level
            else:
                score = consciousness_level  # 默认使用意识水平
            
            return float(score)
            
        except Exception as e:
            self.logger.warning(f"参数评估失败: {e}")
            return 0.0

class ModelCompressor:
    """模型压缩器"""
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ModelCompressor")
    
    def prune_model(self, model: EnhancedCEPEITP, config: CompressionConfig) -> EnhancedCEPEITP:
        """模型剪枝"""
        self.logger.info(f"开始模型剪枝，目标压缩比例: {config.target_ratio}")
        
        # 计算剪枝阈值
        all_weights = []
        for name, param in model.named_parameters():
            if 'weight' in name and param.numel() > 0:
                all_weights.append(param.data.abs().view(-1))
        
        if not all_weights:
            self.logger.warning("没有找到可剪枝的权重参数")
            return model
        
        all_weights = torch.cat(all_weights)
        threshold = torch.quantile(all_weights, config.target_ratio)
        
        # 应用剪枝
        pruned_params = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = param.data.abs() > threshold
                param.data *= mask.float()
                
                pruned_params += (mask == 0).sum().item()
                total_params += param.numel()
        
        compression_ratio = pruned_params / total_params
        self.logger.info(f"剪枝完成，压缩比例: {compression_ratio:.3f}")
        
        return model
    
    def quantize_model(self, model: EnhancedCEPEITP, config: CompressionConfig) -> EnhancedCEPEITP:
        """模型量化"""
        self.logger.info("开始模型量化")
        
        # 简单的权重量化
        for name, param in model.named_parameters():
            if 'weight' in name:
                # 量化到8位
                param.data = torch.round(param.data * 127) / 127
        
        self.logger.info("量化完成")
        return model
    
    def compress_model(self, model: EnhancedCEPEITP, config: CompressionConfig) -> Dict:
        """压缩模型"""
        self.logger.info(f"开始模型压缩: {config.method}")
        
        original_size = self._calculate_model_size(model)
        
        if config.method == "pruning":
            compressed_model = self.prune_model(model, config)
        elif config.method == "quantization":
            compressed_model = self.quantize_model(model, config)
        else:
            self.logger.warning(f"未知压缩方法: {config.method}")
            return {'error': f'未知压缩方法: {config.method}'}
        
        compressed_size = self._calculate_model_size(compressed_model)
        compression_ratio = (original_size - compressed_size) / original_size
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'method': config.method
        }
    
    def _calculate_model_size(self, model: EnhancedCEPEITP) -> int:
        """计算模型大小（参数数量）"""
        total_params = 0
        for param in model.parameters():
            total_params += param.numel()
        return total_params

class OptimizationToolsManager:
    """优化工具管理器"""
    
    def __init__(self):
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.model_compressor = ModelCompressor()
        self.advanced_manager = AdvancedFeaturesManager()
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("OptimizationToolsManager")
    
    def optimize_hyperparameters(self, config: OptimizationConfig) -> Dict:
        """优化超参数"""
        self.logger.info("开始超参数优化")
        
        if config.method == "bayesian":
            result = self.hyperparameter_optimizer.bayesian_optimization(config)
        elif config.method == "grid":
            result = self.hyperparameter_optimizer.grid_search(config)
        elif config.method == "random":
            result = self.hyperparameter_optimizer.random_search(config)
        else:
            raise ValueError(f"未知优化方法: {config.method}")
        
        return result
    
    def compress_model(self, model: EnhancedCEPEITP, config: CompressionConfig) -> Dict:
        """压缩模型"""
        return self.model_compressor.compress_model(model, config)
    
    def run_optimization_pipeline(self, optimization_config: OptimizationConfig, 
                                 compression_config: CompressionConfig) -> Dict:
        """运行优化流水线"""
        self.logger.info("🚀 开始优化流水线")
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'stages': {}
        }
        
        try:
            # 1. 超参数优化
            self.logger.info("📊 阶段1: 超参数优化")
            hyperopt_result = self.optimize_hyperparameters(optimization_config)
            pipeline_results['stages']['hyperparameter_optimization'] = hyperopt_result
            
            # 2. 使用优化后的参数创建模型
            self.logger.info("🏗️ 阶段2: 创建优化模型")
            best_params = hyperopt_result['best_params']
            cep_params = CEPParameters(**best_params)
            optimized_model = EnhancedCEPEITP(
                input_dim=784,
                hidden_dims=[512, 256, 128],
                output_dim=10,
                cep_params=cep_params
            )
            
            # 3. 模型压缩
            self.logger.info("🗜️ 阶段3: 模型压缩")
            compression_result = self.compress_model(optimized_model, compression_config)
            pipeline_results['stages']['model_compression'] = compression_result
            
            # 4. 性能评估
            self.logger.info("📈 阶段4: 性能评估")
            performance = self._evaluate_optimized_model(optimized_model)
            pipeline_results['stages']['performance_evaluation'] = performance
            
            pipeline_results['status'] = 'success'
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            self.logger.info("🎉 优化流水线完成!")
            
        except Exception as e:
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now().isoformat()
            self.logger.error(f"❌ 优化流水线失败: {e}")
        
        return pipeline_results
    
    def _evaluate_optimized_model(self, model: EnhancedCEPEITP) -> Dict:
        """评估优化后的模型"""
        model.eval()
        
        # 生成测试数据
        test_input = torch.randn(100, 784)
        
        with torch.no_grad():
            start_time = time.time()
            output, metrics = model(test_input)
            inference_time = time.time() - start_time
        
        return {
            'consciousness_level': metrics['consciousness_metrics'].consciousness_level,
            'inference_time': inference_time,
            'model_size': self.model_compressor._calculate_model_size(model),
            'memory_usage': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        }

def main():
    """主函数"""
    print("🔧 Enhanced CEP-EIT-P Optimization Tools")
    print("=" * 50)
    
    # 创建优化工具管理器
    optimization_manager = OptimizationToolsManager()
    
    # 创建优化配置
    optimization_config = OptimizationConfig(
        method="random",  # 使用随机搜索进行快速测试
        max_trials=20,
        target_metric="consciousness_level",
        target_value=3.0
    )
    
    # 创建压缩配置
    compression_config = CompressionConfig(
        method="pruning",
        target_ratio=0.3,
        preserve_accuracy=True
    )
    
    # 运行优化流水线
    results = optimization_manager.run_optimization_pipeline(
        optimization_config, 
        compression_config
    )
    
    # 显示结果
    print(f"\n📊 优化结果:")
    print(f"状态: {results['status']}")
    
    if results['status'] == 'success':
        hyperopt = results['stages']['hyperparameter_optimization']
        compression = results['stages']['model_compression']
        performance = results['stages']['performance_evaluation']
        
        print(f"\n🎯 超参数优化:")
        print(f"  最佳分数: {hyperopt['best_score']:.4f}")
        print(f"  总试验数: {hyperopt['total_trials']}")
        print(f"  最佳参数: {hyperopt['best_params']}")
        
        print(f"\n🗜️ 模型压缩:")
        print(f"  原始大小: {compression['original_size']}")
        print(f"  压缩后大小: {compression['compressed_size']}")
        print(f"  压缩比例: {compression['compression_ratio']:.3f}")
        
        print(f"\n📈 性能评估:")
        print(f"  意识水平: {performance['consciousness_level']:.3f}")
        print(f"  推理时间: {performance['inference_time']:.4f}s")
        print(f"  模型大小: {performance['model_size']}")
        print(f"  内存使用: {performance['memory_usage']:.2f}MB")
    
    print("🎉 优化工具测试完成!")

if __name__ == "__main__":
    main()