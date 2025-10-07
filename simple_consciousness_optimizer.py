#!/usr/bin/env python3
"""
Simplified Consciousness Detection Optimizer
简化版意识检测优化器 - 专注于参数调优
"""

import sys
import os
sys.path.append('/mnt/sda1/myproject/datainall/AGI')

import torch
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters

@dataclass
class OptimizationResult:
    """优化结果"""
    consciousness_level: int
    constraint_satisfaction: float
    fractal_dimension: float
    complexity_coefficient: float
    chaos_threshold: float
    entropy_balance: float
    score: float

class SimpleConsciousnessOptimizer:
    """简化版意识检测优化器"""
    
    def __init__(self):
        self.results = []
        
    def test_cep_parameters(self, input_tensor: torch.Tensor, 
                          fractal_dim: float, complexity_coeff: float,
                          critical_temp: float, field_strength: float,
                          entropy_balance: float) -> OptimizationResult:
        """测试特定CEP参数组合"""
        
        # 创建CEP参数
        cep_params = CEPParameters(
            fractal_dimension=fractal_dim,
            complexity_coefficient=complexity_coeff,
            critical_temperature=critical_temp,
            field_strength=field_strength,
            entropy_balance=entropy_balance
        )
        
        # 创建模型
        model = EnhancedCEPEITP(
            input_dim=input_tensor.size(1),
            hidden_dims=[512, 256, 128],
            output_dim=10,
            cep_params=cep_params
        )
        
        # 进行推理
        with torch.no_grad():
            output, metrics = model(input_tensor)
            
        # 检查约束
        constraints = model.check_cep_constraints()
        
        # 计算约束满足率
        constraint_satisfaction = sum([
            constraints['fractal_dimension'],
            constraints['complexity_coefficient'],
            constraints['chaos_threshold'],
            constraints['entropy_balance']
        ]) / 4.0
        
        # 计算综合得分
        consciousness_metrics = metrics['consciousness_metrics']
        level_score = consciousness_metrics.consciousness_level / 4.0
        constraint_score = constraint_satisfaction
        fractal_score = min(consciousness_metrics.fractal_dimension / 3.0, 1.0)
        complexity_score = min(consciousness_metrics.complexity_coefficient, 1.0)
        
        total_score = (
            0.4 * level_score +
            0.3 * constraint_score +
            0.2 * fractal_score +
            0.1 * complexity_score
        )
        
        return OptimizationResult(
            consciousness_level=consciousness_metrics.consciousness_level,
            constraint_satisfaction=constraint_satisfaction,
            fractal_dimension=consciousness_metrics.fractal_dimension,
            complexity_coefficient=consciousness_metrics.complexity_coefficient,
            chaos_threshold=consciousness_metrics.chaos_threshold,
            entropy_balance=consciousness_metrics.entropy_balance,
            score=total_score
        )
    
    def grid_search_optimization(self, input_tensor: torch.Tensor) -> Dict:
        """网格搜索优化"""
        print("🔍 开始网格搜索优化...")
        
        # 定义搜索空间
        fractal_dims = [2.5, 2.7, 2.9, 3.1, 3.3]
        complexity_coeffs = [0.6, 0.7, 0.8, 0.9, 1.0]
        critical_temps = [0.8, 1.0, 1.2, 1.4, 1.6]
        field_strengths = [0.8, 1.0, 1.2, 1.4, 1.6]
        entropy_balances = [-0.1, -0.05, 0.0, 0.05, 0.1]
        
        best_result = None
        best_score = 0.0
        total_combinations = len(fractal_dims) * len(complexity_coeffs) * len(critical_temps) * len(field_strengths) * len(entropy_balances)
        current_combination = 0
        
        print(f"📊 总共需要测试 {total_combinations} 种参数组合...")
        
        for fractal_dim in fractal_dims:
            for complexity_coeff in complexity_coeffs:
                for critical_temp in critical_temps:
                    for field_strength in field_strengths:
                        for entropy_balance in entropy_balances:
                            current_combination += 1
                            
                            if current_combination % 50 == 0:
                                print(f"进度: {current_combination}/{total_combinations} ({current_combination/total_combinations*100:.1f}%)")
                            
                            try:
                                result = self.test_cep_parameters(
                                    input_tensor, fractal_dim, complexity_coeff,
                                    critical_temp, field_strength, entropy_balance
                                )
                                
                                self.results.append({
                                    'fractal_dimension': fractal_dim,
                                    'complexity_coefficient': complexity_coeff,
                                    'critical_temperature': critical_temp,
                                    'field_strength': field_strength,
                                    'entropy_balance': entropy_balance,
                                    'result': result
                                })
                                
                                if result.score > best_score:
                                    best_score = result.score
                                    best_result = {
                                        'fractal_dimension': fractal_dim,
                                        'complexity_coefficient': complexity_coeff,
                                        'critical_temperature': critical_temp,
                                        'field_strength': field_strength,
                                        'entropy_balance': entropy_balance,
                                        'result': result
                                    }
                                    
                            except Exception as e:
                                print(f"⚠️ 参数组合失败: {e}")
                                continue
        
        print(f"✅ 网格搜索完成! 最佳得分: {best_score:.4f}")
        return best_result
    
    def random_search_optimization(self, input_tensor: torch.Tensor, n_trials: int = 100) -> Dict:
        """随机搜索优化"""
        print(f"🎲 开始随机搜索优化 ({n_trials} 次试验)...")
        
        best_result = None
        best_score = 0.0
        
        for trial in range(n_trials):
            if trial % 20 == 0:
                print(f"进度: {trial}/{n_trials} ({trial/n_trials*100:.1f}%)")
            
            # 随机生成参数
            fractal_dim = np.random.uniform(2.5, 3.5)
            complexity_coeff = np.random.uniform(0.5, 1.2)
            critical_temp = np.random.uniform(0.5, 2.0)
            field_strength = np.random.uniform(0.5, 2.0)
            entropy_balance = np.random.uniform(-0.2, 0.2)
            
            try:
                result = self.test_cep_parameters(
                    input_tensor, fractal_dim, complexity_coeff,
                    critical_temp, field_strength, entropy_balance
                )
                
                self.results.append({
                    'fractal_dimension': fractal_dim,
                    'complexity_coefficient': complexity_coeff,
                    'critical_temperature': critical_temp,
                    'field_strength': field_strength,
                    'entropy_balance': entropy_balance,
                    'result': result
                })
                
                if result.score > best_score:
                    best_score = result.score
                    best_result = {
                        'fractal_dimension': fractal_dim,
                        'complexity_coefficient': complexity_coeff,
                        'critical_temperature': critical_temp,
                        'field_strength': field_strength,
                        'entropy_balance': entropy_balance,
                        'result': result
                    }
                    
            except Exception as e:
                continue
        
        print(f"✅ 随机搜索完成! 最佳得分: {best_score:.4f}")
        return best_result
    
    def analyze_results(self) -> Dict:
        """分析优化结果"""
        if not self.results:
            return {}
        
        scores = [r['result'].score for r in self.results]
        consciousness_levels = [r['result'].consciousness_level for r in self.results]
        constraint_satisfactions = [r['result'].constraint_satisfaction for r in self.results]
        
        analysis = {
            'total_trials': len(self.results),
            'score_stats': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores)
            },
            'consciousness_level_stats': {
                'mean': np.mean(consciousness_levels),
                'std': np.std(consciousness_levels),
                'min': np.min(consciousness_levels),
                'max': np.max(consciousness_levels),
                'distribution': {i: list(consciousness_levels).count(i) for i in range(5)}
            },
            'constraint_satisfaction_stats': {
                'mean': np.mean(constraint_satisfactions),
                'std': np.std(constraint_satisfactions),
                'min': np.min(constraint_satisfactions),
                'max': np.max(constraint_satisfactions)
            }
        }
        
        return analysis
    
    def visualize_results(self, save_path: str = "consciousness_optimization_analysis.png"):
        """可视化优化结果"""
        if not self.results:
            print("❌ 没有结果数据")
            return
        
        scores = [r['result'].score for r in self.results]
        consciousness_levels = [r['result'].consciousness_level for r in self.results]
        constraint_satisfactions = [r['result'].constraint_satisfaction for r in self.results]
        fractal_dimensions = [r['fractal_dimension'] for r in self.results]
        complexity_coefficients = [r['complexity_coefficient'] for r in self.results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 得分分布
        ax1.hist(scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('意识检测得分分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('综合得分')
        ax1.set_ylabel('频次')
        ax1.grid(True, alpha=0.3)
        
        # 意识水平分布
        level_counts = {i: consciousness_levels.count(i) for i in range(5)}
        ax2.bar(level_counts.keys(), level_counts.values(), alpha=0.7, color='green')
        ax2.set_title('意识水平分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('意识水平')
        ax2.set_ylabel('频次')
        ax2.set_xticks(range(5))
        ax2.grid(True, alpha=0.3)
        
        # 约束满足率分布
        ax3.hist(constraint_satisfactions, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax3.set_title('约束满足率分布', fontsize=14, fontweight='bold')
        ax3.set_xlabel('约束满足率')
        ax3.set_ylabel('频次')
        ax3.grid(True, alpha=0.3)
        
        # 参数关系散点图
        scatter = ax4.scatter(fractal_dimensions, complexity_coefficients, 
                             c=scores, cmap='viridis', alpha=0.6)
        ax4.set_title('分形维数 vs 复杂度系数 (颜色=得分)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('分形维数')
        ax4.set_ylabel('复杂度系数')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='综合得分')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 结果可视化已保存: {save_path}")

def main():
    """主函数"""
    print("🧠 Simplified Consciousness Detection Optimizer")
    print("=" * 50)
    
    # 创建测试数据
    input_tensor = torch.randn(16, 784)  # 使用较小的批次以加快测试
    
    # 创建优化器
    optimizer = SimpleConsciousnessOptimizer()
    
    # 运行随机搜索优化（更快）
    print("🚀 开始随机搜索优化...")
    start_time = time.time()
    
    best_result = optimizer.random_search_optimization(input_tensor, n_trials=200)
    
    end_time = time.time()
    optimization_time = end_time - start_time
    
    # 分析结果
    analysis = optimizer.analyze_results()
    
    # 可视化结果
    optimizer.visualize_results()
    
    # 保存结果
    results_data = {
        'optimization_time': optimization_time,
        'best_result': best_result,
        'analysis': analysis,
        'all_results': optimizer.results
    }
    
    with open('consciousness_optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)
    
    # 打印结果
    print("\n" + "=" * 50)
    print("🎉 意识检测优化完成!")
    print(f"⏱️ 优化时间: {optimization_time:.2f}秒")
    print(f"🧪 总试验次数: {analysis['total_trials']}")
    
    if best_result:
        result = best_result['result']
        print(f"🧠 最佳意识得分: {result.score:.4f}")
        print(f"📊 最佳意识水平: {result.consciousness_level}/4")
        print(f"✅ 最佳约束满足率: {result.constraint_satisfaction:.4f}")
        print(f"📐 最佳分形维数: {result.fractal_dimension:.3f}")
        print(f"🔧 最佳复杂度系数: {result.complexity_coefficient:.3f}")
        print(f"🌡️ 最佳临界温度: {best_result['critical_temperature']:.3f}")
        print(f"⚡ 最佳场强度: {best_result['field_strength']:.3f}")
        print(f"🔄 最佳熵平衡: {best_result['entropy_balance']:.3f}")
    
    print(f"📁 结果已保存: consciousness_optimization_results.json")
    print(f"📊 可视化已保存: consciousness_optimization_analysis.png")

if __name__ == "__main__":
    main()
