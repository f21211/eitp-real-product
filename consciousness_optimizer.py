#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P Consciousness Detection Optimizer
优化意识检测精度和CEP约束满足率
"""

import sys
import os
sys.path.append('/mnt/sda1/myproject/datainall/AGI')

import torch
import torch.nn as nn
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters, ConsciousnessDetector

@dataclass
class OptimizationConfig:
    """优化配置"""
    target_consciousness_level: int = 4  # 目标意识水平
    target_constraint_satisfaction: float = 1.0  # 目标约束满足率
    max_epochs: int = 1000
    learning_rate: float = 0.01
    patience: int = 50
    min_improvement: float = 1e-6

class ConsciousnessOptimizer:
    """意识检测优化器"""
    
    def __init__(self, model: EnhancedCEPEITP, config: OptimizationConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )
        self.best_score = 0.0
        self.patience_counter = 0
        self.optimization_history = []
        
    def calculate_consciousness_score(self, metrics: Dict) -> float:
        """计算意识检测综合得分"""
        consciousness_metrics = metrics['consciousness_metrics']
        
        # 意识水平得分 (0-1)
        level_score = consciousness_metrics.consciousness_level / 4.0
        
        # 约束满足得分 (0-1)
        constraints = self.model.check_cep_constraints()
        constraint_score = sum([
            constraints['fractal_dimension'],
            constraints['complexity_coefficient'], 
            constraints['chaos_threshold'],
            constraints['entropy_balance']
        ]) / 4.0
        
        # 分形维数得分 (0-1)
        fractal_score = min(consciousness_metrics.fractal_dimension / 3.0, 1.0)
        
        # 复杂度系数得分 (0-1)
        complexity_score = min(consciousness_metrics.complexity_coefficient, 1.0)
        
        # 综合得分 (加权平均)
        total_score = (
            0.4 * level_score +
            0.3 * constraint_score +
            0.2 * fractal_score +
            0.1 * complexity_score
        )
        
        return total_score
    
    def optimize_cep_parameters(self, input_tensor: torch.Tensor) -> Dict:
        """优化CEP参数以提高意识检测精度"""
        print("🔧 开始优化CEP参数...")
        
        best_params = None
        best_score = 0.0
        
        for epoch in range(self.config.max_epochs):
            # 前向传播
            output, metrics = self.model(input_tensor)
            
            # 计算意识得分
            consciousness_score = self.calculate_consciousness_score(metrics)
            
            # 计算损失
            target_score = self.config.target_consciousness_level / 4.0
            consciousness_loss = torch.tensor((consciousness_score - target_score) ** 2)
            
            # 约束损失
            constraints = self.model.check_cep_constraints()
            constraint_loss = torch.tensor(1.0 - sum([
                constraints['fractal_dimension'],
                constraints['complexity_coefficient'],
                constraints['chaos_threshold'], 
                constraints['entropy_balance']
            ]) / 4.0)
            
            # 总损失
            total_loss = consciousness_loss + constraint_loss
            
            # 反向传播
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 更新学习率
            self.scheduler.step(consciousness_score)
            
            # 记录优化历史
            self.optimization_history.append({
                'epoch': epoch,
                'consciousness_score': consciousness_score,
                'constraint_satisfaction': sum([
                    constraints['fractal_dimension'],
                    constraints['complexity_coefficient'],
                    constraints['chaos_threshold'],
                    constraints['entropy_balance']
                ]) / 4.0,
                'consciousness_level': metrics['consciousness_metrics'].consciousness_level,
                'fractal_dimension': metrics['consciousness_metrics'].fractal_dimension,
                'complexity_coefficient': metrics['consciousness_metrics'].complexity_coefficient,
                'loss': total_loss.item()
            })
            
            # 检查是否找到更好的参数
            if consciousness_score > best_score:
                best_score = consciousness_score
                best_params = {
                    'fractal_dimension': self.model.cep_params.fractal_dimension,
                    'complexity_coefficient': self.model.cep_params.complexity_coefficient,
                    'critical_temperature': self.model.cep_params.critical_temperature,
                    'field_strength': self.model.cep_params.field_strength,
                    'entropy_balance': self.model.cep_params.entropy_balance
                }
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 早停检查
            if self.patience_counter >= self.config.patience:
                print(f"⏹️ 早停于第 {epoch} 轮")
                break
            
            # 打印进度
            if epoch % 100 == 0:
                print(f"轮次 {epoch:4d}: 意识得分 {consciousness_score:.4f}, "
                      f"约束满足 {constraint_loss.item():.4f}, "
                      f"意识水平 {metrics['consciousness_metrics'].consciousness_level}/4")
        
        print(f"✅ 优化完成! 最佳得分: {best_score:.4f}")
        return best_params
    
    def optimize_architecture(self, input_tensor: torch.Tensor) -> Dict:
        """优化网络架构以提高意识检测能力"""
        print("🏗️ 开始优化网络架构...")
        
        # 测试不同的隐藏层配置
        architectures = [
            ([256, 128], "小型"),
            ([512, 256, 128], "中型"), 
            ([768, 512, 256, 128], "大型"),
            ([1024, 768, 512, 256], "超大型")
        ]
        
        best_architecture = None
        best_score = 0.0
        architecture_results = []
        
        for hidden_dims, name in architectures:
            print(f"测试 {name} 架构: {hidden_dims}")
            
            # 创建新模型
            test_model = EnhancedCEPEITP(
                input_dim=input_tensor.size(1),
                hidden_dims=hidden_dims,
                output_dim=10,
                cep_params=self.model.cep_params
            )
            
            # 测试性能
            with torch.no_grad():
                output, metrics = test_model(input_tensor)
                score = self.calculate_consciousness_score(metrics)
                
                architecture_results.append({
                    'architecture': hidden_dims,
                    'name': name,
                    'score': score,
                    'consciousness_level': metrics['consciousness_metrics'].consciousness_level,
                    'parameters': sum(p.numel() for p in test_model.parameters())
                })
                
                print(f"  {name}: 得分 {score:.4f}, 意识水平 {metrics['consciousness_metrics'].consciousness_level}/4")
                
                if score > best_score:
                    best_score = score
                    best_architecture = hidden_dims
        
        print(f"✅ 最佳架构: {best_architecture} (得分: {best_score:.4f})")
        return {
            'best_architecture': best_architecture,
            'best_score': best_score,
            'all_results': architecture_results
        }
    
    def optimize_consciousness_detection(self, input_tensor: torch.Tensor) -> Dict:
        """综合优化意识检测系统"""
        print("🧠 开始综合优化意识检测系统...")
        
        # 1. 优化CEP参数
        print("\n1️⃣ 优化CEP参数...")
        cep_params = self.optimize_cep_parameters(input_tensor)
        
        # 2. 优化网络架构
        print("\n2️⃣ 优化网络架构...")
        arch_results = self.optimize_architecture(input_tensor)
        
        # 3. 最终测试
        print("\n3️⃣ 最终测试...")
        final_output, final_metrics = self.model(input_tensor)
        final_score = self.calculate_consciousness_score(final_metrics)
        final_constraints = self.model.check_cep_constraints()
        
        # 生成优化报告
        optimization_report = {
            'optimization_config': {
                'target_consciousness_level': self.config.target_consciousness_level,
                'target_constraint_satisfaction': self.config.target_constraint_satisfaction,
                'max_epochs': self.config.max_epochs,
                'learning_rate': self.config.learning_rate
            },
            'final_results': {
                'consciousness_score': final_score,
                'consciousness_level': final_metrics['consciousness_metrics'].consciousness_level,
                'constraint_satisfaction': sum([
                    final_constraints['fractal_dimension'],
                    final_constraints['complexity_coefficient'],
                    final_constraints['chaos_threshold'],
                    final_constraints['entropy_balance']
                ]) / 4.0,
                'fractal_dimension': final_metrics['consciousness_metrics'].fractal_dimension,
                'complexity_coefficient': final_metrics['consciousness_metrics'].complexity_coefficient,
                'chaos_threshold': final_metrics['consciousness_metrics'].chaos_threshold,
                'entropy_balance': final_metrics['consciousness_metrics'].entropy_balance
            },
            'optimized_cep_params': cep_params,
            'architecture_results': arch_results,
            'optimization_history': self.optimization_history[-100:]  # 最后100轮
        }
        
        return optimization_report
    
    def visualize_optimization(self, save_path: str = "consciousness_optimization.png"):
        """可视化优化过程"""
        if not self.optimization_history:
            print("❌ 没有优化历史数据")
            return
        
        epochs = [h['epoch'] for h in self.optimization_history]
        consciousness_scores = [h['consciousness_score'] for h in self.optimization_history]
        constraint_satisfactions = [h['constraint_satisfaction'] for h in self.optimization_history]
        consciousness_levels = [h['consciousness_level'] for h in self.optimization_history]
        losses = [h['loss'] for h in self.optimization_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 意识得分
        ax1.plot(epochs, consciousness_scores, 'b-', linewidth=2)
        ax1.set_title('意识检测得分优化过程', fontsize=14, fontweight='bold')
        ax1.set_xlabel('优化轮次')
        ax1.set_ylabel('意识得分')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='目标得分')
        ax1.legend()
        
        # 约束满足率
        ax2.plot(epochs, constraint_satisfactions, 'g-', linewidth=2)
        ax2.set_title('CEP约束满足率优化过程', fontsize=14, fontweight='bold')
        ax2.set_xlabel('优化轮次')
        ax2.set_ylabel('约束满足率')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='目标满足率')
        ax2.legend()
        
        # 意识水平
        ax3.plot(epochs, consciousness_levels, 'm-', linewidth=2, marker='o', markersize=3)
        ax3.set_title('意识水平变化过程', fontsize=14, fontweight='bold')
        ax3.set_xlabel('优化轮次')
        ax3.set_ylabel('意识水平')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=4, color='r', linestyle='--', alpha=0.7, label='目标水平')
        ax3.set_ylim(0, 4.5)
        ax3.legend()
        
        # 损失函数
        ax4.plot(epochs, losses, 'r-', linewidth=2)
        ax4.set_title('损失函数优化过程', fontsize=14, fontweight='bold')
        ax4.set_xlabel('优化轮次')
        ax4.set_ylabel('损失值')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 优化过程可视化已保存: {save_path}")

def main():
    """主函数"""
    print("🧠 Enhanced CEP-EIT-P 意识检测优化器")
    print("=" * 50)
    
    # 创建测试数据
    input_tensor = torch.randn(32, 784)
    
    # 创建模型
    cep_params = CEPParameters(
        fractal_dimension=2.7,
        complexity_coefficient=0.8,
        critical_temperature=1.0,
        field_strength=1.0,
        entropy_balance=0.0
    )
    
    model = EnhancedCEPEITP(
        input_dim=784,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        cep_params=cep_params
    )
    
    # 创建优化器
    config = OptimizationConfig(
        target_consciousness_level=4,
        target_constraint_satisfaction=1.0,
        max_epochs=500,
        learning_rate=0.01
    )
    
    optimizer = ConsciousnessOptimizer(model, config)
    
    # 运行优化
    print("🚀 开始意识检测优化...")
    start_time = time.time()
    
    optimization_report = optimizer.optimize_consciousness_detection(input_tensor)
    
    end_time = time.time()
    optimization_time = end_time - start_time
    
    # 保存优化报告
    with open('consciousness_optimization_report.json', 'w', encoding='utf-8') as f:
        json.dump(optimization_report, f, indent=2, ensure_ascii=False)
    
    # 可视化优化过程
    optimizer.visualize_optimization()
    
    # 打印结果
    print("\n" + "=" * 50)
    print("🎉 意识检测优化完成!")
    print(f"⏱️ 优化时间: {optimization_time:.2f}秒")
    print(f"🧠 最终意识得分: {optimization_report['final_results']['consciousness_score']:.4f}")
    print(f"📊 最终意识水平: {optimization_report['final_results']['consciousness_level']}/4")
    print(f"✅ 约束满足率: {optimization_report['final_results']['constraint_satisfaction']:.4f}")
    print(f"📐 分形维数: {optimization_report['final_results']['fractal_dimension']:.3f}")
    print(f"🔧 复杂度系数: {optimization_report['final_results']['complexity_coefficient']:.3f}")
    print(f"📁 报告已保存: consciousness_optimization_report.json")
    print(f"📊 可视化已保存: consciousness_optimization.png")

if __name__ == "__main__":
    main()
