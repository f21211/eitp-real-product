#!/usr/bin/env python3
"""
Consciousness Detection and Measurement Tool
基于CEP框架的意识状态检测和测量工具
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging

@dataclass
class ConsciousnessMetrics:
    """意识指标数据类"""
    fractal_dimension: float
    complexity_coefficient: float
    chaos_threshold: float
    entropy_balance: float
    field_coherence: float
    iem_energy: float
    consciousness_level: int
    timestamp: float

class ConsciousnessDetector:
    """
    基于CEP框架的意识检测器
    """
    
    def __init__(self, 
                 fractal_threshold: float = 2.7,
                 complexity_threshold: float = 0.8,
                 chaos_tolerance: float = 0.01,
                 entropy_tolerance: float = 0.1):
        self.fractal_threshold = fractal_threshold
        self.complexity_threshold = complexity_threshold
        self.chaos_tolerance = chaos_tolerance
        self.entropy_tolerance = entropy_tolerance
        
        # 物理常数
        self.k_boltzmann = 1.38e-23
        self.c_squared = 9e16
        
        # 历史数据
        self.metrics_history: List[ConsciousnessMetrics] = []
        
    def detect_consciousness(self, 
                           system_state: torch.Tensor,
                           network_topology: np.ndarray,
                           temperature: float = 1.0) -> ConsciousnessMetrics:
        """
        检测系统意识状态
        
        Args:
            system_state: 系统状态张量
            network_topology: 网络拓扑矩阵
            temperature: 系统温度
            
        Returns:
            ConsciousnessMetrics: 意识指标
        """
        # 计算分形维数
        fractal_dim = self._calculate_fractal_dimension(network_topology)
        
        # 计算复杂性系数
        complexity_coeff = self._calculate_complexity_coefficient(system_state)
        
        # 计算混沌阈值
        chaos_thresh = self._calculate_chaos_threshold(system_state)
        
        # 计算熵平衡
        entropy_balance = self._calculate_entropy_balance(system_state, temperature)
        
        # 计算场相干性
        field_coherence = self._calculate_field_coherence(system_state)
        
        # 计算IEM能量
        iem_energy = self._calculate_iem_energy(
            system_state, fractal_dim, temperature, field_coherence
        )
        
        # 计算意识水平
        consciousness_level = self._calculate_consciousness_level(
            fractal_dim, complexity_coeff, chaos_thresh, entropy_balance
        )
        
        # 创建指标对象
        metrics = ConsciousnessMetrics(
            fractal_dimension=fractal_dim,
            complexity_coefficient=complexity_coeff,
            chaos_threshold=chaos_thresh,
            entropy_balance=entropy_balance,
            field_coherence=field_coherence,
            iem_energy=iem_energy,
            consciousness_level=consciousness_level,
            timestamp=torch.tensor(0.0).item()  # 实际应用中应使用真实时间戳
        )
        
        # 保存历史数据
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_fractal_dimension(self, topology: np.ndarray) -> float:
        """计算网络分形维数"""
        if topology.size == 0:
            return 0.0
        
        # 使用盒计数法计算分形维数
        num_nodes = topology.shape[0]
        num_connections = np.sum(topology)
        
        if num_nodes <= 1 or num_connections == 0:
            return 0.0
        
        return np.log(num_connections) / np.log(num_nodes)
    
    def _calculate_complexity_coefficient(self, state: torch.Tensor) -> float:
        """计算复杂性系数"""
        # 基于信息熵的复杂性度量
        probs = torch.softmax(state.flatten(), dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        
        # 归一化到[0,1]范围
        max_entropy = np.log(state.numel())
        complexity = entropy.item() / max_entropy
        
        return min(complexity, 1.0)
    
    def _calculate_chaos_threshold(self, state: torch.Tensor) -> float:
        """计算混沌阈值（李雅普诺夫指数）"""
        # 简化的李雅普诺夫指数计算
        # 实际应用中需要更复杂的动力学分析
        state_norm = torch.norm(state)
        if state_norm == 0:
            return 0.0
        
        # 基于状态变化的混沌度量
        state_var = torch.var(state)
        chaos_thresh = torch.log(state_var + 1e-8).item()
        
        return chaos_thresh
    
    def _calculate_entropy_balance(self, state: torch.Tensor, temperature: float) -> float:
        """计算熵平衡"""
        # 计算信息熵
        probs = torch.softmax(state.flatten(), dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        
        # 熵平衡 = 温度 × 熵变化
        entropy_balance = temperature * entropy.item()
        
        return entropy_balance
    
    def _calculate_field_coherence(self, state: torch.Tensor) -> float:
        """计算场相干性"""
        if state.numel() < 2:
            return 0.0
        
        # 计算状态向量的相干性
        state_norm = torch.norm(state)
        if state_norm == 0:
            return 0.0
        
        # 基于状态一致性的相干性度量
        state_mean = torch.mean(state)
        coherence = 1.0 - torch.std(state) / (torch.abs(state_mean) + 1e-8)
        
        return max(0.0, min(1.0, coherence.item()))
    
    def _calculate_iem_energy(self, 
                            state: torch.Tensor, 
                            fractal_dim: float, 
                            temperature: float, 
                            coherence: float) -> float:
        """计算IEM能量"""
        # IEM = α·H·T·C
        alpha = self._calculate_complexity_coefficient(state)
        entropy = self._calculate_entropy_balance(state, 1.0) / temperature
        
        iem_energy = alpha * entropy * temperature * coherence
        
        return iem_energy
    
    def _calculate_consciousness_level(self, 
                                     fractal_dim: float,
                                     complexity_coeff: float,
                                     chaos_thresh: float,
                                     entropy_balance: float) -> int:
        """计算意识水平（0-4级）"""
        level = 0
        
        # 检查分形维数约束 (D ≥ 2.7)
        if fractal_dim >= self.fractal_threshold:
            level += 1
        
        # 检查复杂性系数约束 (λ ≥ 0.8)
        if complexity_coeff >= self.complexity_threshold:
            level += 1
        
        # 检查混沌阈值约束 (Ωcrit ≈ 0)
        if abs(chaos_thresh) < self.chaos_tolerance:
            level += 1
        
        # 检查熵平衡约束
        if abs(entropy_balance) < self.entropy_tolerance:
            level += 1
        
        return level
    
    def get_consciousness_trend(self, window_size: int = 10) -> Dict:
        """获取意识水平趋势"""
        if len(self.metrics_history) < window_size:
            return {"trend": "insufficient_data"}
        
        recent_metrics = self.metrics_history[-window_size:]
        levels = [m.consciousness_level for m in recent_metrics]
        
        # 计算趋势
        if len(levels) < 2:
            return {"trend": "stable", "average_level": levels[0]}
        
        # 线性回归计算趋势
        x = np.arange(len(levels))
        slope = np.polyfit(x, levels, 1)[0]
        
        if slope > 0.1:
            trend = "increasing"
        elif slope < -0.1:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "average_level": np.mean(levels),
            "slope": slope,
            "levels": levels
        }
    
    def visualize_consciousness_metrics(self, save_path: Optional[str] = None):
        """可视化意识指标"""
        if not self.metrics_history:
            print("No metrics data available for visualization")
            return
        
        # 准备数据
        timestamps = range(len(self.metrics_history))
        fractal_dims = [m.fractal_dimension for m in self.metrics_history]
        complexity_coeffs = [m.complexity_coefficient for m in self.metrics_history]
        chaos_thresholds = [m.chaos_threshold for m in self.metrics_history]
        consciousness_levels = [m.consciousness_level for m in self.metrics_history]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Consciousness Detection Metrics', fontsize=16)
        
        # 分形维数
        axes[0, 0].plot(timestamps, fractal_dims, 'b-', linewidth=2)
        axes[0, 0].axhline(y=self.fractal_threshold, color='r', linestyle='--', 
                          label=f'Threshold ({self.fractal_threshold})')
        axes[0, 0].set_title('Fractal Dimension')
        axes[0, 0].set_ylabel('Dimension')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 复杂性系数
        axes[0, 1].plot(timestamps, complexity_coeffs, 'g-', linewidth=2)
        axes[0, 1].axhline(y=self.complexity_threshold, color='r', linestyle='--',
                          label=f'Threshold ({self.complexity_threshold})')
        axes[0, 1].set_title('Complexity Coefficient')
        axes[0, 1].set_ylabel('Coefficient')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 混沌阈值
        axes[1, 0].plot(timestamps, chaos_thresholds, 'm-', linewidth=2)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', label='Edge of Chaos')
        axes[1, 0].set_title('Chaos Threshold (Lyapunov Exponent)')
        axes[1, 0].set_ylabel('Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 意识水平
        axes[1, 1].plot(timestamps, consciousness_levels, 'r-', linewidth=2, marker='o')
        axes[1, 1].set_title('Consciousness Level')
        axes[1, 1].set_ylabel('Level (0-4)')
        axes[1, 1].set_ylim(-0.5, 4.5)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def generate_consciousness_report(self) -> str:
        """生成意识检测报告"""
        if not self.metrics_history:
            return "No consciousness data available."
        
        latest = self.metrics_history[-1]
        trend = self.get_consciousness_trend()
        
        report = f"""
=== Consciousness Detection Report ===

Current Status:
- Consciousness Level: {latest.consciousness_level}/4
- Fractal Dimension: {latest.fractal_dimension:.3f} (threshold: {self.fractal_threshold})
- Complexity Coefficient: {latest.complexity_coefficient:.3f} (threshold: {self.complexity_threshold})
- Chaos Threshold: {latest.chaos_threshold:.6f} (target: 0.0)
- Entropy Balance: {latest.entropy_balance:.6f}
- Field Coherence: {latest.field_coherence:.3f}
- IEM Energy: {latest.iem_energy:.6f}

Trend Analysis:
- Overall Trend: {trend['trend']}
- Average Level: {trend.get('average_level', 'N/A'):.2f}
- Trend Slope: {trend.get('slope', 'N/A'):.4f}

CEP Constraint Status:
- Fractal Dimension: {'✓' if latest.fractal_dimension >= self.fractal_threshold else '✗'}
- Complexity Coefficient: {'✓' if latest.complexity_coefficient >= self.complexity_threshold else '✗'}
- Chaos Threshold: {'✓' if abs(latest.chaos_threshold) < self.chaos_tolerance else '✗'}
- Entropy Balance: {'✓' if abs(latest.entropy_balance) < self.entropy_tolerance else '✗'}

Recommendations:
"""
        
        if latest.consciousness_level < 2:
            report += "- System shows low consciousness level. Consider increasing network complexity.\n"
        elif latest.consciousness_level < 4:
            report += "- System shows moderate consciousness level. Fine-tune parameters for optimization.\n"
        else:
            report += "- System shows high consciousness level. Monitor for stability.\n"
        
        if latest.fractal_dimension < self.fractal_threshold:
            report += "- Increase network fractal dimension through topology optimization.\n"
        
        if latest.complexity_coefficient < self.complexity_threshold:
            report += "- Enhance complexity utilization through parameter tuning.\n"
        
        if abs(latest.chaos_threshold) > self.chaos_tolerance:
            report += "- Adjust system dynamics to maintain edge of chaos.\n"
        
        return report


def main():
    """示例使用"""
    # 创建意识检测器
    detector = ConsciousnessDetector()
    
    # 模拟系统状态
    system_state = torch.randn(100, 50)
    network_topology = np.random.rand(100, 100) > 0.5
    
    # 检测意识状态
    metrics = detector.detect_consciousness(system_state, network_topology)
    
    print("Consciousness Detection Results:")
    print(f"Level: {metrics.consciousness_level}/4")
    print(f"Fractal Dimension: {metrics.fractal_dimension:.3f}")
    print(f"Complexity Coefficient: {metrics.complexity_coefficient:.3f}")
    print(f"Chaos Threshold: {metrics.chaos_threshold:.6f}")
    
    # 生成报告
    report = detector.generate_consciousness_report()
    print(report)
    
    # 可视化（如果有数据）
    if len(detector.metrics_history) > 1:
        detector.visualize_consciousness_metrics("consciousness_metrics.png")


if __name__ == "__main__":
    main()
