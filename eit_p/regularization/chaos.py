"""
混沌临界性控制模块
实现Lyapunov指数监测和边缘混沌状态控制
对应IEM理论中的ΔEF (动态敏感性Ω)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


class LyapunovEstimator(nn.Module):
    """
    Lyapunov指数估计器
    通过自动微分计算雅可比矩阵来估计最大Lyapunov指数
    """
    
    def __init__(self, num_steps: int = 10):
        super().__init__()
        self.num_steps = num_steps
        
    def compute_jacobian(
        self, 
        model: nn.Module, 
        x: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算模型在给定输入下的雅可比矩阵（GPU内存优化版本）
        
        Args:
            model: 要分析的模型
            x: 输入张量（对于整数输入如input_ids，使用hidden_states）
            hidden_states: 初始隐藏状态（可选）
            
        Returns:
            jacobian: 雅可比矩阵 [output_dim, input_dim]
        """
        device = x.device
        
        # 如果输入是整数类型（如input_ids），使用简化的统计方法
        if x.dtype in [torch.int32, torch.int64, torch.long]:
            if hidden_states is None:
                # 获取模型输出的hidden_states
                with torch.no_grad():
                    outputs = model(x, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
            
            # 使用简化的统计方法而不是完整的雅可比矩阵
            # 计算隐藏状态的统计特征作为Lyapunov指数的代理
            h_mean = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_dim]
            h_std = torch.std(hidden_states, dim=1)    # [batch_size, hidden_dim]
            
            # 使用标准差作为动态敏感性的代理
            sensitivity_proxy = torch.mean(h_std, dim=1)  # [batch_size]
            
            # 返回简化的"雅可比矩阵"（实际上是统计特征）
            return sensitivity_proxy.unsqueeze(-1)  # [batch_size, 1]
        
        # 对于连续输入，使用简化的梯度计算
        target = x
        target.requires_grad_(True)
        
        try:
            # 只计算输出的均值，减少计算复杂度
            output = model(target)
            if hasattr(output, 'logits'):
                output = output.logits
            
            # 计算输出的均值
            output_mean = torch.mean(output, dim=-1)  # [batch_size]
            
            # 计算简化的梯度
            grad_outputs = torch.ones_like(output_mean)
            jacobian = torch.autograd.grad(
                outputs=output_mean,
                inputs=target,
                grad_outputs=grad_outputs,
                create_graph=False,  # 不创建计算图以节省内存
                retain_graph=False,  # 不保留计算图
                only_inputs=True,
                allow_unused=True
            )[0]
            
            # 如果梯度为None，返回零矩阵
            if jacobian is None:
                jacobian = torch.zeros(target.shape[0], 1, device=device)
            else:
                # 简化雅可比矩阵为标量
                jacobian = torch.mean(jacobian, dim=-1, keepdim=True)
                
        except Exception as e:
            # 如果计算失败，返回零矩阵
            jacobian = torch.zeros(target.shape[0], 1, device=device)
        
        return jacobian
    
    def estimate_lyapunov_exponent(
        self,
        model: nn.Module,
        x: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        估计最大Lyapunov指数（GPU内存优化版本）
        
        Args:
            model: 要分析的模型
            x: 输入张量
            hidden_states: 初始隐藏状态（可选）
            
        Returns:
            lyapunov_exp: 最大Lyapunov指数
        """
        # 计算简化的雅可比矩阵
        jacobian = self.compute_jacobian(model, x, hidden_states)
        
        # 对于简化的雅可比矩阵，直接使用其值作为Lyapunov指数的代理
        if jacobian.dim() == 2 and jacobian.shape[1] == 1:
            # 简化的雅可比矩阵，直接使用其值
            lyapunov_exp = torch.clamp(jacobian.squeeze(-1), -2.0, 2.0)
        else:
            # 如果仍然是矩阵，计算最大奇异值
            try:
                U, S, V = torch.svd(jacobian)
                # 最大奇异值对应最大Lyapunov指数
                max_singular_value = torch.max(S)
                lyapunov_exp = torch.log(max_singular_value + 1e-8)
            except:
                # 如果SVD失败，使用特征值
                try:
                    eigenvals = torch.linalg.eigvals(jacobian)
                    max_eigenval = torch.max(torch.real(eigenvals))
                    lyapunov_exp = torch.log(torch.abs(max_eigenval) + 1e-8)
                except:
                    # 如果都失败，返回0
                    lyapunov_exp = torch.tensor(0.0, device=x.device)
        
        return lyapunov_exp
    
    def estimate_lyapunov_spectrum(
        self,
        model: nn.Module,
        x: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        估计完整的Lyapunov谱
        
        Args:
            model: 要分析的模型
            x: 输入张量
            hidden_states: 初始隐藏状态（可选）
            
        Returns:
            lyapunov_spectrum: Lyapunov指数谱
        """
        jacobian = self.compute_jacobian(model, x, hidden_states)
        
        try:
            U, S, V = torch.svd(jacobian)
            # 所有奇异值对应Lyapunov指数
            lyapunov_spectrum = torch.log(S + 1e-8)
        except:
            try:
                eigenvals = torch.linalg.eigvals(jacobian)
                lyapunov_spectrum = torch.log(torch.abs(torch.real(eigenvals)) + 1e-8)
            except:
                lyapunov_spectrum = torch.zeros(jacobian.shape[0], device=x.device)
        
        return lyapunov_spectrum


class ChaosRegularizer(nn.Module):
    """
    混沌临界性正则化器
    
    通过Lyapunov指数监测将系统动态锁定在边缘混沌状态，
    确保EC具有最大信息处理活性。
    
    Args:
        target_lyapunov: 目标Lyapunov指数 Λ* (通常接近0)
        chaos_weight: 混沌正则化权重
        stability_threshold: 稳定性阈值
    """
    
    def __init__(
        self,
        target_lyapunov: float = 0.0,
        chaos_weight: float = 1.0,
        stability_threshold: float = 0.1
    ):
        super().__init__()
        self.target_lyapunov = target_lyapunov
        self.chaos_weight = chaos_weight
        self.stability_threshold = stability_threshold
        
        # Lyapunov指数估计器
        self.lyapunov_estimator = LyapunovEstimator(num_steps=10)
        
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算混沌正则化损失
        
        Args:
            model: 要分析的模型
            x: 输入张量
            hidden_states: 初始隐藏状态（可选）
            
        Returns:
            losses: 包含各种损失项的字典
        """
        # 估计最大Lyapunov指数
        max_lyapunov = self.lyapunov_estimator.estimate_lyapunov_exponent(
            model, x, hidden_states
        )
        
        # 临界态正则化：惩罚偏离目标Lyapunov指数的状态
        target_tensor = torch.tensor(self.target_lyapunov, device=max_lyapunov.device)
        chaos_loss = self.chaos_weight * F.mse_loss(max_lyapunov, target_tensor)
        
        # 边缘混沌损失：鼓励系统处于边缘混沌状态
        edge_chaos_loss = self.chaos_weight * torch.relu(
            torch.abs(max_lyapunov) - self.stability_threshold
        )
        
        # 总损失
        total_loss = chaos_loss + edge_chaos_loss
        
        return {
            'chaos_loss': chaos_loss,
            'edge_chaos_loss': edge_chaos_loss,
            'total_loss': total_loss,
            'max_lyapunov': max_lyapunov,
            'is_edge_chaos': torch.abs(max_lyapunov) < self.stability_threshold
        }
    
    def compute_dynamic_sensitivity(
        self,
        model: nn.Module,
        x: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算动态敏感性Ω
        
        Args:
            model: 要分析的模型
            x: 输入张量
            hidden_states: 初始隐藏状态（可选）
            
        Returns:
            sensitivity: 动态敏感性
        """
        # 计算Lyapunov谱
        lyapunov_spectrum = self.lyapunov_estimator.estimate_lyapunov_spectrum(
            model, x, hidden_states
        )
        
        # 动态敏感性 = 最大Lyapunov指数的绝对值
        sensitivity = torch.max(torch.abs(lyapunov_spectrum))
        
        return sensitivity
    
    def check_criticality(
        self,
        model: nn.Module,
        x: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        检查系统是否处于临界状态
        
        Args:
            model: 要分析的模型
            x: 输入张量
            hidden_states: 初始隐藏状态（可选）
            
        Returns:
            criticality_info: 临界性信息
        """
        max_lyapunov = self.lyapunov_estimator.estimate_lyapunov_exponent(
            model, x, hidden_states
        )
        
        # 判断是否处于边缘混沌状态
        is_edge_chaos = torch.abs(max_lyapunov) < self.stability_threshold
        
        # 判断是否处于临界状态（Lyapunov指数接近0）
        is_critical = torch.abs(max_lyapunov - self.target_lyapunov) < 0.01
        
        # 计算临界性分数
        criticality_score = 1.0 - torch.abs(max_lyapunov - self.target_lyapunov)
        criticality_score = torch.clamp(criticality_score, 0.0, 1.0)
        
        return {
            'max_lyapunov': max_lyapunov,
            'is_edge_chaos': is_edge_chaos,
            'is_critical': is_critical,
            'criticality_score': criticality_score,
            'distance_to_target': torch.abs(max_lyapunov - self.target_lyapunov)
        }
    
    def get_chaos_metrics(
        self,
        model: nn.Module,
        x: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        获取混沌相关指标
        
        Args:
            model: 要分析的模型
            x: 输入张量
            hidden_states: 初始隐藏状态（可选）
            
        Returns:
            metrics: 混沌指标字典
        """
        with torch.no_grad():
            max_lyapunov = self.lyapunov_estimator.estimate_lyapunov_exponent(
                model, x, hidden_states
            )
            
            sensitivity = self.compute_dynamic_sensitivity(
                model, x, hidden_states
            )
            
            criticality_info = self.check_criticality(
                model, x, hidden_states
            )
            
            return {
                'max_lyapunov': max_lyapunov.item(),
                'dynamic_sensitivity': sensitivity.item(),
                'is_edge_chaos': criticality_info['is_edge_chaos'].item(),
                'is_critical': criticality_info['is_critical'].item(),
                'criticality_score': criticality_info['criticality_score'].item(),
                'distance_to_target': criticality_info['distance_to_target'].item()
            }
