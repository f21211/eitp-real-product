"""
微分熵正则化模块
实现基于KNIFE的微分熵估计和正则化
对应IEM理论中的ΔES (负熵吸收)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class KNIFEEstimator(nn.Module):
    """
    KNIFE (Kernelized Neural Information Estimator) 实现
    用于估计微分熵和互信息
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_kernels: int = 5,
        kernel_bandwidth: float = 1.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_kernels = num_kernels
        self.kernel_bandwidth = kernel_bandwidth
        
        # 核函数参数
        self.kernel_centers = nn.Parameter(
            torch.randn(num_kernels, input_dim) * 0.1
        )
        self.kernel_weights = nn.Parameter(
            torch.ones(num_kernels) / num_kernels
        )
        self.kernel_scales = nn.Parameter(
            torch.ones(num_kernels) * kernel_bandwidth
        )
        
    def gaussian_kernel(self, x: torch.Tensor, center: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        高斯核函数
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            center: 核中心 [input_dim]
            scale: 核尺度 [1]
            
        Returns:
            kernel_values: 核函数值 [batch_size]
        """
        # 计算欧几里得距离
        diff = x - center.unsqueeze(0)  # [batch_size, input_dim]
        dist_sq = torch.sum(diff ** 2, dim=1)  # [batch_size]
        
        # 高斯核
        kernel_values = torch.exp(-dist_sq / (2 * scale ** 2))
        return kernel_values
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算核密度估计
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            
        Returns:
            density: 密度估计 [batch_size]
        """
        batch_size = x.shape[0]
        density = torch.zeros(batch_size, device=x.device)
        
        for i in range(self.num_kernels):
            center = self.kernel_centers[i].to(x.device)
            weight = self.kernel_weights[i].to(x.device)
            scale = self.kernel_scales[i].to(x.device)
            
            kernel_values = self.gaussian_kernel(x, center, scale)
            density += weight * kernel_values
        
        return density
    
    def compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算微分熵 H(X) = -E[log p(X)]
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            
        Returns:
            entropy: 微分熵估计
        """
        density = self.forward(x)
        log_density = torch.log(density + 1e-8)
        entropy = -torch.mean(log_density)
        return entropy
    
    def compute_mutual_information(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算互信息 I(X;Y) = H(X) + H(Y) - H(X,Y)
        
        Args:
            x: 第一个变量 [batch_size, x_dim]
            y: 第二个变量 [batch_size, y_dim]
            
        Returns:
            mi: 互信息估计
        """
        # 计算边际熵
        h_x = self.compute_entropy(x)
        h_y = self.compute_entropy(y)
        
        # 计算联合熵
        xy = torch.cat([x, y], dim=1)
        h_xy = self.compute_entropy(xy)
        
        # 互信息
        mi = h_x + h_y - h_xy
        return mi


class EntropyRegularizer(nn.Module):
    """
    微分熵正则化器
    
    实现基于KNIFE的微分熵估计和正则化，用于最小化隐藏状态的微分熵，
    实现信息表示的低熵压缩。
    
    Args:
        hidden_dim: 隐藏状态维度
        beta: 熵正则化权重
        gamma: 互信息正则化权重
        target_entropy: 目标熵值
    """
    
    def __init__(
        self,
        hidden_dim: int,
        beta: float = 1.0,
        gamma: float = 0.1,
        target_entropy: float = 0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.gamma = gamma
        self.target_entropy = target_entropy
        
        # 创建KNIFE估计器
        self.knife_estimator = KNIFEEstimator(
            input_dim=hidden_dim,
            hidden_dim=64,
            num_kernels=10,
            kernel_bandwidth=1.0
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        input_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算熵正则化损失
        
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_dim]
            input_embeddings: 输入嵌入 [batch_size, seq_len, hidden_dim]
            
        Returns:
            losses: 包含各种损失项的字典
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 重塑为 [batch_size * seq_len, hidden_dim]
        h_flat = hidden_states.view(-1, hidden_dim)
        
        # 计算隐藏状态的微分熵
        h_entropy = self.knife_estimator.compute_entropy(h_flat)
        
        # 熵正则化损失：最小化熵（鼓励低熵压缩）
        entropy_loss = self.beta * h_entropy
        
        # 目标熵损失：鼓励达到目标熵值
        target_entropy_loss = F.mse_loss(
            h_entropy,
            torch.tensor(self.target_entropy, device=h_entropy.device)
        )
        
        total_loss = entropy_loss + target_entropy_loss
        
        losses = {
            'entropy_loss': entropy_loss,
            'target_entropy_loss': target_entropy_loss,
            'total_loss': total_loss,
            'h_entropy': h_entropy
        }
        
        # 如果提供了输入嵌入，计算互信息
        if input_embeddings is not None:
            x_flat = input_embeddings.view(-1, hidden_dim)
            mutual_info = self.knife_estimator.compute_mutual_information(x_flat, h_flat)
            
            # 互信息正则化：最大化互信息（保持信息传递）
            mi_loss = -self.gamma * mutual_info
            total_loss += mi_loss
            
            losses.update({
                'mi_loss': mi_loss,
                'mutual_info': mutual_info
            })
        
        losses['total_loss'] = total_loss
        return losses
    
    def compute_thermodynamic_balance(
        self, 
        hidden_states: torch.Tensor,
        complexity_energy: torch.Tensor
    ) -> torch.Tensor:
        """
        计算热力学平衡损失
        
        确保负熵吸收率(ΔS)与复杂度能量储存(λEC)之间的平衡
        
        Args:
            hidden_states: 隐藏状态
            complexity_energy: 复杂度能量 λ·EC
            
        Returns:
            balance_loss: 热力学平衡损失
        """
        h_flat = hidden_states.view(-1, hidden_states.shape[-1])
        h_entropy = self.knife_estimator.compute_entropy(h_flat)
        
        # 负熵吸收率 ΔS = -H(Z)
        negative_entropy_rate = -h_entropy
        
        # 平衡损失：ΔS ≈ λ·EC
        balance_loss = F.mse_loss(negative_entropy_rate, complexity_energy)
        
        return balance_loss
    
    def get_entropy_metrics(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """
        获取熵相关指标
        
        Args:
            hidden_states: 隐藏状态
            
        Returns:
            metrics: 熵指标字典
        """
        with torch.no_grad():
            h_flat = hidden_states.view(-1, hidden_states.shape[-1])
            h_entropy = self.knife_estimator.compute_entropy(h_flat)
            
            return {
                'h_entropy': h_entropy.item(),
                'negative_entropy_rate': -h_entropy.item(),
                'compression_ratio': 1.0 / (1.0 + h_entropy.item())
            }
