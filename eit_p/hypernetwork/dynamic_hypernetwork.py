"""
动态权重超网络
实现用于动态生成复杂度系数λ的超网络
基于HyperMAML和Meta-Learning思想
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


class DynamicHypernetwork(nn.Module):
    """
    动态权重超网络
    
    接收内部特征(Λmax, H(Z)等)，动态生成复杂度系数λ，
    用于调控Path-Norm正则化的强度。
    
    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度（λ的维度）
        num_layers: 网络层数
        dropout: Dropout概率
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 构建网络层
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.Sigmoid())  # 确保λ在[0,1]范围内
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        lyapunov_exp: torch.Tensor,
        entropy: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        additional_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播，生成动态λ系数
        
        Args:
            lyapunov_exp: 最大Lyapunov指数 [batch_size]
            entropy: 微分熵 [batch_size]
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_dim]
            additional_features: 额外特征 [batch_size, feature_dim]
            
        Returns:
            lambda_coeff: 动态生成的λ系数 [batch_size, output_dim]
        """
        batch_size = lyapunov_exp.shape[0]
        
        # 准备输入特征
        features = [lyapunov_exp.unsqueeze(-1), entropy.unsqueeze(-1)]
        
        # 添加隐藏状态统计特征
        if hidden_states is not None:
            # 计算隐藏状态的统计特征
            h_mean = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_dim]
            h_std = torch.std(hidden_states, dim=1)    # [batch_size, hidden_dim]
            h_norm = torch.norm(hidden_states, dim=2)  # [batch_size, seq_len]
            h_norm_mean = torch.mean(h_norm, dim=1).unsqueeze(-1)  # [batch_size, 1]
            
            features.extend([h_mean, h_std, h_norm_mean])
        
        # 添加额外特征
        if additional_features is not None:
            features.append(additional_features)
        
        # 拼接所有特征
        input_features = torch.cat(features, dim=-1)
        
        # 通过超网络生成λ系数
        lambda_coeff = self.network(input_features)
        
        return lambda_coeff
    
    def compute_meta_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        lyapunov_exp: torch.Tensor,
        entropy: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算Meta-Learning损失
        
        Args:
            model: 主模型
            x: 输入数据
            y: 目标数据
            lyapunov_exp: Lyapunov指数
            entropy: 微分熵
            hidden_states: 隐藏状态
            
        Returns:
            meta_loss: Meta-Learning损失
        """
        # 生成动态λ系数
        lambda_coeff = self.forward(lyapunov_exp, entropy, hidden_states)
        
        # 使用λ系数计算正则化损失
        # 这里需要与主模型的损失函数结合
        # 具体实现取决于主模型的结构
        
        # 计算λ的稳定性损失（鼓励λ变化平滑）
        lambda_stability_loss = F.mse_loss(
            lambda_coeff[1:], 
            lambda_coeff[:-1]
        )
        
        return lambda_stability_loss
    
    def get_lambda_statistics(self, lambda_coeff: torch.Tensor) -> Dict[str, float]:
        """
        获取λ系数的统计信息
        
        Args:
            lambda_coeff: λ系数张量
            
        Returns:
            stats: 统计信息字典
        """
        with torch.no_grad():
            return {
                'lambda_mean': torch.mean(lambda_coeff).item(),
                'lambda_std': torch.std(lambda_coeff).item(),
                'lambda_min': torch.min(lambda_coeff).item(),
                'lambda_max': torch.max(lambda_coeff).item(),
                'lambda_range': (torch.max(lambda_coeff) - torch.min(lambda_coeff)).item()
            }


class AdaptiveHypernetwork(DynamicHypernetwork):
    """
    自适应超网络
    
    在基础超网络基础上，添加自适应机制，
    根据训练进度和历史信息调整λ生成策略。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        memory_size: int = 100
    ):
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.memory_size = memory_size
        
        # 历史信息存储
        self.register_buffer('lambda_history', torch.zeros(memory_size, output_dim))
        self.register_buffer('loss_history', torch.zeros(memory_size))
        self.register_buffer('step_count', torch.tensor(0))
        
        # 自适应权重
        self.adaptive_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        lyapunov_exp: torch.Tensor,
        entropy: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        additional_features: Optional[torch.Tensor] = None,
        current_loss: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        自适应前向传播
        
        Args:
            lyapunov_exp: 最大Lyapunov指数
            entropy: 微分熵
            hidden_states: 隐藏状态
            additional_features: 额外特征
            current_loss: 当前损失（用于自适应调整）
            
        Returns:
            lambda_coeff: 自适应生成的λ系数
        """
        # 基础λ生成
        base_lambda = super().forward(
            lyapunov_exp, entropy, hidden_states, additional_features
        )
        
        # 如果提供了当前损失，进行自适应调整
        if current_loss is not None:
            # 更新历史信息
            self._update_history(base_lambda, current_loss)
            
            # 计算自适应调整
            adaptive_adjustment = self._compute_adaptive_adjustment()
            
            # 应用自适应调整
            lambda_coeff = base_lambda + self.adaptive_weight * adaptive_adjustment
        else:
            lambda_coeff = base_lambda
        
        return lambda_coeff
    
    def _update_history(self, lambda_coeff: torch.Tensor, loss: torch.Tensor):
        """更新历史信息"""
        step = self.step_count.item()
        idx = step % self.memory_size
        
        self.lambda_history[idx] = lambda_coeff.detach()
        self.loss_history[idx] = loss.detach()
        self.step_count += 1
    
    def _compute_adaptive_adjustment(self) -> torch.Tensor:
        """计算自适应调整"""
        # 基于历史信息计算调整量
        recent_lambda = self.lambda_history[-10:]  # 最近10步的λ
        recent_loss = self.loss_history[-10:]      # 最近10步的损失
        
        if len(recent_lambda) < 2:
            return torch.zeros_like(self.lambda_history[0])
        
        # 计算损失趋势
        loss_trend = torch.mean(torch.diff(recent_loss))
        
        # 计算λ趋势
        lambda_trend = torch.mean(torch.diff(recent_lambda, dim=0), dim=0)
        
        # 自适应调整：如果损失上升，调整λ策略
        if loss_trend > 0:
            adjustment = -lambda_trend * 0.1  # 反向调整
        else:
            adjustment = lambda_trend * 0.05  # 继续当前趋势
        
        return adjustment
