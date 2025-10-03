"""
总损失函数
实现EIT-P的复合损失函数LTotal，整合所有IEM理论要求的损失项
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math

from ..regularization import PathNormRegularizer, EntropyRegularizer, ChaosRegularizer
from .coherence_loss import CoherenceLoss
from .thermodynamic_loss import ThermodynamicLoss


class TotalLoss(nn.Module):
    """
    EIT-P总损失函数
    
    实现IEM理论要求的复合损失函数：
    LTotal(θ,λ) = LCE(θ) + REntropy + RChaos + λ·RPath-Norm + LE-Cost
    
    Args:
        vocab_size: 词汇表大小
        hidden_dim: 隐藏状态维度
        target_lyapunov: 目标Lyapunov指数
        target_fractal_dim: 目标分形维度
        beta: 熵正则化权重
        gamma: 互信息正则化权重
        delta: 混沌正则化权重
        temperature: 温度参数
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        target_lyapunov: float = 0.0,
        target_fractal_dim: float = 2.7,
        beta: float = 1.0,
        gamma: float = 0.1,
        delta: float = 1.0,
        temperature: float = 1.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.target_lyapunov = target_lyapunov
        self.target_fractal_dim = target_fractal_dim
        
        # 初始化各个正则化器
        self.path_norm_reg = PathNormRegularizer(
            target_fractal_dim=target_fractal_dim,
            path_norm_weight=1.0,
            temperature=temperature
        )
        
        self.entropy_reg = EntropyRegularizer(
            hidden_dim=hidden_dim,
            beta=beta,
            gamma=gamma,
            target_entropy=0.0
        )
        
        self.chaos_reg = ChaosRegularizer(
            target_lyapunov=target_lyapunov,
            chaos_weight=delta,
            stability_threshold=0.1
        )
        
        # 其他损失函数
        self.coherence_loss = CoherenceLoss()
        self.thermodynamic_loss = ThermodynamicLoss()
        
        # 损失权重
        self.ce_weight = 1.0
        self.path_norm_weight = 1.0
        self.entropy_weight = 1.0
        self.chaos_weight = 1.0
        self.coherence_weight = 0.1
        self.thermodynamic_weight = 0.1
    
    def forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        lambda_coeff: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        input_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            model: 主模型
            input_ids: 输入token IDs [batch_size, seq_len]
            labels: 标签 [batch_size, seq_len]
            lambda_coeff: 动态复杂度系数λ [batch_size, 1]
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_dim]
            input_embeddings: 输入嵌入 [batch_size, seq_len, hidden_dim]
            
        Returns:
            losses: 包含所有损失项的字典
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # 1. 交叉熵损失 LCE(θ)
        if hasattr(model, 'transformer'):
            # GPT-2模型
            outputs = model(input_ids, labels=labels)
            ce_loss = outputs.loss
            if hidden_states is None:
                hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
        else:
            # 其他模型
            outputs = model(input_ids)
            # 处理outputs可能是对象或张量的情况
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif hasattr(outputs, 'view'):
                logits = outputs
            else:
                # 如果outputs是CausalLMOutputWithCrossAttentions对象
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            if hidden_states is None:
                hidden_states = logits
        
        # 2. Path-Norm正则化 λ·RPath-Norm
        if lambda_coeff is None:
            lambda_coeff = torch.ones(batch_size, 1, device=device)
        
        path_norm_losses = self.path_norm_reg(model, lambda_coeff)
        path_norm_loss = path_norm_losses['total_loss']
        
        # 3. 微分熵正则化 REntropy
        if hidden_states is not None:
            entropy_losses = self.entropy_reg(hidden_states, input_embeddings)
            entropy_loss = entropy_losses['total_loss']
        else:
            entropy_loss = torch.tensor(0.0, device=device)
        
        # 4. 混沌正则化 RChaos
        chaos_losses = self.chaos_reg(model, input_ids, hidden_states)
        chaos_loss = chaos_losses['total_loss']
        
        # 5. 连贯性损失
        coherence_loss = self.coherence_loss(input_ids, labels, model)
        
        # 6. 热力学损失
        thermodynamic_loss = self.thermodynamic_loss(
            model, hidden_states, lambda_coeff
        )
        
        # 计算总损失
        total_loss = (
            self.ce_weight * ce_loss +
            self.path_norm_weight * path_norm_loss +
            self.entropy_weight * entropy_loss +
            self.chaos_weight * chaos_loss +
            self.coherence_weight * coherence_loss +
            self.thermodynamic_weight * thermodynamic_loss
        )
        
        # 返回详细损失信息
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'path_norm_loss': path_norm_loss,
            'entropy_loss': entropy_loss,
            'chaos_loss': chaos_loss,
            'coherence_loss': coherence_loss,
            'thermodynamic_loss': thermodynamic_loss,
            'lambda_mean': torch.mean(lambda_coeff).item(),
            'lambda_std': torch.std(lambda_coeff).item(),
            # 详细的正则化损失
            'path_norm_details': path_norm_losses,
            'entropy_details': entropy_losses if hidden_states is not None else {},
            'chaos_details': chaos_losses
        }
    
    def compute_iem_metrics(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        计算IEM理论相关的指标
        
        Args:
            model: 主模型
            input_ids: 输入token IDs
            hidden_states: 隐藏状态
            
        Returns:
            metrics: IEM指标字典
        """
        device = input_ids.device
        
        if hidden_states is None:
            if hasattr(model, 'transformer'):
                outputs = model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = model(input_ids)
        
        # 获取各模块的指标
        path_norm_metrics = self.path_norm_reg.get_complexity_metrics(model)
        entropy_metrics = self.entropy_reg.get_entropy_metrics(hidden_states)
        chaos_metrics = self.chaos_reg.get_chaos_metrics(model, input_ids, hidden_states)
        
        # 计算IEM理论指标
        complexity_energy = path_norm_metrics['avg_path_norm']
        entropy_change = entropy_metrics['negative_entropy_rate']
        dynamic_sensitivity = chaos_metrics['dynamic_sensitivity']
        
        # 计算修正质能方程 E = mc² + ΔEF + ΔES + λ·EC
        # 这里简化为相对指标
        iem_balance = entropy_change + complexity_energy - dynamic_sensitivity
        
        return {
            'complexity_energy': complexity_energy,
            'entropy_change': entropy_change,
            'dynamic_sensitivity': dynamic_sensitivity,
            'iem_balance': iem_balance,
            'fractal_dimension': path_norm_metrics['avg_fractal_dim'],
            'self_similarity': path_norm_metrics['avg_ss_rate'],
            'lyapunov_exponent': chaos_metrics['max_lyapunov'],
            'is_edge_chaos': chaos_metrics['is_edge_chaos'],
            'is_critical': chaos_metrics['is_critical'],
            'criticality_score': chaos_metrics['criticality_score']
        }
    
    def update_weights(
        self,
        ce_weight: Optional[float] = None,
        path_norm_weight: Optional[float] = None,
        entropy_weight: Optional[float] = None,
        chaos_weight: Optional[float] = None,
        coherence_weight: Optional[float] = None,
        thermodynamic_weight: Optional[float] = None
    ):
        """
        更新损失权重
        
        Args:
            ce_weight: 交叉熵损失权重
            path_norm_weight: Path-Norm正则化权重
            entropy_weight: 熵正则化权重
            chaos_weight: 混沌正则化权重
            coherence_weight: 连贯性损失权重
            thermodynamic_weight: 热力学损失权重
        """
        if ce_weight is not None:
            self.ce_weight = ce_weight
        if path_norm_weight is not None:
            self.path_norm_weight = path_norm_weight
        if entropy_weight is not None:
            self.entropy_weight = entropy_weight
        if chaos_weight is not None:
            self.chaos_weight = chaos_weight
        if coherence_weight is not None:
            self.coherence_weight = coherence_weight
        if thermodynamic_weight is not None:
            self.thermodynamic_weight = thermodynamic_weight
    
    def get_loss_weights(self) -> Dict[str, float]:
        """
        获取当前损失权重
        
        Returns:
            weights: 损失权重字典
        """
        return {
            'ce_weight': self.ce_weight,
            'path_norm_weight': self.path_norm_weight,
            'entropy_weight': self.entropy_weight,
            'chaos_weight': self.chaos_weight,
            'coherence_weight': self.coherence_weight,
            'thermodynamic_weight': self.thermodynamic_weight
        }
