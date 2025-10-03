"""
Path-Norm正则化模块
实现1-路径范数计算，作为正则化项，鼓励网络权重空间的有序化拓扑
对应IEM理论中的 λ·EC (结构复杂度D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class PathNormRegularizer(nn.Module):
    """
    Path-Norm正则化器
    
    实现1-路径范数计算，用于鼓励网络权重空间的有序化拓扑结构。
    这是IEM理论中复杂度有序能(EC)调控的核心组件。
    
    Args:
        target_fractal_dim: 目标分形维度 (D≥2.7)
        path_norm_weight: 路径范数权重系数
        temperature: 温度参数，用于控制正则化强度
    """
    
    def __init__(
        self,
        target_fractal_dim: float = 2.7,
        path_norm_weight: float = 1.0,
        temperature: float = 1.0
    ):
        super().__init__()
        self.target_fractal_dim = target_fractal_dim
        self.path_norm_weight = path_norm_weight
        self.temperature = temperature
        
    def compute_path_norm(self, weights: torch.Tensor) -> torch.Tensor:
        """
        计算1-路径范数
        
        Args:
            weights: 权重张量 [out_features, in_features]
            
        Returns:
            path_norm: 1-路径范数值
        """
        # 计算每行的L1范数
        row_norms = torch.norm(weights, p=1, dim=1)  # [out_features]
        
        # 1-路径范数 = sum of row L1 norms
        path_norm = torch.sum(row_norms)
        
        return path_norm
    
    def compute_fractal_dimension(self, weights: torch.Tensor) -> torch.Tensor:
        """
        估算权重矩阵的分形维度
        
        Args:
            weights: 权重张量 [out_features, in_features]
            
        Returns:
            fractal_dim: 估算的分形维度
        """
        # 将权重矩阵转换为二值化表示
        binary_weights = (weights != 0).float()
        
        # 计算不同尺度下的盒子计数
        scales = [1, 2, 4, 8, 16]
        box_counts = []
        
        for scale in scales:
            if scale >= min(weights.shape):
                continue
                
            # 下采样到指定尺度
            h, w = weights.shape
            new_h, new_w = h // scale, w // scale
            
            if new_h == 0 or new_w == 0:
                continue
                
            # 使用平均池化进行下采样
            downsampled = F.avg_pool2d(
                binary_weights.unsqueeze(0).unsqueeze(0),
                kernel_size=scale,
                stride=scale
            ).squeeze()
            
            # 计算非零盒子数量
            box_count = torch.sum(downsampled > 0).float()
            box_counts.append(box_count)
        
        if len(box_counts) < 2:
            return torch.tensor(1.0, device=weights.device)
        
        # 使用最小二乘法拟合log-log关系
        scales_tensor = torch.tensor(scales[:len(box_counts)], 
                                   dtype=torch.float32, 
                                   device=weights.device)
        box_counts_tensor = torch.stack(box_counts)
        
        # 避免log(0)
        log_scales = torch.log(scales_tensor)
        log_counts = torch.log(box_counts_tensor + 1e-8)
        
        # 线性回归计算斜率
        n = len(log_scales)
        sum_x = torch.sum(log_scales)
        sum_y = torch.sum(log_counts)
        sum_xy = torch.sum(log_scales * log_counts)
        sum_x2 = torch.sum(log_scales ** 2)
        
        # 斜率 = -分形维度
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        fractal_dim = -slope
        
        return torch.clamp(fractal_dim, min=1.0, max=3.0)
    
    def compute_self_similarity(self, weights: torch.Tensor) -> torch.Tensor:
        """
        计算权重矩阵的自相似性率
        
        Args:
            weights: 权重张量 [out_features, in_features]
            
        Returns:
            ss_rate: 自相似性率
        """
        # 计算不同尺度下的相似性
        scales = [1, 2, 4]
        similarities = []
        
        for scale in scales:
            if scale >= min(weights.shape):
                continue
                
            h, w = weights.shape
            new_h, new_w = h // scale, w // scale
            
            if new_h == 0 or new_w == 0:
                continue
                
            # 下采样
            downsampled = F.avg_pool2d(
                weights.unsqueeze(0).unsqueeze(0),
                kernel_size=scale,
                stride=scale
            ).squeeze()
            
            # 上采样回原尺寸
            upsampled = F.interpolate(
                downsampled.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # 计算相似性
            similarity = F.cosine_similarity(
                weights.flatten(),
                upsampled.flatten(),
                dim=0
            )
            similarities.append(similarity)
        
        if not similarities:
            return torch.tensor(0.0, device=weights.device)
        
        # 平均相似性率
        ss_rate = torch.mean(torch.stack(similarities))
        return ss_rate
    
    def forward(
        self, 
        model: nn.Module, 
        lambda_coeff: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算Path-Norm正则化损失
        
        Args:
            model: 要正则化的模型
            lambda_coeff: 动态复杂度系数λ，如果为None则使用默认权重
            
        Returns:
            losses: 包含各种损失项的字典
        """
        if lambda_coeff is None:
            lambda_coeff = torch.tensor(self.path_norm_weight, device=next(model.parameters()).device)
        
        total_path_norm = 0.0
        total_fractal_dim = 0.0
        total_ss_rate = 0.0
        layer_count = 0
        
        # 遍历模型的所有线性层
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                weights = module.weight
                
                # 计算路径范数
                path_norm = self.compute_path_norm(weights)
                total_path_norm += path_norm
                
                # 计算分形维度
                fractal_dim = self.compute_fractal_dimension(weights)
                total_fractal_dim += fractal_dim
                
                # 计算自相似性率
                ss_rate = self.compute_self_similarity(weights)
                total_ss_rate += ss_rate
                
                layer_count += 1
        
        if layer_count == 0:
            return {
                'path_norm_loss': torch.tensor(0.0, device=next(model.parameters()).device),
                'fractal_dim_loss': torch.tensor(0.0, device=next(model.parameters()).device),
                'ss_rate_loss': torch.tensor(0.0, device=next(model.parameters()).device),
                'total_loss': torch.tensor(0.0, device=next(model.parameters()).device)
            }
        
        # 平均化
        avg_path_norm = total_path_norm / layer_count
        avg_fractal_dim = total_fractal_dim / layer_count
        avg_ss_rate = total_ss_rate / layer_count
        
        # 计算各项损失
        path_norm_loss = lambda_coeff * avg_path_norm
        
        # 分形维度损失：鼓励达到目标分形维度
        fractal_dim_loss = F.mse_loss(
            avg_fractal_dim, 
            torch.tensor(self.target_fractal_dim, device=avg_fractal_dim.device)
        )
        
        # 自相似性率损失：鼓励高自相似性
        ss_rate_loss = -avg_ss_rate  # 负号表示最大化
        
        # 总损失
        total_loss = path_norm_loss + fractal_dim_loss + ss_rate_loss
        
        return {
            'path_norm_loss': path_norm_loss,
            'fractal_dim_loss': fractal_dim_loss,
            'ss_rate_loss': ss_rate_loss,
            'total_loss': total_loss,
            'avg_fractal_dim': avg_fractal_dim,
            'avg_ss_rate': avg_ss_rate
        }
    
    def get_complexity_metrics(self, model: nn.Module) -> Dict[str, float]:
        """
        获取模型的复杂度指标
        
        Args:
            model: 要分析的模型
            
        Returns:
            metrics: 复杂度指标字典
        """
        total_path_norm = 0.0
        total_fractal_dim = 0.0
        total_ss_rate = 0.0
        layer_count = 0
        
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    weights = module.weight
                    
                    path_norm = self.compute_path_norm(weights)
                    total_path_norm += path_norm.item()
                    
                    fractal_dim = self.compute_fractal_dimension(weights)
                    total_fractal_dim += fractal_dim.item()
                    
                    ss_rate = self.compute_self_similarity(weights)
                    total_ss_rate += ss_rate.item()
                    
                    layer_count += 1
        
        if layer_count == 0:
            return {
                'avg_path_norm': 0.0,
                'avg_fractal_dim': 0.0,
                'avg_ss_rate': 0.0,
                'layer_count': 0
            }
        
        return {
            'avg_path_norm': total_path_norm / layer_count,
            'avg_fractal_dim': total_fractal_dim / layer_count,
            'avg_ss_rate': total_ss_rate / layer_count,
            'layer_count': layer_count
        }
