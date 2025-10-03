"""
压缩工具模块
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import numpy as np

from ..utils import get_global_logger


class CompressionUtils:
    """压缩工具类"""
    
    def __init__(self):
        self.logger = get_global_logger()
    
    def get_model_size(self, model: nn.Module) -> int:
        """获取模型大小（字节）"""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size
    
    def get_model_parameters(self, model: nn.Module) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in model.parameters())
    
    def calculate_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """计算压缩比"""
        if compressed_size == 0:
            return float('inf')
        return original_size / compressed_size
    
    def analyze_model_structure(self, model: nn.Module) -> Dict[str, Any]:
        """分析模型结构"""
        structure = {
            'total_parameters': self.get_model_parameters(model),
            'total_size': self.get_model_size(model),
            'layers': []
        }
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                layer_info = {
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters()),
                    'size': sum(p.numel() * p.element_size() for p in module.parameters())
                }
                structure['layers'].append(layer_info)
        
        return structure
