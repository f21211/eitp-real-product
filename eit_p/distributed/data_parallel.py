"""
数据并行EIT-P训练器
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel as DP
from typing import Dict, Any, Optional

from ..utils import get_global_logger


class DataParallelEITP(DP):
    """数据并行EIT-P训练器"""
    
    def __init__(self, module=None, device_ids=None, output_device=None, dim=0):
        # 如果没有提供module，创建一个占位符
        if module is None:
            module = nn.Linear(1, 1)  # 简单的占位符模块
        
        super().__init__(module, device_ids, output_device, dim)
        self.logger = get_global_logger()
        self.logger.info("数据并行EIT-P训练器初始化完成")
    
    def forward(self, *inputs, **kwargs):
        """前向传播"""
        return super().forward(*inputs, **kwargs)
    
    def get_device_count(self):
        """获取可用设备数量"""
        return torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    def is_parallel_enabled(self):
        """检查是否启用了并行"""
        return len(self.device_ids) > 1
