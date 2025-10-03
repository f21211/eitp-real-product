"""
模型并行EIT-P训练器
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from ..utils import get_global_logger


class ModelParallelEITP(nn.Module):
    """模型并行EIT-P训练器"""
    
    def __init__(self, module, device_ids=None):
        super().__init__()
        self.module = module
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.logger = get_global_logger()
    
    def forward(self, *inputs, **kwargs):
        """前向传播"""
        return self.module(*inputs, **kwargs)
