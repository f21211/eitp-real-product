"""
EIT-P 模型压缩模块
提供模型量化、剪枝和蒸馏功能
"""

from .quantization import QuantizationManager, QuantizedEITP
from .pruning import PruningManager, PrunedEITP
from .distillation import DistillationManager, DistilledEITP
from .compression_utils import CompressionUtils

__all__ = [
    "QuantizationManager",
    "QuantizedEITP",
    "PruningManager", 
    "PrunedEITP",
    "DistillationManager",
    "DistilledEITP",
    "CompressionUtils"
]
