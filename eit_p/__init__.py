"""
EIT-P: Emergent Intelligence Transformer Prototype
基于物理学原理的涌现智能变压器原型

核心模块:
- regularization: 正则化模块 (Path-Norm, Entropy, Chaos)
- hypernetwork: 动态权重超网络
- losses: 损失函数集合
- evaluation: 评估指标
- training: 训练循环和优化器
"""

from .regularization import PathNormRegularizer, EntropyRegularizer, ChaosRegularizer
from .hypernetwork import DynamicHypernetwork
from .losses import TotalLoss, CoherenceLoss, ThermodynamicLoss
from .evaluation import EmergenceEvaluator
from .training import EITPTrainer

__version__ = "1.0.0"
__all__ = [
    "PathNormRegularizer",
    "EntropyRegularizer", 
    "ChaosRegularizer",
    "DynamicHypernetwork",
    "TotalLoss",
    "CoherenceLoss",
    "ThermodynamicLoss",
    "EmergenceEvaluator",
    "EITPTrainer"
]
