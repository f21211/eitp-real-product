"""
EIT-P 简化核心模块
基于IEM理论的涌现智能框架 - 简化实现
"""

__version__ = "2.0.0"
__author__ = "EIT-P Team"
__description__ = "Emergent Intelligence Transformer Prototype - 基于IEM理论的AI训练框架"

# 导入核心模块
from .experiments import ExperimentManager
from .models import ModelRegistry
from .metrics import MetricsTracker
from .security import SecurityManager
from .compression import ModelCompressor
from .optimization import HyperparameterOptimizer
from .distributed import DistributedTrainer
from .ab_testing import ABTestManager

__all__ = [
    'ExperimentManager',
    'ModelRegistry', 
    'MetricsTracker',
    'SecurityManager',
    'ModelCompressor',
    'HyperparameterOptimizer',
    'DistributedTrainer',
    'ABTestManager'
]
