"""
EIT-P 分布式训练模块
提供多GPU、多节点分布式训练支持
"""

from .distributed_trainer import DistributedEITPTrainer
from .data_parallel import DataParallelEITP
from .model_parallel import ModelParallelEITP
from .communication import CommunicationManager
from .synchronization import SynchronizationManager

__all__ = [
    "DistributedEITPTrainer",
    "DataParallelEITP", 
    "ModelParallelEITP",
    "CommunicationManager",
    "SynchronizationManager"
]
