"""
分布式通信管理器
处理多GPU、多节点间的通信和同步
"""

import torch
import torch.distributed as dist
from typing import Dict, Any, List, Optional
import time
import json
from pathlib import Path

from ..utils import get_global_logger


class CommunicationManager:
    """分布式通信管理器"""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.logger = get_global_logger()
        
        # 通信统计
        self.communication_stats = {
            'all_reduce_calls': 0,
            'broadcast_calls': 0,
            'gather_calls': 0,
            'scatter_calls': 0,
            'total_communication_time': 0.0
        }
    
    def all_reduce(self, tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        """全归约操作"""
        start_time = time.time()
        
        try:
            dist.all_reduce(tensor, op=op)
            self.communication_stats['all_reduce_calls'] += 1
        except Exception as e:
            self.logger.error(f"All-reduce操作失败: {e}")
            raise
        
        end_time = time.time()
        self.communication_stats['total_communication_time'] += (end_time - start_time)
        
        return tensor
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """广播操作"""
        start_time = time.time()
        
        try:
            dist.broadcast(tensor, src=src)
            self.communication_stats['broadcast_calls'] += 1
        except Exception as e:
            self.logger.error(f"Broadcast操作失败: {e}")
            raise
        
        end_time = time.time()
        self.communication_stats['total_communication_time'] += (end_time - start_time)
        
        return tensor
    
    def gather(self, tensor: torch.Tensor, dst: int = 0) -> Optional[List[torch.Tensor]]:
        """收集操作"""
        start_time = time.time()
        
        try:
            if self.rank == dst:
                gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
                dist.gather(tensor, gather_list, dst=dst)
                result = gather_list
            else:
                dist.gather(tensor, dst=dst)
                result = None
            
            self.communication_stats['gather_calls'] += 1
        except Exception as e:
            self.logger.error(f"Gather操作失败: {e}")
            raise
        
        end_time = time.time()
        self.communication_stats['total_communication_time'] += (end_time - start_time)
        
        return result
    
    def scatter(self, tensor_list: List[torch.Tensor], src: int = 0) -> torch.Tensor:
        """分散操作"""
        start_time = time.time()
        
        try:
            if self.rank == src:
                output_tensor = torch.zeros_like(tensor_list[0])
                dist.scatter(output_tensor, tensor_list, src=src)
            else:
                output_tensor = torch.zeros_like(tensor_list[0])
                dist.scatter(output_tensor, src=src)
            
            self.communication_stats['scatter_calls'] += 1
        except Exception as e:
            self.logger.error(f"Scatter操作失败: {e}")
            raise
        
        end_time = time.time()
        self.communication_stats['total_communication_time'] += (end_time - start_time)
        
        return output_tensor
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """全收集操作"""
        start_time = time.time()
        
        try:
            gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(gather_list, tensor)
            
            self.communication_stats['gather_calls'] += 1
        except Exception as e:
            self.logger.error(f"All-gather操作失败: {e}")
            raise
        
        end_time = time.time()
        self.communication_stats['total_communication_time'] += (end_time - start_time)
        
        return gather_list
    
    def reduce_scatter(self, tensor_list: List[torch.Tensor], op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        """归约分散操作"""
        start_time = time.time()
        
        try:
            output_tensor = torch.zeros_like(tensor_list[0])
            dist.reduce_scatter(output_tensor, tensor_list, op=op)
            
            self.communication_stats['all_reduce_calls'] += 1
        except Exception as e:
            self.logger.error(f"Reduce-scatter操作失败: {e}")
            raise
        
        end_time = time.time()
        self.communication_stats['total_communication_time'] += (end_time - start_time)
        
        return output_tensor
    
    def barrier(self):
        """同步屏障"""
        start_time = time.time()
        
        try:
            dist.barrier()
        except Exception as e:
            self.logger.error(f"Barrier操作失败: {e}")
            raise
        
        end_time = time.time()
        self.communication_stats['total_communication_time'] += (end_time - start_time)
    
    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0):
        """发送操作"""
        start_time = time.time()
        
        try:
            dist.send(tensor, dst=dst, tag=tag)
        except Exception as e:
            self.logger.error(f"Send操作失败: {e}")
            raise
        
        end_time = time.time()
        self.communication_stats['total_communication_time'] += (end_time - start_time)
    
    def recv(self, tensor: torch.Tensor, src: int, tag: int = 0):
        """接收操作"""
        start_time = time.time()
        
        try:
            dist.recv(tensor, src=src, tag=tag)
        except Exception as e:
            self.logger.error(f"Recv操作失败: {e}")
            raise
        
        end_time = time.time()
        self.communication_stats['total_communication_time'] += (end_time - start_time)
    
    def isend(self, tensor: torch.Tensor, dst: int, tag: int = 0):
        """异步发送操作"""
        start_time = time.time()
        
        try:
            return dist.isend(tensor, dst=dst, tag=tag)
        except Exception as e:
            self.logger.error(f"Async send操作失败: {e}")
            raise
    
    def irecv(self, tensor: torch.Tensor, src: int, tag: int = 0):
        """异步接收操作"""
        start_time = time.time()
        
        try:
            return dist.irecv(tensor, src=src, tag=tag)
        except Exception as e:
            self.logger.error(f"Async recv操作失败: {e}")
            raise
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """获取通信统计信息"""
        return {
            'rank': self.rank,
            'world_size': self.world_size,
            'stats': self.communication_stats.copy(),
            'avg_communication_time': (
                self.communication_stats['total_communication_time'] / 
                max(1, sum([
                    self.communication_stats['all_reduce_calls'],
                    self.communication_stats['broadcast_calls'],
                    self.communication_stats['gather_calls'],
                    self.communication_stats['scatter_calls']
                ]))
            )
        }
    
    def reset_stats(self):
        """重置通信统计"""
        self.communication_stats = {
            'all_reduce_calls': 0,
            'broadcast_calls': 0,
            'gather_calls': 0,
            'scatter_calls': 0,
            'total_communication_time': 0.0
        }
    
    def log_communication_stats(self):
        """记录通信统计"""
        stats = self.get_communication_stats()
        self.logger.info(f"通信统计 - Rank {self.rank}: {json.dumps(stats, indent=2)}")
    
    def optimize_communication(self, tensor_size: int, world_size: int) -> Dict[str, Any]:
        """优化通信策略"""
        # 根据张量大小和世界大小选择最优通信策略
        if tensor_size < 1024:  # 小张量
            return {
                'strategy': 'all_reduce',
                'chunk_size': tensor_size,
                'overlap': False
            }
        elif tensor_size < 1024 * 1024:  # 中等张量
            return {
                'strategy': 'all_reduce',
                'chunk_size': min(1024, tensor_size),
                'overlap': True
            }
        else:  # 大张量
            return {
                'strategy': 'reduce_scatter',
                'chunk_size': tensor_size // world_size,
                'overlap': True
            }
    
    def benchmark_communication(self, tensor_sizes: List[int], iterations: int = 10) -> Dict[str, Any]:
        """通信性能基准测试"""
        results = {}
        
        for size in tensor_sizes:
            tensor = torch.randn(size, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # 测试all_reduce
            times = []
            for _ in range(iterations):
                start = time.time()
                self.all_reduce(tensor.clone())
                times.append(time.time() - start)
            
            results[f'all_reduce_{size}'] = {
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        
        return results
