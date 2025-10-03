"""
分布式同步管理器
处理多GPU、多节点间的同步和协调
"""

import torch
import torch.distributed as dist
from typing import Dict, Any, List, Optional, Callable
import time
import threading
from collections import defaultdict

from ..utils import get_global_logger


class SynchronizationManager:
    """分布式同步管理器"""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.logger = get_global_logger()
        
        # 同步状态
        self.sync_state = {
            'barriers': 0,
            'reductions': 0,
            'broadcasts': 0,
            'total_sync_time': 0.0
        }
        
        # 同步锁
        self.sync_lock = threading.Lock()
        
        # 同步回调
        self.sync_callbacks = defaultdict(list)
    
    def barrier(self, name: str = "default"):
        """同步屏障"""
        with self.sync_lock:
            start_time = time.time()
            
            try:
                dist.barrier()
                self.sync_state['barriers'] += 1
                
                # 执行同步回调
                for callback in self.sync_callbacks.get('barrier', []):
                    try:
                        callback(name, self.rank, self.world_size)
                    except Exception as e:
                        self.logger.warning(f"同步回调执行失败: {e}")
                
            except Exception as e:
                self.logger.error(f"Barrier同步失败: {e}")
                raise
            
            end_time = time.time()
            self.sync_state['total_sync_time'] += (end_time - start_time)
    
    def reduce(self, tensor: torch.Tensor, dst: int = 0, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        """归约操作"""
        with self.sync_lock:
            start_time = time.time()
            
            try:
                dist.reduce(tensor, dst=dst, op=op)
                self.sync_state['reductions'] += 1
                
                # 执行归约回调
                for callback in self.sync_callbacks.get('reduce', []):
                    try:
                        callback(tensor, dst, op, self.rank, self.world_size)
                    except Exception as e:
                        self.logger.warning(f"归约回调执行失败: {e}")
                
            except Exception as e:
                self.logger.error(f"Reduce同步失败: {e}")
                raise
            
            end_time = time.time()
            self.sync_state['total_sync_time'] += (end_time - start_time)
            
            return tensor
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """广播操作"""
        with self.sync_lock:
            start_time = time.time()
            
            try:
                dist.broadcast(tensor, src=src)
                self.sync_state['broadcasts'] += 1
                
                # 执行广播回调
                for callback in self.sync_callbacks.get('broadcast', []):
                    try:
                        callback(tensor, src, self.rank, self.world_size)
                    except Exception as e:
                        self.logger.warning(f"广播回调执行失败: {e}")
                
            except Exception as e:
                self.logger.error(f"Broadcast同步失败: {e}")
                raise
            
            end_time = time.time()
            self.sync_state['total_sync_time'] += (end_time - start_time)
            
            return tensor
    
    def all_reduce(self, tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        """全归约操作"""
        with self.sync_lock:
            start_time = time.time()
            
            try:
                dist.all_reduce(tensor, op=op)
                self.sync_state['reductions'] += 1
                
                # 执行全归约回调
                for callback in self.sync_callbacks.get('all_reduce', []):
                    try:
                        callback(tensor, op, self.rank, self.world_size)
                    except Exception as e:
                        self.logger.warning(f"全归约回调执行失败: {e}")
                
            except Exception as e:
                self.logger.error(f"All-reduce同步失败: {e}")
                raise
            
            end_time = time.time()
            self.sync_state['total_sync_time'] += (end_time - start_time)
            
            return tensor
    
    def gather(self, tensor: torch.Tensor, dst: int = 0) -> Optional[List[torch.Tensor]]:
        """收集操作"""
        with self.sync_lock:
            start_time = time.time()
            
            try:
                if self.rank == dst:
                    gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
                    dist.gather(tensor, gather_list, dst=dst)
                    result = gather_list
                else:
                    dist.gather(tensor, dst=dst)
                    result = None
                
                # 执行收集回调
                for callback in self.sync_callbacks.get('gather', []):
                    try:
                        callback(tensor, dst, result, self.rank, self.world_size)
                    except Exception as e:
                        self.logger.warning(f"收集回调执行失败: {e}")
                
            except Exception as e:
                self.logger.error(f"Gather同步失败: {e}")
                raise
            
            end_time = time.time()
            self.sync_state['total_sync_time'] += (end_time - start_time)
            
            return result
    
    def scatter(self, tensor_list: List[torch.Tensor], src: int = 0) -> torch.Tensor:
        """分散操作"""
        with self.sync_lock:
            start_time = time.time()
            
            try:
                if self.rank == src:
                    output_tensor = torch.zeros_like(tensor_list[0])
                    dist.scatter(output_tensor, tensor_list, src=src)
                else:
                    output_tensor = torch.zeros_like(tensor_list[0])
                    dist.scatter(output_tensor, src=src)
                
                # 执行分散回调
                for callback in self.sync_callbacks.get('scatter', []):
                    try:
                        callback(tensor_list, src, output_tensor, self.rank, self.world_size)
                    except Exception as e:
                        self.logger.warning(f"分散回调执行失败: {e}")
                
            except Exception as e:
                self.logger.error(f"Scatter同步失败: {e}")
                raise
            
            end_time = time.time()
            self.sync_state['total_sync_time'] += (end_time - start_time)
            
            return output_tensor
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """全收集操作"""
        with self.sync_lock:
            start_time = time.time()
            
            try:
                gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
                dist.all_gather(gather_list, tensor)
                
                # 执行全收集回调
                for callback in self.sync_callbacks.get('all_gather', []):
                    try:
                        callback(tensor, gather_list, self.rank, self.world_size)
                    except Exception as e:
                        self.logger.warning(f"全收集回调执行失败: {e}")
                
            except Exception as e:
                self.logger.error(f"All-gather同步失败: {e}")
                raise
            
            end_time = time.time()
            self.sync_state['total_sync_time'] += (end_time - start_time)
            
            return gather_list
    
    def reduce_scatter(self, tensor_list: List[torch.Tensor], op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        """归约分散操作"""
        with self.sync_lock:
            start_time = time.time()
            
            try:
                output_tensor = torch.zeros_like(tensor_list[0])
                dist.reduce_scatter(output_tensor, tensor_list, op=op)
                
                # 执行归约分散回调
                for callback in self.sync_callbacks.get('reduce_scatter', []):
                    try:
                        callback(tensor_list, op, output_tensor, self.rank, self.world_size)
                    except Exception as e:
                        self.logger.warning(f"归约分散回调执行失败: {e}")
                
            except Exception as e:
                self.logger.error(f"Reduce-scatter同步失败: {e}")
                raise
            
            end_time = time.time()
            self.sync_state['total_sync_time'] += (end_time - start_time)
            
            return output_tensor
    
    def add_sync_callback(self, sync_type: str, callback: Callable):
        """添加同步回调"""
        self.sync_callbacks[sync_type].append(callback)
        self.logger.debug(f"添加同步回调: {sync_type}")
    
    def remove_sync_callback(self, sync_type: str, callback: Callable):
        """移除同步回调"""
        if callback in self.sync_callbacks[sync_type]:
            self.sync_callbacks[sync_type].remove(callback)
            self.logger.debug(f"移除同步回调: {sync_type}")
    
    def clear_sync_callbacks(self, sync_type: Optional[str] = None):
        """清空同步回调"""
        if sync_type:
            self.sync_callbacks[sync_type].clear()
            self.logger.debug(f"清空同步回调: {sync_type}")
        else:
            self.sync_callbacks.clear()
            self.logger.debug("清空所有同步回调")
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """获取同步统计信息"""
        return {
            'rank': self.rank,
            'world_size': self.world_size,
            'stats': self.sync_state.copy(),
            'avg_sync_time': (
                self.sync_state['total_sync_time'] / 
                max(1, sum([
                    self.sync_state['barriers'],
                    self.sync_state['reductions'],
                    self.sync_state['broadcasts']
                ]))
            )
        }
    
    def reset_stats(self):
        """重置同步统计"""
        self.sync_state = {
            'barriers': 0,
            'reductions': 0,
            'broadcasts': 0,
            'total_sync_time': 0.0
        }
    
    def log_sync_stats(self):
        """记录同步统计"""
        stats = self.get_sync_stats()
        self.logger.info(f"同步统计 - Rank {self.rank}: {stats}")
    
    def wait_for_all(self, timeout: Optional[float] = None):
        """等待所有进程完成"""
        if timeout:
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    self.barrier("wait_for_all")
                    return True
                except:
                    time.sleep(0.1)
            return False
        else:
            self.barrier("wait_for_all")
            return True
    
    def synchronize_gradients(self, model: torch.nn.Module):
        """同步模型梯度"""
        for param in model.parameters():
            if param.grad is not None:
                self.all_reduce(param.grad)
    
    def synchronize_parameters(self, model: torch.nn.Module):
        """同步模型参数"""
        for param in model.parameters():
            self.broadcast(param.data)
    
    def create_sync_group(self, ranks: List[int]) -> dist.ProcessGroup:
        """创建同步组"""
        try:
            group = dist.new_group(ranks)
            self.logger.info(f"创建同步组: {ranks}")
            return group
        except Exception as e:
            self.logger.error(f"创建同步组失败: {e}")
            raise
    
    def destroy_sync_group(self, group: dist.ProcessGroup):
        """销毁同步组"""
        try:
            dist.destroy_process_group(group)
            self.logger.info("销毁同步组")
        except Exception as e:
            self.logger.error(f"销毁同步组失败: {e}")
            raise
