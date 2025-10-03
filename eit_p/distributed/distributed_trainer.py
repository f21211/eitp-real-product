"""
分布式EIT-P训练器
支持多GPU、多节点分布式训练
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Any, Optional, List
import time
import json
from pathlib import Path

from ..training import EITPTrainer
from ..utils import get_global_logger
from .communication import CommunicationManager
from .synchronization import SynchronizationManager


class DistributedEITPTrainer(EITPTrainer):
    """分布式EIT-P训练器"""
    
    def __init__(self, model=None, args=None, train_dataset=None, eval_dataset=None, 
                 hypernetwork_params=None, config_manager=None, **kwargs):
        # 如果参数不足，创建一个简化的实例
        if model is None or args is None or train_dataset is None or eval_dataset is None or hypernetwork_params is None:
            self.logger = get_global_logger()
            self.logger.info("创建分布式训练器（简化模式）")
        else:
            super().__init__(model, args, train_dataset, eval_dataset, 
                            hypernetwork_params, config_manager, **kwargs)
        
        self.logger = get_global_logger()
        
        # 分布式训练参数
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.is_distributed = self.world_size > 1
        
        # 通信管理器
        self.comm_manager = CommunicationManager(self.rank, self.world_size)
        
        # 同步管理器
        self.sync_manager = SynchronizationManager(self.rank, self.world_size)
        
        # 分布式训练状态
        self.distributed_metrics = {}
        self.global_step = 0
        
        if self.is_distributed:
            self._setup_distributed()
    
    def _setup_distributed(self):
        """设置分布式训练环境"""
        try:
            # 初始化进程组
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl' if torch.cuda.is_available() else 'gloo',
                    init_method='env://',
                    world_size=self.world_size,
                    rank=self.rank
                )
            
            # 设置设备
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = f'cuda:{self.local_rank}'
            else:
                self.device = 'cpu'
            
            # 移动模型到设备
            self.model = self.model.to(self.device)
            
            # 包装模型为DDP
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=True
            )
            
            # 设置分布式采样器
            if hasattr(self.train_dataset, 'sampler'):
                self.train_dataset.sampler = DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True
                )
            
            self.logger.info(f"分布式训练初始化完成 - Rank: {self.rank}/{self.world_size}")
            
        except Exception as e:
            self.logger.error(f"分布式训练初始化失败: {e}")
            raise
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """分布式训练主循环"""
        if not self.is_distributed:
            return super().train(resume_from_checkpoint)
        
        try:
            self.logger.info(f"开始分布式训练 - Rank {self.rank}")
            
            # 同步所有进程
            self.sync_manager.barrier()
            
            # 训练循环
            for epoch in range(self.args.num_train_epochs):
                if hasattr(self.train_dataset, 'sampler'):
                    self.train_dataset.sampler.set_epoch(epoch)
                
                # 分布式训练步骤
                self._distributed_train_epoch(epoch)
                
                # 同步所有进程
                self.sync_manager.barrier()
                
                # 只在主进程保存检查点
                if self.rank == 0:
                    self._save_checkpoint(epoch)
            
            # 最终同步
            self.sync_manager.barrier()
            
            self.logger.info(f"分布式训练完成 - Rank {self.rank}")
            
        except Exception as e:
            self.logger.error(f"分布式训练失败 - Rank {self.rank}: {e}")
            raise
        finally:
            # 清理分布式环境
            if dist.is_initialized():
                dist.destroy_process_group()
    
    def _distributed_train_epoch(self, epoch: int):
        """分布式训练一个epoch"""
        self.model.train()
        
        for step, batch in enumerate(self.train_dataloader):
            # 移动数据到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(**batch)
            
            # 计算损失
            loss = self.compute_loss(outputs, batch)
            
            # 反向传播
            loss.backward()
            
            # 梯度同步
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.args.max_grad_norm
                    )
                
                # 优化器步骤
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # 更新学习率
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                
                # 更新全局步数
                self.global_step += 1
                
                # 记录指标
                self._log_distributed_metrics(epoch, step, loss.item())
                
                # 同步指标
                self._sync_metrics()
    
    def _log_distributed_metrics(self, epoch: int, step: int, loss: float):
        """记录分布式训练指标"""
        if self.rank == 0:  # 只在主进程记录
            self.distributed_metrics.update({
                'epoch': epoch,
                'step': step,
                'global_step': self.global_step,
                'loss': loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'world_size': self.world_size
            })
            
            # 记录到日志
            if step % self.args.logging_steps == 0:
                self.logger.info(
                    f"Epoch {epoch}, Step {step}, Global Step {self.global_step}, "
                    f"Loss: {loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
    
    def _sync_metrics(self):
        """同步分布式训练指标"""
        if not self.is_distributed:
            return
        
        # 收集所有进程的指标
        metrics_tensor = torch.tensor([
            self.distributed_metrics.get('loss', 0.0),
            self.distributed_metrics.get('learning_rate', 0.0),
            float(self.global_step)
        ], device=self.device)
        
        # 同步指标
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        # 计算平均值
        avg_metrics = metrics_tensor / self.world_size
        
        # 更新指标
        self.distributed_metrics.update({
            'avg_loss': avg_metrics[0].item(),
            'avg_learning_rate': avg_metrics[1].item(),
            'total_global_steps': int(avg_metrics[2].item())
        })
    
    def _save_checkpoint(self, epoch: int):
        """保存检查点（仅主进程）"""
        if self.rank != 0:
            return
        
        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型状态
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        torch.save(model_state, checkpoint_dir / "pytorch_model.bin")
        
        # 保存优化器状态
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        
        # 保存训练状态
        training_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'metrics': self.distributed_metrics,
            'world_size': self.world_size
        }
        
        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)
        
        self.logger.info(f"检查点已保存: {checkpoint_dir}")
    
    def evaluate(self, eval_dataset=None, ignore_keys=None):
        """分布式评估"""
        if not self.is_distributed:
            return super().evaluate(eval_dataset, ignore_keys)
        
        # 同步所有进程
        self.sync_manager.barrier()
        
        # 设置评估模式
        self.model.eval()
        
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            return {}
        
        # 分布式评估
        eval_metrics = {}
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in eval_dataset:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = self.compute_loss(outputs, batch)
                
                total_loss += loss.item() * batch['input_ids'].size(0)
                num_samples += batch['input_ids'].size(0)
        
        # 同步评估结果
        loss_tensor = torch.tensor(total_loss, device=self.device)
        samples_tensor = torch.tensor(num_samples, device=self.device)
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        
        # 计算平均损失
        avg_loss = loss_tensor.item() / samples_tensor.item()
        
        eval_metrics = {
            'eval_loss': avg_loss,
            'eval_samples': samples_tensor.item(),
            'world_size': self.world_size
        }
        
        # 同步所有进程
        self.sync_manager.barrier()
        
        return eval_metrics
    
    def get_distributed_info(self) -> Dict[str, Any]:
        """获取分布式训练信息"""
        return {
            'is_distributed': self.is_distributed,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'device': self.device,
            'global_step': self.global_step,
            'metrics': self.distributed_metrics
        }


def setup_distributed_training(backend: str = 'nccl'):
    """设置分布式训练环境"""
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        raise RuntimeError("分布式训练环境变量未设置")
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # 初始化进程组
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    return rank, world_size


def cleanup_distributed_training():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
