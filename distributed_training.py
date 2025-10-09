#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P Distributed Training
分布式训练模块
"""

import sys
import os
sys.path.append('/mnt/sda1/myproject/datainall/AGI')

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import logging
from typing import Dict, List
from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters

class DistributedTrainer:
    """分布式训练器"""
    
    def __init__(self, world_size: int = 1, rank: int = 0):
        self.world_size = world_size
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"DistributedTrainer-{self.rank}")
    
    def setup_distributed(self):
        """设置分布式环境"""
        if self.world_size > 1:
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="env://",
                world_size=self.world_size,
                rank=self.rank
            )
            torch.cuda.set_device(self.rank)
    
    def train_distributed(self, model: EnhancedCEPEITP, 
                         train_data: torch.Tensor, 
                         epochs: int = 100,
                         learning_rate: float = 0.001) -> Dict:
        """分布式训练"""
        self.logger.info(f"开始分布式训练 (Rank {self.rank}/{self.world_size})")
        
        # 设置分布式
        if self.world_size > 1:
            self.setup_distributed()
            model = model.to(self.device)
            model = DDP(model, device_ids=[self.rank])
        
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        # 训练循环
        training_history = []
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # 前向传播
            output, metrics = model(train_data)
            
            # 计算损失
            target = torch.randn_like(output)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录训练历史
            epoch_time = time.time() - start_time
            training_history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'consciousness_level': metrics['consciousness_metrics'].consciousness_level,
                'time': epoch_time
            })
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Loss={loss.item():.6f}, "
                               f"Consciousness={metrics['consciousness_metrics'].consciousness_level}/4")
        
        training_time = time.time() - start_time
        
        # 清理分布式环境
        if self.world_size > 1:
            dist.destroy_process_group()
        
        return {
            'training_time': training_time,
            'final_loss': loss.item(),
            'final_consciousness_level': metrics['consciousness_metrics'].consciousness_level,
            'training_history': training_history
        }

def run_distributed_training(world_size: int = 2, epochs: int = 100) -> Dict:
    """运行分布式训练"""
    print(f"🚀 开始分布式训练 (World Size: {world_size})")
    
    # 创建训练数据
    train_data = torch.randn(1000, 784)
    
    if world_size > 1:
        # 多进程分布式训练
        mp.spawn(_train_worker, 
                args=(world_size, train_data, epochs),
                nprocs=world_size,
                join=True)
    else:
        # 单进程训练
        trainer = DistributedTrainer(world_size=1, rank=0)
        model = EnhancedCEPEITP(
            input_dim=784,
            hidden_dims=[512, 256, 128],
            output_dim=10,
            cep_params=CEPParameters()
        )
        return trainer.train_distributed(model, train_data, epochs)

def _train_worker(rank: int, world_size: int, 
                 train_data: torch.Tensor, epochs: int):
    """训练工作进程"""
    trainer = DistributedTrainer(world_size=world_size, rank=rank)
    model = EnhancedCEPEITP(
        input_dim=784,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        cep_params=CEPParameters()
    )
    
    results = trainer.train_distributed(model, train_data, epochs)
    
    if rank == 0:
        print(f"✅ 分布式训练完成: {results['training_time']:.2f}s")
        print(f"📊 最终损失: {results['final_loss']:.6f}")
        print(f"🧠 意识水平: {results['final_consciousness_level']}/4")

if __name__ == "__main__":
    # 测试分布式训练
    results = run_distributed_training(world_size=1, epochs=50)
    print("🎉 分布式训练测试完成!")
