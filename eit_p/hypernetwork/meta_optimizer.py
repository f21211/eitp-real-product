"""
Meta优化器
实现基于MAML和HyperMAML的双层优化循环
用于训练动态超网络
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Callable
import copy


class MetaOptimizer:
    """
    Meta优化器
    
    实现基于MAML的双层优化循环，用于训练动态超网络。
    支持HyperMAML扩展，实现超网络参数的元学习。
    """
    
    def __init__(
        self,
        hypernetwork: nn.Module,
        main_model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        device: str = 'cuda'
    ):
        self.hypernetwork = hypernetwork
        self.main_model = main_model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.device = device
        
        # 优化器
        self.outer_optimizer = optim.Adam(
            self.hypernetwork.parameters(), 
            lr=outer_lr
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.outer_optimizer, T_max=1000
        )
        
        # 历史信息
        self.meta_loss_history = []
        self.lambda_history = []
    
    def meta_update(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        query_data: Tuple[torch.Tensor, torch.Tensor],
        regularization_loss_fn: Callable,
        coherence_loss_fn: Callable
    ) -> Dict[str, float]:
        """
        执行一次Meta更新
        
        Args:
            support_data: 支持集数据 (x_support, y_support)
            query_data: 查询集数据 (x_query, y_query)
            regularization_loss_fn: 正则化损失函数
            coherence_loss_fn: 连贯性损失函数
            
        Returns:
            metrics: 训练指标
        """
        x_support, y_support = support_data
        x_query, y_query = query_data
        
        # 保存原始超网络参数
        original_params = {
            name: param.clone() 
            for name, param in self.hypernetwork.named_parameters()
        }
        
        # 内层循环：在支持集上快速适应
        inner_losses = []
        for step in range(self.num_inner_steps):
            # 前向传播
            with torch.enable_grad():
                # 获取当前状态特征
                lyapunov_exp, entropy, hidden_states = self._extract_features(
                    x_support
                )
                
                # 生成动态λ
                lambda_coeff = self.hypernetwork(
                    lyapunov_exp, entropy, hidden_states
                )
                
                # 计算损失
                main_loss = self._compute_main_loss(x_support, y_support)
                reg_loss = regularization_loss_fn(
                    self.main_model, lambda_coeff, lyapunov_exp, entropy, hidden_states
                )
                coherence_loss = coherence_loss_fn(x_support, y_support)
                
                total_loss = main_loss + reg_loss + coherence_loss
                inner_losses.append(total_loss.item())
            
            # 内层梯度更新
            self.outer_optimizer.zero_grad()
            total_loss.backward()
            
            # 使用较小的学习率进行内层更新
            for param in self.hypernetwork.parameters():
                if param.grad is not None:
                    param.data -= self.inner_lr * param.grad
        
        # 外层循环：在查询集上评估并更新
        with torch.enable_grad():
            # 使用适应后的参数在查询集上评估
            lyapunov_exp, entropy, hidden_states = self._extract_features(x_query)
            lambda_coeff = self.hypernetwork(lyapunov_exp, entropy, hidden_states)
            
            # 计算查询集损失
            main_loss = self._compute_main_loss(x_query, y_query)
            reg_loss = regularization_loss_fn(
                self.main_model, lambda_coeff, lyapunov_exp, entropy, hidden_states
            )
            coherence_loss = coherence_loss_fn(x_query, y_query)
            
            meta_loss = main_loss + reg_loss + coherence_loss
        
        # 恢复原始参数
        for name, param in self.hypernetwork.named_parameters():
            param.data = original_params[name]
        
        # 外层梯度更新
        self.outer_optimizer.zero_grad()
        meta_loss.backward()
        self.outer_optimizer.step()
        
        # 更新学习率
        self.scheduler.step()
        
        # 记录历史
        self.meta_loss_history.append(meta_loss.item())
        self.lambda_history.append(lambda_coeff.detach().cpu().numpy())
        
        return {
            'meta_loss': meta_loss.item(),
            'main_loss': main_loss.item(),
            'reg_loss': reg_loss.item(),
            'coherence_loss': coherence_loss.item(),
            'inner_losses': inner_losses,
            'lambda_mean': torch.mean(lambda_coeff).item(),
            'lambda_std': torch.std(lambda_coeff).item()
        }
    
    def _extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        提取用于超网络的特征
        
        Args:
            x: 输入数据
            
        Returns:
            lyapunov_exp: Lyapunov指数
            entropy: 微分熵
            hidden_states: 隐藏状态
        """
        with torch.no_grad():
            # 获取模型输出和隐藏状态
            if hasattr(self.main_model, 'transformer'):
                # GPT-2模型
                outputs = self.main_model.transformer(x, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # 最后一层隐藏状态
            else:
                # 其他模型
                hidden_states = self.main_model(x)
            
            # 计算Lyapunov指数（简化版本）
            lyapunov_exp = torch.zeros(x.shape[0], device=x.device)
            
            # 计算微分熵（简化版本）
            h_flat = hidden_states.view(-1, hidden_states.shape[-1])
            entropy = torch.mean(torch.std(h_flat, dim=1))
        
        return lyapunov_exp, entropy, hidden_states
    
    def _compute_main_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算主模型损失
        
        Args:
            x: 输入数据
            y: 目标数据
            
        Returns:
            loss: 主模型损失
        """
        if hasattr(self.main_model, 'transformer'):
            # GPT-2模型
            outputs = self.main_model(x, labels=y)
            loss = outputs.loss
        else:
            # 其他模型
            logits = self.main_model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        return loss
    
    def hypermaml_update(
        self,
        tasks: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
        regularization_loss_fn: Callable,
        coherence_loss_fn: Callable
    ) -> Dict[str, float]:
        """
        执行HyperMAML更新
        
        Args:
            tasks: 任务列表，每个任务包含(support_data, query_data)
            regularization_loss_fn: 正则化损失函数
            coherence_loss_fn: 连贯性损失函数
            
        Returns:
            metrics: 训练指标
        """
        total_meta_loss = 0.0
        total_main_loss = 0.0
        total_reg_loss = 0.0
        total_coherence_loss = 0.0
        
        for support_data, query_data in tasks:
            metrics = self.meta_update(
                support_data, query_data, 
                regularization_loss_fn, coherence_loss_fn
            )
            
            total_meta_loss += metrics['meta_loss']
            total_main_loss += metrics['main_loss']
            total_reg_loss += metrics['reg_loss']
            total_coherence_loss += metrics['coherence_loss']
        
        num_tasks = len(tasks)
        
        return {
            'avg_meta_loss': total_meta_loss / num_tasks,
            'avg_main_loss': total_main_loss / num_tasks,
            'avg_reg_loss': total_reg_loss / num_tasks,
            'avg_coherence_loss': total_coherence_loss / num_tasks
        }
    
    def get_hypernetwork_state(self) -> Dict[str, torch.Tensor]:
        """
        获取超网络状态
        
        Returns:
            state: 超网络状态字典
        """
        return {
            name: param.detach().cpu() 
            for name, param in self.hypernetwork.named_parameters()
        }
    
    def load_hypernetwork_state(self, state: Dict[str, torch.Tensor]):
        """
        加载超网络状态
        
        Args:
            state: 超网络状态字典
        """
        for name, param in self.hypernetwork.named_parameters():
            if name in state:
                param.data = state[name].to(self.device)
    
    def get_training_history(self) -> Dict[str, List]:
        """
        获取训练历史
        
        Returns:
            history: 训练历史字典
        """
        return {
            'meta_loss_history': self.meta_loss_history,
            'lambda_history': self.lambda_history
        }
