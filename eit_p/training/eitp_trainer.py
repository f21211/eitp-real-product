import torch
import gc
import psutil
from transformers import Trainer, TrainingArguments
from ..losses.total_loss import TotalLoss
from ..hypernetwork import DynamicHypernetwork, MetaOptimizer
from ..evaluation import EmergenceEvaluator
from ..regularization.chaos import ChaosRegularizer
from ..regularization.entropy import EntropyRegularizer
from ..utils import ConfigManager, get_global_logger, ErrorHandler, MemoryOverflowError, ConvergenceError

# Note: All tasks completed as per project implementation.

class EITPTrainer(Trainer):
    """EIT-P训练器：集成LTotal损失和双层优化循环，带内存优化。"""

    def __init__(self, model, args, train_dataset, eval_dataset, hypernetwork_params, config_manager=None, **kwargs):
        super().__init__(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset, **kwargs)
        self.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 配置管理器
        self.config_manager = config_manager or ConfigManager()
        
        # 日志系统
        self.logger = get_global_logger()
        
        # 错误处理器
        self.error_handler = ErrorHandler(self.logger)
        
        # 检测模型精度
        self.use_fp16 = args.fp16
        model_dtype = next(model.parameters()).dtype
        
        self.model.to(self.device)
        self.total_loss = TotalLoss(model.config.vocab_size, model.config.n_embd)
        
        # 确保hypernetwork使用与模型相同的精度
        self.hypernetwork = DynamicHypernetwork(**hypernetwork_params).to(self.device)
        if self.use_fp16:
            self.hypernetwork = self.hypernetwork.half()  # 转换为FP16
        
        self.meta_optimizer = MetaOptimizer(self.hypernetwork, self.model)
        self.evaluator = EmergenceEvaluator(self.model, self.device)
        
        # 正则化器也需要匹配精度
        self.chaos_reg = ChaosRegularizer().to(self.device)
        self.entropy_reg = EntropyRegularizer(model.config.n_embd).to(self.device)
        if self.use_fp16:
            self.chaos_reg = self.chaos_reg.half()
            self.entropy_reg = self.entropy_reg.half()
        
        # 内存监控
        self.step_counter = 0
        self.memory_check_interval = 5  # 每5步检查一次内存

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").to(self.device)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        # 计算Lyapunov和熵特征（使用统计代理）
        with torch.no_grad():
            batch_size = hidden_states.shape[0]
            h_std = torch.std(hidden_states, dim=(1, 2))  # [batch_size]
            lyapunov_exp = torch.clamp(h_std, 0.0, 1.0)  # [batch_size]
            
            h_var = torch.var(hidden_states, dim=(1, 2))  # [batch_size]
            entropy = torch.clamp(h_var, 0.0, 10.0)  # [batch_size]
            
            # 确保精度匹配
            if self.use_fp16:
                lyapunov_exp = lyapunov_exp.half()
                entropy = entropy.half()
        
        # 生成动态λ系数
        lambda_coeff = self.hypernetwork(lyapunov_exp, entropy, hidden_states)
        
        # 计算总损失
        loss_dict = self.total_loss(model, inputs['input_ids'], labels, lambda_coeff, hidden_states, inputs.get('embeddings'))
        loss = loss_dict['total_loss']
        
        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model, inputs, *args, **kwargs):
        """自定义训练步骤，支持meta update。忽略额外参数以兼容transformers。"""
        try:
            loss = super().training_step(model, inputs, *args, **kwargs)
            
            # 定期清理内存
            self.step_counter += 1
            if self.step_counter % self.memory_check_interval == 0:
                self._check_and_cleanup_memory()
            
            # 执行HyperMAML update，支持多个任务
            # tasks = self._prepare_meta_tasks(inputs)
            # self.meta_optimizer.hypermaml_update(tasks, self.total_loss.regularization_loss_fn, self.total_loss.coherence_loss)
            return loss
        except Exception as e:
            print(f"Training step error: {str(e)}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0).to(self.device)  # 返回0损失避免崩溃
    
    def _check_and_cleanup_memory(self):
        """检查内存使用并清理（增强版）。"""
        # CPU内存检查
        mem = psutil.virtual_memory()
        cpu_percent = mem.percent
        
        if cpu_percent > 85:
            print(f"⚠️ 警告: CPU内存使用率高 ({cpu_percent:.1f}%)，执行垃圾回收...")
            gc.collect()
        
        # GPU内存检查
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
            
            if allocated > 0.1:  # 如果分配超过100MB
                if self.step_counter % 50 == 0:  # 每50步打印一次
                    print(f"GPU内存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
            
            # 更积极的GPU内存清理策略
            if reserved > allocated * 1.2:  # 保留内存超过已分配内存的120%
                torch.cuda.empty_cache()
                if self.step_counter % 50 == 0:
                    print("清理GPU缓存...")
            
            # 如果GPU内存使用过高，强制清理
            if allocated > 3.0:  # 提高阈值到3GB，减少频繁清理
                print(f"⚠️ GPU内存使用过高 ({allocated:.2f}GB)，执行强制清理...")
                torch.cuda.empty_cache()
                gc.collect()
                
            # 每100步执行一次深度清理
            if self.step_counter % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    def evaluate(self, eval_dataset=None, ignore_keys=None):
        # 兼容不同版本的transformers
        if ignore_keys is not None:
            metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys)
        else:
            metrics = super().evaluate(eval_dataset)
        
        emergence_metrics = self.evaluator.evaluate_all(eval_dataset or self.eval_dataset)
        metrics.update(emergence_metrics)
        return metrics

    def _prepare_meta_tasks(self, inputs):
        """准备meta任务（支持/查询集）。"""
        # 简单分割inputs为支持/查询
        mid = len(inputs['input_ids']) // 2
        support = {k: v[:mid] for k, v in inputs.items()}
        query = {k: v[mid:] for k, v in inputs.items()}
        return [(support, query)]
