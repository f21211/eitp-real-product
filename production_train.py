#!/usr/bin/env python3
"""
EIT-P 生产级训练脚本
集成实验管理、模型注册和指标跟踪
"""

import os
import sys
import time
import json
import torch
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, TrainingArguments
from eit_p.training import EITPTrainer
from eit_p.experiments import ExperimentManager, ModelRegistry, MetricsTracker
from eit_p.experiments.experiment_manager import ExperimentConfig
from eit_p.experiments.model_registry import ModelMetadata
from eit_p.utils import get_global_logger, ConfigManager


class ProductionTrainer:
    """生产级训练器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.logger = get_global_logger()
        
        # 初始化管理器
        self.experiment_manager = ExperimentManager()
        self.model_registry = ModelRegistry()
        
        # 当前实验
        self.current_experiment_id = None
        self.metrics_tracker = None
        
        # 训练状态
        self.training_started = False
        self.training_completed = False
        
    def create_experiment(self, experiment_name: str, description: str, 
                         model_name: str, dataset_name: str, 
                         hyperparameters: Dict[str, Any]) -> str:
        """创建新实验"""
        try:
            # 创建实验配置
            config = ExperimentConfig(
                name=experiment_name,
                description=description,
                model_name=model_name,
                dataset_name=dataset_name,
                hyperparameters=hyperparameters,
                training_config=self.config_manager.get_training_config(),
                created_at=datetime.now().isoformat(),
                created_by="production_trainer",
                tags=["production", "eit-p"]
            )
            
            # 创建实验
            experiment_id = self.experiment_manager.create_experiment(config)
            self.current_experiment_id = experiment_id
            
            # 初始化指标跟踪器
            self.metrics_tracker = MetricsTracker(experiment_id)
            self.metrics_tracker.start_monitoring()
            
            self.logger.info(f"创建实验: {experiment_id} - {experiment_name}")
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"创建实验失败: {e}")
            raise
    
    def setup_model_and_data(self, model_name: str, dataset_path: str):
        """设置模型和数据"""
        try:
            self.logger.info("设置模型和数据...")
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                block_size=self.config_manager.get('model.block_size', 16),
                output_hidden_states=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
            
            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 加载数据集
            train_dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=dataset_path,
                block_size=self.config_manager.get('model.block_size', 16)
            )
            
            # 数据整理器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            self.logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
            self.logger.info(f"训练样本数量: {len(train_dataset):,}")
            
            return model, tokenizer, train_dataset, data_collator
            
        except Exception as e:
            self.logger.error(f"设置模型和数据失败: {e}")
            raise
    
    def train(self, experiment_id: str, model: torch.nn.Module, 
              train_dataset: TextDataset, data_collator: DataCollatorForLanguageModeling,
              hypernetwork_params: Dict[str, Any]) -> Dict[str, Any]:
        """执行训练"""
        try:
            self.logger.info(f"开始训练实验: {experiment_id}")
            
            # 开始实验
            self.experiment_manager.start_experiment(experiment_id)
            self.training_started = True
            
            # 设置训练参数
            training_args = TrainingArguments(
                output_dir=f"./experiments/experiments/{experiment_id}",
                **self.config_manager.get_training_config()
            )
            
            # 创建训练器
            trainer = EITPTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator,
                hypernetwork_params=hypernetwork_params,
                config_manager=self.config_manager
            )
            
            # 添加指标跟踪回调
            trainer.add_callback(MetricsCallback(self.metrics_tracker))
            
            # 开始训练
            start_time = time.time()
            trainer.train()
            training_time = time.time() - start_time
            
            # 训练完成
            self.training_completed = True
            
            # 收集最终结果
            results = {
                'training_time': training_time,
                'final_loss': trainer.state.log_history[-1].get('train_loss', 0),
                'total_steps': trainer.state.global_step,
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'completed_at': datetime.now().isoformat()
            }
            
            # 完成实验
            self.experiment_manager.complete_experiment(experiment_id, results)
            
            # 注册模型
            self.register_model(experiment_id, model, results)
            
            self.logger.info(f"训练完成: {experiment_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            self.logger.error(traceback.format_exc())
            
            # 标记实验失败
            if self.current_experiment_id:
                self.experiment_manager.fail_experiment(experiment_id, str(e))
            
            raise
    
    def register_model(self, experiment_id: str, model: torch.nn.Module, 
                      results: Dict[str, Any]) -> str:
        """注册模型"""
        try:
            # 创建模型元数据
            model_id = f"eitp_{experiment_id}_{int(time.time())}"
            
            metadata = ModelMetadata(
                model_id=model_id,
                name=f"EIT-P Model {experiment_id}",
                version="1.0.0",
                description=f"EIT-P模型 - 实验 {experiment_id}",
                model_type="eit-p",
                architecture="gpt2",
                parameters=sum(p.numel() for p in model.parameters()),
                size_mb=sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2,
                created_at=datetime.now().isoformat(),
                created_by="production_trainer",
                experiment_id=experiment_id,
                performance_metrics={
                    'final_loss': results.get('final_loss', 0),
                    'training_time': results.get('training_time', 0)
                },
                tags=["eit-p", "production"]
            )
            
            # 注册模型
            model_id = self.model_registry.register_model(model, metadata, experiment_id)
            
            self.logger.info(f"注册模型: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"注册模型失败: {e}")
            raise
    
    def run_training_pipeline(self, experiment_name: str, description: str,
                            model_name: str, dataset_path: str,
                            hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """运行完整训练流水线"""
        try:
            self.logger.info("🚀 启动EIT-P生产级训练流水线")
            
            # 1. 创建实验
            experiment_id = self.create_experiment(
                experiment_name, description, model_name, 
                "custom_dataset", hyperparameters
            )
            
            # 2. 设置模型和数据
            model, tokenizer, train_dataset, data_collator = self.setup_model_and_data(
                model_name, dataset_path
            )
            
            # 3. 设置超网络参数
            hypernetwork_params = self.config_manager.get_hypernetwork_config()
            
            # 4. 执行训练
            results = self.train(
                experiment_id, model, train_dataset, 
                data_collator, hypernetwork_params
            )
            
            # 5. 返回结果
            return {
                'experiment_id': experiment_id,
                'status': 'success',
                'results': results,
                'message': '训练完成'
            }
            
        except Exception as e:
            self.logger.error(f"训练流水线失败: {e}")
            return {
                'experiment_id': self.current_experiment_id,
                'status': 'failed',
                'error': str(e),
                'message': '训练失败'
            }
        
        finally:
            # 清理资源
            if self.metrics_tracker:
                self.metrics_tracker.stop_monitoring()


class MetricsCallback:
    """指标回调"""
    
    def __init__(self, metrics_tracker: MetricsTracker):
        self.metrics_tracker = metrics_tracker
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """训练日志回调"""
        if logs:
            # 记录指标
            self.metrics_tracker.log_metrics_batch(
                logs, 
                step=state.global_step,
                epoch=state.epoch
            )
            
            # 设置当前步数和轮次
            self.metrics_tracker.set_current_step(state.global_step)
            self.metrics_tracker.set_current_epoch(int(state.epoch))


def main():
    """主函数"""
    print("🚀 EIT-P 生产级训练脚本")
    print("=" * 50)
    
    # 检查参数
    if len(sys.argv) < 4:
        print("用法: python production_train.py <experiment_name> <model_name> <dataset_path>")
        print("示例: python production_train.py 'my_experiment' 'gpt2' './data/train.txt'")
        sys.exit(1)
    
    experiment_name = sys.argv[1]
    model_name = sys.argv[2]
    dataset_path = sys.argv[3]
    
    # 检查数据集文件
    if not Path(dataset_path).exists():
        print(f"❌ 数据集文件不存在: {dataset_path}")
        sys.exit(1)
    
    # 创建训练器
    trainer = ProductionTrainer()
    
    # 设置超参数
    hyperparameters = {
        'learning_rate': 5e-5,
        'batch_size': 1,
        'gradient_accumulation_steps': 16,
        'max_grad_norm': 0.1,
        'warmup_steps': 10,
        'num_epochs': 1
    }
    
    # 运行训练
    try:
        results = trainer.run_training_pipeline(
            experiment_name=experiment_name,
            description=f"EIT-P实验: {experiment_name}",
            model_name=model_name,
            dataset_path=dataset_path,
            hyperparameters=hyperparameters
        )
        
        print("\n🎉 训练完成!")
        print("=" * 50)
        print(f"实验ID: {results['experiment_id']}")
        print(f"状态: {results['status']}")
        
        if results['status'] == 'success':
            print(f"最终损失: {results['results']['final_loss']:.4f}")
            print(f"训练时间: {results['results']['training_time']:.2f}秒")
            print(f"总步数: {results['results']['total_steps']}")
        else:
            print(f"错误: {results['error']}")
        
        print("\n📊 查看结果:")
        print(f"  实验详情: http://localhost:8083/api/experiments/{results['experiment_id']}")
        print(f"  监控仪表板: http://localhost:8082")
        print(f"  模型列表: http://localhost:8083/api/models")
        
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
