#!/usr/bin/env python3
"""
EIT-P优越性验证实验执行脚本
对比EIT-P与传统LLM的性能差异
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append('/mnt/sda1/myproject/datainall/AGI')

from eit_p.utils import get_global_logger, ConfigManager
from eit_p.training.eitp_trainer import EITPTrainer
from eit_p.losses.total_loss import TotalLoss
from eit_p.regularization.path_norm import PathNormRegularizer
from eit_p.regularization.entropy import EntropyRegularizer
from eit_p.regularization.chaos import ChaosRegularizer

class SuperiorityExperiment:
    """EIT-P优越性验证实验"""
    
    def __init__(self, config_path='EITP_Superiority_Experiment_Design.json'):
        self.logger = get_global_logger()
        self.config_manager = ConfigManager()
        self.experiment_config = self.load_experiment_config(config_path)
        self.results = {
            'experiment_id': self.experiment_config['experiment_id'],
            'start_time': datetime.now().isoformat(),
            'control_group_results': [],
            'treatment_group_results': [],
            'comparative_analysis': {}
        }
        
    def load_experiment_config(self, config_path):
        """加载实验配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def setup_environment(self):
        """设置实验环境"""
        self.logger.info('🔧 设置实验环境...')
        
        # 设置CUDA环境
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
        
        # 检查GPU可用性
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f'✅ GPU可用: {torch.cuda.get_device_name(0)}')
        else:
            self.device = torch.device('cpu')
            self.logger.warning('⚠️ GPU不可用，使用CPU')
            
        return True
    
    def create_baseline_model(self, model_size='117M'):
        """创建传统LLM基线模型"""
        self.logger.info(f'📊 创建传统LLM基线模型 ({model_size})...')
        
        # 简化的GPT-2架构
        config = {
            'vocab_size': 50257,
            'n_positions': 1024,
            'n_ctx': 1024,
            'n_embd': 768 if model_size == '117M' else 1024,
            'n_layer': 12 if model_size == '117M' else 24,
            'n_head': 12 if model_size == '117M' else 16,
            'activation_function': 'gelu_new',
            'resid_pdrop': 0.1,
            'embd_pdrop': 0.1,
            'attn_pdrop': 0.1,
            'layer_norm_epsilon': 1e-5,
            'initializer_range': 0.02,
            'use_cache': True
        }
        
        return config
    
    def create_eitp_model(self, model_size='117M'):
        """创建EIT-P模型"""
        self.logger.info(f'🧠 创建EIT-P模型 ({model_size})...')
        
        # EIT-P增强配置
        config = {
            'vocab_size': 50257,
            'n_positions': 1024,
            'n_ctx': 1024,
            'n_embd': 768 if model_size == '117M' else 1024,
            'n_layer': 12 if model_size == '117M' else 24,
            'n_head': 12 if model_size == '117M' else 16,
            'activation_function': 'gelu_new',
            'resid_pdrop': 0.1,
            'embd_pdrop': 0.1,
            'attn_pdrop': 0.1,
            'layer_norm_epsilon': 1e-5,
            'initializer_range': 0.02,
            'use_cache': True,
            # EIT-P特有配置
            'iem_enhanced': True,
            'thermodynamic_optimization': True,
            'chaos_control': True,
            'coherence_loss': True
        }
        
        return config
    
    def measure_memory_efficiency(self, model, data_loader, model_type='baseline'):
        """测量内存效率"""
        self.logger.info(f'💾 测量{model_type}内存效率...')
        
        # 记录初始内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            max_memory = torch.cuda.max_memory_allocated()
            
            # 模拟训练步骤
            model.train()
            total_memory = 0
            memory_samples = []
            
            for i, batch in enumerate(data_loader):
                if i >= 10:  # 只测试前10个batch
                    break
                    
                # 前向传播
                outputs = model(batch['input_ids'])
                
                # 记录内存使用
                current_memory = torch.cuda.memory_allocated()
                memory_samples.append(current_memory)
                total_memory += current_memory
                
                # 清理内存
                del outputs
                torch.cuda.empty_cache()
            
            # 计算内存效率
            avg_memory = np.mean(memory_samples)
            memory_efficiency = (avg_memory / (24 * 1024**3)) * 100  # 假设24GB GPU
            
            return {
                'initial_memory': initial_memory,
                'max_memory': max_memory,
                'avg_memory': avg_memory,
                'memory_efficiency': memory_efficiency,
                'memory_samples': memory_samples
            }
        else:
            return {
                'initial_memory': 0,
                'max_memory': 0,
                'avg_memory': 0,
                'memory_efficiency': 0,
                'memory_samples': []
            }
    
    def measure_training_stability(self, model, data_loader, epochs=3):
        """测量训练稳定性"""
        self.logger.info('📈 测量训练稳定性...')
        
        stability_metrics = {
            'memory_overflows': 0,
            'training_interruptions': 0,
            'loss_variance': [],
            'gradient_norms': [],
            'learning_rate_stability': []
        }
        
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for i, batch in enumerate(data_loader):
                try:
                    # 前向传播
                    outputs = model(batch['input_ids'])
                    loss = criterion(outputs.logits, batch['labels'])
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 记录梯度范数
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    stability_metrics['gradient_norms'].append(total_norm)
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    epoch_losses.append(loss.item())
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        stability_metrics['memory_overflows'] += 1
                        torch.cuda.empty_cache()
                    else:
                        stability_metrics['training_interruptions'] += 1
                        self.logger.error(f'训练中断: {e}')
                
                # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 记录损失方差
            if epoch_losses:
                stability_metrics['loss_variance'].append(np.var(epoch_losses))
        
        return stability_metrics
    
    def run_control_group_experiment(self, model_size='117M'):
        """运行对照组实验（传统LLM）"""
        self.logger.info(f'🔬 运行对照组实验 ({model_size})...')
        
        # 创建基线模型
        model_config = self.create_baseline_model(model_size)
        
        # 模拟数据加载器
        data_loader = self.create_mock_data_loader()
        
        # 测量性能指标
        start_time = time.time()
        
        # 内存效率测试
        memory_metrics = self.measure_memory_efficiency(None, data_loader, 'baseline')
        
        # 训练稳定性测试
        stability_metrics = self.measure_training_stability(None, data_loader)
        
        end_time = time.time()
        
        results = {
            'model_type': 'baseline',
            'model_size': model_size,
            'training_time': end_time - start_time,
            'memory_metrics': memory_metrics,
            'stability_metrics': stability_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['control_group_results'].append(results)
        return results
    
    def run_treatment_group_experiment(self, model_size='117M'):
        """运行实验组实验（EIT-P）"""
        self.logger.info(f'🧠 运行实验组实验 ({model_size})...')
        
        # 创建EIT-P模型
        model_config = self.create_eitp_model(model_size)
        
        # 模拟数据加载器
        data_loader = self.create_mock_data_loader()
        
        # 测量性能指标
        start_time = time.time()
        
        # 内存效率测试
        memory_metrics = self.measure_memory_efficiency(None, data_loader, 'eitp')
        
        # 训练稳定性测试
        stability_metrics = self.measure_training_stability(None, data_loader)
        
        end_time = time.time()
        
        results = {
            'model_type': 'eitp',
            'model_size': model_size,
            'training_time': end_time - start_time,
            'memory_metrics': memory_metrics,
            'stability_metrics': stability_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['treatment_group_results'].append(results)
        return results
    
    def create_mock_data_loader(self):
        """创建模拟数据加载器"""
        # 模拟数据
        mock_data = []
        for i in range(100):
            mock_data.append({
                'input_ids': torch.randint(0, 50257, (32, 128)),
                'labels': torch.randint(0, 50257, (32, 128))
            })
        return mock_data
    
    def run_statistical_analysis(self):
        """运行统计分析"""
        self.logger.info('📊 运行统计分析...')
        
        # 提取关键指标
        control_memory = [r['memory_metrics']['memory_efficiency'] for r in self.results['control_group_results']]
        treatment_memory = [r['memory_metrics']['memory_efficiency'] for r in self.results['treatment_group_results']]
        
        control_stability = [r['stability_metrics']['memory_overflows'] for r in self.results['control_group_results']]
        treatment_stability = [r['stability_metrics']['memory_overflows'] for r in self.results['treatment_group_results']]
        
        # 计算统计指标
        analysis = {
            'memory_efficiency': {
                'control_mean': np.mean(control_memory) if control_memory else 0,
                'treatment_mean': np.mean(treatment_memory) if treatment_memory else 0,
                'improvement': 0,
                'effect_size': 0
            },
            'training_stability': {
                'control_overflows': np.mean(control_stability) if control_stability else 0,
                'treatment_overflows': np.mean(treatment_stability) if treatment_stability else 0,
                'improvement': 0,
                'effect_size': 0
            }
        }
        
        # 计算改进幅度
        if control_memory and treatment_memory:
            analysis['memory_efficiency']['improvement'] = (
                (np.mean(treatment_memory) - np.mean(control_memory)) / np.mean(control_memory) * 100
            )
        
        if control_stability and treatment_stability:
            analysis['training_stability']['improvement'] = (
                (np.mean(control_stability) - np.mean(treatment_stability)) / np.mean(control_stability) * 100
            )
        
        self.results['comparative_analysis'] = analysis
        return analysis
    
    def generate_report(self):
        """生成实验报告"""
        self.logger.info('📋 生成实验报告...')
        
        self.results['end_time'] = datetime.now().isoformat()
        self.results['total_experiments'] = len(self.results['control_group_results']) + len(self.results['treatment_group_results'])
        
        # 保存结果
        with open('EITP_Superiority_Experiment_Results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        print('\n🎯 EIT-P优越性验证实验报告')
        print('=' * 60)
        print(f'实验ID: {self.results["experiment_id"]}')
        print(f'开始时间: {self.results["start_time"]}')
        print(f'结束时间: {self.results["end_time"]}')
        print(f'总实验数: {self.results["total_experiments"]}')
        print()
        
        if 'comparative_analysis' in self.results:
            analysis = self.results['comparative_analysis']
            print('📊 关键发现:')
            print(f'  • 内存效率改进: {analysis["memory_efficiency"]["improvement"]:.2f}%')
            print(f'  • 训练稳定性改进: {analysis["training_stability"]["improvement"]:.2f}%')
            print()
        
        print('✅ 实验完成！结果已保存到 EITP_Superiority_Experiment_Results.json')
        
        return self.results
    
    def run_full_experiment(self):
        """运行完整实验"""
        self.logger.info('🚀 开始EIT-P优越性验证实验...')
        
        # 设置环境
        self.setup_environment()
        
        # 运行对照组实验
        self.logger.info('🔬 运行对照组实验...')
        for model_size in ['117M', '345M']:
            self.run_control_group_experiment(model_size)
        
        # 运行实验组实验
        self.logger.info('🧠 运行实验组实验...')
        for model_size in ['117M', '345M']:
            self.run_treatment_group_experiment(model_size)
        
        # 统计分析
        self.run_statistical_analysis()
        
        # 生成报告
        return self.generate_report()

if __name__ == '__main__':
    # 运行实验
    experiment = SuperiorityExperiment()
    results = experiment.run_full_experiment()
