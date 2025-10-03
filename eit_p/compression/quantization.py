"""
模型量化模块
提供INT8、FP16等量化功能
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pathlib import Path
import json

from ..utils import get_global_logger


class QuantizationManager:
    """量化管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = get_global_logger()
        
        # 量化统计
        self.quantization_stats = {
            'original_size': 0,
            'quantized_size': 0,
            'compression_ratio': 0.0,
            'accuracy_drop': 0.0
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认量化配置"""
        return {
            'quantization_type': 'int8',  # int8, int16, fp16
            'calibration_method': 'minmax',  # minmax, histogram, kl_divergence
            'per_channel': True,
            'symmetric': True,
            'preserve_sparsity': True,
            'target_accuracy_drop': 0.01,  # 1%
            'calibration_samples': 1000
        }
    
    def quantize_model(self, model: nn.Module, calibration_data: Optional[List[torch.Tensor]] = None) -> nn.Module:
        """量化模型"""
        try:
            self.logger.info("开始模型量化...")
            
            # 记录原始模型大小
            self.quantization_stats['original_size'] = self._get_model_size(model)
            
            # 设置量化配置
            self._setup_quantization_config(model)
            
            # 准备量化
            model.eval()
            model.qconfig = self._get_qconfig()
            
            # 准备量化模型
            prepared_model = quantization.prepare(model)
            
            # 校准（如果提供数据）
            if calibration_data:
                self._calibrate_model(prepared_model, calibration_data)
            
            # 转换模型
            quantized_model = quantization.convert(prepared_model)
            
            # 记录量化后模型大小
            self.quantization_stats['quantized_size'] = self._get_model_size(quantized_model)
            self.quantization_stats['compression_ratio'] = (
                self.quantization_stats['original_size'] / 
                self.quantization_stats['quantized_size']
            )
            
            self.logger.info(f"模型量化完成 - 压缩比: {self.quantization_stats['compression_ratio']:.2f}x")
            
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"模型量化失败: {e}")
            raise
    
    def _setup_quantization_config(self, model: nn.Module):
        """设置量化配置"""
        if self.config['quantization_type'] == 'int8':
            if self.config['per_channel']:
                qconfig = quantization.QConfig(
                    activation=quantization.observer.MinMaxObserver.with_args(
                        dtype=torch.qint8,
                        qscheme=torch.per_tensor_symmetric
                    ),
                    weight=quantization.default_weight_observer
                )
            else:
                qconfig = quantization.QConfig(
                    activation=quantization.observer.MinMaxObserver.with_args(
                        dtype=torch.qint8,
                        qscheme=torch.per_tensor_symmetric
                    ),
                    weight=quantization.observer.MinMaxObserver.with_args(
                        dtype=torch.qint8,
                        qscheme=torch.per_tensor_symmetric
                    )
                )
        elif self.config['quantization_type'] == 'int16':
            qconfig = quantization.QConfig(
                activation=quantization.observer.MinMaxObserver.with_args(
                    dtype=torch.qint16,
                    qscheme=torch.per_tensor_symmetric
                ),
                weight=quantization.observer.MinMaxObserver.with_args(
                    dtype=torch.qint16,
                    qscheme=torch.per_tensor_symmetric
                )
            )
        else:
            raise ValueError(f"不支持的量化类型: {self.config['quantization_type']}")
        
        return qconfig
    
    def _get_qconfig(self):
        """获取量化配置"""
        if self.config['quantization_type'] == 'int8':
            return quantization.get_default_qconfig('fbgemm')
        elif self.config['quantization_type'] == 'int16':
            return quantization.get_default_qconfig('qnnpack')
        else:
            raise ValueError(f"不支持的量化类型: {self.config['quantization_type']}")
    
    def _calibrate_model(self, model: nn.Module, calibration_data: List[torch.Tensor]):
        """校准模型"""
        self.logger.info("开始模型校准...")
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(calibration_data):
                if i >= self.config['calibration_samples']:
                    break
                
                # 前向传播进行校准
                if isinstance(data, (list, tuple)):
                    model(*data)
                else:
                    model(data)
        
        self.logger.info("模型校准完成")
    
    def _get_model_size(self, model: nn.Module) -> int:
        """获取模型大小（字节）"""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size
    
    def evaluate_quantization_impact(self, original_model: nn.Module, 
                                   quantized_model: nn.Module, 
                                   test_data: List[torch.Tensor]) -> Dict[str, float]:
        """评估量化影响"""
        try:
            self.logger.info("评估量化影响...")
            
            # 设置模型为评估模式
            original_model.eval()
            quantized_model.eval()
            
            original_outputs = []
            quantized_outputs = []
            
            with torch.no_grad():
                for data in test_data:
                    # 原始模型输出
                    if isinstance(data, (list, tuple)):
                        orig_out = original_model(*data)
                        quant_out = quantized_model(*data)
                    else:
                        orig_out = original_model(data)
                        quant_out = quantized_model(data)
                    
                    original_outputs.append(orig_out)
                    quantized_outputs.append(quant_out)
            
            # 计算指标
            mse_loss = self._calculate_mse_loss(original_outputs, quantized_outputs)
            cosine_similarity = self._calculate_cosine_similarity(original_outputs, quantized_outputs)
            relative_error = self._calculate_relative_error(original_outputs, quantized_outputs)
            
            # 更新统计
            self.quantization_stats['accuracy_drop'] = mse_loss
            
            results = {
                'mse_loss': mse_loss,
                'cosine_similarity': cosine_similarity,
                'relative_error': relative_error,
                'compression_ratio': self.quantization_stats['compression_ratio']
            }
            
            self.logger.info(f"量化影响评估完成: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"量化影响评估失败: {e}")
            raise
    
    def _calculate_mse_loss(self, original_outputs: List[torch.Tensor], 
                          quantized_outputs: List[torch.Tensor]) -> float:
        """计算MSE损失"""
        total_mse = 0.0
        total_samples = 0
        
        for orig, quant in zip(original_outputs, quantized_outputs):
            if isinstance(orig, (list, tuple)):
                for o, q in zip(orig, quant):
                    mse = torch.nn.functional.mse_loss(o, q)
                    total_mse += mse.item()
                    total_samples += 1
            else:
                mse = torch.nn.functional.mse_loss(orig, quant)
                total_mse += mse.item()
                total_samples += 1
        
        return total_mse / total_samples if total_samples > 0 else 0.0
    
    def _calculate_cosine_similarity(self, original_outputs: List[torch.Tensor], 
                                   quantized_outputs: List[torch.Tensor]) -> float:
        """计算余弦相似度"""
        total_similarity = 0.0
        total_samples = 0
        
        for orig, quant in zip(original_outputs, quantized_outputs):
            if isinstance(orig, (list, tuple)):
                for o, q in zip(orig, quant):
                    similarity = torch.nn.functional.cosine_similarity(
                        o.flatten(), q.flatten(), dim=0
                    )
                    total_similarity += similarity.item()
                    total_samples += 1
            else:
                similarity = torch.nn.functional.cosine_similarity(
                    orig.flatten(), quant.flatten(), dim=0
                )
                total_similarity += similarity.item()
                total_samples += 1
        
        return total_similarity / total_samples if total_samples > 0 else 0.0
    
    def _calculate_relative_error(self, original_outputs: List[torch.Tensor], 
                                quantized_outputs: List[torch.Tensor]) -> float:
        """计算相对误差"""
        total_error = 0.0
        total_samples = 0
        
        for orig, quant in zip(original_outputs, quantized_outputs):
            if isinstance(orig, (list, tuple)):
                for o, q in zip(orig, quant):
                    relative_error = torch.abs(o - q) / (torch.abs(o) + 1e-8)
                    total_error += relative_error.mean().item()
                    total_samples += 1
            else:
                relative_error = torch.abs(orig - quant) / (torch.abs(orig) + 1e-8)
                total_error += relative_error.mean().item()
                total_samples += 1
        
        return total_error / total_samples if total_samples > 0 else 0.0
    
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """保存量化模型"""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存模型
            torch.save(model.state_dict(), save_path / "quantized_model.pt")
            
            # 保存量化配置
            with open(save_path / "quantization_config.json", 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # 保存量化统计
            with open(save_path / "quantization_stats.json", 'w') as f:
                json.dump(self.quantization_stats, f, indent=2)
            
            self.logger.info(f"量化模型已保存: {save_path}")
            
        except Exception as e:
            self.logger.error(f"保存量化模型失败: {e}")
            raise
    
    def load_quantized_model(self, model: nn.Module, load_path: str) -> nn.Module:
        """加载量化模型"""
        try:
            load_path = Path(load_path)
            
            # 加载模型状态
            state_dict = torch.load(load_path / "quantized_model.pt", map_location='cpu')
            model.load_state_dict(state_dict)
            
            # 加载量化配置
            with open(load_path / "quantization_config.json", 'r') as f:
                self.config = json.load(f)
            
            # 加载量化统计
            with open(load_path / "quantization_stats.json", 'r') as f:
                self.quantization_stats = json.load(f)
            
            self.logger.info(f"量化模型已加载: {load_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"加载量化模型失败: {e}")
            raise
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """获取量化统计信息"""
        return self.quantization_stats.copy()
    
    def optimize_quantization_config(self, model: nn.Module, 
                                   test_data: List[torch.Tensor],
                                   target_accuracy_drop: float = 0.01) -> Dict[str, Any]:
        """优化量化配置"""
        self.logger.info("优化量化配置...")
        
        best_config = None
        best_score = float('inf')
        
        # 测试不同的量化配置
        configs_to_test = [
            {'quantization_type': 'int8', 'per_channel': True, 'symmetric': True},
            {'quantization_type': 'int8', 'per_channel': False, 'symmetric': True},
            {'quantization_type': 'int8', 'per_channel': True, 'symmetric': False},
            {'quantization_type': 'int16', 'per_channel': True, 'symmetric': True},
        ]
        
        for config in configs_to_test:
            try:
                # 临时更新配置
                original_config = self.config.copy()
                self.config.update(config)
                
                # 量化模型
                quantized_model = self.quantize_model(model, test_data)
                
                # 评估影响
                impact = self.evaluate_quantization_impact(model, quantized_model, test_data)
                
                # 计算得分（平衡压缩比和精度损失）
                score = impact['mse_loss'] + (1.0 / impact['compression_ratio'])
                
                if score < best_score and impact['mse_loss'] <= target_accuracy_drop:
                    best_score = score
                    best_config = config.copy()
                
                # 恢复原始配置
                self.config = original_config
                
            except Exception as e:
                self.logger.warning(f"测试配置失败: {config} - {e}")
                continue
        
        if best_config:
            self.config.update(best_config)
            self.logger.info(f"最优量化配置: {best_config}")
        else:
            self.logger.warning("未找到满足要求的量化配置")
        
        return best_config or self.config


class QuantizedEITP(nn.Module):
    """量化EIT-P模型"""
    
    def __init__(self, original_model: nn.Module, quantization_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.original_model = original_model
        self.quantization_manager = QuantizationManager(quantization_config)
        self.quantized_model = None
        
    def quantize(self, calibration_data: Optional[List[torch.Tensor]] = None):
        """量化模型"""
        self.quantized_model = self.quantization_manager.quantize_model(
            self.original_model, calibration_data
        )
        return self.quantized_model
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        if self.quantized_model is not None:
            return self.quantized_model(*args, **kwargs)
        else:
            return self.original_model(*args, **kwargs)
    
    def get_compression_ratio(self) -> float:
        """获取压缩比"""
        return self.quantization_manager.get_quantization_stats()['compression_ratio']
    
    def save(self, save_path: str):
        """保存量化模型"""
        if self.quantized_model is not None:
            self.quantization_manager.save_quantized_model(self.quantized_model, save_path)
    
    def load(self, load_path: str):
        """加载量化模型"""
        self.quantized_model = self.quantization_manager.load_quantized_model(
            self.original_model, load_path
        )
        return self.quantized_model
