#!/usr/bin/env python3
"""
EIT-P Production API Server
Emergent Intelligence Framework based on IEM Theory - Real Product Implementation
"""

import os
import sys
import json
import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import traceback

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入核心依赖
import torch
import numpy as np
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import jwt
from functools import wraps
import psutil
import yaml

# 导入transformers
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    GPT2Config,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# 导入EIT-P核心模块
try:
    from eit_p_simple.experiments import ExperimentManager
    from eit_p_simple.models import ModelRegistry
    from eit_p_simple.metrics import MetricsTracker
    from eit_p_simple.security import SecurityManager
    from eit_p_simple.compression import ModelCompressor
    from eit_p_simple.optimization import HyperparameterOptimizer
    from eit_p_simple.distributed import DistributedTrainer
    from eit_p_simple.ab_testing import ABTestManager
    print("✅ EIT-P简化模块导入成功")
except ImportError as e:
    print(f"警告: 无法导入EIT-P模块: {e}")
    print("将使用简化版本...")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eitp_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealEITPAPI:
    """真实的EIT-P API服务器 - 基于IEM理论的涌现智能框架"""
    
    def __init__(self, host='0.0.0.0', port=8085):
        self.app = Flask(__name__)
        CORS(self.app)
        self.host = host
        self.port = port
        
        # 初始化核心组件
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 配置
        self.config = self._load_config()
        self.jwt_secret = self.config.get('jwt_secret', 'eitp_jwt_secret_2025')
        
        # 初始化EIT-P核心模块
        self._init_eitp_modules()
        
        # 设置路由
        self._setup_routes()
        
        logger.info(f"EIT-P API服务器初始化完成 - 设备: {self.device}")
    
    def _load_config(self) -> Dict:
        """加载配置"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'production.yaml')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"无法加载配置文件: {e}, 使用默认配置")
            return {
                'jwt_secret': 'eitp_jwt_secret_2025',
                'model_name': 'gpt2',
                'max_length': 512,
                'temperature': 0.7,
                'top_p': 0.9
            }
    
    def _init_eitp_modules(self):
        """初始化EIT-P核心模块"""
        try:
            # 初始化实验管理器
            self.experiment_manager = ExperimentManager()
            logger.info("✅ 实验管理器初始化成功")
            
            # 初始化模型注册表
            self.model_registry = ModelRegistry()
            logger.info("✅ 模型注册表初始化成功")
            
            # 初始化指标跟踪器
            self.metrics_tracker = MetricsTracker()
            logger.info("✅ 指标跟踪器初始化成功")
            
            # 初始化安全管理器
            self.security_manager = SecurityManager()
            logger.info("✅ 安全管理器初始化成功")
            
            # 初始化模型压缩器
            self.model_compressor = ModelCompressor()
            logger.info("✅ 模型压缩器初始化成功")
            
            # 初始化超参数优化器
            self.hyperparameter_optimizer = HyperparameterOptimizer()
            logger.info("✅ 超参数优化器初始化成功")
            
            # 初始化分布式训练器
            self.distributed_trainer = DistributedTrainer()
            logger.info("✅ 分布式训练器初始化成功")
            
            # 初始化A/B测试管理器
            self.ab_test_manager = ABTestManager()
            logger.info("✅ A/B测试管理器初始化成功")
            
        except Exception as e:
            logger.error(f"EIT-P模块初始化失败: {e}")
            logger.info("使用简化模式运行...")
    
    def _load_model(self):
        """加载GPT-2模型"""
        try:
            model_name = self.config.get('model_name', 'gpt2')
            logger.info(f"正在加载模型: {model_name}")
            
            # 加载tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ 模型加载成功: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """健康检查"""
            try:
                # 检查系统资源
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                gpu_available = torch.cuda.is_available()
                
                health_data = {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'version': '2.0.0',
                    'system': {
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'gpu_available': gpu_available,
                        'device': str(self.device)
                    },
                    'eitp_modules': {
                        'experiment_manager': hasattr(self, 'experiment_manager'),
                        'model_registry': hasattr(self, 'model_registry'),
                        'metrics_tracker': hasattr(self, 'metrics_tracker'),
                        'security_manager': hasattr(self, 'security_manager'),
                        'model_compressor': hasattr(self, 'model_compressor'),
                        'hyperparameter_optimizer': hasattr(self, 'hyperparameter_optimizer'),
                        'distributed_trainer': hasattr(self, 'distributed_trainer'),
                        'ab_test_manager': hasattr(self, 'ab_test_manager')
                    }
                }
                
                return jsonify(health_data), 200
                
            except Exception as e:
                logger.error(f"健康检查失败: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/inference', methods=['POST'])
        def inference():
            """推理接口 - 基于IEM理论的智能生成"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': '无效的请求数据'}), 400
                
                prompt = data.get('prompt', '')
                max_length = data.get('max_length', self.config.get('max_length', 512))
                temperature = data.get('temperature', self.config.get('temperature', 0.7))
                top_p = data.get('top_p', self.config.get('top_p', 0.9))
                
                if not prompt:
                    return jsonify({'error': '缺少prompt参数'}), 400
                
                # 确保模型已加载
                if self.model is None:
                    if not self._load_model():
                        return jsonify({'error': '模型加载失败'}), 500
                
                # 执行推理
                start_time = time.time()
                
                # 编码输入
                inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                # 生成文本
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # 解码输出
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_text = generated_text[len(prompt):]
                
                inference_time = time.time() - start_time
                
                # 记录指标
                if hasattr(self, 'metrics_tracker'):
                    self.metrics_tracker.record_inference_metrics({
                        'prompt_length': len(prompt),
                        'response_length': len(response_text),
                        'inference_time': inference_time,
                        'temperature': temperature,
                        'top_p': top_p
                    })
                
                return jsonify({
                    'status': 'success',
                    'prompt': prompt,
                    'response': response_text,
                    'full_text': generated_text,
                    'metrics': {
                        'inference_time': inference_time,
                        'prompt_length': len(prompt),
                        'response_length': len(response_text),
                        'temperature': temperature,
                        'top_p': top_p
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"推理失败: {e}")
                return jsonify({'error': f'推理失败: {str(e)}'}), 500
        
        @self.app.route('/api/experiments', methods=['POST'])
        def create_experiment():
            """创建实验"""
            try:
                if not hasattr(self, 'experiment_manager'):
                    return jsonify({'error': '实验管理器未初始化'}), 500
                
                data = request.get_json()
                if not data:
                    return jsonify({'error': '无效的请求数据'}), 400
                
                experiment_id = self.experiment_manager.create_experiment(data)
                
                return jsonify({
                    'status': 'success',
                    'experiment_id': experiment_id,
                    'message': '实验创建成功'
                }), 201
                
            except Exception as e:
                logger.error(f"创建实验失败: {e}")
                return jsonify({'error': f'创建实验失败: {str(e)}'}), 500
        
        @self.app.route('/api/experiments/<experiment_id>', methods=['GET'])
        def get_experiment(experiment_id):
            """获取实验信息"""
            try:
                if not hasattr(self, 'experiment_manager'):
                    return jsonify({'error': '实验管理器未初始化'}), 500
                
                experiment = self.experiment_manager.get_experiment(experiment_id)
                if not experiment:
                    return jsonify({'error': '实验不存在'}), 404
                
                return jsonify({
                    'status': 'success',
                    'experiment': experiment
                }), 200
                
            except Exception as e:
                logger.error(f"获取实验失败: {e}")
                return jsonify({'error': f'获取实验失败: {str(e)}'}), 500
        
        @self.app.route('/api/models', methods=['GET'])
        def list_models():
            """列出所有模型"""
            try:
                if not hasattr(self, 'model_registry'):
                    return jsonify({'error': '模型注册表未初始化'}), 500
                
                models = self.model_registry.list_models()
                
                return jsonify({
                    'status': 'success',
                    'models': models
                }), 200
                
            except Exception as e:
                logger.error(f"列出模型失败: {e}")
                return jsonify({'error': f'列出模型失败: {str(e)}'}), 500
        
        @self.app.route('/api/metrics', methods=['GET'])
        def get_metrics():
            """获取系统指标"""
            try:
                if not hasattr(self, 'metrics_tracker'):
                    return jsonify({'error': '指标跟踪器未初始化'}), 500
                
                metrics = self.metrics_tracker.get_metrics()
                
                return jsonify({
                    'status': 'success',
                    'metrics': metrics
                }), 200
                
            except Exception as e:
                logger.error(f"获取指标失败: {e}")
                return jsonify({'error': f'获取指标失败: {str(e)}'}), 500
        
        @self.app.route('/api/compress', methods=['POST'])
        def compress_model():
            """模型压缩"""
            try:
                if not hasattr(self, 'model_compressor'):
                    return jsonify({'error': '模型压缩器未初始化'}), 500
                
                data = request.get_json()
                if not data:
                    return jsonify({'error': '无效的请求数据'}), 400
                
                model_id = data.get('model_id')
                compression_ratio = data.get('compression_ratio', 0.5)
                
                if not model_id:
                    return jsonify({'error': '缺少model_id参数'}), 400
                
                # 执行模型压缩
                compressed_model = self.model_compressor.compress_model(
                    model_id, compression_ratio
                )
                
                return jsonify({
                    'status': 'success',
                    'compressed_model': compressed_model,
                    'compression_ratio': compression_ratio
                }), 200
                
            except Exception as e:
                logger.error(f"模型压缩失败: {e}")
                return jsonify({'error': f'模型压缩失败: {str(e)}'}), 500
        
        @self.app.route('/api/optimize', methods=['POST'])
        def optimize_hyperparameters():
            """超参数优化"""
            try:
                if not hasattr(self, 'hyperparameter_optimizer'):
                    return jsonify({'error': '超参数优化器未初始化'}), 500
                
                data = request.get_json()
                if not data:
                    return jsonify({'error': '无效的请求数据'}), 400
                
                # 执行超参数优化
                best_params = self.hyperparameter_optimizer.optimize(data)
                
                return jsonify({
                    'status': 'success',
                    'best_parameters': best_params
                }), 200
                
            except Exception as e:
                logger.error(f"超参数优化失败: {e}")
                return jsonify({'error': f'超参数优化失败: {str(e)}'}), 500
        
        @self.app.route('/api/ab_test', methods=['POST'])
        def create_ab_test():
            """创建A/B测试"""
            try:
                if not hasattr(self, 'ab_test_manager'):
                    return jsonify({'error': 'A/B测试管理器未初始化'}), 500
                
                data = request.get_json()
                if not data:
                    return jsonify({'error': '无效的请求数据'}), 400
                
                test_id = self.ab_test_manager.create_test(data)
                
                return jsonify({
                    'status': 'success',
                    'test_id': test_id,
                    'message': 'A/B测试创建成功'
                }), 201
                
            except Exception as e:
                logger.error(f"创建A/B测试失败: {e}")
                return jsonify({'error': f'创建A/B测试失败: {str(e)}'}), 500
        
        @self.app.route('/', methods=['GET'])
        def welcome():
            """欢迎页面"""
            return jsonify({
                'message': '🎉 欢迎使用EIT-P框架！',
                'description': '基于IEM理论的涌现智能框架 - 真实产品',
                'version': '2.0.0',
                'status': 'running',
                'api_endpoints': {
                    'health': '/api/health',
                    'inference': '/api/inference',
                    'experiments': '/api/experiments',
                    'models': '/api/models',
                    'metrics': '/api/metrics',
                    'compress': '/api/compress',
                    'optimize': '/api/optimize',
                    'ab_test': '/api/ab_test',
                    'status': '/api/status'
                },
                'documentation': '请查看 README.md 了解详细使用方法',
                'timestamp': datetime.now().isoformat()
            }), 200
        
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            """获取系统状态"""
            try:
                # 系统资源状态
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # GPU状态
                gpu_info = []
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        gpu_info.append({
                            'device_id': i,
                            'name': torch.cuda.get_device_name(i),
                            'memory_allocated': torch.cuda.memory_allocated(i),
                            'memory_reserved': torch.cuda.memory_reserved(i),
                            'memory_total': torch.cuda.get_device_properties(i).total_memory
                        })
                
                status = {
                    'timestamp': datetime.now().isoformat(),
                    'system': {
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'disk_percent': disk.percent,
                        'gpu_count': torch.cuda.device_count(),
                        'gpu_info': gpu_info
                    },
                    'eitp_status': {
                        'model_loaded': self.model is not None,
                        'modules_initialized': hasattr(self, 'experiment_manager'),
                        'device': str(self.device)
                    }
                }
                
                return jsonify({
                    'status': 'success',
                    'data': status
                }), 200
                
            except Exception as e:
                logger.error(f"获取状态失败: {e}")
                return jsonify({'error': f'获取状态失败: {str(e)}'}), 500
    
    def run(self, debug=False):
        """运行API服务器"""
        try:
            logger.info(f"🚀 启动EIT-P API服务器...")
            logger.info(f"📍 地址: http://{self.host}:{self.port}")
            logger.info(f"🔧 设备: {self.device}")
            logger.info(f"🧠 模型: {self.config.get('model_name', 'gpt2')}")
            
            self.app.run(
                host=self.host,
                port=self.port,
                debug=debug,
                threaded=True
            )
            
        except Exception as e:
            logger.error(f"API服务器启动失败: {e}")
            raise

def main():
    """主函数"""
    try:
        # 创建API服务器实例
        api_server = RealEITPAPI(host='0.0.0.0', port=8085)
        
        # 运行服务器
        api_server.run(debug=False)
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"服务器运行失败: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
