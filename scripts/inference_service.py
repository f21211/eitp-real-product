#!/usr/bin/env python3
"""
EIT-P 模型推理服务
提供高性能的模型推理API，支持多种模型类型和推理模式
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForCausalLM,
        AutoModelForSequenceClassification, pipeline
    )
    import yaml
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: pip install flask flask-cors torch transformers")
    sys.exit(1)

from eit_p.utils import get_global_logger


class ModelCache:
    """模型缓存管理器"""
    
    def __init__(self, max_models=5):
        self.max_models = max_models
        self.models = {}
        self.access_times = {}
        self.logger = get_global_logger()
    
    def get_model(self, model_id: str):
        """获取模型"""
        if model_id in self.models:
            self.access_times[model_id] = time.time()
            return self.models[model_id]
        return None
    
    def add_model(self, model_id: str, model_data: Dict[str, Any]):
        """添加模型到缓存"""
        if len(self.models) >= self.max_models:
            oldest_model = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self.remove_model(oldest_model)
        
        self.models[model_id] = model_data
        self.access_times[model_id] = time.time()
        self.logger.info(f"模型 {model_id} 已添加到缓存")
    
    def remove_model(self, model_id: str):
        """从缓存中移除模型"""
        if model_id in self.models:
            if 'model' in self.models[model_id]:
                del self.models[model_id]['model']
            if 'tokenizer' in self.models[model_id]:
                del self.models[model_id]['tokenizer']
            
            del self.models[model_id]
            del self.access_times[model_id]
            torch.cuda.empty_cache()
            self.logger.info(f"模型 {model_id} 已从缓存中移除")


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_global_logger()
        self.model_cache = ModelCache(max_models=config.get('model_cache_size', 5))
    
    def load_model(self, model_id: str, model_type: str, model_path: str):
        """加载模型"""
        cached_model = self.model_cache.get_model(model_id)
        if cached_model:
            return cached_model
        
        try:
            if model_type == 'text_generation':
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path)
                if torch.cuda.is_available():
                    model = model.cuda()
                
                model_data = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'type': 'text_generation'
                }
            elif model_type == 'text_classification':
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                if torch.cuda.is_available():
                    model = model.cuda()
                
                model_data = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'type': 'text_classification'
                }
            elif model_type == 'sentiment_analysis':
                pipe = pipeline("sentiment-analysis", model=model_path)
                model_data = {
                    'pipeline': pipe,
                    'type': 'sentiment_analysis'
                }
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            self.model_cache.add_model(model_id, model_data)
            return model_data
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            return None
    
    def text_generation(self, model_id: str, text: str, options: Dict[str, Any] = None):
        """文本生成推理"""
        model_data = self.model_cache.get_model(model_id)
        if not model_data or model_data['type'] != 'text_generation':
            raise ValueError(f"模型 {model_id} 未加载或类型不匹配")
        
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        
        inputs = tokenizer.encode(text, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=100, do_sample=True)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'generated_text': generated_text,
            'input_text': text,
            'model_id': model_id
        }
    
    def text_classification(self, model_id: str, text: str, options: Dict[str, Any] = None):
        """文本分类推理"""
        model_data = self.model_cache.get_model(model_id)
        if not model_data or model_data['type'] != 'text_classification':
            raise ValueError(f"模型 {model_id} 未加载或类型不匹配")
        
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class_id].item()
        
        return {
            'predicted_class': f"class_{predicted_class_id}",
            'confidence': confidence,
            'input_text': text,
            'model_id': model_id
        }
    
    def sentiment_analysis(self, model_id: str, text: str, options: Dict[str, Any] = None):
        """情感分析推理"""
        model_data = self.model_cache.get_model(model_id)
        if not model_data or model_data['type'] != 'sentiment_analysis':
            raise ValueError(f"模型 {model_id} 未加载或类型不匹配")
        
        pipeline = model_data['pipeline']
        result = pipeline(text)
        
        return {
            'label': result[0]['label'],
            'score': result[0]['score'],
            'input_text': text,
            'model_id': model_id
        }


class InferenceService:
    """推理服务"""
    
    def __init__(self, host='0.0.0.0', port=8086, config_path=None):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.host = host
        self.port = port
        self.logger = get_global_logger()
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化推理引擎
        self.inference_engine = InferenceEngine(self.config)
        
        # 设置路由
        self._setup_routes()
    
    def _load_config(self, config_path):
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return {
                'model_cache_size': 5,
                'max_workers': 4
            }
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'service': 'EIT-P Inference Service',
                'timestamp': datetime.now().isoformat(),
                'gpu_available': torch.cuda.is_available(),
                'cached_models': len(self.inference_engine.model_cache.models)
            })
        
        @self.app.route('/api/models/load', methods=['POST'])
        def load_model():
            """加载模型"""
            try:
                data = request.get_json()
                model_id = data.get('model_id')
                model_type = data.get('model_type')
                model_path = data.get('model_path')
                
                if not all([model_id, model_type, model_path]):
                    return jsonify({'status': 'error', 'message': '缺少必需参数'}), 400
                
                model_data = self.inference_engine.load_model(model_id, model_type, model_path)
                if model_data:
                    return jsonify({
                        'status': 'success',
                        'message': f'模型 {model_id} 加载成功'
                    })
                else:
                    return jsonify({'status': 'error', 'message': '模型加载失败'}), 500
                    
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/inference/text_generation', methods=['POST'])
        def text_generation():
            """文本生成推理"""
            try:
                data = request.get_json()
                model_id = data.get('model_id')
                text = data.get('text')
                options = data.get('options', {})
                
                if not model_id or not text:
                    return jsonify({'status': 'error', 'message': '缺少必需参数'}), 400
                
                start_time = time.time()
                result = self.inference_engine.text_generation(model_id, text, options)
                processing_time = time.time() - start_time
                
                result['processing_time'] = processing_time
                result['timestamp'] = datetime.now().isoformat()
                
                return jsonify({
                    'status': 'success',
                    'data': result
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/inference/text_classification', methods=['POST'])
        def text_classification():
            """文本分类推理"""
            try:
                data = request.get_json()
                model_id = data.get('model_id')
                text = data.get('text')
                options = data.get('options', {})
                
                if not model_id or not text:
                    return jsonify({'status': 'error', 'message': '缺少必需参数'}), 400
                
                start_time = time.time()
                result = self.inference_engine.text_classification(model_id, text, options)
                processing_time = time.time() - start_time
                
                result['processing_time'] = processing_time
                result['timestamp'] = datetime.now().isoformat()
                
                return jsonify({
                    'status': 'success',
                    'data': result
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/inference/sentiment_analysis', methods=['POST'])
        def sentiment_analysis():
            """情感分析推理"""
            try:
                data = request.get_json()
                model_id = data.get('model_id')
                text = data.get('text')
                options = data.get('options', {})
                
                if not model_id or not text:
                    return jsonify({'status': 'error', 'message': '缺少必需参数'}), 400
                
                start_time = time.time()
                result = self.inference_engine.sentiment_analysis(model_id, text, options)
                processing_time = time.time() - start_time
                
                result['processing_time'] = processing_time
                result['timestamp'] = datetime.now().isoformat()
                
                return jsonify({
                    'status': 'success',
                    'data': result
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/system/status', methods=['GET'])
        def system_status():
            """系统状态"""
            try:
                import psutil
                
                status = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'gpu_available': torch.cuda.is_available(),
                    'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                    'cached_models': len(self.inference_engine.model_cache.models)
                }
                
                return jsonify({
                    'status': 'success',
                    'data': status
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def run(self, debug=False):
        """运行推理服务"""
        self.logger.info(f"启动EIT-P推理服务: http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)


def main():
    """主函数"""
    print("🚀 启动EIT-P推理服务...")
    
    # 检查配置文件
    config_path = project_root / "config" / "production.yaml"
    if not config_path.exists():
        print(f"警告: 配置文件不存在 {config_path}，使用默认配置")
        config_path = None
    
    # 启动推理服务
    service = InferenceService(config_path=str(config_path) if config_path else None)
    service.run(debug=False)


if __name__ == "__main__":
    main()