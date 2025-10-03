#!/usr/bin/env python3
"""
EIT-P 增强版REST API服务器
提供完整的实验管理、模型管理、推理服务、用户认证和监控数据的API接口
"""

import os
import sys
import json
import time
import hashlib
import jwt
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import wraps

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from flask import Flask, request, jsonify, send_file, g
    from flask_cors import CORS
    import torch
    import psutil
    import numpy as np
    from werkzeug.utils import secure_filename
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: pip install flask flask-cors pyjwt pyyaml numpy")
    sys.exit(1)

from eit_p.experiments import ExperimentManager, ModelRegistry, MetricsTracker
from eit_p.utils import get_global_logger


class EITPAPIServer:
    """EIT-P 增强版API服务器"""
    
    def __init__(self, host='0.0.0.0', port=8085, config_path=None):
        self.app = Flask(__name__)
        CORS(self.app)  # 启用跨域支持
        
        self.host = host
        self.port = port
        self.logger = get_global_logger()
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化管理器
        self.experiment_manager = ExperimentManager()
        self.model_registry = ModelRegistry()
        
        # 用户认证
        self.users = self._load_users()
        self.jwt_secret = self.config.get('security', {}).get('jwt_secret', 'eitp_jwt_secret_2025')
        
        # 速率限制
        self.rate_limits = {}
        
        # 设置路由
        self._setup_routes()
    
    def _load_config(self, config_path):
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 默认配置
            return {
                'security': {
                    'jwt_secret': 'eitp_jwt_secret_2025',
                    'rate_limit': {
                        'api_requests_per_minute': 1000,
                        'inference_requests_per_minute': 100
                    }
                }
            }
    
    def _load_users(self):
        """加载用户数据"""
        # 这里应该从数据库加载，暂时使用内存存储
        return {
            'admin': {
                'password': hashlib.sha256('admin123'.encode()).hexdigest(),
                'role': 'admin',
                'permissions': ['read', 'write', 'admin']
            },
            'user': {
                'password': hashlib.sha256('user123'.encode()).hexdigest(),
                'role': 'user',
                'permissions': ['read', 'write']
            }
        }
    
    def _check_rate_limit(self, client_ip, endpoint_type='api'):
        """检查速率限制"""
        now = time.time()
        minute = int(now // 60)
        
        key = f"{client_ip}:{endpoint_type}:{minute}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = 0
        
        # 获取限制配置
        limits = self.config.get('security', {}).get('rate_limit', {})
        if endpoint_type == 'inference':
            limit = limits.get('inference_requests_per_minute', 100)
        else:
            limit = limits.get('api_requests_per_minute', 1000)
        
        if self.rate_limits[key] >= limit:
            return False
        
        self.rate_limits[key] += 1
        return True
    
    def _require_auth(self, f):
        """认证装饰器"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'status': 'error', 'message': '缺少认证令牌'}), 401
            
            try:
                if token.startswith('Bearer '):
                    token = token[7:]
                
                data = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
                g.current_user = data['username']
                g.user_role = data['role']
            except jwt.ExpiredSignatureError:
                return jsonify({'status': 'error', 'message': '令牌已过期'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'status': 'error', 'message': '无效令牌'}), 401
            
            return f(*args, **kwargs)
        return decorated_function
    
    def _require_permission(self, permission):
        """权限检查装饰器"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if not hasattr(g, 'current_user'):
                    return jsonify({'status': 'error', 'message': '需要认证'}), 401
                
                user_permissions = self.users.get(g.current_user, {}).get('permissions', [])
                if permission not in user_permissions:
                    return jsonify({'status': 'error', 'message': '权限不足'}), 403
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def _setup_routes(self):
        """设置API路由"""
        
        # 认证相关API
        @self.app.route('/api/auth/login', methods=['POST'])
        def login():
            """用户登录"""
            try:
                data = request.get_json()
                username = data.get('username')
                password = data.get('password')
                
                if not username or not password:
                    return jsonify({'status': 'error', 'message': '用户名和密码不能为空'}), 400
                
                # 验证用户
                if username not in self.users:
                    return jsonify({'status': 'error', 'message': '用户不存在'}), 401
                
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                if self.users[username]['password'] != password_hash:
                    return jsonify({'status': 'error', 'message': '密码错误'}), 401
                
                # 生成JWT令牌
                payload = {
                    'username': username,
                    'role': self.users[username]['role'],
                    'exp': datetime.utcnow() + timedelta(hours=24)
                }
                token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'token': token,
                        'username': username,
                        'role': self.users[username]['role'],
                        'permissions': self.users[username]['permissions']
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/auth/logout', methods=['POST'])
        @self._require_auth
        def logout():
            """用户登出"""
            return jsonify({'status': 'success', 'message': '登出成功'})
        
        @self.app.route('/api/auth/profile', methods=['GET'])
        @self._require_auth
        def get_profile():
            """获取用户信息"""
            return jsonify({
                'status': 'success',
                'data': {
                    'username': g.current_user,
                    'role': g.user_role,
                    'permissions': self.users[g.current_user]['permissions']
                }
            })
        
        # 健康检查API
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0',
                'service': 'EIT-P Enhanced API Server'
            })
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """健康检查端点"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'service': 'EIT-P API Server'
            })
        
        @self.app.route('/', methods=['GET'])
        def root():
            """根路径"""
            return jsonify({
                'message': 'EIT-P API Server',
                'version': '1.0.0',
                'endpoints': {
                    'health': '/health',
                    'experiments': '/api/experiments',
                    'models': '/api/models',
                    'system': '/api/system/status'
                }
            })
        
        # 增强的推理API
        @self.app.route('/api/inference', methods=['POST'])
        def model_inference():
            """模型推理接口"""
            try:
                # 检查速率限制
                client_ip = request.remote_addr
                if not self._check_rate_limit(client_ip, 'inference'):
                    return jsonify({'status': 'error', 'message': '请求过于频繁'}), 429
                
                data = request.get_json()
                text = data.get('text', '')
                model_id = data.get('model_id', 'default')
                options = data.get('options', {})
                
                if not text:
                    return jsonify({'status': 'error', 'message': '缺少输入文本'}), 400
                
                # 这里应该调用实际的模型推理
                # 暂时返回模拟结果
                start_time = time.time()
                
                # 模拟推理处理
                time.sleep(0.1)  # 模拟处理时间
                
                result = {
                    'input_text': text,
                    'model_id': model_id,
                    'output': f'模拟推理结果: {text[:50]}...',
                    'confidence': 0.95,
                    'processing_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat(),
                    'options': options
                }
                
                return jsonify({
                    'status': 'success',
                    'data': result
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/inference/batch', methods=['POST'])
        def batch_inference():
            """批量推理接口"""
            try:
                # 检查速率限制
                client_ip = request.remote_addr
                if not self._check_rate_limit(client_ip, 'inference'):
                    return jsonify({'status': 'error', 'message': '请求过于频繁'}), 429
                
                data = request.get_json()
                texts = data.get('texts', [])
                model_id = data.get('model_id', 'default')
                options = data.get('options', {})
                
                if not texts or not isinstance(texts, list):
                    return jsonify({'status': 'error', 'message': '缺少输入文本列表'}), 400
                
                if len(texts) > 100:  # 限制批量大小
                    return jsonify({'status': 'error', 'message': '批量大小不能超过100'}), 400
                
                start_time = time.time()
                results = []
                
                for i, text in enumerate(texts):
                    # 模拟批量推理处理
                    time.sleep(0.01)  # 模拟处理时间
                    
                    result = {
                        'index': i,
                        'input_text': text,
                        'output': f'批量推理结果 {i+1}: {text[:30]}...',
                        'confidence': 0.95 - (i * 0.001),  # 模拟不同的置信度
                        'processing_time': 0.01
                    }
                    results.append(result)
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'results': results,
                        'total_count': len(texts),
                        'total_processing_time': time.time() - start_time,
                        'timestamp': datetime.now().isoformat()
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/training/status', methods=['GET'])
        def get_training_status():
            """获取训练状态"""
            try:
                # 检查是否有正在运行的训练进程
                import psutil
                training_processes = []
                
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if 'python' in proc.info['name'] and any('train' in arg for arg in proc.info['cmdline']):
                            training_processes.append({
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cmdline': ' '.join(proc.info['cmdline'])
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'is_training': len(training_processes) > 0,
                        'training_processes': training_processes,
                        'timestamp': datetime.now().isoformat()
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/training/start', methods=['POST'])
        def start_training():
            """开始训练"""
            try:
                data = request.get_json()
                experiment_id = data.get('experiment_id')
                
                if not experiment_id:
                    return jsonify({'status': 'error', 'message': '缺少实验ID'}), 400
                
                # 这里应该启动实际的训练进程
                # 暂时返回成功状态
                return jsonify({
                    'status': 'success',
                    'message': '训练已开始',
                    'data': {
                        'experiment_id': experiment_id,
                        'status': 'running',
                        'timestamp': datetime.now().isoformat()
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/training/stop', methods=['POST'])
        def stop_training():
            """停止训练"""
            try:
                data = request.get_json()
                experiment_id = data.get('experiment_id')
                
                if not experiment_id:
                    return jsonify({'status': 'error', 'message': '缺少实验ID'}), 400
                
                # 这里应该停止实际的训练进程
                # 暂时返回成功状态
                return jsonify({
                    'status': 'success',
                    'message': '训练已停止',
                    'data': {
                        'experiment_id': experiment_id,
                        'status': 'stopped',
                        'timestamp': datetime.now().isoformat()
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        # 实验管理API
        @self.app.route('/api/experiments', methods=['GET'])
        def list_experiments():
            """获取实验列表"""
            try:
                status = request.args.get('status')
                experiments = self.experiment_manager.list_experiments(status)
                return jsonify({
                    'status': 'success',
                    'data': experiments,
                    'count': len(experiments)
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/experiments', methods=['POST'])
        def create_experiment():
            """创建新实验"""
            try:
                data = request.get_json()
                
                # 验证必需字段
                required_fields = ['name', 'description', 'model_name', 'dataset_name', 'hyperparameters']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'status': 'error', 'message': f'缺少必需字段: {field}'}), 400
                
                # 创建实验配置
                from eit_p.experiments.experiment_manager import ExperimentConfig
                config = ExperimentConfig(
                    name=data['name'],
                    description=data['description'],
                    model_name=data['model_name'],
                    dataset_name=data['dataset_name'],
                    hyperparameters=data['hyperparameters'],
                    training_config=data.get('training_config', {}),
                    created_at=datetime.now().isoformat(),
                    created_by=data.get('created_by', 'api_user'),
                    tags=data.get('tags', [])
                )
                
                experiment_id = self.experiment_manager.create_experiment(config)
                
                return jsonify({
                    'status': 'success',
                    'data': {'experiment_id': experiment_id},
                    'message': '实验创建成功'
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/experiments/<experiment_id>', methods=['GET'])
        def get_experiment(experiment_id):
            """获取实验详情"""
            try:
                config = self.experiment_manager.get_experiment_config(experiment_id)
                if not config:
                    return jsonify({'status': 'error', 'message': '实验不存在'}), 404
                
                results = self.experiment_manager.get_experiment_results(experiment_id)
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'config': config.__dict__,
                        'results': results
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/experiments/<experiment_id>/start', methods=['POST'])
        def start_experiment(experiment_id):
            """开始实验"""
            try:
                success = self.experiment_manager.start_experiment(experiment_id)
                if success:
                    return jsonify({
                        'status': 'success',
                        'message': '实验已开始'
                    })
                else:
                    return jsonify({'status': 'error', 'message': '无法开始实验'}), 400
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/experiments/<experiment_id>/complete', methods=['POST'])
        def complete_experiment(experiment_id):
            """完成实验"""
            try:
                data = request.get_json()
                results = data.get('results', {})
                
                success = self.experiment_manager.complete_experiment(experiment_id, results)
                if success:
                    return jsonify({
                        'status': 'success',
                        'message': '实验已完成'
                    })
                else:
                    return jsonify({'status': 'error', 'message': '无法完成实验'}), 400
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/experiments/<experiment_id>/fail', methods=['POST'])
        def fail_experiment(experiment_id):
            """标记实验失败"""
            try:
                data = request.get_json()
                error_message = data.get('error_message', '未知错误')
                
                success = self.experiment_manager.fail_experiment(experiment_id, error_message)
                if success:
                    return jsonify({
                        'status': 'success',
                        'message': '实验已标记为失败'
                    })
                else:
                    return jsonify({'status': 'error', 'message': '无法标记实验失败'}), 400
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        # 指标管理API
        @self.app.route('/api/experiments/<experiment_id>/metrics', methods=['GET'])
        def get_experiment_metrics(experiment_id):
            """获取实验指标"""
            try:
                metrics_tracker = MetricsTracker(experiment_id)
                metrics_summary = metrics_tracker.get_all_metrics_summary()
                
                return jsonify({
                    'status': 'success',
                    'data': metrics_summary
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/experiments/<experiment_id>/metrics/<metric_name>', methods=['GET'])
        def get_metric_history(experiment_id, metric_name):
            """获取指标历史"""
            try:
                last_n = request.args.get('last_n', type=int)
                
                metrics_tracker = MetricsTracker(experiment_id)
                history = metrics_tracker.get_metric_history(metric_name, last_n)
                
                data = [{
                    'timestamp': point.timestamp,
                    'step': point.step,
                    'epoch': point.epoch,
                    'value': point.value,
                    'tags': point.tags
                } for point in history]
                
                return jsonify({
                    'status': 'success',
                    'data': data,
                    'count': len(data)
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/experiments/<experiment_id>/metrics', methods=['POST'])
        def log_metrics(experiment_id):
            """记录指标"""
            try:
                data = request.get_json()
                metrics = data.get('metrics', {})
                step = data.get('step')
                epoch = data.get('epoch')
                tags = data.get('tags', {})
                
                metrics_tracker = MetricsTracker(experiment_id)
                metrics_tracker.log_metrics_batch(metrics, step, epoch, tags)
                
                return jsonify({
                    'status': 'success',
                    'message': '指标记录成功'
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        # 模型管理API
        @self.app.route('/api/models', methods=['GET'])
        def list_models():
            """获取模型列表"""
            try:
                status = request.args.get('status')
                model_type = request.args.get('model_type')
                
                models = self.model_registry.list_models(status, model_type)
                
                return jsonify({
                    'status': 'success',
                    'data': models,
                    'count': len(models)
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/models/<model_id>', methods=['GET'])
        def get_model(model_id):
            """获取模型详情"""
            try:
                metadata = self.model_registry.get_model_metadata(model_id)
                if not metadata:
                    return jsonify({'status': 'error', 'message': '模型不存在'}), 404
                
                return jsonify({
                    'status': 'success',
                    'data': metadata.__dict__
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/models/<model_id>/status', methods=['PUT'])
        def update_model_status(model_id):
            """更新模型状态"""
            try:
                data = request.get_json()
                status = data.get('status')
                
                if not status:
                    return jsonify({'status': 'error', 'message': '缺少状态字段'}), 400
                
                success = self.model_registry.update_model_status(model_id, status)
                if success:
                    return jsonify({
                        'status': 'success',
                        'message': '模型状态更新成功'
                    })
                else:
                    return jsonify({'status': 'error', 'message': '无法更新模型状态'}), 400
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/models/<model_id1>/compare/<model_id2>', methods=['GET'])
        def compare_models(model_id1, model_id2):
            """比较两个模型"""
            try:
                comparison = self.model_registry.compare_models(model_id1, model_id2)
                
                return jsonify({
                    'status': 'success',
                    'data': comparison
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        # 系统状态API
        @self.app.route('/api/system/status', methods=['GET'])
        def get_system_status():
            """获取系统状态"""
            try:
                import psutil
                
                status = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage_percent': psutil.disk_usage('/').percent,
                    'gpu_available': torch.cuda.is_available(),
                    'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                    'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
                }
                
                return jsonify({
                    'status': 'success',
                    'data': status
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        # 文件上传API
        @self.app.route('/api/upload', methods=['POST'])
        @self._require_auth
        @self._require_permission('write')
        def upload_file():
            """文件上传"""
            try:
                if 'file' not in request.files:
                    return jsonify({'status': 'error', 'message': '没有文件'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'status': 'error', 'message': '没有选择文件'}), 400
                
                # 检查文件类型
                allowed_extensions = {'.txt', '.json', '.csv', '.py', '.yaml', '.yml'}
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in allowed_extensions:
                    return jsonify({'status': 'error', 'message': '不支持的文件类型'}), 400
                
                # 保存文件
                filename = secure_filename(file.filename)
                upload_dir = Path('./uploads')
                upload_dir.mkdir(exist_ok=True)
                
                file_path = upload_dir / f"{int(time.time())}_{filename}"
                file.save(str(file_path))
                
                # 获取文件信息
                file_size = file_path.stat().st_size
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'filename': filename,
                        'file_path': str(file_path),
                        'file_size': file_size,
                        'upload_time': datetime.now().isoformat()
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        # 统计分析API
        @self.app.route('/api/analytics/experiments', methods=['GET'])
        @self._require_auth
        def get_experiment_analytics():
            """获取实验统计分析"""
            try:
                experiments = self.experiment_manager.list_experiments()
                
                # 统计分析
                total_experiments = len(experiments)
                running_experiments = len([e for e in experiments if e.get('status') == 'running'])
                completed_experiments = len([e for e in experiments if e.get('status') == 'completed'])
                failed_experiments = len([e for e in experiments if e.get('status') == 'failed'])
                
                # 按模型类型统计
                model_stats = {}
                for exp in experiments:
                    model_name = exp.get('model_name', 'unknown')
                    model_stats[model_name] = model_stats.get(model_name, 0) + 1
                
                # 按创建时间统计（最近30天）
                now = datetime.now()
                thirty_days_ago = now - timedelta(days=30)
                
                recent_experiments = 0
                for exp in experiments:
                    created_at = exp.get('created_at', '')
                    if created_at:
                        try:
                            exp_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            if exp_date >= thirty_days_ago:
                                recent_experiments += 1
                        except:
                            pass
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'total_experiments': total_experiments,
                        'running_experiments': running_experiments,
                        'completed_experiments': completed_experiments,
                        'failed_experiments': failed_experiments,
                        'recent_experiments': recent_experiments,
                        'model_distribution': model_stats,
                        'success_rate': completed_experiments / total_experiments if total_experiments > 0 else 0
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/analytics/performance', methods=['GET'])
        @self._require_auth
        def get_performance_analytics():
            """获取性能分析"""
            try:
                # 获取系统性能数据
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # GPU信息
                gpu_info = {
                    'available': torch.cuda.is_available(),
                    'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    'memory_allocated': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                    'memory_reserved': torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
                }
                
                # 进程信息
                process_count = len(psutil.pids())
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'system': {
                            'cpu_percent': cpu_percent,
                            'memory_percent': memory.percent,
                            'disk_percent': disk.percent,
                            'process_count': process_count
                        },
                        'gpu': gpu_info,
                        'timestamp': datetime.now().isoformat()
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        # 批量操作API
        @self.app.route('/api/experiments/batch', methods=['POST'])
        @self._require_auth
        @self._require_permission('write')
        def batch_experiment_operations():
            """批量实验操作"""
            try:
                data = request.get_json()
                operation = data.get('operation')
                experiment_ids = data.get('experiment_ids', [])
                
                if not operation or not experiment_ids:
                    return jsonify({'status': 'error', 'message': '缺少操作类型或实验ID列表'}), 400
                
                results = []
                
                for exp_id in experiment_ids:
                    try:
                        if operation == 'start':
                            success = self.experiment_manager.start_experiment(exp_id)
                        elif operation == 'stop':
                            success = self.experiment_manager.stop_experiment(exp_id)
                        elif operation == 'delete':
                            success = self.experiment_manager.delete_experiment(exp_id)
                        else:
                            success = False
                        
                        results.append({
                            'experiment_id': exp_id,
                            'success': success,
                            'message': '操作成功' if success else '操作失败'
                        })
                    except Exception as e:
                        results.append({
                            'experiment_id': exp_id,
                            'success': False,
                            'message': str(e)
                        })
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'operation': operation,
                        'results': results,
                        'total_count': len(experiment_ids),
                        'success_count': len([r for r in results if r['success']])
                    }
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        # 文件下载API
        @self.app.route('/api/experiments/<experiment_id>/download/<file_type>', methods=['GET'])
        def download_experiment_file(experiment_id, file_type):
            """下载实验文件"""
            try:
                experiment_dir = Path(f"./experiments/experiments/{experiment_id}")
                
                if file_type == 'config':
                    file_path = experiment_dir / "config.json"
                elif file_type == 'results':
                    file_path = experiment_dir / "results" / "final_results.json"
                elif file_type == 'metrics':
                    file_path = experiment_dir / "metrics.json"
                elif file_type == 'logs':
                    file_path = experiment_dir / "logs.txt"
                else:
                    return jsonify({'status': 'error', 'message': '不支持的文件类型'}), 400
                
                if not file_path.exists():
                    return jsonify({'status': 'error', 'message': '文件不存在'}), 404
                
                return send_file(str(file_path), as_attachment=True)
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        # 错误处理
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'status': 'error', 'message': '接口不存在'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'status': 'error', 'message': '服务器内部错误'}), 500
    
    def run(self, debug=False):
        """运行API服务器"""
        self.logger.info(f"启动EIT-P增强版API服务器: http://{self.host}:{self.port}")
        self.logger.info("可用端点:")
        self.logger.info("  - 认证: /api/auth/login, /api/auth/logout, /api/auth/profile")
        self.logger.info("  - 推理: /api/inference, /api/inference/batch")
        self.logger.info("  - 实验: /api/experiments/*")
        self.logger.info("  - 模型: /api/models/*")
        self.logger.info("  - 分析: /api/analytics/*")
        self.logger.info("  - 文件: /api/upload, /api/experiments/*/download/*")
        self.logger.info("  - 系统: /api/system/status, /api/health")
        
        self.app.run(host=self.host, port=self.port, debug=debug)


def main():
    """主函数"""
    print("🚀 启动EIT-P增强版API服务器...")
    
    # 检查配置文件
    config_path = project_root / "config" / "production.yaml"
    if not config_path.exists():
        print(f"警告: 配置文件不存在 {config_path}，使用默认配置")
        config_path = None
    
    # 启动API服务器
    server = EITPAPIServer(config_path=str(config_path) if config_path else None)
    server.run(debug=False)


if __name__ == "__main__":
    main()
