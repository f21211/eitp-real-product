#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P API Server V2
增强版API服务器 - 添加高级特性
"""

import sys
import os
sys.path.append('/mnt/sda1/myproject/datainall/AGI')

from flask import Flask, request, jsonify
import torch
import numpy as np
import time
import json
import threading
from datetime import datetime, timedelta
from collections import deque
import logging
from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters

class EnhancedAPIServerV2:
    """增强版API服务器V2"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.model = None
        self.request_history = deque(maxlen=1000)
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_inference_time': 0.0,
            'avg_inference_time': 0.0,
            'requests_per_second': 0.0,
            'last_reset': datetime.now().isoformat()
        }
        self.consciousness_history = deque(maxlen=100)
        self.energy_history = deque(maxlen=100)
        self.setup_routes()
        self.setup_logging()
        self.initialize_model()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_model(self):
        """初始化模型"""
        try:
            cep_params = CEPParameters(
                fractal_dimension=2.7,
                complexity_coefficient=0.8,
                critical_temperature=1.0,
                field_strength=1.0,
                entropy_balance=0.0
            )
            
            self.model = EnhancedCEPEITP(
                input_dim=784,
                hidden_dims=[512, 256, 128],
                output_dim=10,
                cep_params=cep_params
            )
            
            self.logger.info("✅ Enhanced CEP-EIT-P模型初始化成功")
        except Exception as e:
            self.logger.error(f"❌ 模型初始化失败: {e}")
    
    def setup_routes(self):
        """设置路由"""
        
        @self.app.route('/', methods=['GET'])
        def welcome():
            """欢迎页面"""
            return jsonify({
                'message': '🎉 Enhanced CEP-EIT-P API Server V2',
                'description': '基于CEP理论的涌现智能框架 - 增强版',
                'version': '2.0.0',
                'status': 'running',
                'features': [
                    '实时意识检测',
                    '高级能量分析', 
                    '批量处理',
                    '历史数据查询',
                    '性能监控',
                    '模型优化',
                    '统计分析'
                ],
                'endpoints': {
                    'health': '/api/health',
                    'model_info': '/api/model_info',
                    'inference': '/api/inference',
                    'batch_inference': '/api/batch_inference',
                    'consciousness': '/api/consciousness',
                    'energy_analysis': '/api/energy_analysis',
                    'performance': '/api/performance',
                    'optimize': '/api/optimize',
                    'history': '/api/history',
                    'statistics': '/api/statistics',
                    'reset_metrics': '/api/reset_metrics'
                },
                'timestamp': datetime.now().isoformat()
            }), 200
        
        @self.app.route('/api/health', methods=['GET'])
        def health():
            """健康检查"""
            uptime = (datetime.now() - datetime.fromisoformat(self.performance_metrics['last_reset'])).total_seconds()
            return jsonify({
                'status': 'healthy',
                'model_initialized': self.model is not None,
                'uptime': uptime,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        @self.app.route('/api/model_info', methods=['GET'])
        def model_info():
            """模型信息"""
            if not self.model:
                return jsonify({'error': '模型未初始化'}), 500
            
            total_params = sum(p.numel() for p in self.model.parameters())
            
            return jsonify({
                'model_name': 'Enhanced CEP-EIT-P V2',
                'architecture': {
                    'input_dim': 784,
                    'hidden_dims': [512, 256, 128],
                    'output_dim': 10
                },
                'total_parameters': total_params,
                'cep_params': {
                    'fractal_dimension': self.model.cep_params.fractal_dimension,
                    'complexity_coefficient': self.model.cep_params.complexity_coefficient,
                    'critical_temperature': self.model.cep_params.critical_temperature,
                    'field_strength': self.model.cep_params.field_strength,
                    'entropy_balance': self.model.cep_params.entropy_balance
                },
                'features': [
                    '意识检测',
                    '能量分析',
                    '忆阻器网络',
                    '分形拓扑',
                    '混沌控制'
                ],
                'timestamp': datetime.now().isoformat()
            }), 200
        
        @self.app.route('/api/inference', methods=['POST'])
        def inference():
            """推理服务"""
            try:
                data = request.get_json()
                if not data or 'input' not in data:
                    return jsonify({'error': '缺少input参数'}), 400
                
                input_data = np.array(data['input'], dtype=np.float32)
                if len(input_data) != 784:
                    return jsonify({'error': '输入维度必须为784'}), 400
                
                input_tensor = torch.tensor(input_data).unsqueeze(0)
                
                start_time = time.time()
                output, metrics = self.model(input_tensor)
                inference_time = time.time() - start_time
                
                # 更新性能指标
                self.update_performance_metrics(inference_time, True)
                
                # 记录历史数据
                self.record_inference_data(metrics, inference_time)
                
                # 转换输出为可序列化格式
                output_list = output.detach().cpu().numpy().tolist()[0]
                
                return jsonify({
                    'success': True,
                    'output': output_list,
                    'consciousness_metrics': {
                        'level': metrics['consciousness_metrics'].consciousness_level,
                        'fractal_dimension': float(metrics['consciousness_metrics'].fractal_dimension),
                        'complexity_coefficient': float(metrics['consciousness_metrics'].complexity_coefficient),
                        'chaos_threshold': float(metrics['consciousness_metrics'].chaos_threshold),
                        'entropy_balance': float(metrics['consciousness_metrics'].entropy_balance)
                    },
                    'cep_energies': {
                        'mass_energy': float(metrics['cep_energies']['mass_energy']),
                        'field_energy': float(metrics['cep_energies']['field_energy']),
                        'entropy_energy': float(metrics['cep_energies']['entropy_energy']),
                        'complexity_energy': float(metrics['cep_energies']['complexity_energy']),
                        'total_energy': float(metrics['cep_energies']['total_energy'])
                    },
                    'iem_energy': float(metrics['iem_energy']),
                    'inference_time': inference_time,
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                self.update_performance_metrics(0, False)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/batch_inference', methods=['POST'])
        def batch_inference():
            """批量推理服务"""
            try:
                data = request.get_json()
                if not data or 'inputs' not in data:
                    return jsonify({'error': '缺少inputs参数'}), 400
                
                inputs = data['inputs']
                if not isinstance(inputs, list) or len(inputs) == 0:
                    return jsonify({'error': 'inputs必须是非空列表'}), 400
                
                if len(inputs) > 100:
                    return jsonify({'error': '批量大小不能超过100'}), 400
                
                # 处理输入数据
                input_tensors = []
                for i, input_data in enumerate(inputs):
                    if len(input_data) != 784:
                        return jsonify({'error': f'第{i+1}个输入维度必须为784'}), 400
                    input_tensors.append(torch.tensor(input_data, dtype=torch.float32))
                
                batch_tensor = torch.stack(input_tensors)
                
                start_time = time.time()
                outputs, metrics = self.model(batch_tensor)
                inference_time = time.time() - start_time
                
                # 更新性能指标
                self.update_performance_metrics(inference_time, True)
                
                # 记录历史数据
                self.record_inference_data(metrics, inference_time)
                
                # 转换输出
                outputs_list = outputs.detach().cpu().numpy().tolist()
                
                return jsonify({
                    'success': True,
                    'outputs': outputs_list,
                    'batch_size': len(inputs),
                    'consciousness_metrics': {
                        'avg_level': float(metrics['consciousness_metrics'].consciousness_level),
                        'max_level': int(metrics['consciousness_metrics'].consciousness_level),
                        'min_level': int(metrics['consciousness_metrics'].consciousness_level)
                    },
                    'inference_time': inference_time,
                    'avg_inference_time': inference_time / len(inputs),
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                self.update_performance_metrics(0, False)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/consciousness', methods=['GET'])
        def consciousness():
            """意识分析"""
            if not self.consciousness_history:
                return jsonify({'error': '没有意识检测历史数据'}), 404
            
            recent_data = list(self.consciousness_history)[-50:]  # 最近50条
            
            levels = [d['consciousness_level'] for d in recent_data]
            fractal_dims = [d['fractal_dimension'] for d in recent_data]
            complexity_coeffs = [d['complexity_coefficient'] for d in recent_data]
            
            return jsonify({
                'analysis': {
                    'avg_consciousness_level': float(np.mean(levels)),
                    'max_consciousness_level': int(np.max(levels)),
                    'min_consciousness_level': int(np.min(levels)),
                    'avg_fractal_dimension': float(np.mean(fractal_dims)),
                    'avg_complexity_coefficient': float(np.mean(complexity_coeffs)),
                    'level_distribution': {str(i): int(np.sum(np.array(levels) == i)) for i in range(5)},
                    'samples_count': len(recent_data)
                },
                'timestamp': datetime.now().isoformat()
            }), 200
        
        @self.app.route('/api/energy_analysis', methods=['POST'])
        def energy_analysis():
            """能量分析"""
            try:
                data = request.get_json()
                input_data = data.get('input', [0.1] * 784)
                
                input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
                output, metrics = self.model(input_tensor)
                
                cep_energies = metrics['cep_energies']
                iem_energy = metrics['iem_energy']
                
                # 计算能量效率
                mass_energy = cep_energies['mass_energy']
                total_energy = cep_energies['total_energy']
                efficiency = total_energy / (mass_energy + 1e-8) if mass_energy != 0 else 0
                
                # 记录能量历史
                self.energy_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'cep_energies': cep_energies,
                    'iem_energy': float(iem_energy),
                    'efficiency': float(efficiency)
                })
                
                return jsonify({
                    'energy_analysis': {
                        'cep_energies': {
                            'mass_energy': float(cep_energies['mass_energy']),
                            'field_energy': float(cep_energies['field_energy']),
                            'entropy_energy': float(cep_energies['entropy_energy']),
                            'complexity_energy': float(cep_energies['complexity_energy']),
                            'total_energy': float(cep_energies['total_energy'])
                        },
                        'iem_energy': float(iem_energy),
                        'efficiency': float(efficiency),
                        'energy_breakdown': {
                            'mass_energy_ratio': float(mass_energy / total_energy) if total_energy != 0 else 0,
                            'field_energy_ratio': float(cep_energies['field_energy'] / total_energy) if total_energy != 0 else 0,
                            'entropy_energy_ratio': float(cep_energies['entropy_energy'] / total_energy) if total_energy != 0 else 0,
                            'complexity_energy_ratio': float(cep_energies['complexity_energy'] / total_energy) if total_energy != 0 else 0
                        }
                    },
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/performance', methods=['GET'])
        def performance():
            """性能指标"""
            return jsonify({
                'performance': self.performance_metrics,
                'consciousness_history_length': len(self.consciousness_history),
                'energy_history_length': len(self.energy_history),
                'timestamp': datetime.now().isoformat()
            }), 200
        
        @self.app.route('/api/optimize', methods=['POST'])
        def optimize():
            """模型优化"""
            try:
                data = request.get_json() or {}
                epochs = data.get('epochs', 10)
                
                if not self.model:
                    return jsonify({'error': '模型未初始化'}), 500
                
                # 简单的参数优化
                start_time = time.time()
                
                # 更新CEP参数
                self.model.cep_params.fractal_dimension = min(3.0, self.model.cep_params.fractal_dimension + 0.01)
                self.model.cep_params.complexity_coefficient = min(1.0, self.model.cep_params.complexity_coefficient + 0.01)
                
                optimization_time = time.time() - start_time
                
                return jsonify({
                    'success': True,
                    'optimization_time': optimization_time,
                    'updated_params': {
                        'fractal_dimension': self.model.cep_params.fractal_dimension,
                        'complexity_coefficient': self.model.cep_params.complexity_coefficient
                    },
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/history', methods=['GET'])
        def history():
            """历史数据查询"""
            data_type = request.args.get('type', 'consciousness')
            limit = min(int(request.args.get('limit', 50)), 100)
            
            if data_type == 'consciousness':
                data = list(self.consciousness_history)[-limit:]
            elif data_type == 'energy':
                data = list(self.energy_history)[-limit:]
            else:
                return jsonify({'error': '无效的数据类型'}), 400
            
            return jsonify({
                'data_type': data_type,
                'count': len(data),
                'data': data,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        @self.app.route('/api/statistics', methods=['GET'])
        def statistics():
            """统计分析"""
            if not self.consciousness_history:
                return jsonify({'error': '没有足够的历史数据'}), 404
            
            levels = [d['consciousness_level'] for d in self.consciousness_history]
            inference_times = [d['inference_time'] for d in self.consciousness_history]
            
            return jsonify({
                'statistics': {
                    'consciousness_level': {
                        'mean': float(np.mean(levels)),
                        'std': float(np.std(levels)),
                        'min': int(np.min(levels)),
                        'max': int(np.max(levels)),
                        'median': float(np.median(levels))
                    },
                    'inference_time': {
                        'mean': float(np.mean(inference_times)),
                        'std': float(np.std(inference_times)),
                        'min': float(np.min(inference_times)),
                        'max': float(np.max(inference_times)),
                        'median': float(np.median(inference_times))
                    },
                    'total_samples': len(self.consciousness_history),
                    'data_period': {
                        'start': self.consciousness_history[0]['timestamp'],
                        'end': self.consciousness_history[-1]['timestamp']
                    }
                },
                'timestamp': datetime.now().isoformat()
            }), 200
        
        @self.app.route('/api/reset_metrics', methods=['POST'])
        def reset_metrics():
            """重置指标"""
            self.performance_metrics = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_inference_time': 0.0,
                'avg_inference_time': 0.0,
                'requests_per_second': 0.0,
                'last_reset': datetime.now().isoformat()
            }
            self.consciousness_history.clear()
            self.energy_history.clear()
            
            return jsonify({
                'success': True,
                'message': '指标已重置',
                'timestamp': datetime.now().isoformat()
            }), 200
    
    def update_performance_metrics(self, inference_time: float, success: bool):
        """更新性能指标"""
        self.performance_metrics['total_requests'] += 1
        
        if success:
            self.performance_metrics['successful_requests'] += 1
            self.performance_metrics['total_inference_time'] += inference_time
            self.performance_metrics['avg_inference_time'] = (
                self.performance_metrics['total_inference_time'] / 
                self.performance_metrics['successful_requests']
            )
        else:
            self.performance_metrics['failed_requests'] += 1
        
        # 计算每秒请求数
        time_since_reset = (datetime.now() - datetime.fromisoformat(self.performance_metrics['last_reset'])).total_seconds()
        if time_since_reset > 0:
            self.performance_metrics['requests_per_second'] = self.performance_metrics['total_requests'] / time_since_reset
    
    def record_inference_data(self, metrics: dict, inference_time: float):
        """记录推理数据"""
        consciousness_data = {
            'timestamp': datetime.now().isoformat(),
            'consciousness_level': metrics['consciousness_metrics'].consciousness_level,
            'fractal_dimension': float(metrics['consciousness_metrics'].fractal_dimension),
            'complexity_coefficient': float(metrics['consciousness_metrics'].complexity_coefficient),
            'chaos_threshold': float(metrics['consciousness_metrics'].chaos_threshold),
            'entropy_balance': float(metrics['consciousness_metrics'].entropy_balance),
            'inference_time': inference_time
        }
        
        self.consciousness_history.append(consciousness_data)
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """运行服务器"""
        self.logger.info(f"🚀 Enhanced CEP-EIT-P API Server V2 启动在 {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def main():
    """主函数"""
    server = EnhancedAPIServerV2()
    server.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()
