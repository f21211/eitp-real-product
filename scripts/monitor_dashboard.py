#!/usr/bin/env python3
"""
EIT-P 监控仪表板
提供实时训练监控和系统状态展示
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    import psutil
    import torch
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: pip install flask flask-socketio")
    sys.exit(1)

from eit_p.experiments import ExperimentManager, ModelRegistry, MetricsTracker
from eit_p.utils import get_global_logger


class MonitorDashboard:
    """监控仪表板"""
    
    def __init__(self, host='0.0.0.0', port=8082):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'eitp_monitor_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.host = host
        self.port = port
        self.logger = get_global_logger()
        
        # 初始化管理器
        self.experiment_manager = ExperimentManager()
        self.model_registry = ModelRegistry()
        
        # 当前监控的实验
        self.current_experiment_id = None
        self.metrics_tracker = None
        
        # 系统监控数据
        self.system_data = {
            'cpu_percent': 0,
            'memory_percent': 0,
            'gpu_memory_allocated': 0,
            'gpu_memory_reserved': 0,
            'gpu_utilization': 0,
            'disk_usage_percent': 0
        }
        
        # 设置路由
        self._setup_routes()
        self._setup_socketio()
        
        # 启动系统监控
        self._start_system_monitoring()
    

    def _setup_routes(self):
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'service': 'EIT-P Monitor Dashboard',
                'timestamp': datetime.now().isoformat(),
                'system_data': self.system_data
            })

        """设置路由"""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/api/experiments')
        def get_experiments():
            """获取实验列表"""
            experiments = self.experiment_manager.list_experiments()
            return jsonify(experiments)
        
        @self.app.route('/api/experiments/<experiment_id>')
        def get_experiment(experiment_id):
            """获取实验详情"""
            config = self.experiment_manager.get_experiment_config(experiment_id)
            if not config:
                return jsonify({'error': '实验不存在'}), 404
            
            results = self.experiment_manager.get_experiment_results(experiment_id)
            
            return jsonify({
                'config': config.__dict__,
                'results': results
            })
        
        @self.app.route('/api/experiments/<experiment_id>/metrics')
        def get_experiment_metrics(experiment_id):
            """获取实验指标"""
            if not self.metrics_tracker or self.current_experiment_id != experiment_id:
                self.metrics_tracker = MetricsTracker(experiment_id)
                self.current_experiment_id = experiment_id
            
            metrics_summary = self.metrics_tracker.get_all_metrics_summary()
            return jsonify(metrics_summary)
        
        @self.app.route('/api/experiments/<experiment_id>/metrics/<metric_name>')
        def get_metric_history(experiment_id, metric_name):
            """获取指标历史"""
            if not self.metrics_tracker or self.current_experiment_id != experiment_id:
                self.metrics_tracker = MetricsTracker(experiment_id)
                self.current_experiment_id = experiment_id
            
            history = self.metrics_tracker.get_metric_history(metric_name)
            return jsonify([{
                'timestamp': point.timestamp,
                'step': point.step,
                'epoch': point.epoch,
                'value': point.value
            } for point in history])
        
        @self.app.route('/api/models')
        def get_models():
            """获取模型列表"""
            models = self.model_registry.list_models()
            return jsonify(models)
        
        @self.app.route('/api/system')
        def get_system_status():
            """获取系统状态"""
            return jsonify(self.system_data)
        
        @self.app.route('/api/start_monitoring/<experiment_id>')
        def start_monitoring(experiment_id):
            """开始监控实验"""
            self.metrics_tracker = MetricsTracker(experiment_id)
            self.current_experiment_id = experiment_id
            self.metrics_tracker.start_monitoring()
            
            return jsonify({'status': 'success', 'message': f'开始监控实验 {experiment_id}'})
        
        @self.app.route('/api/stop_monitoring')
        def stop_monitoring():
            """停止监控"""
            if self.metrics_tracker:
                self.metrics_tracker.stop_monitoring()
                self.metrics_tracker = None
                self.current_experiment_id = None
            
            return jsonify({'status': 'success', 'message': '停止监控'})
        
        # 新增功能端点
        @self.app.route('/api/training/status')
        def get_training_status():
            """获取训练状态"""
            try:
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
                    'is_training': len(training_processes) > 0,
                    'training_processes': training_processes,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/gpu/status')
        def get_gpu_status():
            """获取GPU状态"""
            try:
                gpu_info = {
                    'available': torch.cuda.is_available(),
                    'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
                    'memory_allocated': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                    'memory_reserved': torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0,
                    'memory_cached': torch.cuda.memory_cached() / 1024**3 if torch.cuda.is_available() else 0,
                    'timestamp': datetime.now().isoformat()
                }
                
                # 获取GPU设备信息
                if torch.cuda.is_available():
                    gpu_info['devices'] = []
                    for i in range(torch.cuda.device_count()):
                        device_info = {
                            'id': i,
                            'name': torch.cuda.get_device_name(i),
                            'memory_total': torch.cuda.get_device_properties(i).total_memory / 1024**3,
                            'memory_allocated': torch.cuda.memory_allocated(i) / 1024**3,
                            'memory_reserved': torch.cuda.memory_reserved(i) / 1024**3
                        }
                        gpu_info['devices'].append(device_info)
                
                return jsonify(gpu_info)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/experiments/<experiment_id>/logs')
        def get_experiment_logs(experiment_id):
            """获取实验日志"""
            try:
                log_file = Path(f'./experiments/experiments/{experiment_id}/logs.txt')
                if not log_file.exists():
                    return jsonify({'error': '日志文件不存在'}), 404
                
                # 读取最后1000行日志
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    last_lines = lines[-1000:] if len(lines) > 1000 else lines
                
                return jsonify({
                    'logs': last_lines,
                    'total_lines': len(lines),
                    'showing_lines': len(last_lines)
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/experiments/<experiment_id>/download')
        def download_experiment_data(experiment_id):
            """下载实验数据"""
            try:
                from flask import send_file
                experiment_dir = Path(f'./experiments/experiments/{experiment_id}')
                
                # 创建压缩包
                import zipfile
                import tempfile
                
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
                    with zipfile.ZipFile(tmp_file.name, 'w') as zip_file:
                        for file_path in experiment_dir.rglob('*'):
                            if file_path.is_file():
                                zip_file.write(file_path, file_path.relative_to(experiment_dir))
                    
                    return send_file(tmp_file.name, as_attachment=True, 
                                   download_name=f'experiment_{experiment_id}.zip')
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    
    def _setup_socketio(self):
        """设置SocketIO事件"""
        
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info('客户端连接')
            emit('status', {'message': '连接成功'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info('客户端断开连接')
        
        @self.socketio.on('request_system_status')
        def handle_system_status():
            emit('system_status', self.system_data)
        
        @self.socketio.on('request_metrics')
        def handle_metrics(data):
            experiment_id = data.get('experiment_id')
            if not experiment_id:
                return
            
            if not self.metrics_tracker or self.current_experiment_id != experiment_id:
                self.metrics_tracker = MetricsTracker(experiment_id)
                self.current_experiment_id = experiment_id
            
            metrics_summary = self.metrics_tracker.get_all_metrics_summary()
            emit('metrics_update', metrics_summary)
    
    def _start_system_monitoring(self):
        """启动系统监控线程"""
        def monitor_system():
            while True:
                try:
                    # 更新系统数据
                    self.system_data.update({
                        'cpu_percent': psutil.cpu_percent(),
                        'memory_percent': psutil.virtual_memory().percent,
                        'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                        'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0,
                        'gpu_utilization': 0,  # 需要nvidia-ml-py
                        'disk_usage_percent': psutil.disk_usage('/').percent,
                        'timestamp': time.time()
                    })
                    
                    # 通过WebSocket发送更新
                    self.socketio.emit('system_status', self.system_data)
                    
                    # 发送指标更新
                    if self.metrics_tracker:
                        metrics_summary = self.metrics_tracker.get_all_metrics_summary()
                        self.socketio.emit('metrics_update', metrics_summary)
                    
                    time.sleep(1)  # 每秒更新一次
                    
                except Exception as e:
                    self.logger.error(f"系统监控错误: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def run(self, debug=False):
        """运行仪表板"""
        self.logger.info(f"启动监控仪表板: http://{self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)


def create_dashboard_template():
    """创建仪表板HTML模板"""
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)
    
    dashboard_html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EIT-P 监控仪表板</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-top: 0;
            color: #333;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .metric-value {
            font-weight: bold;
            color: #007bff;
        }
        .status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        .status.running { background: #d4edda; color: #155724; }
        .status.completed { background: #d1ecf1; color: #0c5460; }
        .status.failed { background: #f8d7da; color: #721c24; }
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }
        .experiment-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .experiment-item {
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 10px 0;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .experiment-item:hover {
            background-color: #f8f9fa;
        }
        .experiment-item.active {
            background-color: #e3f2fd;
            border-color: #2196f3;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 EIT-P 监控仪表板</h1>
            <p>实时训练监控和系统状态展示</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📊 系统状态</h3>
                <div id="system-metrics">
                    <div class="metric">
                        <span>CPU使用率:</span>
                        <span class="metric-value" id="cpu-percent">0%</span>
                    </div>
                    <div class="metric">
                        <span>内存使用率:</span>
                        <span class="metric-value" id="memory-percent">0%</span>
                    </div>
                    <div class="metric">
                        <span>GPU内存:</span>
                        <span class="metric-value" id="gpu-memory">0 GB</span>
                    </div>
                    <div class="metric">
                        <span>磁盘使用率:</span>
                        <span class="metric-value" id="disk-percent">0%</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>🧪 当前实验</h3>
                <div id="current-experiment">
                    <p>未选择实验</p>
                </div>
            </div>
            
            <div class="card">
                <h3>📈 训练指标</h3>
                <div id="training-metrics">
                    <p>暂无数据</p>
                </div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📋 实验列表</h3>
                <div class="experiment-list" id="experiment-list">
                    <p>加载中...</p>
                </div>
            </div>
            
            <div class="card">
                <h3>📊 指标图表</h3>
                <div class="chart-container">
                    <canvas id="metrics-chart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 初始化Socket.IO连接
        const socket = io();
        
        // 初始化图表
        const ctx = document.getElementById('metrics-chart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        
        // 当前选中的实验
        let currentExperimentId = null;
        
        // 连接事件
        socket.on('connect', function() {
            console.log('已连接到服务器');
            loadExperiments();
        });
        
        socket.on('system_status', function(data) {
            updateSystemMetrics(data);
        });
        
        socket.on('metrics_update', function(data) {
            updateTrainingMetrics(data);
        });
        
        // 更新系统指标
        function updateSystemMetrics(data) {
            document.getElementById('cpu-percent').textContent = data.cpu_percent.toFixed(1) + '%';
            document.getElementById('memory-percent').textContent = data.memory_percent.toFixed(1) + '%';
            document.getElementById('gpu-memory').textContent = data.gpu_memory_allocated.toFixed(2) + ' GB';
            document.getElementById('disk-percent').textContent = data.disk_usage_percent.toFixed(1) + '%';
        }
        
        // 更新训练指标
        function updateTrainingMetrics(data) {
            const container = document.getElementById('training-metrics');
            container.innerHTML = '';
            
            for (const [metricName, summary] of Object.entries(data)) {
                if (summary.count > 0) {
                    const div = document.createElement('div');
                    div.className = 'metric';
                    div.innerHTML = `
                        <span>${metricName}:</span>
                        <span class="metric-value">${summary.latest.toFixed(4)}</span>
                    `;
                    container.appendChild(div);
                }
            }
        }
        
        // 加载实验列表
        function loadExperiments() {
            fetch('/api/experiments')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('experiment-list');
                    container.innerHTML = '';
                    
                    data.forEach(experiment => {
                        const div = document.createElement('div');
                        div.className = 'experiment-item';
                        div.innerHTML = `
                            <h4>${experiment.name}</h4>
                            <p>状态: <span class="status ${experiment.status}">${experiment.status}</span></p>
                            <p>创建时间: ${new Date(experiment.created_at).toLocaleString()}</p>
                            <p>模型: ${experiment.model_name}</p>
                        `;
                        div.onclick = () => selectExperiment(experiment.experiment_id);
                        container.appendChild(div);
                    });
                })
                .catch(error => {
                    console.error('加载实验列表失败:', error);
                });
        }
        
        // 选择实验
        function selectExperiment(experimentId) {
            currentExperimentId = experimentId;
            
            // 更新UI
            document.querySelectorAll('.experiment-item').forEach(item => {
                item.classList.remove('active');
            });
            event.target.closest('.experiment-item').classList.add('active');
            
            // 开始监控
            fetch(`/api/start_monitoring/${experimentId}`)
                .then(response => response.json())
                .then(data => {
                    console.log('开始监控:', data.message);
                });
            
            // 更新当前实验信息
            fetch(`/api/experiments/${experimentId}`)
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('current-experiment');
                    container.innerHTML = `
                        <h4>${data.config.name}</h4>
                        <p>状态: <span class="status ${data.config.status}">${data.config.status}</span></p>
                        <p>描述: ${data.config.description}</p>
                        <p>模型: ${data.config.model_name}</p>
                        <p>数据集: ${data.config.dataset_name}</p>
                    `;
                });
        }
        
        // 定期刷新实验列表
        setInterval(loadExperiments, 30000); // 每30秒刷新一次
    </script>
</body>
</html>
    """
    
    with open(template_dir / "dashboard.html", "w", encoding="utf-8") as f:
        f.write(dashboard_html)


def main():
    """主函数"""
    print("🚀 启动EIT-P监控仪表板...")
    
    # 创建模板文件
    create_dashboard_template()
    
    # 启动仪表板
    dashboard = MonitorDashboard()
    dashboard.run(debug=False)


if __name__ == "__main__":
    main()
