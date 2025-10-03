#!/usr/bin/env python3
"""
EIT-P 高级监控和告警系统
提供实时监控、告警、性能分析和系统健康检查
"""

import os
import sys
import json
import time
import threading
import smtplib
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import psutil
    import torch
    import yaml
    from flask import Flask, jsonify, request
    from flask_socketio import SocketIO, emit
    import numpy as np
    from collections import deque
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: pip install flask flask-socketio pyyaml numpy")
    sys.exit(1)

from eit_p.utils import get_global_logger


@dataclass
class AlertThreshold:
    """告警阈值配置"""
    cpu_percent: float = 85.0
    memory_percent: float = 85.0
    disk_percent: float = 90.0
    gpu_memory_percent: float = 85.0
    gpu_temperature: float = 80.0
    response_time_ms: float = 1000.0
    error_rate_percent: float = 5.0


@dataclass
class SystemMetrics:
    """系统指标数据"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    gpu_memory_allocated: float
    gpu_memory_reserved: float
    gpu_temperature: float
    gpu_utilization: float
    network_sent: float
    network_recv: float
    process_count: int
    load_average: List[float]


class AlertManager:
    """告警管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_global_logger()
        self.alert_history = deque(maxlen=1000)
        self.active_alerts = {}
        
        # 告警阈值
        thresholds = config.get('monitoring', {}).get('alert_thresholds', {})
        self.thresholds = AlertThreshold(
            cpu_percent=thresholds.get('cpu_percent', 85.0),
            memory_percent=thresholds.get('memory_percent', 85.0),
            disk_percent=thresholds.get('disk_percent', 90.0),
            gpu_memory_percent=thresholds.get('gpu_memory_percent', 85.0),
            gpu_temperature=thresholds.get('gpu_temperature', 80.0),
            response_time_ms=thresholds.get('response_time_ms', 1000.0),
            error_rate_percent=thresholds.get('error_rate_percent', 5.0)
        )
        
        # 告警通道配置
        self.alert_channels = config.get('monitoring', {}).get('alert_channels', {})
    
    def check_metrics(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """检查指标并生成告警"""
        alerts = []
        
        # CPU使用率告警
        if metrics.cpu_percent > self.thresholds.cpu_percent:
            alerts.append(self._create_alert(
                'cpu_high',
                f'CPU使用率过高: {metrics.cpu_percent:.1f}%',
                'warning' if metrics.cpu_percent < 95 else 'critical',
                {'cpu_percent': metrics.cpu_percent, 'threshold': self.thresholds.cpu_percent}
            ))
        
        # 内存使用率告警
        if metrics.memory_percent > self.thresholds.memory_percent:
            alerts.append(self._create_alert(
                'memory_high',
                f'内存使用率过高: {metrics.memory_percent:.1f}%',
                'warning' if metrics.memory_percent < 95 else 'critical',
                {'memory_percent': metrics.memory_percent, 'threshold': self.thresholds.memory_percent}
            ))
        
        # 磁盘使用率告警
        if metrics.disk_percent > self.thresholds.disk_percent:
            alerts.append(self._create_alert(
                'disk_high',
                f'磁盘使用率过高: {metrics.disk_percent:.1f}%',
                'warning' if metrics.disk_percent < 98 else 'critical',
                {'disk_percent': metrics.disk_percent, 'threshold': self.thresholds.disk_percent}
            ))
        
        # GPU内存告警
        if metrics.gpu_memory_allocated > 0:
            gpu_memory_percent = (metrics.gpu_memory_allocated / (metrics.gpu_memory_allocated + metrics.gpu_memory_reserved)) * 100
            if gpu_memory_percent > self.thresholds.gpu_memory_percent:
                alerts.append(self._create_alert(
                    'gpu_memory_high',
                    f'GPU内存使用率过高: {gpu_memory_percent:.1f}%',
                    'warning' if gpu_memory_percent < 95 else 'critical',
                    {'gpu_memory_percent': gpu_memory_percent, 'threshold': self.thresholds.gpu_memory_percent}
                ))
        
        # GPU温度告警
        if metrics.gpu_temperature > self.thresholds.gpu_temperature:
            alerts.append(self._create_alert(
                'gpu_temperature_high',
                f'GPU温度过高: {metrics.gpu_temperature:.1f}°C',
                'warning' if metrics.gpu_temperature < 90 else 'critical',
                {'gpu_temperature': metrics.gpu_temperature, 'threshold': self.thresholds.gpu_temperature}
            ))
        
        return alerts
    
    def _create_alert(self, alert_type: str, message: str, severity: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """创建告警"""
        alert = {
            'id': f"{alert_type}_{int(time.time())}",
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'status': 'active'
        }
        
        # 添加到历史记录
        self.alert_history.append(alert)
        
        # 更新活跃告警
        self.active_alerts[alert_type] = alert
        
        return alert
    
    def send_alert(self, alert: Dict[str, Any]):
        """发送告警通知"""
        try:
            # 发送邮件告警
            if self.alert_channels.get('email', {}).get('enabled', False):
                self._send_email_alert(alert)
            
            # 发送Webhook告警
            if self.alert_channels.get('webhook', {}).get('enabled', False):
                self._send_webhook_alert(alert)
            
            self.logger.info(f"告警已发送: {alert['message']}")
            
        except Exception as e:
            self.logger.error(f"发送告警失败: {e}")
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """发送邮件告警"""
        email_config = self.alert_channels['email']
        
        msg = MIMEMultipart()
        msg['From'] = email_config['username']
        msg['To'] = ', '.join(email_config['recipients'])
        msg['Subject'] = f"[EIT-P告警] {alert['severity'].upper()}: {alert['message']}"
        
        body = f"""
        告警类型: {alert['type']}
        严重程度: {alert['severity']}
        时间: {alert['timestamp']}
        消息: {alert['message']}
        
        详细信息:
        {json.dumps(alert['data'], indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['username'], email_config['password'])
        server.send_message(msg)
        server.quit()
    
    def _send_webhook_alert(self, alert: Dict[str, Any]):
        """发送Webhook告警"""
        webhook_config = self.alert_channels['webhook']
        
        payload = {
            'text': f"[EIT-P告警] {alert['severity'].upper()}: {alert['message']}",
            'attachments': [{
                'color': 'danger' if alert['severity'] == 'critical' else 'warning',
                'fields': [
                    {'title': '告警类型', 'value': alert['type'], 'short': True},
                    {'title': '严重程度', 'value': alert['severity'], 'short': True},
                    {'title': '时间', 'value': alert['timestamp'], 'short': True},
                    {'title': '详细信息', 'value': json.dumps(alert['data'], indent=2), 'short': False}
                ]
            }]
        }
        
        requests.post(
            webhook_config['url'],
            json=payload,
            timeout=webhook_config.get('timeout', 10)
        )
    
    def resolve_alert(self, alert_type: str):
        """解决告警"""
        if alert_type in self.active_alerts:
            alert = self.active_alerts[alert_type]
            alert['status'] = 'resolved'
            alert['resolved_at'] = datetime.now().isoformat()
            del self.active_alerts[alert_type]
            self.logger.info(f"告警已解决: {alert_type}")


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.logger = get_global_logger()
    
    def add_metrics(self, metrics: SystemMetrics):
        """添加指标数据"""
        self.metrics_history.append(metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics_history:
            return {}
        
        # 计算统计信息
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_percent for m in self.metrics_history]
        gpu_memory_values = [m.gpu_memory_allocated for m in self.metrics_history]
        
        return {
            'cpu': {
                'current': cpu_values[-1] if cpu_values else 0,
                'average': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'trend': self._calculate_trend(cpu_values)
            },
            'memory': {
                'current': memory_values[-1] if memory_values else 0,
                'average': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'trend': self._calculate_trend(memory_values)
            },
            'gpu_memory': {
                'current': gpu_memory_values[-1] if gpu_memory_values else 0,
                'average': np.mean(gpu_memory_values),
                'max': np.max(gpu_memory_values),
                'min': np.min(gpu_memory_values),
                'trend': self._calculate_trend(gpu_memory_values)
            },
            'sample_count': len(self.metrics_history),
            'time_range': {
                'start': self.metrics_history[0].timestamp if self.metrics_history else 0,
                'end': self.metrics_history[-1].timestamp if self.metrics_history else 0
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return 'stable'
        
        # 使用线性回归计算趋势
        x = np.arange(len(values))
        y = np.array(values)
        
        # 计算斜率
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_anomalies(self, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """检测异常值"""
        if len(self.metrics_history) < 10:
            return []
        
        anomalies = []
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_percent for m in self.metrics_history]
        
        # 使用Z-score检测异常
        cpu_mean = np.mean(cpu_values)
        cpu_std = np.std(cpu_values)
        memory_mean = np.mean(memory_values)
        memory_std = np.std(memory_values)
        
        for i, metrics in enumerate(self.metrics_history):
            cpu_z_score = abs((metrics.cpu_percent - cpu_mean) / cpu_std) if cpu_std > 0 else 0
            memory_z_score = abs((metrics.memory_percent - memory_mean) / memory_std) if memory_std > 0 else 0
            
            if cpu_z_score > threshold or memory_z_score > threshold:
                anomalies.append({
                    'timestamp': metrics.timestamp,
                    'cpu_z_score': cpu_z_score,
                    'memory_z_score': memory_z_score,
                    'cpu_percent': metrics.cpu_percent,
                    'memory_percent': metrics.memory_percent
                })
        
        return anomalies


class AdvancedMonitor:
    """高级监控系统"""
    
    def __init__(self, config_path: str = None):
        self.logger = get_global_logger()
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        # 初始化组件
        self.alert_manager = AlertManager(self.config)
        self.performance_analyzer = PerformanceAnalyzer()
        
        # 系统监控数据
        self.current_metrics = None
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Flask应用
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'eitp_advanced_monitor'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # 设置路由
        self._setup_routes()
        self._setup_socketio()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'monitoring': {
                'metrics_interval': 1,
                'alert_thresholds': {
                    'cpu_percent': 85.0,
                    'memory_percent': 85.0,
                    'disk_percent': 90.0,
                    'gpu_memory_percent': 85.0,
                    'gpu_temperature': 80.0
                },
                'alert_channels': {
                    'email': {'enabled': False},
                    'webhook': {'enabled': False}
                }
            }
        }
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'service': 'EIT-P Advanced Monitor',
                'timestamp': datetime.now().isoformat(),
                'monitoring_active': self.monitoring_active,
                'metrics_count': len(self.performance_analyzer.metrics_history)
            })
        
        @self.app.route('/api/metrics/current', methods=['GET'])
        def get_current_metrics():
            """获取当前指标"""
            if self.current_metrics:
                return jsonify(self.current_metrics.__dict__)
            else:
                return jsonify({'error': 'No metrics available'}), 404
        
        @self.app.route('/api/metrics/history', methods=['GET'])
        def get_metrics_history():
            """获取指标历史"""
            limit = request.args.get('limit', 100, type=int)
            metrics_list = list(self.performance_analyzer.metrics_history)[-limit:]
            return jsonify([m.__dict__ for m in metrics_list])
        
        @self.app.route('/api/performance/summary', methods=['GET'])
        def get_performance_summary():
            """获取性能摘要"""
            return jsonify(self.performance_analyzer.get_performance_summary())
        
        @self.app.route('/api/performance/anomalies', methods=['GET'])
        def get_anomalies():
            """获取异常检测结果"""
            threshold = request.args.get('threshold', 2.0, type=float)
            return jsonify(self.performance_analyzer.get_anomalies(threshold))
        
        @self.app.route('/api/alerts/active', methods=['GET'])
        def get_active_alerts():
            """获取活跃告警"""
            return jsonify(list(self.alert_manager.active_alerts.values()))
        
        @self.app.route('/api/alerts/history', methods=['GET'])
        def get_alert_history():
            """获取告警历史"""
            limit = request.args.get('limit', 100, type=int)
            alerts = list(self.alert_manager.alert_history)[-limit:]
            return jsonify(alerts)
        
        @self.app.route('/api/alerts/<alert_type>/resolve', methods=['POST'])
        def resolve_alert(alert_type):
            """解决告警"""
            self.alert_manager.resolve_alert(alert_type)
            return jsonify({'status': 'success', 'message': f'Alert {alert_type} resolved'})
        
        @self.app.route('/api/monitoring/start', methods=['POST'])
        def start_monitoring():
            """开始监控"""
            if not self.monitoring_active:
                self.start_monitoring()
                return jsonify({'status': 'success', 'message': 'Monitoring started'})
            else:
                return jsonify({'status': 'already_running', 'message': 'Monitoring already active'})
        
        @self.app.route('/api/monitoring/stop', methods=['POST'])
        def stop_monitoring():
            """停止监控"""
            if self.monitoring_active:
                self.stop_monitoring()
                return jsonify({'status': 'success', 'message': 'Monitoring stopped'})
            else:
                return jsonify({'status': 'not_running', 'message': 'Monitoring not active'})
    
    def _setup_socketio(self):
        """设置SocketIO事件"""
        
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info('客户端连接到高级监控')
            emit('status', {'message': '已连接到EIT-P高级监控系统'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info('客户端断开高级监控连接')
        
        @self.socketio.on('request_metrics')
        def handle_metrics():
            if self.current_metrics:
                emit('metrics_update', self.current_metrics.__dict__)
        
        @self.socketio.on('request_alerts')
        def handle_alerts():
            emit('alerts_update', list(self.alert_manager.active_alerts.values()))
    
    def _collect_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            # 基础系统指标
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            
            # GPU指标
            gpu_memory_allocated = 0
            gpu_memory_reserved = 0
            gpu_temperature = 0
            gpu_utilization = 0
            
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
                
                # 尝试获取GPU温度（需要nvidia-ml-py）
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    gpu_temperature = 0
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                gpu_memory_allocated=gpu_memory_allocated,
                gpu_memory_reserved=gpu_memory_reserved,
                gpu_temperature=gpu_temperature,
                gpu_utilization=gpu_utilization,
                network_sent=network.bytes_sent / 1024**2,  # MB
                network_recv=network.bytes_recv / 1024**2,  # MB
                process_count=len(psutil.pids()),
                load_average=list(load_avg)
            )
            
        except Exception as e:
            self.logger.error(f"收集指标失败: {e}")
            return None
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集指标
                metrics = self._collect_metrics()
                if metrics:
                    self.current_metrics = metrics
                    self.performance_analyzer.add_metrics(metrics)
                    
                    # 检查告警
                    alerts = self.alert_manager.check_metrics(metrics)
                    for alert in alerts:
                        self.alert_manager.send_alert(alert)
                        # 通过WebSocket发送告警
                        self.socketio.emit('alert', alert)
                    
                    # 发送指标更新
                    self.socketio.emit('metrics_update', metrics.__dict__)
                
                # 等待下次收集
                time.sleep(self.config.get('monitoring', {}).get('metrics_interval', 1))
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(5)
    
    def start_monitoring(self):
        """开始监控"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("高级监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            self.logger.info("高级监控已停止")
    
    def run(self, host='0.0.0.0', port=8089, debug=False):
        """运行监控系统"""
        self.logger.info(f"启动EIT-P高级监控系统: http://{host}:{port}")
        
        # 自动开始监控
        self.start_monitoring()
        
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            self.logger.info("收到停止信号，正在关闭监控系统...")
            self.stop_monitoring()


def main():
    """主函数"""
    print("🚀 启动EIT-P高级监控系统...")
    
    # 检查配置文件
    config_path = project_root / "config" / "production.yaml"
    if not config_path.exists():
        print(f"警告: 配置文件不存在 {config_path}，使用默认配置")
        config_path = None
    
    # 启动监控系统
    monitor = AdvancedMonitor(str(config_path) if config_path else None)
    monitor.run(debug=False)


if __name__ == "__main__":
    main()
