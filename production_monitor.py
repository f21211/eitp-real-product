#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P Production Monitor
生产环境监控和告警系统
"""

import sys
import os
sys.path.append('/mnt/sda1/myproject/datainall/AGI')

import time
import json
import psutil
import requests
import threading
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    load_average: List[float]

@dataclass
class APIMetrics:
    """API指标"""
    timestamp: str
    endpoint: str
    response_time: float
    status_code: int
    success: bool
    error_message: Optional[str] = None

@dataclass
class ModelMetrics:
    """模型指标"""
    timestamp: str
    consciousness_level: int
    constraint_satisfaction: float
    fractal_dimension: float
    complexity_coefficient: float
    inference_time: float
    memory_usage_mb: float

@dataclass
class Alert:
    """告警"""
    timestamp: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    category: str  # SYSTEM, API, MODEL, PERFORMANCE
    message: str
    details: Dict
    resolved: bool = False

class ProductionMonitor:
    """生产环境监控器"""
    
    def __init__(self, api_base_url: str = "http://localhost:5000", 
                 check_interval: int = 30, alert_thresholds: Dict = None):
        self.api_base_url = api_base_url
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread = None
        
        # 默认告警阈值
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'response_time': 5.0,
            'error_rate': 0.1,
            'consciousness_level': 1,
            'constraint_satisfaction': 0.3
        }
        
        # 数据存储
        self.system_metrics_history = deque(maxlen=1000)
        self.api_metrics_history = deque(maxlen=1000)
        self.model_metrics_history = deque(maxlen=1000)
        self.alerts = []
        
        # 设置日志
        self.setup_logging()
        
        # 告警回调
        self.alert_callbacks = []
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('production_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            # CPU和内存
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # 磁盘使用
            disk = psutil.disk_usage('/')
            
            # 网络统计
            network = psutil.net_io_counters()
            
            # 负载平均值
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_sent_mb=network.bytes_sent / (1024 * 1024),
                network_recv_mb=network.bytes_recv / (1024 * 1024),
                load_average=list(load_avg)
            )
            
            self.system_metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"收集系统指标失败: {e}")
            return None
    
    def collect_api_metrics(self) -> List[APIMetrics]:
        """收集API指标"""
        api_metrics = []
        endpoints = [
            '/api/health',
            '/api/model_info',
            '/api/consciousness',
            '/api/performance'
        ]
        
        for endpoint in endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{self.api_base_url}{endpoint}", timeout=10)
                response_time = time.time() - start_time
                
                metric = APIMetrics(
                    timestamp=datetime.now().isoformat(),
                    endpoint=endpoint,
                    response_time=response_time,
                    status_code=response.status_code,
                    success=response.status_code == 200,
                    error_message=None if response.status_code == 200 else f"HTTP {response.status_code}"
                )
                
                api_metrics.append(metric)
                self.api_metrics_history.append(metric)
                
            except Exception as e:
                metric = APIMetrics(
                    timestamp=datetime.now().isoformat(),
                    endpoint=endpoint,
                    response_time=0.0,
                    status_code=0,
                    success=False,
                    error_message=str(e)
                )
                
                api_metrics.append(metric)
                self.api_metrics_history.append(metric)
        
        return api_metrics
    
    def collect_model_metrics(self) -> Optional[ModelMetrics]:
        """收集模型指标"""
        try:
            # 测试推理
            test_input = [0.1] * 784
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_base_url}/api/inference",
                json={'input': test_input},
                timeout=10
            )
            
            inference_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                consciousness_metrics = data['consciousness_metrics']
                
                # 获取内存使用
                process = psutil.Process()
                memory_usage = process.memory_info().rss / (1024 * 1024)
                
                metric = ModelMetrics(
                    timestamp=datetime.now().isoformat(),
                    consciousness_level=consciousness_metrics['level'],
                    constraint_satisfaction=consciousness_metrics.get('constraint_satisfaction', 0.0),
                    fractal_dimension=consciousness_metrics['fractal_dimension'],
                    complexity_coefficient=consciousness_metrics['complexity_coefficient'],
                    inference_time=inference_time,
                    memory_usage_mb=memory_usage
                )
                
                self.model_metrics_history.append(metric)
                return metric
            else:
                self.logger.warning(f"模型推理失败: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"收集模型指标失败: {e}")
            return None
    
    def check_alerts(self, system_metrics: SystemMetrics, 
                    api_metrics: List[APIMetrics], 
                    model_metrics: Optional[ModelMetrics]):
        """检查告警条件"""
        current_time = datetime.now().isoformat()
        
        # 系统告警
        if system_metrics:
            if system_metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
                self.create_alert(
                    level="WARNING",
                    category="SYSTEM",
                    message=f"CPU使用率过高: {system_metrics.cpu_percent:.1f}%",
                    details={'cpu_percent': system_metrics.cpu_percent}
                )
            
            if system_metrics.memory_percent > self.alert_thresholds['memory_percent']:
                self.create_alert(
                    level="WARNING",
                    category="SYSTEM",
                    message=f"内存使用率过高: {system_metrics.memory_percent:.1f}%",
                    details={'memory_percent': system_metrics.memory_percent}
                )
            
            if system_metrics.disk_usage_percent > self.alert_thresholds['disk_usage_percent']:
                self.create_alert(
                    level="ERROR",
                    category="SYSTEM",
                    message=f"磁盘使用率过高: {system_metrics.disk_usage_percent:.1f}%",
                    details={'disk_usage_percent': system_metrics.disk_usage_percent}
                )
        
        # API告警
        if api_metrics:
            failed_requests = [m for m in api_metrics if not m.success]
            if failed_requests:
                error_rate = len(failed_requests) / len(api_metrics)
                if error_rate > self.alert_thresholds['error_rate']:
                    self.create_alert(
                        level="ERROR",
                        category="API",
                        message=f"API错误率过高: {error_rate:.1%}",
                        details={'error_rate': error_rate, 'failed_requests': len(failed_requests)}
                    )
            
            avg_response_time = np.mean([m.response_time for m in api_metrics])
            if avg_response_time > self.alert_thresholds['response_time']:
                self.create_alert(
                    level="WARNING",
                    category="API",
                    message=f"API响应时间过长: {avg_response_time:.2f}s",
                    details={'avg_response_time': avg_response_time}
                )
        
        # 模型告警
        if model_metrics:
            if model_metrics.consciousness_level < self.alert_thresholds['consciousness_level']:
                self.create_alert(
                    level="WARNING",
                    category="MODEL",
                    message=f"意识水平过低: {model_metrics.consciousness_level}/4",
                    details={'consciousness_level': model_metrics.consciousness_level}
                )
            
            if model_metrics.constraint_satisfaction < self.alert_thresholds['constraint_satisfaction']:
                self.create_alert(
                    level="WARNING",
                    category="MODEL",
                    message=f"约束满足率过低: {model_metrics.constraint_satisfaction:.1%}",
                    details={'constraint_satisfaction': model_metrics.constraint_satisfaction}
                )
    
    def create_alert(self, level: str, category: str, message: str, details: Dict):
        """创建告警"""
        alert = Alert(
            timestamp=datetime.now().isoformat(),
            level=level,
            category=category,
            message=message,
            details=details
        )
        
        self.alerts.append(alert)
        self.logger.warning(f"[{level}] {category}: {message}")
        
        # 调用告警回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"告警回调执行失败: {e}")
    
    def monitor_loop(self):
        """监控循环"""
        self.logger.info("开始生产环境监控...")
        
        while self.running:
            try:
                # 收集指标
                system_metrics = self.collect_system_metrics()
                api_metrics = self.collect_api_metrics()
                model_metrics = self.collect_model_metrics()
                
                # 检查告警
                self.check_alerts(system_metrics, api_metrics, model_metrics)
                
                # 打印状态
                if system_metrics:
                    self.logger.info(
                        f"系统状态 - CPU: {system_metrics.cpu_percent:.1f}%, "
                        f"内存: {system_metrics.memory_percent:.1f}%, "
                        f"磁盘: {system_metrics.disk_usage_percent:.1f}%"
                    )
                
                if model_metrics:
                    self.logger.info(
                        f"模型状态 - 意识水平: {model_metrics.consciousness_level}/4, "
                        f"推理时间: {model_metrics.inference_time:.3f}s"
                    )
                
                # 等待下次检查
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(self.check_interval)
    
    def start_monitoring(self):
        """开始监控"""
        if self.running:
            self.logger.warning("监控已在运行中")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("生产环境监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("生产环境监控已停止")
    
    def get_status_report(self) -> Dict:
        """获取状态报告"""
        current_time = datetime.now()
        
        # 最近1小时的指标
        recent_time = current_time - timedelta(hours=1)
        recent_system = [m for m in self.system_metrics_history 
                        if datetime.fromisoformat(m.timestamp) > recent_time]
        recent_api = [m for m in self.api_metrics_history 
                     if datetime.fromisoformat(m.timestamp) > recent_time]
        recent_model = [m for m in self.model_metrics_history 
                       if datetime.fromisoformat(m.timestamp) > recent_time]
        
        # 计算统计信息
        report = {
            'timestamp': current_time.isoformat(),
            'monitoring_status': 'running' if self.running else 'stopped',
            'system_metrics': {
                'total_samples': len(self.system_metrics_history),
                'recent_samples': len(recent_system),
                'avg_cpu_percent': np.mean([m.cpu_percent for m in recent_system]) if recent_system else 0,
                'avg_memory_percent': np.mean([m.memory_percent for m in recent_system]) if recent_system else 0,
                'avg_disk_usage_percent': np.mean([m.disk_usage_percent for m in recent_system]) if recent_system else 0
            },
            'api_metrics': {
                'total_samples': len(self.api_metrics_history),
                'recent_samples': len(recent_api),
                'success_rate': len([m for m in recent_api if m.success]) / len(recent_api) if recent_api else 0,
                'avg_response_time': np.mean([m.response_time for m in recent_api]) if recent_api else 0
            },
            'model_metrics': {
                'total_samples': len(self.model_metrics_history),
                'recent_samples': len(recent_model),
                'avg_consciousness_level': np.mean([m.consciousness_level for m in recent_model]) if recent_model else 0,
                'avg_inference_time': np.mean([m.inference_time for m in recent_model]) if recent_model else 0
            },
            'alerts': {
                'total_alerts': len(self.alerts),
                'recent_alerts': len([a for a in self.alerts 
                                    if datetime.fromisoformat(a.timestamp) > recent_time]),
                'unresolved_alerts': len([a for a in self.alerts if not a.resolved]),
                'alert_levels': {
                    'INFO': len([a for a in self.alerts if a.level == 'INFO']),
                    'WARNING': len([a for a in self.alerts if a.level == 'WARNING']),
                    'ERROR': len([a for a in self.alerts if a.level == 'ERROR']),
                    'CRITICAL': len([a for a in self.alerts if a.level == 'CRITICAL'])
                }
            }
        }
        
        return report
    
    def save_metrics_to_file(self, filename: str = None):
        """保存指标到文件"""
        if filename is None:
            filename = f"production_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'system_metrics': [asdict(m) for m in self.system_metrics_history],
            'api_metrics': [asdict(m) for m in self.api_metrics_history],
            'model_metrics': [asdict(m) for m in self.model_metrics_history],
            'alerts': [asdict(a) for a in self.alerts]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"指标数据已保存到: {filename}")
        return filename
    
    def create_dashboard_visualization(self, save_path: str = "production_dashboard.png"):
        """创建监控仪表板可视化"""
        if not self.system_metrics_history:
            self.logger.warning("没有足够的指标数据创建可视化")
            return
        
        # 准备数据
        timestamps = [datetime.fromisoformat(m.timestamp) for m in self.system_metrics_history]
        cpu_data = [m.cpu_percent for m in self.system_metrics_history]
        memory_data = [m.memory_percent for m in self.system_metrics_history]
        disk_data = [m.disk_usage_percent for m in self.system_metrics_history]
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU使用率
        ax1.plot(timestamps, cpu_data, 'b-', linewidth=2, label='CPU使用率')
        ax1.axhline(y=self.alert_thresholds['cpu_percent'], color='r', linestyle='--', alpha=0.7, label='告警阈值')
        ax1.set_title('CPU使用率监控', fontsize=14, fontweight='bold')
        ax1.set_ylabel('CPU使用率 (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 内存使用率
        ax2.plot(timestamps, memory_data, 'g-', linewidth=2, label='内存使用率')
        ax2.axhline(y=self.alert_thresholds['memory_percent'], color='r', linestyle='--', alpha=0.7, label='告警阈值')
        ax2.set_title('内存使用率监控', fontsize=14, fontweight='bold')
        ax2.set_ylabel('内存使用率 (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 磁盘使用率
        ax3.plot(timestamps, disk_data, 'm-', linewidth=2, label='磁盘使用率')
        ax3.axhline(y=self.alert_thresholds['disk_usage_percent'], color='r', linestyle='--', alpha=0.7, label='告警阈值')
        ax3.set_title('磁盘使用率监控', fontsize=14, fontweight='bold')
        ax3.set_ylabel('磁盘使用率 (%)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 告警统计
        alert_levels = ['INFO', 'WARNING', 'ERROR', 'CRITICAL']
        alert_counts = [len([a for a in self.alerts if a.level == level]) for level in alert_levels]
        colors = ['blue', 'orange', 'red', 'darkred']
        
        ax4.bar(alert_levels, alert_counts, color=colors, alpha=0.7)
        ax4.set_title('告警统计', fontsize=14, fontweight='bold')
        ax4.set_ylabel('告警数量')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"监控仪表板已保存到: {save_path}")

def alert_email_callback(alert: Alert):
    """邮件告警回调示例"""
    print(f"📧 邮件告警: [{alert.level}] {alert.category} - {alert.message}")

def alert_slack_callback(alert: Alert):
    """Slack告警回调示例"""
    print(f"💬 Slack告警: [{alert.level}] {alert.category} - {alert.message}")

def main():
    """主函数"""
    print("🔍 Enhanced CEP-EIT-P Production Monitor")
    print("=" * 50)
    
    # 创建监控器
    monitor = ProductionMonitor(
        api_base_url="http://localhost:5000",
        check_interval=30
    )
    
    # 添加告警回调
    monitor.add_alert_callback(alert_email_callback)
    monitor.add_alert_callback(alert_slack_callback)
    
    try:
        # 开始监控
        monitor.start_monitoring()
        
        # 运行一段时间
        print("🚀 监控已启动，按 Ctrl+C 停止...")
        
        while True:
            time.sleep(60)  # 每分钟打印一次状态报告
            
            # 打印状态报告
            report = monitor.get_status_report()
            print(f"\n📊 状态报告 - {report['timestamp']}")
            print(f"系统: CPU {report['system_metrics']['avg_cpu_percent']:.1f}%, "
                  f"内存 {report['system_metrics']['avg_memory_percent']:.1f}%")
            print(f"API: 成功率 {report['api_metrics']['success_rate']:.1%}, "
                  f"响应时间 {report['api_metrics']['avg_response_time']:.3f}s")
            print(f"模型: 意识水平 {report['model_metrics']['avg_consciousness_level']:.1f}, "
                  f"推理时间 {report['model_metrics']['avg_inference_time']:.3f}s")
            print(f"告警: 总计 {report['alerts']['total_alerts']}, "
                  f"未解决 {report['alerts']['unresolved_alerts']}")
            
    except KeyboardInterrupt:
        print("\n⏹️ 停止监控...")
        monitor.stop_monitoring()
        
        # 保存指标数据
        filename = monitor.save_metrics_to_file()
        
        # 创建可视化
        monitor.create_dashboard_visualization()
        
        print(f"📁 指标数据已保存: {filename}")
        print("📊 监控仪表板已生成: production_dashboard.png")
        print("🎉 监控已停止")

if __name__ == "__main__":
    main()
