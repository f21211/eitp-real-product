#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P Web Dashboard
Web监控仪表板
"""

import sys
import os
sys.path.append('/mnt/sda1/myproject/datainall/AGI')

from flask import Flask, render_template_string, jsonify, request
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List
import threading
from advanced_features_manager import AdvancedFeaturesManager

app = Flask(__name__)

# 全局变量
advanced_manager = AdvancedFeaturesManager()
dashboard_data = {
    'system_status': {},
    'model_versions': [],
    'ab_tests': [],
    'performance_metrics': {},
    'last_update': None
}

def update_dashboard_data():
    """更新仪表板数据"""
    global dashboard_data
    
    while True:
        try:
            # 更新系统状态
            dashboard_data['system_status'] = advanced_manager.get_system_status()
            
            # 更新模型版本
            dashboard_data['model_versions'] = advanced_manager.list_model_versions()
            
            # 更新A/B测试
            dashboard_data['ab_tests'] = advanced_manager.list_ab_tests()
            
            # 更新性能指标
            dashboard_data['performance_metrics'] = {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': 45.2,  # 模拟数据
                'memory_usage': 67.8,
                'gpu_usage': 23.4,
                'disk_usage': 34.5,
                'network_io': 12.3
            }
            
            dashboard_data['last_update'] = datetime.now().isoformat()
            
        except Exception as e:
            print(f"更新仪表板数据失败: {e}")
        
        time.sleep(5)  # 每5秒更新一次

# 启动后台更新线程
update_thread = threading.Thread(target=update_dashboard_data, daemon=True)
update_thread.start()

# HTML模板
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced CEP-EIT-P 监控仪表板</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .header p {
            color: #7f8c8d;
            font-size: 1.2em;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-running { background-color: #27ae60; }
        .status-warning { background-color: #f39c12; }
        .status-error { background-color: #e74c3c; }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        }
        
        .card h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            font-weight: 600;
            color: #34495e;
        }
        
        .metric-value {
            font-weight: bold;
            color: #2c3e50;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: #ecf0f1;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            transition: width 0.3s ease;
        }
        
        .table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .table th,
        .table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .table tr:hover {
            background-color: #f8f9fa;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .badge-success { background-color: #d4edda; color: #155724; }
        .badge-warning { background-color: #fff3cd; color: #856404; }
        .badge-danger { background-color: #f8d7da; color: #721c24; }
        .badge-info { background-color: #d1ecf1; color: #0c5460; }
        
        .refresh-btn {
            background: linear-gradient(45deg, #3498db, #2ecc71);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: all 0.3s ease;
            margin: 20px 0;
        }
        
        .refresh-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        
        .footer {
            text-align: center;
            color: rgba(255, 255, 255, 0.8);
            margin-top: 30px;
        }
        
        .loading {
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
        }
        
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Enhanced CEP-EIT-P 监控仪表板</h1>
            <p>基于CEP理论的涌现智能框架 - 实时监控系统</p>
            <div style="margin-top: 15px;">
                <span class="status-indicator status-running"></span>
                <span>系统运行中</span>
                <span style="margin-left: 20px;">最后更新: <span id="lastUpdate">加载中...</span></span>
            </div>
        </div>
        
        <div style="text-align: center;">
            <button class="refresh-btn" onclick="refreshData()">🔄 刷新数据</button>
        </div>
        
        <div class="grid">
            <!-- 系统状态卡片 -->
            <div class="card">
                <h2>📊 系统状态</h2>
                <div id="systemStatus" class="loading">加载中...</div>
            </div>
            
            <!-- 性能指标卡片 -->
            <div class="card">
                <h2>⚡ 性能指标</h2>
                <div id="performanceMetrics" class="loading">加载中...</div>
            </div>
            
            <!-- 模型版本卡片 -->
            <div class="card">
                <h2>🤖 模型版本</h2>
                <div id="modelVersions" class="loading">加载中...</div>
            </div>
            
            <!-- A/B测试卡片 -->
            <div class="card">
                <h2>🧪 A/B测试</h2>
                <div id="abTests" class="loading">加载中...</div>
            </div>
        </div>
        
        <div class="footer">
            <p>Enhanced CEP-EIT-P Framework © 2024 | 基于CEP理论的涌现智能框架</p>
        </div>
    </div>
    
    <script>
        function refreshData() {
            location.reload();
        }
        
        function updateDashboard() {
            fetch('/api/dashboard_data')
                .then(response => response.json())
                .then(data => {
                    updateSystemStatus(data.system_status);
                    updatePerformanceMetrics(data.performance_metrics);
                    updateModelVersions(data.model_versions);
                    updateABTests(data.ab_tests);
                    document.getElementById('lastUpdate').textContent = data.last_update || '未知';
                })
                .catch(error => {
                    console.error('更新数据失败:', error);
                });
        }
        
        function updateSystemStatus(status) {
            const container = document.getElementById('systemStatus');
            if (!status || Object.keys(status).length === 0) {
                container.innerHTML = '<div class="loading">暂无数据</div>';
                return;
            }
            
            let html = '';
            for (const [key, value] of Object.entries(status)) {
                const statusClass = value === 'running' ? 'status-running' : 
                                  value === 'warning' ? 'status-warning' : 'status-error';
                html += `
                    <div class="metric">
                        <span class="metric-label">${key}</span>
                        <span class="metric-value">
                            <span class="status-indicator ${statusClass}"></span>
                            ${value}
                        </span>
                    </div>
                `;
            }
            container.innerHTML = html;
        }
        
        function updatePerformanceMetrics(metrics) {
            const container = document.getElementById('performanceMetrics');
            if (!metrics || Object.keys(metrics).length === 0) {
                container.innerHTML = '<div class="loading">暂无数据</div>';
                return;
            }
            
            let html = '';
            const metricsToShow = ['cpu_usage', 'memory_usage', 'gpu_usage', 'disk_usage'];
            
            for (const metric of metricsToShow) {
                if (metrics[metric] !== undefined) {
                    const value = metrics[metric];
                    const percentage = Math.min(100, Math.max(0, value));
                    html += `
                        <div class="metric">
                            <span class="metric-label">${metric.replace('_', ' ').toUpperCase()}</span>
                            <span class="metric-value">${value.toFixed(1)}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${percentage}%"></div>
                        </div>
                    `;
                }
            }
            container.innerHTML = html;
        }
        
        function updateModelVersions(versions) {
            const container = document.getElementById('modelVersions');
            if (!versions || versions.length === 0) {
                container.innerHTML = '<div class="loading">暂无模型版本</div>';
                return;
            }
            
            let html = '<table class="table">';
            html += '<tr><th>版本</th><th>状态</th><th>创建时间</th><th>标签</th></tr>';
            
            versions.slice(0, 5).forEach(version => {
                const statusClass = version.status === 'active' ? 'badge-success' : 'badge-warning';
                html += `
                    <tr>
                        <td>${version.version_id || 'N/A'}</td>
                        <td><span class="badge ${statusClass}">${version.status || 'unknown'}</span></td>
                        <td>${version.created_at ? new Date(version.created_at).toLocaleString() : 'N/A'}</td>
                        <td>${(version.tags || []).join(', ')}</td>
                    </tr>
                `;
            });
            
            html += '</table>';
            if (versions.length > 5) {
                html += `<p style="text-align: center; margin-top: 10px; color: #7f8c8d;">显示前5个版本，共${versions.length}个版本</p>`;
            }
            
            container.innerHTML = html;
        }
        
        function updateABTests(tests) {
            const container = document.getElementById('abTests');
            if (!tests || tests.length === 0) {
                container.innerHTML = '<div class="loading">暂无A/B测试</div>';
                return;
            }
            
            let html = '<table class="table">';
            html += '<tr><th>测试ID</th><th>状态</th><th>开始时间</th><th>参与用户</th></tr>';
            
            tests.slice(0, 5).forEach(test => {
                const statusClass = test.status === 'running' ? 'badge-success' : 
                                  test.status === 'completed' ? 'badge-info' : 'badge-warning';
                html += `
                    <tr>
                        <td>${test.test_id || 'N/A'}</td>
                        <td><span class="badge ${statusClass}">${test.status || 'unknown'}</span></td>
                        <td>${test.start_time ? new Date(test.start_time).toLocaleString() : 'N/A'}</td>
                        <td>${test.participants || 0}</td>
                    </tr>
                `;
            });
            
            html += '</table>';
            if (tests.length > 5) {
                html += `<p style="text-align: center; margin-top: 10px; color: #7f8c8d;">显示前5个测试，共${tests.length}个测试</p>`;
            }
            
            container.innerHTML = html;
        }
        
        // 页面加载时更新数据
        updateDashboard();
        
        // 每30秒自动更新
        setInterval(updateDashboard, 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """主仪表板页面"""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/dashboard_data')
def get_dashboard_data():
    """获取仪表板数据API"""
    return jsonify(dashboard_data)

@app.route('/api/system_status')
def get_system_status():
    """获取系统状态API"""
    return jsonify(advanced_manager.get_system_status())

@app.route('/api/model_versions')
def get_model_versions():
    """获取模型版本API"""
    return jsonify(advanced_manager.list_model_versions())

@app.route('/api/ab_tests')
def get_ab_tests():
    """获取A/B测试API"""
    return jsonify(advanced_manager.list_ab_tests())

@app.route('/api/performance_metrics')
def get_performance_metrics():
    """获取性能指标API"""
    return jsonify(dashboard_data['performance_metrics'])

if __name__ == '__main__':
    print("🌐 启动Enhanced CEP-EIT-P Web监控仪表板...")
    print("访问地址: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)