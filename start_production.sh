#!/bin/bash
# EIT-P Production Startup Script
# Generated on: 2025-10-03T09:39:38.687602

echo "🚀 启动EIT-P生产环境..."
echo "================================================================================="

# 设置环境变量
export 

# 创建必要目录
mkdir -p logs data experiments/models experiments/experiments

# 启动API服务器
echo "📋 启动API服务器..."
python3 scripts/api_server.py &
API_PID=

# 启动监控仪表板
echo "📋 启动监控仪表板..."
python3 scripts/monitor_dashboard.py &
DASHBOARD_PID=

# 保存进程ID
echo  > .api_pid
echo  > .dashboard_pid

echo "✅ 生产环境启动完成"
echo "  • API服务器: http://localhost:8085 (PID: )"
echo "  • 监控仪表板: http://localhost:8082 (PID: )"
echo "  • 进程ID文件: .api_pid, .dashboard_pid"
