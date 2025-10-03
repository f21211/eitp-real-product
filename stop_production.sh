#!/bin/bash
# EIT-P Production Stop Script
# Generated on: 2025-10-03T09:39:38.687621

echo "🛑 停止EIT-P生产环境..."
echo "================================================================================="

# 读取进程ID
if [ -f .api_pid ]; then
    API_PID=
    echo "📋 停止API服务器 (PID: )..."
    kill  2>/dev/null || echo "API服务器已停止"
    rm .api_pid
fi

if [ -f .dashboard_pid ]; then
    DASHBOARD_PID=
    echo "📋 停止监控仪表板 (PID: )..."
    kill  2>/dev/null || echo "监控仪表板已停止"
    rm .dashboard_pid
fi

# 清理其他相关进程
pkill -f "python3 scripts/api_server.py" 2>/dev/null || true
pkill -f "python3 scripts/monitor_dashboard.py" 2>/dev/null || true

echo "✅ 生产环境已停止"
