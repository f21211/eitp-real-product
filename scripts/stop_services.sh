#!/bin/bash
# EIT-P 服务停止脚本

echo "🛑 EIT-P 服务停止脚本"
echo "========================"

# 停止API服务器
if [ -f "logs/api_server.pid" ]; then
    API_PID=$(cat logs/api_server.pid)
    if kill -0 $API_PID 2>/dev/null; then
        echo "停止API服务器 (PID: $API_PID)..."
        kill $API_PID
        sleep 2
        if kill -0 $API_PID 2>/dev/null; then
            echo "强制停止API服务器..."
            kill -9 $API_PID
        fi
        echo "✅ API服务器已停止"
    else
        echo "⚠️ API服务器未运行"
    fi
    rm -f logs/api_server.pid
else
    echo "⚠️ 未找到API服务器PID文件"
fi

# 停止监控仪表板
if [ -f "logs/monitor_dashboard.pid" ]; then
    DASHBOARD_PID=$(cat logs/monitor_dashboard.pid)
    if kill -0 $DASHBOARD_PID 2>/dev/null; then
        echo "停止监控仪表板 (PID: $DASHBOARD_PID)..."
        kill $DASHBOARD_PID
        sleep 2
        if kill -0 $DASHBOARD_PID 2>/dev/null; then
            echo "强制停止监控仪表板..."
            kill -9 $DASHBOARD_PID
        fi
        echo "✅ 监控仪表板已停止"
    else
        echo "⚠️ 监控仪表板未运行"
    fi
    rm -f logs/monitor_dashboard.pid
else
    echo "⚠️ 未找到监控仪表板PID文件"
fi

# 清理端口
echo "🧹 清理端口..."
lsof -ti:8082,8083 | xargs -r kill -9 2>/dev/null || true

echo ""
echo "✅ 所有服务已停止"
