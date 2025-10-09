#!/bin/bash

# Enhanced CEP-EIT-P Web Dashboard 停止脚本
# Web监控仪表板停止脚本

echo "🛑 停止Enhanced CEP-EIT-P Web监控仪表板..."

# 检查PID文件
if [ -f "web_dashboard.pid" ]; then
    WEB_DASHBOARD_PID=$(cat web_dashboard.pid)
    
    if ps -p $WEB_DASHBOARD_PID > /dev/null; then
        echo "🔍 找到运行中的Web仪表板进程 (PID: $WEB_DASHBOARD_PID)"
        
        # 优雅停止
        echo "📤 发送停止信号..."
        kill $WEB_DASHBOARD_PID
        
        # 等待进程结束
        sleep 2
        
        # 检查是否还在运行
        if ps -p $WEB_DASHBOARD_PID > /dev/null; then
            echo "⚠️ 进程仍在运行，强制停止..."
            kill -9 $WEB_DASHBOARD_PID
            sleep 1
        fi
        
        # 再次检查
        if ps -p $WEB_DASHBOARD_PID > /dev/null; then
            echo "❌ 无法停止Web仪表板进程"
            exit 1
        else
            echo "✅ Web仪表板已成功停止"
        fi
    else
        echo "ℹ️ PID文件存在但进程未运行"
    fi
    
    # 删除PID文件
    rm -f web_dashboard.pid
    echo "🗑️ 已删除PID文件"
else
    echo "⚠️ 未找到PID文件，尝试通过进程名停止..."
    
    # 通过进程名查找并停止
    PIDS=$(pgrep -f "web_dashboard.py")
    if [ -n "$PIDS" ]; then
        echo "🔍 找到Web仪表板进程: $PIDS"
        kill $PIDS
        sleep 2
        
        # 检查是否还在运行
        REMAINING_PIDS=$(pgrep -f "web_dashboard.py")
        if [ -n "$REMAINING_PIDS" ]; then
            echo "⚠️ 进程仍在运行，强制停止..."
            kill -9 $REMAINING_PIDS
        fi
        
        echo "✅ Web仪表板已停止"
    else
        echo "ℹ️ 未找到运行中的Web仪表板进程"
    fi
fi

# 清理日志文件（可选）
if [ "$1" = "--clean-logs" ]; then
    echo "🧹 清理日志文件..."
    rm -f web_dashboard.log
    echo "✅ 日志文件已清理"
fi

echo "🎉 Web仪表板停止完成!"
