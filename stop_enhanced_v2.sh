#!/bin/bash
# Enhanced CEP-EIT-P API Server V2 停止脚本

echo "⏹️ 停止 Enhanced CEP-EIT-P API Server V2..."

# 检查PID文件
if [ -f "enhanced_api_v2.pid" ]; then
    API_PID=$(cat enhanced_api_v2.pid)
    echo "📊 找到PID: $API_PID"
    
    # 检查进程是否存在
    if ps -p $API_PID > /dev/null 2>&1; then
        echo "🔄 正在停止进程 $API_PID..."
        kill $API_PID
        
        # 等待进程停止
        sleep 3
        
        # 检查是否已停止
        if ps -p $API_PID > /dev/null 2>&1; then
            echo "⚠️ 进程未正常停止，强制终止..."
            kill -9 $API_PID
            sleep 1
        fi
        
        echo "✅ 进程已停止"
    else
        echo "⚠️ 进程不存在或已停止"
    fi
    
    # 删除PID文件
    rm -f enhanced_api_v2.pid
else
    echo "⚠️ 未找到PID文件，尝试通过进程名停止..."
    pkill -f "enhanced_api_server_v2"
fi

# 检查端口是否释放
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️ 端口5000仍被占用，尝试强制释放..."
    fuser -k 5000/tcp
    sleep 2
fi

# 最终检查
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "❌ 无法释放端口5000"
    exit 1
else
    echo "✅ Enhanced CEP-EIT-P API Server V2 已完全停止"
    echo "🌐 端口5000已释放"
fi
