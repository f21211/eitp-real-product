#!/bin/bash

# Enhanced CEP-EIT-P Web Dashboard 启动脚本
# Web监控仪表板启动脚本

echo "🌐 启动Enhanced CEP-EIT-P Web监控仪表板..."

# 检查虚拟环境
if [ ! -d "/mnt/sda1/myproject/datainall/AGI/venv" ]; then
    echo "❌ 虚拟环境不存在，请先创建虚拟环境"
    exit 1
fi

# 激活虚拟环境
source /mnt/sda1/myproject/datainall/AGI/venv/bin/activate

# 检查Flask是否安装
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 安装Flask..."
    pip install flask
fi

# 启动Web仪表板
echo "🚀 启动Web仪表板服务..."
echo "访问地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务"

# 使用nohup在后台运行
nohup python web_dashboard.py > web_dashboard.log 2>&1 &
WEB_DASHBOARD_PID=$!

# 保存PID
echo $WEB_DASHBOARD_PID > web_dashboard.pid

echo "✅ Web仪表板已启动 (PID: $WEB_DASHBOARD_PID)"
echo "📝 日志文件: web_dashboard.log"
echo "🆔 PID文件: web_dashboard.pid"

# 等待几秒检查服务状态
sleep 3
if ps -p $WEB_DASHBOARD_PID > /dev/null; then
    echo "🎉 Web仪表板运行正常!"
    echo "🌐 请访问: http://localhost:5000"
else
    echo "❌ Web仪表板启动失败，请检查日志"
    cat web_dashboard.log
fi
