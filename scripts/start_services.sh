#!/bin/bash
# EIT-P 服务启动脚本

set -e

echo "🚀 EIT-P 服务启动脚本"
echo "========================"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在，请先运行 setup.sh"
    exit 1
fi

# 激活虚拟环境
source venv/bin/activate

# 检查依赖
echo "🔍 检查依赖..."
python3 -c "import torch, transformers, flask" 2>/dev/null || {
    echo "❌ 缺少依赖，请运行: pip install -r requirements.txt"
    exit 1
}

# 创建必要目录
echo "📁 创建目录..."
mkdir -p experiments models metrics logs data results

# 设置权限
chmod +x scripts/*.sh

# 启动服务
echo "🚀 启动服务..."

# 启动API服务器（后台）
echo "启动API服务器..."
nohup python3 scripts/api_server.py > logs/api_server.log 2>&1 &
API_PID=$!
echo "API服务器PID: $API_PID"

# 等待API服务器启动
sleep 5

# 启动监控仪表板（后台）
echo "启动监控仪表板..."
nohup python3 scripts/monitor_dashboard.py > logs/monitor_dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "监控仪表板PID: $DASHBOARD_PID"

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo "🔍 检查服务状态..."

# 检查API服务器
if curl -s http://localhost:8083/api/health > /dev/null; then
    echo "✅ API服务器运行正常"
else
    echo "❌ API服务器启动失败"
fi

# 检查监控仪表板
if curl -s http://localhost:8082 > /dev/null; then
    echo "✅ 监控仪表板运行正常"
else
    echo "❌ 监控仪表板启动失败"
fi

# 保存PID
echo "$API_PID" > logs/api_server.pid
echo "$DASHBOARD_PID" > logs/monitor_dashboard.pid

echo ""
echo "🎉 服务启动完成！"
echo "========================"
echo "📊 监控仪表板: http://localhost:8082"
echo "🔌 REST API: http://localhost:8083"
echo "📈 健康检查: http://localhost:8083/api/health"
echo ""
echo "📋 管理命令:"
echo "  查看日志: tail -f logs/*.log"
echo "  停止服务: ./scripts/stop_services.sh"
echo "  重启服务: ./scripts/restart_services.sh"
echo "  查看状态: ./scripts/status_services.sh"
echo ""
echo "🔧 开始训练:"
echo "  python3 production_train.py 'my_experiment' 'gpt2' './data/train.txt'"
