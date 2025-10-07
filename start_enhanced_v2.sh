#!/bin/bash
# Enhanced CEP-EIT-P API Server V2 启动脚本

echo "🚀 启动 Enhanced CEP-EIT-P API Server V2..."

# 检查虚拟环境
if [ ! -d "/mnt/sda1/myproject/datainall/AGI/venv" ]; then
    echo "❌ 虚拟环境不存在，请先创建虚拟环境"
    exit 1
fi

# 激活虚拟环境
source /mnt/sda1/myproject/datainall/AGI/venv/bin/activate

# 检查端口是否被占用
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️ 端口5000已被占用，正在停止现有服务..."
    pkill -f "enhanced_api_server"
    sleep 2
fi

# 启动API服务器
echo "🎯 启动API服务器..."
nohup python3 enhanced_api_server_v2.py > enhanced_api_v2.log 2>&1 &

# 获取进程ID
API_PID=$!
echo $API_PID > enhanced_api_v2.pid

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 5

# 检查服务状态
if curl -s http://localhost:5000/api/health > /dev/null; then
    echo "✅ Enhanced CEP-EIT-P API Server V2 启动成功!"
    echo "📊 PID: $API_PID"
    echo "🌐 访问地址: http://localhost:5000"
    echo "📝 日志文件: enhanced_api_v2.log"
    echo "🆔 PID文件: enhanced_api_v2.pid"
    
    # 显示API端点
    echo ""
    echo "📋 可用API端点:"
    echo "  - GET  /                    # 欢迎页面"
    echo "  - GET  /api/health          # 健康检查"
    echo "  - GET  /api/model_info      # 模型信息"
    echo "  - POST /api/inference       # 推理服务"
    echo "  - POST /api/batch_inference # 批量推理"
    echo "  - GET  /api/consciousness   # 意识分析"
    echo "  - POST /api/energy_analysis # 能量分析"
    echo "  - GET  /api/performance     # 性能指标"
    echo "  - POST /api/optimize        # 模型优化"
    echo "  - GET  /api/history         # 历史数据"
    echo "  - GET  /api/statistics      # 统计分析"
    echo "  - POST /api/reset_metrics   # 重置指标"
    
    # 测试API
    echo ""
    echo "🧪 测试API连接..."
    curl -s http://localhost:5000/api/health | head -3
else
    echo "❌ API服务器启动失败"
    echo "📝 查看日志: tail -f enhanced_api_v2.log"
    exit 1
fi
