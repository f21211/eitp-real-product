#!/bin/bash
# EIT-P 服务状态检查脚本

echo "📊 EIT-P 服务状态"
echo "========================"

# 检查API服务器
echo "🔌 API服务器 (端口 8083):"
if curl -s http://localhost:8083/api/health > /dev/null; then
    echo "  ✅ 运行正常"
    HEALTH_RESPONSE=$(curl -s http://localhost:8083/api/health)
    echo "  📋 响应: $HEALTH_RESPONSE"
else
    echo "  ❌ 未运行或异常"
fi

# 检查监控仪表板
echo ""
echo "📊 监控仪表板 (端口 8082):"
if curl -s http://localhost:8082 > /dev/null; then
    echo "  ✅ 运行正常"
else
    echo "  ❌ 未运行或异常"
fi

# 检查进程
echo ""
echo "🔍 进程状态:"
API_PID=$(ps aux | grep "api_server.py" | grep -v grep | awk '{print $2}' | head -1)
if [ ! -z "$API_PID" ]; then
    echo "  API服务器 PID: $API_PID"
else
    echo "  API服务器: 未运行"
fi

DASHBOARD_PID=$(ps aux | grep "monitor_dashboard.py" | grep -v grep | awk '{print $2}' | head -1)
if [ ! -z "$DASHBOARD_PID" ]; then
    echo "  监控仪表板 PID: $DASHBOARD_PID"
else
    echo "  监控仪表板: 未运行"
fi

# 检查端口
echo ""
echo "🌐 端口使用情况:"
lsof -i :8082,8083 2>/dev/null || echo "  无服务占用端口"

# 检查日志
echo ""
echo "📋 最近日志:"
if [ -f "logs/api_server.log" ]; then
    echo "  API服务器日志 (最后5行):"
    tail -5 logs/api_server.log | sed 's/^/    /'
fi

if [ -f "logs/monitor_dashboard.log" ]; then
    echo "  监控仪表板日志 (最后5行):"
    tail -5 logs/monitor_dashboard.log | sed 's/^/    /'
fi

# 检查实验和模型
echo ""
echo "🧪 实验统计:"
if [ -d "experiments/experiments" ]; then
    EXPERIMENT_COUNT=$(find experiments/experiments -maxdepth 1 -type d | wc -l)
    echo "  实验数量: $((EXPERIMENT_COUNT - 1))"
else
    echo "  实验数量: 0"
fi

if [ -d "models/models" ]; then
    MODEL_COUNT=$(find models/models -maxdepth 1 -type d | wc -l)
    echo "  模型数量: $((MODEL_COUNT - 1))"
else
    echo "  模型数量: 0"
fi

# 系统资源
echo ""
echo "💻 系统资源:"
echo "  CPU使用率: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "  内存使用率: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU状态:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed 's/^/    /'
fi

echo ""
echo "🔗 访问链接:"
echo "  监控仪表板: http://localhost:8082"
echo "  REST API: http://localhost:8083"
echo "  API文档: http://localhost:8083/api/health"
