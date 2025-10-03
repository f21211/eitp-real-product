#!/bin/bash

# EIT-P 生产环境增强启动脚本
# 提供完整的生产级服务启动、监控和健康检查

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="/mnt/sda1/myproject/datainall/AGI"
cd "$PROJECT_ROOT"

# 日志目录
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# PID文件目录
PID_DIR="$PROJECT_ROOT/pids"
mkdir -p "$PID_DIR"

# 配置文件
CONFIG_FILE="$PROJECT_ROOT/config/production.yaml"

# 服务端口配置
API_SERVER_PORT=8085
MONITOR_DASHBOARD_PORT=8082
INFERENCE_SERVICE_PORT=8086
AUTH_SERVICE_PORT=8087
MODEL_MANAGEMENT_PORT=8090
DOCS_SERVER_PORT=8088
ADVANCED_MONITOR_PORT=8089

# 服务状态
declare -A SERVICES=(
    ["api_server"]="API服务器"
    ["monitor_dashboard"]="监控仪表板"
    ["inference_service"]="推理服务"
    ["auth_service"]="认证服务"
    ["model_management"]="模型管理"
    ["docs_server"]="文档服务"
    ["advanced_monitor"]="高级监控"
)

# 函数：打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

# 函数：检查端口是否被占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # 端口被占用
    else
        return 1  # 端口空闲
    fi
}

# 函数：检查服务是否运行
check_service() {
    local service_name=$1
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0  # 服务运行中
        else
            rm -f "$pid_file"
            return 1  # 服务未运行
        fi
    else
        return 1  # 服务未运行
    fi
}

# 函数：启动服务
start_service() {
    local service_name=$1
    local service_script=$2
    local port=$3
    local pid_file="$PID_DIR/${service_name}.pid"
    local log_file="$LOG_DIR/${service_name}.log"
    
    print_message $BLUE "正在启动 ${SERVICES[$service_name]}..."
    
    # 检查端口是否被占用
    if check_port $port; then
        print_message $YELLOW "端口 $port 已被占用，尝试停止现有服务..."
        stop_service $service_name
        sleep 2
    fi
    
    # 检查服务是否已在运行
    if check_service $service_name; then
        print_message $YELLOW "${SERVICES[$service_name]} 已在运行中"
        return 0
    fi
    
    # 启动服务
    nohup python3 "$service_script" > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"
    
    # 等待服务启动
    sleep 3
    
    # 检查服务是否成功启动
    if check_service $service_name; then
        print_message $GREEN "✅ ${SERVICES[$service_name]} 启动成功 (PID: $pid, 端口: $port)"
        return 0
    else
        print_message $RED "❌ ${SERVICES[$service_name]} 启动失败"
        return 1
    fi
}

# 函数：停止服务
stop_service() {
    local service_name=$1
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            print_message $YELLOW "正在停止 ${SERVICES[$service_name]} (PID: $pid)..."
            kill -TERM "$pid"
            
            # 等待进程结束
            local count=0
            while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            # 强制杀死进程
            if ps -p "$pid" > /dev/null 2>&1; then
                print_message $YELLOW "强制停止 ${SERVICES[$service_name]}..."
                kill -KILL "$pid"
            fi
            
            print_message $GREEN "✅ ${SERVICES[$service_name]} 已停止"
        fi
        rm -f "$pid_file"
    else
        print_message $YELLOW "${SERVICES[$service_name]} 未运行"
    fi
}

# 函数：检查服务健康状态
check_health() {
    local service_name=$1
    local port=$2
    local health_url="http://localhost:$port/health"
    
    if curl -s -f "$health_url" > /dev/null 2>&1; then
        print_message $GREEN "✅ ${SERVICES[$service_name]} 健康检查通过"
        return 0
    else
        print_message $RED "❌ ${SERVICES[$service_name]} 健康检查失败"
        return 1
    fi
}

# 函数：显示服务状态
show_status() {
    print_message $BLUE "=== EIT-P 服务状态 ==="
    
    local all_healthy=true
    
    for service_name in "${!SERVICES[@]}"; do
        local port_var="${service_name^^}_PORT"
        local port=${!port_var}
        
        if check_service $service_name; then
            if check_health $service_name $port; then
                print_message $GREEN "✅ ${SERVICES[$service_name]} - 运行中 (端口: $port)"
            else
                print_message $YELLOW "⚠️  ${SERVICES[$service_name]} - 运行中但健康检查失败 (端口: $port)"
                all_healthy=false
            fi
        else
            print_message $RED "❌ ${SERVICES[$service_name]} - 未运行 (端口: $port)"
            all_healthy=false
        fi
    done
    
    echo
    if [ "$all_healthy" = true ]; then
        print_message $GREEN "🎉 所有服务运行正常！"
    else
        print_message $YELLOW "⚠️  部分服务存在问题，请检查日志"
    fi
}

# 函数：启动所有服务
start_all() {
    print_message $BLUE "🚀 启动EIT-P生产环境服务..."
    
    # 检查配置文件
    if [ ! -f "$CONFIG_FILE" ]; then
        print_message $RED "❌ 配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
    
    # 检查Python环境
    if ! command -v python3 &> /dev/null; then
        print_message $RED "❌ Python3 未安装"
        exit 1
    fi
    
    # 检查依赖
    print_message $BLUE "检查依赖包..."
    python3 -c "import flask, torch, psutil, yaml" 2>/dev/null || {
        print_message $RED "❌ 缺少必要的Python包，请运行: pip install -r requirements.txt"
        exit 1
    }
    
    # 启动服务
    local failed_services=()
    
    start_service "api_server" "scripts/api_server.py" $API_SERVER_PORT || failed_services+=("api_server")
    start_service "monitor_dashboard" "scripts/monitor_dashboard.py" $MONITOR_DASHBOARD_PORT || failed_services+=("monitor_dashboard")
    start_service "inference_service" "scripts/inference_service.py" $INFERENCE_SERVICE_PORT || failed_services+=("inference_service")
    start_service "auth_service" "scripts/auth_service.py" $AUTH_SERVICE_PORT || failed_services+=("auth_service")
    start_service "model_management" "scripts/model_management.py" $MODEL_MANAGEMENT_PORT || failed_services+=("model_management")
    start_service "docs_server" "scripts/docs_server.py" $DOCS_SERVER_PORT || failed_services+=("docs_server")
    start_service "advanced_monitor" "scripts/advanced_monitor.py" $ADVANCED_MONITOR_PORT || failed_services+=("advanced_monitor")
    
    # 等待所有服务启动
    print_message $BLUE "等待服务启动完成..."
    sleep 10
    
    # 显示状态
    show_status
    
    # 显示访问信息
    echo
    print_message $BLUE "=== 服务访问地址 ==="
    echo "🌐 API服务器: http://localhost:$API_SERVER_PORT"
    echo "📊 监控仪表板: http://localhost:$MONITOR_DASHBOARD_PORT"
    echo "🤖 推理服务: http://localhost:$INFERENCE_SERVICE_PORT"
    echo "🔐 认证服务: http://localhost:$AUTH_SERVICE_PORT"
    echo "📦 模型管理: http://localhost:$MODEL_MANAGEMENT_PORT"
    echo "📚 API文档: http://localhost:$DOCS_SERVER_PORT"
    echo "🔍 高级监控: http://localhost:$ADVANCED_MONITOR_PORT"
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        print_message $RED "❌ 以下服务启动失败: ${failed_services[*]}"
        print_message $YELLOW "请检查日志文件: $LOG_DIR/"
        exit 1
    else
        print_message $GREEN "🎉 所有服务启动成功！"
    fi
}

# 函数：停止所有服务
stop_all() {
    print_message $BLUE "🛑 停止EIT-P生产环境服务..."
    
    for service_name in "${!SERVICES[@]}"; do
        stop_service $service_name
    done
    
    print_message $GREEN "✅ 所有服务已停止"
}

# 函数：重启服务
restart_service() {
    local service_name=$1
    local port_var="${service_name^^}_PORT"
    local port=${!port_var}
    
    print_message $BLUE "🔄 重启 ${SERVICES[$service_name]}..."
    stop_service $service_name
    sleep 2
    start_service $service_name "scripts/${service_name}.py" $port
}

# 函数：显示日志
show_logs() {
    local service_name=$1
    local log_file="$LOG_DIR/${service_name}.log"
    
    if [ -f "$log_file" ]; then
        print_message $BLUE "显示 ${SERVICES[$service_name]} 日志 (最后50行):"
        echo "----------------------------------------"
        tail -n 50 "$log_file"
    else
        print_message $RED "日志文件不存在: $log_file"
    fi
}

# 函数：清理日志
clean_logs() {
    print_message $BLUE "🧹 清理日志文件..."
    
    # 保留最近7天的日志
    find "$LOG_DIR" -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    # 压缩旧日志
    find "$LOG_DIR" -name "*.log" -mtime +1 -exec gzip {} \; 2>/dev/null || true
    
    print_message $GREEN "✅ 日志清理完成"
}

# 函数：显示帮助信息
show_help() {
    echo "EIT-P 生产环境管理脚本"
    echo
    echo "用法: $0 [命令] [服务名]"
    echo
    echo "命令:"
    echo "  start       启动所有服务"
    echo "  stop        停止所有服务"
    echo "  restart     重启所有服务"
    echo "  status      显示服务状态"
    echo "  health      健康检查"
    echo "  logs        显示服务日志"
    echo "  clean       清理日志"
    echo "  help        显示帮助信息"
    echo
    echo "服务名:"
    for service_name in "${!SERVICES[@]}"; do
        echo "  $service_name - ${SERVICES[$service_name]}"
    done
    echo
    echo "示例:"
    echo "  $0 start                    # 启动所有服务"
    echo "  $0 stop                     # 停止所有服务"
    echo "  $0 restart api_server       # 重启API服务器"
    echo "  $0 logs monitor_dashboard   # 查看监控仪表板日志"
    echo "  $0 status                   # 查看服务状态"
}

# 主逻辑
case "${1:-start}" in
    "start")
        start_all
        ;;
    "stop")
        stop_all
        ;;
    "restart")
        if [ -n "$2" ]; then
            restart_service "$2"
        else
            stop_all
            sleep 3
            start_all
        fi
        ;;
    "status")
        show_status
        ;;
    "health")
        show_status
        ;;
    "logs")
        if [ -n "$2" ]; then
            show_logs "$2"
        else
            print_message $RED "请指定服务名"
            show_help
        fi
        ;;
    "clean")
        clean_logs
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_message $RED "未知命令: $1"
        show_help
        exit 1
        ;;
esac
