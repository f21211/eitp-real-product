#!/bin/bash
# 训练监控脚本 - 监控系统资源并在过载时告警

LOG_FILE="training_monitor.log"
ALERT_THRESHOLD_MEM=85  # CPU内存告警阈值 (%)
ALERT_THRESHOLD_CPU=90  # CPU使用率告警阈值 (%)
CHECK_INTERVAL=5  # 检查间隔(秒)

echo "===== 训练监控开始 =====" | tee -a "$LOG_FILE"
echo "时间: $(date)" | tee -a "$LOG_FILE"

while true; do
    # 获取时间戳
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 检查CPU使用率
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    
    # 检查内存使用率
    MEM_USAGE=$(free | grep Mem | awk '{printf("%.1f"), $3/$2 * 100.0}')
    
    # 检查GPU使用率(如果有NVIDIA GPU)
    if command -v nvidia-smi &> /dev/null; then
        GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ ! -z "$GPU_MEM" ] && [ ! -z "$GPU_MEM_TOTAL" ]; then
            GPU_MEM_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($GPU_MEM/$GPU_MEM_TOTAL)*100}")
        else
            GPU_MEM_PERCENT="N/A"
        fi
    else
        GPU_USAGE="N/A"
        GPU_MEM_PERCENT="N/A"
    fi
    
    # 检查Python进程数量
    PYTHON_PROCS=$(pgrep -c python || echo 0)
    
    # 输出状态
    STATUS="[$TIMESTAMP] CPU: ${CPU_USAGE}% | MEM: ${MEM_USAGE}% | GPU: ${GPU_USAGE}% | GPU_MEM: ${GPU_MEM_PERCENT}% | Python进程: $PYTHON_PROCS"
    echo "$STATUS" | tee -a "$LOG_FILE"
    
    # 告警检查
    if (( $(echo "$MEM_USAGE > $ALERT_THRESHOLD_MEM" | bc -l) )); then
        echo "⚠️  [ALERT] 内存使用率过高: ${MEM_USAGE}%" | tee -a "$LOG_FILE"
    fi
    
    if (( $(echo "$CPU_USAGE > $ALERT_THRESHOLD_CPU" | bc -l) )); then
        echo "⚠️  [ALERT] CPU使用率过高: ${CPU_USAGE}%" | tee -a "$LOG_FILE"
    fi
    
    # 检查训练进程是否存活
    if [ $PYTHON_PROCS -eq 0 ]; then
        echo "⚠️  [ALERT] 未检测到Python训练进程，训练可能已停止！" | tee -a "$LOG_FILE"
    fi
    
    sleep $CHECK_INTERVAL
done

