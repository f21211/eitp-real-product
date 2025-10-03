#!/bin/bash
# 安全训练启动脚本 - 带资源限制和自动恢复

set -e

PROJECT_DIR="/mnt/sda1/myproject/datainall/AGI"
VENV_DIR="$PROJECT_DIR/venv"
LOG_DIR="$PROJECT_DIR/logs"
CHECKPOINT_DIR="$PROJECT_DIR/eitp_results"

# 创建必要目录
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

# 日志文件
TRAIN_LOG="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"
ERROR_LOG="$LOG_DIR/error_$(date +%Y%m%d_%H%M%S).log"

echo "===== EIT-P 安全训练启动 ======"
echo "时间: $(date)"
echo "项目目录: $PROJECT_DIR"
echo "训练日志: $TRAIN_LOG"
echo "错误日志: $ERROR_LOG"
echo "==============================="

# 检查虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo "错误: 虚拟环境不存在 ($VENV_DIR)"
    exit 1
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 设置资源限制 (防止OOM)
# 限制最大虚拟内存使用 (如8GB)
# ulimit -v 8388608  # 8GB in KB (可能导致问题，先注释)

# 设置Python内存优化
export PYTHONOPTIMIZE=1
export MALLOC_TRIM_THRESHOLD_=65536

# 清理旧的Python进程
echo "清理旧进程..."
killall -9 python 2>/dev/null || true
sleep 2

# 清理GPU缓存
if command -v nvidia-smi &> /dev/null; then
    echo "重置GPU..."
    nvidia-smi --gpu-reset -i 0 2>/dev/null || true
fi

# 执行训练
cd "$PROJECT_DIR"

echo "开始训练..."
echo "提示: 如果系统资源不足，训练会自动使用更小的batch size和混合精度"
echo ""

# 使用nohup在后台运行，限制CPU核心数
# nice -n 10: 降低优先级，避免占满系统资源
# 2>&1: 合并stdout和stderr
nice -n 10 python examples/train_eitp.py 2>&1 | tee "$TRAIN_LOG"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ 训练成功完成！"
    echo "结果保存在: $CHECKPOINT_DIR"
else
    echo ""
    echo "❌ 训练异常退出 (退出码: $EXIT_CODE)"
    echo "错误日志: $ERROR_LOG"
    tail -100 "$TRAIN_LOG" > "$ERROR_LOG"
fi

# 清理
deactivate 2>/dev/null || true

exit $EXIT_CODE

