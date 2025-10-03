#!/bin/bash
# EIT-P 生产部署脚本

set -e

echo "🚀 EIT-P 生产部署脚本"
echo "========================"

# 检查Docker和Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 检查NVIDIA Docker支持
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Docker支持未正确配置"
    exit 1
fi

echo "✅ 环境检查通过"

# 创建必要的目录
echo "📁 创建目录结构..."
mkdir -p data results logs models scripts

# 设置权限
chmod +x scripts/*.sh

# 构建镜像
echo "🔨 构建Docker镜像..."
docker-compose build --no-cache

# 启动服务
echo "🚀 启动EIT-P服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 30

# 检查服务状态
echo "🔍 检查服务状态..."
docker-compose ps

# 显示日志
echo "📋 显示训练日志..."
docker-compose logs -f eitp-training &

# 显示访问信息
echo ""
echo "🎉 部署完成！"
echo "========================"
echo "📊 监控仪表板: http://localhost:8082"
echo "🔌 REST API: http://localhost:8083"
echo "📈 训练状态: http://localhost:8081"
echo ""
echo "📋 管理命令:"
echo "  查看日志: docker-compose logs -f"
echo "  停止服务: docker-compose down"
echo "  重启服务: docker-compose restart"
echo "  查看状态: docker-compose ps"
echo ""
echo "🔧 配置文件: config.yaml"
echo "📁 结果目录: ./results"
echo "📁 日志目录: ./logs"
