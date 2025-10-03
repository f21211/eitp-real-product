#!/bin/bash

# EIT-P 生产部署脚本
# EIT-P Production Deployment Script

echo "🚀 开始EIT-P生产部署..."
echo "=================================="

# 1. 检查环境
echo "📋 检查环境..."
if ! command -v git &> /dev/null; then
    echo "❌ Git未安装"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    exit 1
fi

echo "✅ 环境检查通过"

# 2. 创建GitHub仓库
echo ""
echo "📦 创建GitHub仓库..."
echo "请手动执行以下步骤："
echo "1. 访问 https://github.com/new"
echo "2. 仓库名称: eitp-ai-platform"
echo "3. 描述: EIT-P: Enterprise Intelligence Training Platform"
echo "4. 选择: Public"
echo "5. 点击 'Create repository'"
echo ""
read -p "按Enter键继续..."

# 3. 配置Git远程仓库
echo ""
echo "🔧 配置Git远程仓库..."
git remote set-url origin git@github.com:f21211/eitp-ai-platform.git
echo "✅ Git远程仓库配置完成"

# 4. 推送代码
echo ""
echo "📤 推送代码到GitHub..."
if git push origin main; then
    echo "✅ 代码推送成功"
else
    echo "❌ 代码推送失败，请检查网络连接和权限"
    exit 1
fi

# 5. 配置云存储
echo ""
echo "☁️ 配置云存储..."
echo "请选择云存储提供商："
echo "1) AWS S3"
echo "2) Google Cloud Storage"
echo "3) Azure Blob Storage"
echo "4) 阿里云 OSS"
read -p "请输入选择 (1-4): " provider_choice

case $provider_choice in
    1)
        echo "配置AWS S3..."
        echo "请设置以下环境变量："
        echo "export AWS_ACCESS_KEY_ID=your_access_key"
        echo "export AWS_SECRET_ACCESS_KEY=your_secret_key"
        echo "export AWS_DEFAULT_REGION=us-east-1"
        read -p "按Enter键继续..."
        PROVIDER="aws_s3"
        ;;
    2)
        echo "配置Google Cloud Storage..."
        echo "请设置以下环境变量："
        echo "export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json"
        echo "export GOOGLE_CLOUD_PROJECT=your-project-id"
        read -p "按Enter键继续..."
        PROVIDER="google_cloud"
        ;;
    3)
        echo "配置Azure Blob Storage..."
        echo "请设置以下环境变量："
        echo "export AZURE_STORAGE_CONNECTION_STRING=your_connection_string"
        read -p "按Enter键继续..."
        PROVIDER="azure_blob"
        ;;
    4)
        echo "配置阿里云 OSS..."
        echo "请设置以下环境变量："
        echo "export ALIYUN_ACCESS_KEY_ID=your_access_key"
        echo "export ALIYUN_ACCESS_KEY_SECRET=your_secret_key"
        read -p "按Enter键继续..."
        PROVIDER="aliyun_oss"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

# 6. 上传模型到云存储
echo ""
echo "📤 上传模型到云存储..."
if [ -d "models" ]; then
    python3 upload_models_to_cloud.py --provider $PROVIDER --models-dir models/
    if [ $? -eq 0 ]; then
        echo "✅ 模型上传成功"
    else
        echo "❌ 模型上传失败"
    fi
else
    echo "⚠️ 未找到models目录，跳过模型上传"
fi

# 7. 安装依赖
echo ""
echo "📦 安装依赖..."
pip3 install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✅ 依赖安装成功"
else
    echo "❌ 依赖安装失败"
    exit 1
fi

# 8. 启动服务
echo ""
echo "🚀 启动EIT-P服务..."
if [ -f "scripts/start_production_enhanced.sh" ]; then
    chmod +x scripts/start_production_enhanced.sh
    ./scripts/start_production_enhanced.sh
else
    echo "⚠️ 未找到启动脚本，请手动启动服务"
fi

# 9. 验证部署
echo ""
echo "🔍 验证部署..."
sleep 5

# 检查API服务
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo "✅ API服务运行正常"
else
    echo "❌ API服务未运行"
fi

# 检查监控面板
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ 监控面板运行正常"
else
    echo "❌ 监控面板未运行"
fi

# 检查文档服务
if curl -s http://localhost:8002/health > /dev/null; then
    echo "✅ 文档服务运行正常"
else
    echo "❌ 文档服务未运行"
fi

# 10. 显示访问信息
echo ""
echo "🎉 EIT-P部署完成！"
echo "=================================="
echo "📊 服务访问地址："
echo "  API服务: http://localhost:8000"
echo "  监控面板: http://localhost:8001"
echo "  文档服务: http://localhost:8002"
echo ""
echo "📚 文档地址："
echo "  API文档: http://localhost:8002/api/docs"
echo "  用户指南: http://localhost:8002/guide"
echo ""
echo "🔧 管理命令："
echo "  启动服务: ./scripts/start_production_enhanced.sh"
echo "  停止服务: ./scripts/stop_services.sh"
echo "  检查状态: ./scripts/status_services.sh"
echo ""
echo "☁️ 云存储管理："
echo "  上传模型: python3 upload_models_to_cloud.py --provider $PROVIDER"
echo "  下载模型: python3 download_models_from_cloud.py --provider $PROVIDER --list"
echo ""
echo "🎯 EIT-P框架已成功部署并运行！"
