#!/usr/bin/env python3
"""
EIT-P 模型从云存储下载脚本
Download EIT-P Models from Cloud Storage

使用方法:
python download_models_from_cloud.py --provider aws_s3 --model-name eitp_model --version v1.0.0
"""

import argparse
import sys
import os
from pathlib import Path
from scripts.cloud_storage_manager import CloudStorageManager

def download_model_from_cloud(model_name: str, version: str, provider: str = "aws_s3", output_dir: str = "downloaded_models"):
    """从云存储下载模型"""
    
    print("📥 开始从云存储下载EIT-P模型...")
    print(f"🏷️  模型名称: {model_name}")
    print(f"📋 版本: {version}")
    print(f"☁️  云存储提供商: {provider}")
    print(f"📁 输出目录: {output_dir}")
    print("-" * 50)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 初始化云存储管理器
    try:
        manager = CloudStorageManager()
    except Exception as e:
        print(f"❌ 初始化云存储管理器失败: {str(e)}")
        return False
    
    # 构建本地文件路径
    local_file = output_path / f"{model_name}_{version}.pt"
    
    # 下载模型
    print(f"📤 正在下载模型...")
    downloaded_path = manager.download_model(
        model_name=model_name,
        version=version,
        provider=provider,
        local_path=str(local_file)
    )
    
    if downloaded_path:
        print(f"✅ 模型下载成功!")
        print(f"📁 本地路径: {downloaded_path}")
        
        # 显示文件信息
        file_size = Path(downloaded_path).stat().st_size
        print(f"📊 文件大小: {file_size / (1024*1024):.2f} MB")
        
        return True
    else:
        print(f"❌ 模型下载失败!")
        return False

def list_available_models(provider: str = "aws_s3"):
    """列出可用的模型"""
    
    print("📋 列出云存储中的可用模型...")
    print(f"☁️  云存储提供商: {provider}")
    print("-" * 50)
    
    # 初始化云存储管理器
    try:
        manager = CloudStorageManager()
    except Exception as e:
        print(f"❌ 初始化云存储管理器失败: {str(e)}")
        return
    
    # 列出模型
    models = manager.list_models(provider=provider)
    
    if not models:
        print("❌ 未找到任何模型")
        return
    
    print(f"📊 找到 {len(models)} 个模型:")
    print()
    
    for i, model in enumerate(models, 1):
        print(f"{i:2d}. {model['key']}")
        print(f"    大小: {model['size'] / (1024*1024):.2f} MB")
        print(f"    修改时间: {model['last_modified']}")
        print()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从云存储下载EIT-P模型")
    parser.add_argument(
        "--provider", 
        choices=["aws_s3", "google_cloud", "azure_blob", "aliyun_oss"],
        default="aws_s3",
        help="云存储提供商 (默认: aws_s3)"
    )
    parser.add_argument(
        "--model-name",
        default="eitp_model",
        help="模型名称 (默认: eitp_model)"
    )
    parser.add_argument(
        "--version",
        help="模型版本 (必需)"
    )
    parser.add_argument(
        "--output-dir",
        default="downloaded_models",
        help="输出目录 (默认: downloaded_models)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出可用的模型"
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="列出支持的云存储提供商"
    )
    
    args = parser.parse_args()
    
    if args.list_providers:
        print("支持的云存储提供商:")
        print("- aws_s3: Amazon S3")
        print("- google_cloud: Google Cloud Storage")
        print("- azure_blob: Azure Blob Storage")
        print("- aliyun_oss: 阿里云 OSS")
        return
    
    if args.list:
        list_available_models(args.provider)
        return
    
    if not args.version:
        print("❌ 请指定模型版本: --version v1.0.0")
        return
    
    # 检查环境变量
    if args.provider == "aws_s3":
        if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
            print("❌ 请设置AWS环境变量:")
            print("   export AWS_ACCESS_KEY_ID=your_access_key")
            print("   export AWS_SECRET_ACCESS_KEY=your_secret_key")
            return
    
    # 下载模型
    success = download_model_from_cloud(
        model_name=args.model_name,
        version=args.version,
        provider=args.provider,
        output_dir=args.output_dir
    )
    
    if success:
        print("\n🎯 下一步:")
        print("1. 验证下载的模型文件")
        print("2. 加载模型进行推理")
        print("3. 集成到EIT-P框架中")
        sys.exit(0)
    else:
        print("\n❌ 下载失败，请检查配置和网络连接")
        sys.exit(1)

if __name__ == "__main__":
    main()
