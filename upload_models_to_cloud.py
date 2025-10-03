#!/usr/bin/env python3
"""
EIT-P 模型上传到云存储脚本
Upload EIT-P Models to Cloud Storage

使用方法:
python upload_models_to_cloud.py --provider aws_s3 --models-dir models/
"""

import argparse
import sys
import os
from pathlib import Path
from scripts.cloud_storage_manager import CloudStorageManager

def upload_models_to_cloud(models_dir: str, provider: str = "aws_s3"):
    """上传模型到云存储"""
    
    print("🚀 开始上传EIT-P模型到云存储...")
    print(f"📁 模型目录: {models_dir}")
    print(f"☁️  云存储提供商: {provider}")
    print("-" * 50)
    
    # 初始化云存储管理器
    try:
        manager = CloudStorageManager()
    except Exception as e:
        print(f"❌ 初始化云存储管理器失败: {str(e)}")
        return False
    
    # 检查模型目录
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"❌ 模型目录不存在: {models_dir}")
        return False
    
    # 查找所有模型文件
    model_files = []
    for model_dir in models_path.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith('model_'):
            model_file = model_dir / "model.pt"
            if model_file.exists():
                model_files.append((model_dir, model_file))
    
    if not model_files:
        print("❌ 未找到任何模型文件")
        return False
    
    print(f"📊 找到 {len(model_files)} 个模型文件")
    print()
    
    # 上传每个模型
    success_count = 0
    for model_dir, model_file in model_files:
        print(f"📤 上传模型: {model_dir.name}")
        
        # 从目录名提取版本信息
        version = model_dir.name.replace('model_', 'v')
        
        # 准备元数据
        metadata = {
            "description": f"EIT-P模型 - {model_dir.name}",
            "model_type": "EIT-P",
            "framework": "PyTorch",
            "created_by": "EIT-P Training Platform"
        }
        
        # 上传模型
        success = manager.upload_model(
            local_path=str(model_file),
            model_name="eitp_model",
            version=version,
            provider=provider,
            metadata=metadata
        )
        
        if success:
            print(f"  ✅ 上传成功: {version}")
            success_count += 1
        else:
            print(f"  ❌ 上传失败: {version}")
        
        print()
    
    # 输出结果
    print("=" * 50)
    print(f"📊 上传完成!")
    print(f"✅ 成功: {success_count}/{len(model_files)}")
    print(f"❌ 失败: {len(model_files) - success_count}/{len(model_files)}")
    
    if success_count == len(model_files):
        print("🎉 所有模型上传成功！")
        return True
    else:
        print("⚠️  部分模型上传失败，请检查日志")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="上传EIT-P模型到云存储")
    parser.add_argument(
        "--provider", 
        choices=["aws_s3", "google_cloud", "azure_blob", "aliyun_oss"],
        default="aws_s3",
        help="云存储提供商 (默认: aws_s3)"
    )
    parser.add_argument(
        "--models-dir",
        default="models/",
        help="模型目录路径 (默认: models/)"
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
    
    # 检查环境变量
    if args.provider == "aws_s3":
        if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
            print("❌ 请设置AWS环境变量:")
            print("   export AWS_ACCESS_KEY_ID=your_access_key")
            print("   export AWS_SECRET_ACCESS_KEY=your_secret_key")
            return
    
    # 上传模型
    success = upload_models_to_cloud(args.models_dir, args.provider)
    
    if success:
        print("\n🎯 下一步:")
        print("1. 配置云存储访问权限")
        print("2. 更新模型下载链接")
        print("3. 测试模型下载功能")
        sys.exit(0)
    else:
        print("\n❌ 上传失败，请检查配置和网络连接")
        sys.exit(1)

if __name__ == "__main__":
    main()
