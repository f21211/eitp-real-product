#!/usr/bin/env python3
"""
EIT-P 云存储管理器
Cloud Storage Manager for EIT-P Models

支持多种云存储提供商：
- AWS S3
- Google Cloud Storage
- Azure Blob Storage
- 阿里云 OSS
"""

import os
import yaml
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import boto3
from google.cloud import storage
from azure.storage.blob import BlobServiceClient
import oss2

class CloudStorageManager:
    """云存储管理器"""
    
    def __init__(self, config_path: str = "cloud_storage_config.yaml"):
        """初始化云存储管理器"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.cache_dir = Path(self.config['download']['cache_dir'])
        self.cache_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('CloudStorageManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def upload_model(self, 
                    local_path: str, 
                    model_name: str, 
                    version: str,
                    provider: str = "aws_s3",
                    metadata: Optional[Dict] = None) -> bool:
        """上传模型到云存储"""
        try:
            local_file = Path(local_path)
            if not local_file.exists():
                self.logger.error(f"本地文件不存在: {local_path}")
                return False
            
            # 计算文件校验和
            file_hash = self._calculate_checksum(local_path)
            file_size = local_file.stat().st_size
            
            # 准备元数据
            model_metadata = {
                "model_name": model_name,
                "version": version,
                "created_at": datetime.now().isoformat(),
                "file_size": file_size,
                "checksum": file_hash,
                "local_path": str(local_path)
            }
            
            if metadata:
                model_metadata.update(metadata)
            
            # 构建云存储路径
            cloud_path = self._build_cloud_path(model_name, version)
            
            # 根据提供商上传
            if provider == "aws_s3":
                return self._upload_to_s3(local_path, cloud_path, model_metadata)
            elif provider == "google_cloud":
                return self._upload_to_gcs(local_path, cloud_path, model_metadata)
            elif provider == "azure_blob":
                return self._upload_to_azure(local_path, cloud_path, model_metadata)
            elif provider == "aliyun_oss":
                return self._upload_to_oss(local_path, cloud_path, model_metadata)
            else:
                self.logger.error(f"不支持的云存储提供商: {provider}")
                return False
                
        except Exception as e:
            self.logger.error(f"上传模型失败: {str(e)}")
            return False
    
    def download_model(self, 
                      model_name: str, 
                      version: str,
                      provider: str = "aws_s3",
                      local_path: Optional[str] = None) -> Optional[str]:
        """从云存储下载模型"""
        try:
            cloud_path = self._build_cloud_path(model_name, version)
            
            if local_path is None:
                local_path = self.cache_dir / f"{model_name}_{version}.pt"
            
            # 检查缓存
            if Path(local_path).exists():
                self.logger.info(f"模型已存在于缓存: {local_path}")
                return str(local_path)
            
            # 根据提供商下载
            if provider == "aws_s3":
                return self._download_from_s3(cloud_path, str(local_path))
            elif provider == "google_cloud":
                return self._download_from_gcs(cloud_path, str(local_path))
            elif provider == "azure_blob":
                return self._download_from_azure(cloud_path, str(local_path))
            elif provider == "aliyun_oss":
                return self._download_from_oss(cloud_path, str(local_path))
            else:
                self.logger.error(f"不支持的云存储提供商: {provider}")
                return None
                
        except Exception as e:
            self.logger.error(f"下载模型失败: {str(e)}")
            return None
    
    def list_models(self, provider: str = "aws_s3") -> List[Dict]:
        """列出云存储中的模型"""
        try:
            if provider == "aws_s3":
                return self._list_s3_models()
            elif provider == "google_cloud":
                return self._list_gcs_models()
            elif provider == "azure_blob":
                return self._list_azure_models()
            elif provider == "aliyun_oss":
                return self._list_oss_models()
            else:
                self.logger.error(f"不支持的云存储提供商: {provider}")
                return []
                
        except Exception as e:
            self.logger.error(f"列出模型失败: {str(e)}")
            return []
    
    def delete_model(self, model_name: str, version: str, provider: str = "aws_s3") -> bool:
        """删除云存储中的模型"""
        try:
            cloud_path = self._build_cloud_path(model_name, version)
            
            if provider == "aws_s3":
                return self._delete_from_s3(cloud_path)
            elif provider == "google_cloud":
                return self._delete_from_gcs(cloud_path)
            elif provider == "azure_blob":
                return self._delete_from_azure(cloud_path)
            elif provider == "aliyun_oss":
                return self._delete_from_oss(cloud_path)
            else:
                self.logger.error(f"不支持的云存储提供商: {provider}")
                return False
                
        except Exception as e:
            self.logger.error(f"删除模型失败: {str(e)}")
            return False
    
    def _calculate_checksum(self, file_path: str) -> str:
        """计算文件校验和"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _build_cloud_path(self, model_name: str, version: str) -> str:
        """构建云存储路径"""
        base_path = self.config['model_storage']['base_path']
        return f"{base_path}{model_name}/{version}/model.pt"
    
    def _upload_to_s3(self, local_path: str, cloud_path: str, metadata: Dict) -> bool:
        """上传到AWS S3"""
        try:
            s3_config = self.config['cloud_storage']['providers'][0]['config']
            s3_client = boto3.client(
                's3',
                aws_access_key_id=s3_config['access_key'],
                aws_secret_access_key=s3_config['secret_key'],
                region_name=s3_config['region']
            )
            
            s3_client.upload_file(
                local_path,
                s3_config['bucket'],
                cloud_path,
                ExtraArgs={'Metadata': metadata}
            )
            
            self.logger.info(f"模型已上传到S3: s3://{s3_config['bucket']}/{cloud_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"S3上传失败: {str(e)}")
            return False
    
    def _download_from_s3(self, cloud_path: str, local_path: str) -> Optional[str]:
        """从AWS S3下载"""
        try:
            s3_config = self.config['cloud_storage']['providers'][0]['config']
            s3_client = boto3.client(
                's3',
                aws_access_key_id=s3_config['access_key'],
                aws_secret_access_key=s3_config['secret_key'],
                region_name=s3_config['region']
            )
            
            s3_client.download_file(
                s3_config['bucket'],
                cloud_path,
                local_path
            )
            
            self.logger.info(f"模型已从S3下载: {local_path}")
            return local_path
            
        except Exception as e:
            self.logger.error(f"S3下载失败: {str(e)}")
            return None
    
    def _list_s3_models(self) -> List[Dict]:
        """列出S3中的模型"""
        try:
            s3_config = self.config['cloud_storage']['providers'][0]['config']
            s3_client = boto3.client(
                's3',
                aws_access_key_id=s3_config['access_key'],
                aws_secret_access_key=s3_config['secret_key'],
                region_name=s3_config['region']
            )
            
            response = s3_client.list_objects_v2(
                Bucket=s3_config['bucket'],
                Prefix=self.config['model_storage']['base_path']
            )
            
            models = []
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.pt'):
                    models.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat()
                    })
            
            return models
            
        except Exception as e:
            self.logger.error(f"列出S3模型失败: {str(e)}")
            return []
    
    def _delete_from_s3(self, cloud_path: str) -> bool:
        """从S3删除模型"""
        try:
            s3_config = self.config['cloud_storage']['providers'][0]['config']
            s3_client = boto3.client(
                's3',
                aws_access_key_id=s3_config['access_key'],
                aws_secret_access_key=s3_config['secret_key'],
                region_name=s3_config['region']
            )
            
            s3_client.delete_object(
                Bucket=s3_config['bucket'],
                Key=cloud_path
            )
            
            self.logger.info(f"模型已从S3删除: {cloud_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"S3删除失败: {str(e)}")
            return False
    
    # 其他云存储提供商的方法实现...
    def _upload_to_gcs(self, local_path: str, cloud_path: str, metadata: Dict) -> bool:
        """上传到Google Cloud Storage"""
        # 实现GCS上传逻辑
        pass
    
    def _download_from_gcs(self, cloud_path: str, local_path: str) -> Optional[str]:
        """从GCS下载"""
        # 实现GCS下载逻辑
        pass
    
    def _list_gcs_models(self) -> List[Dict]:
        """列出GCS中的模型"""
        # 实现GCS列表逻辑
        pass
    
    def _delete_from_gcs(self, cloud_path: str) -> bool:
        """从GCS删除模型"""
        # 实现GCS删除逻辑
        pass
    
    def _upload_to_azure(self, local_path: str, cloud_path: str, metadata: Dict) -> bool:
        """上传到Azure Blob Storage"""
        # 实现Azure上传逻辑
        pass
    
    def _download_from_azure(self, cloud_path: str, local_path: str) -> Optional[str]:
        """从Azure下载"""
        # 实现Azure下载逻辑
        pass
    
    def _list_azure_models(self) -> List[Dict]:
        """列出Azure中的模型"""
        # 实现Azure列表逻辑
        pass
    
    def _delete_from_azure(self, cloud_path: str) -> bool:
        """从Azure删除模型"""
        # 实现Azure删除逻辑
        pass
    
    def _upload_to_oss(self, local_path: str, cloud_path: str, metadata: Dict) -> bool:
        """上传到阿里云OSS"""
        # 实现OSS上传逻辑
        pass
    
    def _download_from_oss(self, cloud_path: str, local_path: str) -> Optional[str]:
        """从OSS下载"""
        # 实现OSS下载逻辑
        pass
    
    def _list_oss_models(self) -> List[Dict]:
        """列出OSS中的模型"""
        # 实现OSS列表逻辑
        pass
    
    def _delete_from_oss(self, cloud_path: str) -> bool:
        """从OSS删除模型"""
        # 实现OSS删除逻辑
        pass


def main():
    """主函数 - 示例用法"""
    manager = CloudStorageManager()
    
    # 上传模型示例
    print("上传模型到云存储...")
    success = manager.upload_model(
        local_path="models/model_20251003_004528/model.pt",
        model_name="eitp_gpt2",
        version="v1.0.0",
        provider="aws_s3",
        metadata={
            "description": "EIT-P GPT-2模型",
            "performance_metrics": {"accuracy": 0.95, "loss": 0.05}
        }
    )
    
    if success:
        print("✅ 模型上传成功！")
    else:
        print("❌ 模型上传失败！")
    
    # 列出模型示例
    print("\n列出云存储中的模型...")
    models = manager.list_models(provider="aws_s3")
    for model in models:
        print(f"- {model['key']} ({model['size']} bytes)")


if __name__ == "__main__":
    main()
