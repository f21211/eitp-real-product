# ☁️ EIT-P 云存储管理

## 📖 概述

EIT-P框架支持将训练好的模型保存到云存储中，以便于分发、备份和团队协作。模型文件不会包含在Git仓库中，而是通过云存储进行管理。

## 🏗️ 架构设计

```
EIT-P 项目
├── 代码文件 (Git仓库)
│   ├── eit_p/           # 核心框架
│   ├── scripts/         # 微服务
│   ├── config/          # 配置文件
│   └── docs/            # 文档
└── 模型文件 (云存储)
    ├── checkpoints/     # 训练检查点
    ├── final_models/    # 最终模型
    ├── compressed_models/ # 压缩模型
    └── experiments/     # 实验模型
```

## 🚀 支持的云存储提供商

### 1. Amazon S3
- **优势**: 全球覆盖，高可用性
- **配置**: 需要AWS访问密钥
- **成本**: 按存储和传输计费

### 2. Google Cloud Storage
- **优势**: 与Google AI服务集成
- **配置**: 需要GCP服务账户
- **成本**: 按存储和传输计费

### 3. Azure Blob Storage
- **优势**: 与Azure AI服务集成
- **配置**: 需要Azure连接字符串
- **成本**: 按存储和传输计费

### 4. 阿里云 OSS
- **优势**: 国内访问速度快
- **配置**: 需要阿里云访问密钥
- **成本**: 按存储和传输计费

## ⚙️ 配置说明

### 1. 环境变量配置

#### AWS S3
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

#### Google Cloud Storage
```bash
export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
export GOOGLE_CLOUD_PROJECT=your-project-id
```

#### Azure Blob Storage
```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=..."
```

#### 阿里云 OSS
```bash
export ALIYUN_ACCESS_KEY_ID=your_access_key
export ALIYUN_ACCESS_KEY_SECRET=your_secret_key
```

### 2. 配置文件

编辑 `cloud_storage_config.yaml` 文件：

```yaml
cloud_storage:
  providers:
    - name: "aws_s3"
      enabled: true
      config:
        bucket: "eitp-models"
        region: "us-east-1"
        access_key: "${AWS_ACCESS_KEY_ID}"
        secret_key: "${AWS_SECRET_ACCESS_KEY}"
```

## 📤 上传模型

### 1. 自动上传所有模型

```bash
# 上传到AWS S3
python upload_models_to_cloud.py --provider aws_s3 --models-dir models/

# 上传到Google Cloud Storage
python upload_models_to_cloud.py --provider google_cloud --models-dir models/

# 上传到Azure Blob Storage
python upload_models_to_cloud.py --provider azure_blob --models-dir models/

# 上传到阿里云 OSS
python upload_models_to_cloud.py --provider aliyun_oss --models-dir models/
```

### 2. 手动上传单个模型

```python
from scripts.cloud_storage_manager import CloudStorageManager

manager = CloudStorageManager()

# 上传模型
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
```

## 📥 下载模型

### 1. 下载指定模型

```bash
# 从AWS S3下载
python download_models_from_cloud.py --provider aws_s3 --model-name eitp_model --version v1.0.0

# 从Google Cloud Storage下载
python download_models_from_cloud.py --provider google_cloud --model-name eitp_model --version v1.0.0

# 从Azure Blob Storage下载
python download_models_from_cloud.py --provider azure_blob --model-name eitp_model --version v1.0.0

# 从阿里云 OSS下载
python download_models_from_cloud.py --provider aliyun_oss --model-name eitp_model --version v1.0.0
```

### 2. 列出可用模型

```bash
# 列出AWS S3中的模型
python download_models_from_cloud.py --provider aws_s3 --list

# 列出Google Cloud Storage中的模型
python download_models_from_cloud.py --provider google_cloud --list
```

### 3. 手动下载模型

```python
from scripts.cloud_storage_manager import CloudStorageManager

manager = CloudStorageManager()

# 下载模型
local_path = manager.download_model(
    model_name="eitp_model",
    version="v1.0.0",
    provider="aws_s3",
    local_path="downloaded_models/eitp_model_v1.0.0.pt"
)
```

## 🔧 高级功能

### 1. 模型元数据管理

每个上传的模型都会包含以下元数据：

```json
{
    "model_name": "eitp_gpt2",
    "version": "v1.0.0",
    "created_at": "2025-01-27T10:30:00.000000",
    "file_size": 1048576,
    "checksum": "md5_hash_value",
    "description": "EIT-P GPT-2模型",
    "performance_metrics": {
        "accuracy": 0.95,
        "loss": 0.05
    },
    "training_config": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001
    }
}
```

### 2. 模型版本管理

- **语义化版本**: 使用 `v1.0.0` 格式
- **版本比较**: 支持版本大小比较
- **版本回滚**: 可以回滚到之前的版本

### 3. 缓存管理

- **本地缓存**: 下载的模型会缓存在本地
- **缓存清理**: 自动清理过期的缓存文件
- **缓存大小限制**: 可配置最大缓存大小

### 4. 并发控制

- **并发上传**: 支持多文件并发上传
- **并发下载**: 支持多文件并发下载
- **速率限制**: 可配置上传/下载速率

## 📊 监控和日志

### 1. 上传/下载日志

```bash
# 查看上传日志
tail -f logs/cloud_storage_upload.log

# 查看下载日志
tail -f logs/cloud_storage_download.log
```

### 2. 性能监控

- **上传速度**: 监控上传速度
- **下载速度**: 监控下载速度
- **成功率**: 监控上传/下载成功率
- **错误率**: 监控错误率

## 🛡️ 安全考虑

### 1. 访问控制

- **IAM角色**: 使用最小权限原则
- **访问密钥**: 定期轮换访问密钥
- **网络访问**: 限制网络访问范围

### 2. 数据加密

- **传输加密**: 使用HTTPS/TLS加密传输
- **存储加密**: 使用云存储的加密功能
- **密钥管理**: 使用云服务的密钥管理

### 3. 审计日志

- **操作日志**: 记录所有上传/下载操作
- **访问日志**: 记录访问历史
- **错误日志**: 记录错误信息

## 💰 成本优化

### 1. 存储优化

- **压缩**: 上传前压缩模型文件
- **去重**: 避免重复存储相同文件
- **生命周期**: 设置文件生命周期策略

### 2. 传输优化

- **CDN**: 使用CDN加速下载
- **分片上传**: 大文件分片上传
- **断点续传**: 支持断点续传

### 3. 成本监控

- **使用量监控**: 监控存储和传输使用量
- **成本告警**: 设置成本告警阈值
- **成本分析**: 定期分析成本结构

## 🚀 最佳实践

### 1. 模型命名规范

```
模型类型_版本号_日期.pt
例如: eitp_gpt2_v1.0.0_20250127.pt
```

### 2. 版本管理策略

- **主版本**: 重大功能更新
- **次版本**: 功能增强
- **修订版本**: 错误修复

### 3. 备份策略

- **多区域备份**: 在不同区域备份
- **定期备份**: 定期创建备份
- **灾难恢复**: 制定灾难恢复计划

## 📞 技术支持

如有问题，请联系：

- **邮箱**: chen11521@gtiit.edu.cn
- **GitHub**: https://github.com/f21211/eitp-ai-platform
- **文档**: 查看项目文档获取更多信息

---

**EIT-P框架云存储管理 - 让模型管理更简单！** ☁️
