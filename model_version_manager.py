#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P Model Version Manager
模型版本管理模块
"""

import sys
import os
sys.path.append('/mnt/sda1/myproject/datainall/AGI')

import torch
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters

@dataclass
class ModelVersion:
    """模型版本信息"""
    version: str
    model_path: str
    cep_params: Dict
    performance_metrics: Dict
    created_at: str
    description: str
    tags: List[str]

class ModelVersionManager:
    """模型版本管理器"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.versions_file = self.model_dir / "versions.json"
        self.versions = self.load_versions()
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ModelVersionManager")
    
    def load_versions(self) -> Dict[str, ModelVersion]:
        """加载版本信息"""
        if self.versions_file.exists():
            with open(self.versions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {k: ModelVersion(**v) for k, v in data.items()}
        return {}
    
    def save_versions(self):
        """保存版本信息"""
        data = {k: asdict(v) for k, v in self.versions.items()}
        with open(self.versions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def create_version(self, model: EnhancedCEPEITP, 
                      performance_metrics: Dict,
                      description: str = "",
                      tags: List[str] = None) -> str:
        """创建新版本"""
        # 生成版本号
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{timestamp}"
        
        # 创建版本目录
        version_dir = self.model_dir / version
        version_dir.mkdir(exist_ok=True)
        
        # 保存模型
        model_path = version_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'cep_params': asdict(model.cep_params),
            'architecture': {
                'input_dim': model.input_dim,
                'hidden_dims': model.hidden_dims,
                'output_dim': model.output_dim
            }
        }, model_path)
        
        # 创建版本信息
        model_version = ModelVersion(
            version=version,
            model_path=str(model_path),
            cep_params=asdict(model.cep_params),
            performance_metrics=performance_metrics,
            created_at=datetime.now().isoformat(),
            description=description,
            tags=tags or []
        )
        
        # 保存版本信息
        self.versions[version] = model_version
        self.save_versions()
        
        self.logger.info(f"创建模型版本: {version}")
        return version
    
    def load_model(self, version: str) -> Tuple[EnhancedCEPEITP, Dict]:
        """加载指定版本的模型"""
        if version not in self.versions:
            raise ValueError(f"版本 {version} 不存在")
        
        model_version = self.versions[version]
        
        # 加载模型数据
        checkpoint = torch.load(model_version.model_path, map_location='cpu')
        
        # 创建模型
        cep_params = CEPParameters(**checkpoint['cep_params'])
        model = EnhancedCEPEITP(
            input_dim=checkpoint['architecture']['input_dim'],
            hidden_dims=checkpoint['architecture']['hidden_dims'],
            output_dim=checkpoint['architecture']['output_dim'],
            cep_params=cep_params
        )
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, model_version.performance_metrics
    
    def list_versions(self) -> List[Dict]:
        """列出所有版本"""
        return [asdict(v) for v in self.versions.values()]
    
    def get_latest_version(self) -> Optional[str]:
        """获取最新版本"""
        if not self.versions:
            return None
        
        latest = max(self.versions.values(), key=lambda v: v.created_at)
        return latest.version
    
    def delete_version(self, version: str) -> bool:
        """删除版本"""
        if version not in self.versions:
            return False
        
        model_version = self.versions[version]
        
        # 删除文件
        if os.path.exists(model_version.model_path):
            os.remove(model_version.model_path)
        
        # 删除版本目录
        version_dir = Path(model_version.model_path).parent
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        # 从版本列表中删除
        del self.versions[version]
        self.save_versions()
        
        self.logger.info(f"删除模型版本: {version}")
        return True
    
    def get_version_info(self, version: str) -> Optional[Dict]:
        """获取版本信息"""
        if version not in self.versions:
            return None
        return asdict(self.versions[version])
    
    def search_versions(self, tags: List[str] = None, 
                       min_consciousness_level: int = None) -> List[Dict]:
        """搜索版本"""
        results = []
        for version in self.versions.values():
            # 标签匹配
            if tags and not any(tag in version.tags for tag in tags):
                continue
            
            # 意识水平匹配
            if min_consciousness_level is not None:
                consciousness_level = version.performance_metrics.get('consciousness_level', 0)
                if consciousness_level < min_consciousness_level:
                    continue
            
            results.append(asdict(version))
        
        return results

def main():
    """主函数"""
    print("📦 Enhanced CEP-EIT-P Model Version Manager")
    print("=" * 50)
    
    # 创建版本管理器
    manager = ModelVersionManager()
    
    # 创建测试模型
    model = EnhancedCEPEITP(
        input_dim=784,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        cep_params=CEPParameters()
    )
    
    # 创建版本
    performance_metrics = {
        'consciousness_level': 2,
        'inference_time': 0.001,
        'accuracy': 0.95
    }
    
    version = manager.create_version(
        model, 
        performance_metrics,
        description="测试模型版本",
        tags=['test', 'demo']
    )
    
    print(f"✅ 创建版本: {version}")
    
    # 列出版本
    versions = manager.list_versions()
    print(f"📋 版本列表: {len(versions)} 个版本")
    
    # 加载模型
    loaded_model, metrics = manager.load_model(version)
    print(f"✅ 加载模型: {version}")
    print(f"📊 性能指标: {metrics}")

if __name__ == "__main__":
    main()
