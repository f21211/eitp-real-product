"""
模型回滚模块
处理模型版本回滚和恢复功能
"""

import os
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class RollbackStatus(Enum):
    """回滚状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ModelVersion:
    """模型版本信息"""
    version_id: str
    model_path: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    is_active: bool = False
    is_rollback_candidate: bool = False


@dataclass
class RollbackOperation:
    """回滚操作信息"""
    operation_id: str
    from_version: str
    to_version: str
    status: RollbackStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    backup_path: Optional[str] = None


class ModelRollbackManager:
    """模型回滚管理器"""
    
    def __init__(self, models_dir: str = "models", backups_dir: str = "model_backups"):
        self.models_dir = models_dir
        self.backups_dir = backups_dir
        self.logger = logging.getLogger("model_rollback")
        self.versions: Dict[str, ModelVersion] = {}
        self.rollback_operations: List[RollbackOperation] = []
        
        # 创建必要的目录
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.backups_dir, exist_ok=True)
        
        self._load_versions()
        self._load_rollback_operations()
    
    def _load_versions(self):
        """加载模型版本信息"""
        versions_file = os.path.join(self.models_dir, "versions.json")
        try:
            if os.path.exists(versions_file):
                with open(versions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for version_data in data:
                        version = ModelVersion(
                            version_id=version_data['version_id'],
                            model_path=version_data['model_path'],
                            created_at=datetime.fromisoformat(version_data['created_at']),
                            performance_metrics=version_data['performance_metrics'],
                            metadata=version_data['metadata'],
                            is_active=version_data.get('is_active', False),
                            is_rollback_candidate=version_data.get('is_rollback_candidate', False)
                        )
                        self.versions[version.version_id] = version
                self.logger.info(f"Loaded {len(self.versions)} model versions")
        except Exception as e:
            self.logger.error(f"Error loading versions: {e}")
    
    def _save_versions(self):
        """保存模型版本信息"""
        versions_file = os.path.join(self.models_dir, "versions.json")
        try:
            data = []
            for version in self.versions.values():
                data.append({
                    'version_id': version.version_id,
                    'model_path': version.model_path,
                    'created_at': version.created_at.isoformat(),
                    'performance_metrics': version.performance_metrics,
                    'metadata': version.metadata,
                    'is_active': version.is_active,
                    'is_rollback_candidate': version.is_rollback_candidate
                })
            
            with open(versions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving versions: {e}")
    
    def _load_rollback_operations(self):
        """加载回滚操作记录"""
        operations_file = os.path.join(self.models_dir, "rollback_operations.json")
        try:
            if os.path.exists(operations_file):
                with open(operations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for op_data in data:
                        operation = RollbackOperation(
                            operation_id=op_data['operation_id'],
                            from_version=op_data['from_version'],
                            to_version=op_data['to_version'],
                            status=RollbackStatus(op_data['status']),
                            created_at=datetime.fromisoformat(op_data['created_at']),
                            completed_at=datetime.fromisoformat(op_data['completed_at']) if op_data.get('completed_at') else None,
                            error_message=op_data.get('error_message'),
                            backup_path=op_data.get('backup_path')
                        )
                        self.rollback_operations.append(operation)
                self.logger.info(f"Loaded {len(self.rollback_operations)} rollback operations")
        except Exception as e:
            self.logger.error(f"Error loading rollback operations: {e}")
    
    def _save_rollback_operations(self):
        """保存回滚操作记录"""
        operations_file = os.path.join(self.models_dir, "rollback_operations.json")
        try:
            data = []
            for operation in self.rollback_operations:
                data.append({
                    'operation_id': operation.operation_id,
                    'from_version': operation.from_version,
                    'to_version': operation.to_version,
                    'status': operation.status.value,
                    'created_at': operation.created_at.isoformat(),
                    'completed_at': operation.completed_at.isoformat() if operation.completed_at else None,
                    'error_message': operation.error_message,
                    'backup_path': operation.backup_path
                })
            
            with open(operations_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving rollback operations: {e}")
    
    def register_model_version(self, version_id: str, model_path: str,
                             performance_metrics: Dict[str, float],
                             metadata: Dict[str, Any] = None) -> ModelVersion:
        """注册新的模型版本"""
        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            created_at=datetime.now(),
            performance_metrics=performance_metrics,
            metadata=metadata or {}
        )
        
        self.versions[version_id] = version
        self._save_versions()
        
        self.logger.info(f"Registered model version: {version_id}")
        return version
    
    def set_active_version(self, version_id: str) -> bool:
        """设置活跃版本"""
        if version_id not in self.versions:
            self.logger.error(f"Version {version_id} not found")
            return False
        
        # 取消所有其他版本的活跃状态
        for version in self.versions.values():
            version.is_active = False
        
        # 设置新版本为活跃
        self.versions[version_id].is_active = True
        self._save_versions()
        
        self.logger.info(f"Set active version: {version_id}")
        return True
    
    def get_active_version(self) -> Optional[ModelVersion]:
        """获取当前活跃版本"""
        for version in self.versions.values():
            if version.is_active:
                return version
        return None
    
    def get_rollback_candidates(self, min_performance: float = 0.0) -> List[ModelVersion]:
        """获取可回滚的候选版本"""
        candidates = []
        for version in self.versions.values():
            if (not version.is_active and 
                version.is_rollback_candidate and
                any(score >= min_performance for score in version.performance_metrics.values())):
                candidates.append(version)
        
        # 按性能指标排序
        candidates.sort(key=lambda x: max(x.performance_metrics.values()), reverse=True)
        return candidates
    
    def initiate_rollback(self, to_version_id: str, backup_current: bool = True) -> str:
        """启动回滚操作"""
        if to_version_id not in self.versions:
            raise ValueError(f"Version {to_version_id} not found")
        
        active_version = self.get_active_version()
        if not active_version:
            raise ValueError("No active version found")
        
        if active_version.version_id == to_version_id:
            raise ValueError("Cannot rollback to the same version")
        
        operation_id = f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        operation = RollbackOperation(
            operation_id=operation_id,
            from_version=active_version.version_id,
            to_version=to_version_id,
            status=RollbackStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.rollback_operations.append(operation)
        self._save_rollback_operations()
        
        self.logger.info(f"Initiated rollback operation: {operation_id}")
        return operation_id
    
    def execute_rollback(self, operation_id: str) -> bool:
        """执行回滚操作"""
        operation = None
        for op in self.rollback_operations:
            if op.operation_id == operation_id:
                operation = op
                break
        
        if not operation:
            self.logger.error(f"Rollback operation {operation_id} not found")
            return False
        
        if operation.status != RollbackStatus.PENDING:
            self.logger.error(f"Rollback operation {operation_id} is not pending")
            return False
        
        try:
            operation.status = RollbackStatus.IN_PROGRESS
            self._save_rollback_operations()
            
            # 备份当前活跃版本
            if operation.backup_path is None:
                backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                operation.backup_path = os.path.join(
                    self.backups_dir, 
                    f"backup_{operation.from_version}_{backup_timestamp}"
                )
                os.makedirs(operation.backup_path, exist_ok=True)
            
            # 复制当前模型到备份位置
            active_version = self.versions[operation.from_version]
            if os.path.exists(active_version.model_path):
                if os.path.isdir(active_version.model_path):
                    shutil.copytree(active_version.model_path, 
                                  os.path.join(operation.backup_path, "model"))
                else:
                    shutil.copy2(active_version.model_path, operation.backup_path)
            
            # 切换到目标版本
            target_version = self.versions[operation.to_version]
            
            # 更新活跃版本
            self.set_active_version(operation.to_version)
            
            # 标记目标版本为可回滚候选
            target_version.is_rollback_candidate = True
            
            operation.status = RollbackStatus.COMPLETED
            operation.completed_at = datetime.now()
            self._save_rollback_operations()
            
            self.logger.info(f"Successfully completed rollback operation: {operation_id}")
            return True
            
        except Exception as e:
            operation.status = RollbackStatus.FAILED
            operation.error_message = str(e)
            operation.completed_at = datetime.now()
            self._save_rollback_operations()
            
            self.logger.error(f"Rollback operation {operation_id} failed: {e}")
            return False
    
    def cancel_rollback(self, operation_id: str) -> bool:
        """取消回滚操作"""
        for operation in self.rollback_operations:
            if operation.operation_id == operation_id:
                if operation.status == RollbackStatus.PENDING:
                    operation.status = RollbackStatus.CANCELLED
                    operation.completed_at = datetime.now()
                    self._save_rollback_operations()
                    self.logger.info(f"Cancelled rollback operation: {operation_id}")
                    return True
                else:
                    self.logger.error(f"Cannot cancel rollback operation {operation_id} in status {operation.status}")
                    return False
        
        self.logger.error(f"Rollback operation {operation_id} not found")
        return False
    
    def get_rollback_history(self, limit: int = 10) -> List[RollbackOperation]:
        """获取回滚历史"""
        return sorted(self.rollback_operations, 
                     key=lambda x: x.created_at, reverse=True)[:limit]
    
    def restore_from_backup(self, operation_id: str) -> bool:
        """从备份恢复模型"""
        operation = None
        for op in self.rollback_operations:
            if op.operation_id == operation_id:
                operation = op
                break
        
        if not operation:
            self.logger.error(f"Rollback operation {operation_id} not found")
            return False
        
        if not operation.backup_path or not os.path.exists(operation.backup_path):
            self.logger.error(f"Backup path not found for operation {operation_id}")
            return False
        
        try:
            # 恢复模型文件
            active_version = self.get_active_version()
            if active_version and os.path.exists(active_version.model_path):
                if os.path.isdir(active_version.model_path):
                    shutil.rmtree(active_version.model_path)
                else:
                    os.remove(active_version.model_path)
            
            # 从备份恢复
            backup_model_path = os.path.join(operation.backup_path, "model")
            if os.path.exists(backup_model_path):
                shutil.copytree(backup_model_path, active_version.model_path)
            else:
                # 单个文件备份
                shutil.copy2(operation.backup_path, active_version.model_path)
            
            self.logger.info(f"Successfully restored model from backup: {operation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore from backup {operation_id}: {e}")
            return False


# 全局回滚管理器实例
_global_rollback_manager = None


def get_rollback_manager() -> ModelRollbackManager:
    """获取全局回滚管理器实例"""
    global _global_rollback_manager
    if _global_rollback_manager is None:
        _global_rollback_manager = ModelRollbackManager()
    return _global_rollback_manager