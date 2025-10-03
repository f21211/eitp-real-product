"""
流量分割器
管理A/B测试中的流量分配和用户分组
"""

import hashlib
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TrafficSplitStrategy(Enum):
    """流量分割策略"""
    RANDOM = "random"           # 随机分配
    CONSISTENT_HASH = "consistent_hash"  # 一致性哈希
    USER_ID_BASED = "user_id_based"      # 基于用户ID
    SESSION_BASED = "session_based"      # 基于会话


@dataclass
class Variant:
    """变体信息"""
    name: str
    weight: float  # 权重 (0.0 - 1.0)
    description: str = ""
    is_active: bool = True
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


@dataclass
class Experiment:
    """实验信息"""
    experiment_id: str
    name: str
    description: str
    variants: List[Variant]
    strategy: TrafficSplitStrategy
    start_time: datetime
    end_time: Optional[datetime] = None
    is_active: bool = True
    target_audience: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.target_audience is None:
            self.target_audience = {}


@dataclass
class UserAssignment:
    """用户分配信息"""
    user_id: str
    experiment_id: str
    variant: str
    assigned_at: datetime
    session_id: Optional[str] = None


class TrafficSplitter:
    """流量分割器"""
    
    def __init__(self):
        self.logger = logging.getLogger("traffic_splitter")
        self.experiments: Dict[str, Experiment] = {}
        self.user_assignments: Dict[str, UserAssignment] = {}  # user_id -> assignment
        self.session_assignments: Dict[str, UserAssignment] = {}  # session_id -> assignment
    
    def create_experiment(self, experiment_id: str, name: str, description: str,
                         variants: List[Variant], strategy: TrafficSplitStrategy = TrafficSplitStrategy.RANDOM,
                         duration_days: int = 30, target_audience: Dict[str, Any] = None) -> Experiment:
        """创建新实验"""
        # 验证变体权重
        total_weight = sum(variant.weight for variant in variants)
        if abs(total_weight - 1.0) > 0.01:  # 允许小的浮点误差
            raise ValueError(f"Variant weights must sum to 1.0, got {total_weight}")
        
        # 验证变体名称唯一性
        variant_names = [v.name for v in variants]
        if len(variant_names) != len(set(variant_names)):
            raise ValueError("Variant names must be unique")
        
        end_time = datetime.now() + timedelta(days=duration_days)
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            strategy=strategy,
            start_time=datetime.now(),
            end_time=end_time,
            target_audience=target_audience or {}
        )
        
        self.experiments[experiment_id] = experiment
        self.logger.info(f"Created experiment: {experiment_id}")
        return experiment
    
    def get_variant_for_user(self, user_id: str, experiment_id: str,
                           session_id: Optional[str] = None) -> Optional[str]:
        """为用户获取分配的变体"""
        # 检查实验是否存在且活跃
        if experiment_id not in self.experiments:
            self.logger.warning(f"Experiment {experiment_id} not found")
            return None
        
        experiment = self.experiments[experiment_id]
        if not experiment.is_active:
            self.logger.warning(f"Experiment {experiment_id} is not active")
            return None
        
        # 检查实验是否在有效期内
        now = datetime.now()
        if now < experiment.start_time:
            self.logger.warning(f"Experiment {experiment_id} has not started yet")
            return None
        
        if experiment.end_time and now > experiment.end_time:
            self.logger.warning(f"Experiment {experiment_id} has ended")
            return None
        
        # 检查是否已有分配
        assignment_key = f"{user_id}_{experiment_id}"
        if assignment_key in self.user_assignments:
            assignment = self.user_assignments[assignment_key]
            self.logger.debug(f"User {user_id} already assigned to variant {assignment.variant}")
            return assignment.variant
        
        # 检查会话分配
        if session_id and session_id in self.session_assignments:
            assignment = self.session_assignments[session_id]
            if assignment.experiment_id == experiment_id:
                self.logger.debug(f"Session {session_id} already assigned to variant {assignment.variant}")
                return assignment.variant
        
        # 根据策略分配变体
        variant = self._assign_variant(user_id, experiment, session_id)
        
        if variant:
            # 记录分配
            assignment = UserAssignment(
                user_id=user_id,
                experiment_id=experiment_id,
                variant=variant,
                assigned_at=datetime.now(),
                session_id=session_id
            )
            
            self.user_assignments[assignment_key] = assignment
            if session_id:
                self.session_assignments[session_id] = assignment
            
            self.logger.info(f"Assigned user {user_id} to variant {variant} in experiment {experiment_id}")
        
        return variant
    
    def _assign_variant(self, user_id: str, experiment: Experiment, 
                       session_id: Optional[str] = None) -> Optional[str]:
        """根据策略分配变体"""
        if experiment.strategy == TrafficSplitStrategy.RANDOM:
            return self._random_assignment(experiment)
        elif experiment.strategy == TrafficSplitStrategy.CONSISTENT_HASH:
            return self._consistent_hash_assignment(user_id, experiment)
        elif experiment.strategy == TrafficSplitStrategy.USER_ID_BASED:
            return self._user_id_based_assignment(user_id, experiment)
        elif experiment.strategy == TrafficSplitStrategy.SESSION_BASED:
            return self._session_based_assignment(session_id or user_id, experiment)
        else:
            self.logger.error(f"Unknown strategy: {experiment.strategy}")
            return None
    
    def _random_assignment(self, experiment: Experiment) -> str:
        """随机分配"""
        rand = random.random()
        cumulative_weight = 0.0
        
        for variant in experiment.variants:
            if not variant.is_active:
                continue
            cumulative_weight += variant.weight
            if rand <= cumulative_weight:
                return variant.name
        
        # 如果没有找到，返回第一个活跃变体
        for variant in experiment.variants:
            if variant.is_active:
                return variant.name
        
        return experiment.variants[0].name
    
    def _consistent_hash_assignment(self, user_id: str, experiment: Experiment) -> str:
        """一致性哈希分配"""
        hash_input = f"{user_id}_{experiment.experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0  # 0-1范围
        
        cumulative_weight = 0.0
        for variant in experiment.variants:
            if not variant.is_active:
                continue
            cumulative_weight += variant.weight
            if normalized_hash <= cumulative_weight:
                return variant.name
        
        return experiment.variants[0].name
    
    def _user_id_based_assignment(self, user_id: str, experiment: Experiment) -> str:
        """基于用户ID的分配"""
        # 使用用户ID的哈希值确保一致性
        hash_value = hash(user_id) % 10000
        normalized_hash = hash_value / 10000.0
        
        cumulative_weight = 0.0
        for variant in experiment.variants:
            if not variant.is_active:
                continue
            cumulative_weight += variant.weight
            if normalized_hash <= cumulative_weight:
                return variant.name
        
        return experiment.variants[0].name
    
    def _session_based_assignment(self, session_id: str, experiment: Experiment) -> str:
        """基于会话的分配"""
        # 使用会话ID的哈希值
        hash_value = hash(session_id) % 10000
        normalized_hash = hash_value / 10000.0
        
        cumulative_weight = 0.0
        for variant in experiment.variants:
            if not variant.is_active:
                continue
            cumulative_weight += variant.weight
            if normalized_hash <= cumulative_weight:
                return variant.name
        
        return experiment.variants[0].name
    
    def get_experiment_stats(self, experiment_id: str) -> Dict[str, Any]:
        """获取实验统计信息"""
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.experiments[experiment_id]
        
        # 统计各变体的用户数
        variant_counts = {}
        for variant in experiment.variants:
            variant_counts[variant.name] = 0
        
        for assignment in self.user_assignments.values():
            if assignment.experiment_id == experiment_id:
                if assignment.variant in variant_counts:
                    variant_counts[assignment.variant] += 1
        
        # 计算分配比例
        total_users = sum(variant_counts.values())
        variant_ratios = {}
        for variant_name, count in variant_counts.items():
            variant_ratios[variant_name] = count / max(total_users, 1)
        
        return {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "total_users": total_users,
            "variant_counts": variant_counts,
            "variant_ratios": variant_ratios,
            "is_active": experiment.is_active,
            "start_time": experiment.start_time.isoformat(),
            "end_time": experiment.end_time.isoformat() if experiment.end_time else None
        }
    
    def update_variant_weights(self, experiment_id: str, 
                             new_weights: Dict[str, float]) -> bool:
        """更新变体权重"""
        if experiment_id not in self.experiments:
            self.logger.error(f"Experiment {experiment_id} not found")
            return False
        
        experiment = self.experiments[experiment_id]
        
        # 验证权重
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.error(f"Total weight must be 1.0, got {total_weight}")
            return False
        
        # 更新权重
        for variant in experiment.variants:
            if variant.name in new_weights:
                variant.weight = new_weights[variant.name]
        
        self.logger.info(f"Updated variant weights for experiment {experiment_id}")
        return True
    
    def pause_experiment(self, experiment_id: str) -> bool:
        """暂停实验"""
        if experiment_id not in self.experiments:
            self.logger.error(f"Experiment {experiment_id} not found")
            return False
        
        self.experiments[experiment_id].is_active = False
        self.logger.info(f"Paused experiment {experiment_id}")
        return True
    
    def resume_experiment(self, experiment_id: str) -> bool:
        """恢复实验"""
        if experiment_id not in self.experiments:
            self.logger.error(f"Experiment {experiment_id} not found")
            return False
        
        self.experiments[experiment_id].is_active = True
        self.logger.info(f"Resumed experiment {experiment_id}")
        return True
    
    def end_experiment(self, experiment_id: str) -> bool:
        """结束实验"""
        if experiment_id not in self.experiments:
            self.logger.error(f"Experiment {experiment_id} not found")
            return False
        
        self.experiments[experiment_id].is_active = False
        self.experiments[experiment_id].end_time = datetime.now()
        self.logger.info(f"Ended experiment {experiment_id}")
        return True
    
    def get_user_assignments(self, user_id: str) -> List[UserAssignment]:
        """获取用户的所有分配"""
        assignments = []
        for assignment in self.user_assignments.values():
            if assignment.user_id == user_id:
                assignments.append(assignment)
        return assignments
    
    def clear_user_assignment(self, user_id: str, experiment_id: str) -> bool:
        """清除用户分配"""
        assignment_key = f"{user_id}_{experiment_id}"
        if assignment_key in self.user_assignments:
            assignment = self.user_assignments[assignment_key]
            if assignment.session_id and assignment.session_id in self.session_assignments:
                del self.session_assignments[assignment.session_id]
            del self.user_assignments[assignment_key]
            self.logger.info(f"Cleared assignment for user {user_id} in experiment {experiment_id}")
            return True
        return False


# 全局流量分割器实例
_global_splitter = None


def get_traffic_splitter() -> TrafficSplitter:
    """获取全局流量分割器实例"""
    global _global_splitter
    if _global_splitter is None:
        _global_splitter = TrafficSplitter()
    return _global_splitter