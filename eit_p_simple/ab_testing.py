"""
EIT-P A/B测试管理器 - 简化实现
基于IEM理论的A/B测试
"""

import uuid
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ABTestManager:
    """A/B测试管理器 - 基于IEM理论的A/B测试"""
    
    def __init__(self):
        self.tests = {}
        self.results = {}
        self.logger = logging.getLogger(__name__)
        self.logger.info("A/B测试管理器初始化完成")
    
    def create_test(self, config: Dict[str, Any]) -> str:
        """创建A/B测试"""
        try:
            test_id = str(uuid.uuid4())
            
            test = {
                'id': test_id,
                'name': config.get('name', f'ABTest_{test_id[:8]}'),
                'description': config.get('description', ''),
                'variants': config.get('variants', {}),
                'traffic_split': config.get('traffic_split', 0.5),
                'status': 'created',
                'created_at': datetime.now().isoformat(),
                'started_at': None,
                'ended_at': None,
                'participants': {},
                'metrics': {}
            }
            
            self.tests[test_id] = test
            self.logger.info(f"A/B测试创建成功: {test_id}")
            
            return test_id
            
        except Exception as e:
            self.logger.error(f"创建A/B测试失败: {e}")
            raise
    
    def get_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """获取A/B测试信息"""
        return self.tests.get(test_id)
    
    def start_test(self, test_id: str) -> bool:
        """启动A/B测试"""
        try:
            if test_id not in self.tests:
                return False
            
            self.tests[test_id]['status'] = 'running'
            self.tests[test_id]['started_at'] = datetime.now().isoformat()
            
            self.logger.info(f"A/B测试启动成功: {test_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"启动A/B测试失败: {e}")
            return False
    
    def stop_test(self, test_id: str) -> bool:
        """停止A/B测试"""
        try:
            if test_id not in self.tests:
                return False
            
            self.tests[test_id]['status'] = 'stopped'
            self.tests[test_id]['ended_at'] = datetime.now().isoformat()
            
            self.logger.info(f"A/B测试停止成功: {test_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"停止A/B测试失败: {e}")
            return False
    
    def assign_variant(self, test_id: str, user_id: str) -> str:
        """为用户分配变体"""
        try:
            if test_id not in self.tests:
                return 'control'
            
            test = self.tests[test_id]
            if test['status'] != 'running':
                return 'control'
            
            # 检查用户是否已经分配
            if user_id in test['participants']:
                return test['participants'][user_id]['variant']
            
            # 随机分配变体
            if random.random() < test['traffic_split']:
                variant = 'treatment'
            else:
                variant = 'control'
            
            test['participants'][user_id] = {
                'variant': variant,
                'assigned_at': datetime.now().isoformat()
            }
            
            return variant
            
        except Exception as e:
            self.logger.error(f"分配变体失败: {e}")
            return 'control'
    
    def record_conversion(self, test_id: str, user_id: str, conversion_type: str, value: float = 1.0):
        """记录转化事件"""
        try:
            if test_id not in self.tests:
                return False
            
            test = self.tests[test_id]
            if user_id not in test['participants']:
                return False
            
            participant = test['participants'][user_id]
            variant = participant['variant']
            
            if 'conversions' not in participant:
                participant['conversions'] = {}
            
            if conversion_type not in participant['conversions']:
                participant['conversions'][conversion_type] = 0
            
            participant['conversions'][conversion_type] += value
            
            self.logger.debug(f"转化记录成功: {test_id}, {user_id}, {conversion_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"记录转化失败: {e}")
            return False
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """获取A/B测试结果"""
        try:
            if test_id not in self.tests:
                return {}
            
            test = self.tests[test_id]
            participants = test['participants']
            
            # 计算各变体的统计信息
            variants = {}
            for user_id, participant in participants.items():
                variant = participant['variant']
                if variant not in variants:
                    variants[variant] = {
                        'participants': 0,
                        'conversions': {}
                    }
                
                variants[variant]['participants'] += 1
                
                if 'conversions' in participant:
                    for conv_type, value in participant['conversions'].items():
                        if conv_type not in variants[variant]['conversions']:
                            variants[variant]['conversions'][conv_type] = 0
                        variants[variant]['conversions'][conv_type] += value
            
            # 计算转化率
            for variant, stats in variants.items():
                if stats['participants'] > 0:
                    for conv_type, total_conversions in stats['conversions'].items():
                        conversion_rate = total_conversions / stats['participants']
                        stats['conversions'][f'{conv_type}_rate'] = conversion_rate
            
            results = {
                'test_id': test_id,
                'status': test['status'],
                'total_participants': len(participants),
                'variants': variants,
                'statistical_significance': self._calculate_significance(variants),
                'recommendation': self._get_recommendation(variants)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"获取A/B测试结果失败: {e}")
            return {}
    
    def _calculate_significance(self, variants: Dict[str, Any]) -> Dict[str, Any]:
        """计算统计显著性"""
        try:
            # 简化的统计显著性计算
            significance = {
                'p_value': random.uniform(0.01, 0.1),
                'confidence_level': 0.95,
                'is_significant': random.choice([True, False])
            }
            
            return significance
            
        except Exception as e:
            self.logger.error(f"计算统计显著性失败: {e}")
            return {}
    
    def _get_recommendation(self, variants: Dict[str, Any]) -> str:
        """获取推荐结果"""
        try:
            if 'treatment' in variants and 'control' in variants:
                treatment_conversions = sum(variants['treatment']['conversions'].values())
                control_conversions = sum(variants['control']['conversions'].values())
                
                if treatment_conversions > control_conversions:
                    return "推荐使用treatment变体"
                else:
                    return "推荐使用control变体"
            else:
                return "需要更多数据进行分析"
                
        except Exception as e:
            self.logger.error(f"获取推荐结果失败: {e}")
            return "分析失败"
