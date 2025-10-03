"""
安全审计模块
提供安全事件记录、分析和报告功能
"""

import logging
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class SecurityEventType(Enum):
    """安全事件类型"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    MODEL_ACCESS = "model_access"
    CONFIG_CHANGE = "config_change"
    TRAINING_START = "training_start"
    TRAINING_END = "training_end"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class SecurityEvent:
    """安全事件数据类"""
    event_id: str
    event_type: SecurityEventType
    timestamp: datetime
    user_id: Optional[str]
    resource: str
    action: str
    result: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class SecurityAuditor:
    """安全审计器"""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("security_auditor")
        self._setup_logging()
        self.events: List[SecurityEvent] = []
    
    def _setup_logging(self):
        """设置日志记录"""
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _generate_event_id(self, event: SecurityEvent) -> str:
        """生成事件ID"""
        content = f"{event.timestamp.isoformat()}{event.user_id}{event.resource}{event.action}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def log_event(self, event_type: SecurityEventType, user_id: Optional[str], 
                  resource: str, action: str, result: str, 
                  details: Dict[str, Any] = None, ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None) -> str:
        """记录安全事件"""
        event = SecurityEvent(
            event_id="",
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        event.event_id = self._generate_event_id(event)
        self.events.append(event)
        
        # 记录到日志文件
        log_entry = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "user_id": event.user_id,
            "resource": event.resource,
            "action": event.action,
            "result": event.result,
            "details": event.details,
            "ip_address": event.ip_address,
            "user_agent": event.user_agent
        }
        
        self.logger.info(f"Security Event: {json.dumps(log_entry)}")
        return event.event_id
    
    def get_events(self, event_type: Optional[SecurityEventType] = None,
                   user_id: Optional[str] = None, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[SecurityEvent]:
        """获取安全事件"""
        filtered_events = self.events
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        return sorted(filtered_events, key=lambda x: x.timestamp, reverse=True)
    
    def generate_report(self, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """生成安全报告"""
        events = self.get_events(start_time=start_time, end_time=end_time)
        
        # 统计信息
        total_events = len(events)
        event_type_counts = {}
        user_activity = {}
        resource_access = {}
        
        for event in events:
            # 事件类型统计
            event_type_counts[event.event_type.value] = event_type_counts.get(event.event_type.value, 0) + 1
            
            # 用户活动统计
            if event.user_id:
                user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
            
            # 资源访问统计
            resource_access[event.resource] = resource_access.get(event.resource, 0) + 1
        
        # 安全风险分析
        risk_events = [e for e in events if e.result == "failed" or e.event_type == SecurityEventType.ERROR]
        risk_score = len(risk_events) / max(total_events, 1) * 100
        
        return {
            "report_time": datetime.now().isoformat(),
            "period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None
            },
            "summary": {
                "total_events": total_events,
                "risk_events": len(risk_events),
                "risk_score": round(risk_score, 2)
            },
            "event_type_distribution": event_type_counts,
            "user_activity": user_activity,
            "resource_access": resource_access,
            "recent_events": [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp.isoformat(),
                    "type": e.event_type.value,
                    "user": e.user_id,
                    "resource": e.resource,
                    "action": e.action,
                    "result": e.result
                }
                for e in events[:10]  # 最近10个事件
            ]
        }
    
    def export_events(self, filename: str, event_type: Optional[SecurityEventType] = None):
        """导出安全事件到文件"""
        events = self.get_events(event_type=event_type)
        
        export_data = []
        for event in events:
            export_data.append({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "resource": event.resource,
                "action": event.action,
                "result": event.result,
                "details": event.details,
                "ip_address": event.ip_address,
                "user_agent": event.user_agent
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported {len(export_data)} events to {filename}")


# 全局审计器实例
_global_auditor = None


def get_auditor() -> SecurityAuditor:
    """获取全局审计器实例"""
    global _global_auditor
    if _global_auditor is None:
        _global_auditor = SecurityAuditor()
    return _global_auditor


def log_security_event(event_type: SecurityEventType, user_id: Optional[str],
                      resource: str, action: str, result: str,
                      details: Dict[str, Any] = None, **kwargs) -> str:
    """记录安全事件的便捷函数"""
    auditor = get_auditor()
    return auditor.log_event(event_type, user_id, resource, action, result, details, **kwargs)