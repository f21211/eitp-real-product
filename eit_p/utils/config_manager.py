"""
配置管理器
统一管理EIT-P项目的所有配置参数
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """EIT-P配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为项目根目录的config.yaml
        """
        if config_path is None:
            # 默认配置文件路径
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = {}
        self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """加载配置文件"""
        if config_path is not None:
            self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            # 如果配置文件不存在，创建默认配置
            self._create_default_config()
            return
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    def _create_default_config(self) -> None:
        """创建默认配置"""
        self._config = {
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 10,
                'gradient_accumulation_steps': 1
            },
            'model': {
                'hidden_dim': 128,
                'num_layers': 6,
                'dropout': 0.1
            },
            'hypernetwork': {
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.2
            },
            'memory': {
                'max_gpu_usage': 0.8,
                'cleanup_threshold': 0.9
            },
            'loss_weights': {
                'coherence': 1.0,
                'thermodynamic': 0.5,
                'path_norm': 0.1,
                'entropy': 0.1,
                'chaos': 0.1
            },
            'regularization': {
                'path_norm_weight': 0.01,
                'entropy_weight': 0.01,
                'chaos_weight': 0.01
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        # 保存默认配置
        self.save_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的嵌套键
        
        Args:
            key: 配置键，如 'training.batch_size'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self._config
        
        # 导航到目标位置
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.get('training', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.get('model', {})
    
    def get_hypernetwork_config(self) -> Dict[str, Any]:
        """获取超网络配置"""
        return self.get('hypernetwork', {})
    
    def get_memory_config(self) -> Dict[str, Any]:
        """获取内存管理配置"""
        return self.get('memory', {})
    
    def get_loss_weights(self) -> Dict[str, float]:
        """获取损失函数权重"""
        return self.get('loss_weights', {})
    
    def get_regularization_config(self) -> Dict[str, Any]:
        """获取正则化配置"""
        return self.get('regularization', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.get('logging', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """获取监控配置"""
        return self.get('monitoring', {})
    
    def get_safety_config(self) -> Dict[str, Any]:
        """获取安全配置"""
        return self.get('safety', {})
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        保存配置到文件
        
        Args:
            output_path: 输出路径，默认为原配置文件路径
        """
        if output_path is None:
            output_path = self.config_path
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """
        从字典更新配置
        
        Args:
            updates: 更新字典
        """
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self._config = deep_update(self._config, updates)
    
    def validate_config(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            配置是否有效
        """
        required_sections = [
            'training', 'model', 'hypernetwork', 'memory', 
            'loss_weights', 'regularization'
        ]
        
        for section in required_sections:
            if section not in self._config:
                print(f"警告: 缺少必需的配置节: {section}")
                return False
        
        # 验证训练配置
        training_config = self.get_training_config()
        batch_size = training_config.get('batch_size', 0)
        if isinstance(batch_size, str):
            try:
                batch_size = float(batch_size)
            except ValueError:
                print("错误: batch_size格式不正确")
                return False
        
        if batch_size <= 0:
            print("错误: batch_size必须大于0")
            return False
        
        learning_rate = training_config.get('learning_rate', 0)
        if isinstance(learning_rate, str):
            try:
                learning_rate = float(learning_rate)
            except ValueError:
                print("错误: learning_rate格式不正确")
                return False
        
        if learning_rate <= 0:
            print("错误: learning_rate必须大于0")
            return False
        
        # 验证内存配置
        memory_config = self.get_memory_config()
        max_gpu_usage = memory_config.get('max_gpu_usage', 0)
        if isinstance(max_gpu_usage, str):
            try:
                max_gpu_usage = float(max_gpu_usage)
            except ValueError:
                print("错误: max_gpu_usage格式不正确")
                return False
        
        if max_gpu_usage <= 0:
            print("错误: max_gpu_usage必须大于0")
            return False
        
        return True
    
    def __str__(self) -> str:
        """返回配置的字符串表示"""
        return f"ConfigManager(config_path={self.config_path})"
    
    def __repr__(self) -> str:
        """返回配置的详细表示"""
        return f"ConfigManager(config_path={self.config_path}, sections={list(self._config.keys())})"


# 全局配置实例
_global_config = None

def get_global_config() -> ConfigManager:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config

def set_global_config(config: ConfigManager) -> None:
    """设置全局配置实例"""
    global _global_config
    _global_config = config
