# 💻 代码规范与最佳实践

本文档提供EIT-P项目的代码规范、类型注解示例和最佳实践。

**更新日期**: 2025年10月8日

---

## 🎯 代码风格指南

### Python风格
遵循 **PEP 8** 标准

```python
# ✅ 好的命名
class EnhancedCEPEITP:
    def calculate_consciousness_level(self, metrics: Dict) -> float:
        pass

# ❌ 不好的命名
class eitp:
    def calc_cl(self, m):
        pass
```

---

## 📝 类型注解

### 基础类型注解

```python
from typing import List, Dict, Tuple, Optional, Union

def process_data(
    inputs: List[str],
    batch_size: int = 32,
    max_length: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    处理输入数据
    
    Args:
        inputs: 输入文本列表
        batch_size: 批量大小，默认32
        max_length: 最大长度，None表示不限制
    
    Returns:
        (处理后的张量, 统计信息字典)
    """
    ...
    return tensor, stats
```

---

### 复杂类型注解

```python
from typing import Callable, TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class CEPParameters:
    """CEP参数数据类"""
    mass_term: float = 1.0
    field_lambda: float = 0.1
    entropy_lambda: float = 0.05
    complexity_lambda: float = 0.01
    
    def __post_init__(self):
        """参数验证"""
        assert self.mass_term > 0, "质量项必须为正"
        assert 0 <= self.field_lambda <= 1, "场能量权重在[0,1]"

class Model(Generic[T]):
    """泛型模型基类"""
    def predict(self, input: T) -> T:
        ...
```

---

### PyTorch模型类型注解

```python
import torch
import torch.nn as nn
from torch import Tensor

class EnhancedCEPEITP(nn.Module):
    """
    Enhanced CEP-EIT-P模型
    
    Args:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        cep_params: CEP参数配置
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        cep_params: Optional[CEPParameters] = None
    ) -> None:
        super().__init__()
        
        self.input_dim: int = input_dim
        self.hidden_dims: List[int] = hidden_dims
        self.output_dim: int = output_dim
        
        # 构建网络层
        self.layers: nn.ModuleList = self._build_layers()
        self.cep_params: CEPParameters = cep_params or CEPParameters()
    
    def forward(
        self,
        x: Tensor
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_dim)
        
        Returns:
            output: 输出张量 (batch_size, output_dim)
            metrics: 性能指标字典
        """
        batch_size: int = x.size(0)
        
        # 前向传播
        hidden: Tensor = x
        for layer in self.layers:
            hidden = layer(hidden)
        
        output: Tensor = hidden
        
        # 计算指标
        metrics: Dict[str, Any] = self._compute_metrics(x, output)
        
        return output, metrics
    
    def _compute_metrics(
        self,
        input_tensor: Tensor,
        output_tensor: Tensor
    ) -> Dict[str, Any]:
        """计算性能指标"""
        ...
```

---

## 🔧 最佳实践

### 1. 错误处理

```python
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def safe_inference(
    model: nn.Module,
    input_text: str,
    max_retries: int = 3
) -> Optional[str]:
    """
    安全的推理函数，带重试机制
    
    Args:
        model: 模型
        input_text: 输入文本
        max_retries: 最大重试次数
    
    Returns:
        推理结果，失败返回None
    """
    for attempt in range(max_retries):
        try:
            # 验证输入
            if not input_text or len(input_text) > 1000:
                raise ValueError(f"输入文本长度无效: {len(input_text)}")
            
            # 推理
            with torch.no_grad():
                output = model(input_text)
            
            return output
            
        except RuntimeError as e:
            logger.error(f"推理失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if "out of memory" in str(e):
                torch.cuda.empty_cache()  # 清理GPU内存
                time.sleep(1)  # 等待
            else:
                raise
        
        except Exception as e:
            logger.error(f"未预期的错误: {e}")
            raise
    
    logger.error(f"推理失败，已重试{max_retries}次")
    return None
```

---

### 2. 配置管理

```python
from dataclasses import dataclass, field
from typing import List
import yaml

@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 模型参数
    input_dim: int = 768
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    output_dim: int = 10
    
    # 训练参数
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    
    # CEP参数
    cep_lambda_field: float = 0.1
    cep_lambda_entropy: float = 0.05
    cep_lambda_complexity: float = 0.01
    
    # 设备
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """从YAML文件加载配置"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """保存配置到YAML"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)

# 使用
config = TrainingConfig.from_yaml('config.yaml')
model = EnhancedCEPEITP(**config.__dict__)
```

---

### 3. 日志记录

```python
import logging
from pathlib import Path

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    配置日志系统
    
    Args:
        log_file: 日志文件路径，None则只输出到控制台
        level: 日志级别
    
    Returns:
        配置好的logger
    """
    # 创建logger
    logger = logging.getLogger('eitp')
    logger.setLevel(level)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 使用
logger = setup_logging('logs/training.log')
logger.info("开始训练")
logger.debug(f"批量大小: {batch_size}")
logger.warning("GPU内存不足")
logger.error("训练失败")
```

---

### 4. 性能监控

```python
import time
from contextlib import contextmanager
from typing import Generator

@contextmanager
def timer(name: str) -> Generator[None, None, None]:
    """
    计时上下文管理器
    
    使用:
        with timer("前向传播"):
            output = model(input)
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"{name} 耗时: {elapsed:.3f}秒")

# 性能监控类
class PerformanceMonitor:
    """性能监控"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def record(self, name: str, value: float) -> None:
        """记录指标"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """获取统计信息"""
        values = self.metrics.get(name, [])
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    def report(self) -> str:
        """生成报告"""
        lines = ["性能监控报告", "=" * 50]
        for name in self.metrics:
            stats = self.get_stats(name)
            lines.append(f"{name}:")
            lines.append(f"  均值: {stats['mean']:.4f}")
            lines.append(f"  标准差: {stats['std']:.4f}")
            lines.append(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        return "\n".join(lines)

# 使用
monitor = PerformanceMonitor()

for step in range(1000):
    with timer("训练步"):
        loss = train_step()
        monitor.record("loss", loss)
        monitor.record("lr", get_lr())

print(monitor.report())
```

---

### 5. GPU内存管理

```python
class GPUMemoryManager:
    """GPU内存管理器"""
    
    def __init__(self, threshold_gb: float = 3.0):
        self.threshold_gb: float = threshold_gb
        self.logger = logging.getLogger(__name__)
    
    def check_and_clear_memory(self) -> Dict[str, float]:
        """检查并清理GPU内存"""
        if not torch.cuda.is_available():
            return {}
        
        # 获取内存信息
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        usage_percent = (allocated_gb / total_gb) * 100
        
        self.logger.info(
            f"GPU内存 - 已分配: {allocated_gb:.2f}GB, "
            f"已预留: {reserved_gb:.2f}GB, "
            f"总量: {total_gb:.2f}GB, "
            f"使用率: {usage_percent:.1f}%"
        )
        
        # 超过阈值则清理
        if allocated_gb > self.threshold_gb:
            self.logger.warning(
                f"GPU内存使用 ({allocated_gb:.2f}GB) "
                f"超过阈值 ({self.threshold_gb:.2f}GB)，执行清理..."
            )
            torch.cuda.empty_cache()
            self.logger.info("GPU内存已清理")
        
        return {
            'allocated_gb': allocated_gb,
            'reserved_gb': reserved_gb,
            'total_gb': total_gb,
            'usage_percent': usage_percent
        }

# 使用
gpu_manager = GPUMemoryManager(threshold_gb=10.0)

for epoch in range(num_epochs):
    train_one_epoch()
    gpu_manager.check_and_clear_memory()
```

---

### 6. 线程安全模型

```python
import threading
from typing import Any

class ThreadSafeModel:
    """线程安全的模型包装器"""
    
    def __init__(self, model: nn.Module):
        self.model: nn.Module = model
        self.lock: threading.Lock = threading.Lock()
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """线程安全的调用"""
        with self.lock:
            return self.model(*args, **kwargs)
    
    def train(self) -> None:
        """设置为训练模式"""
        with self.lock:
            self.model.train()
    
    def eval(self) -> None:
        """设置为评估模式"""
        with self.lock:
            self.model.eval()

# 使用（多线程场景）
model = EnhancedCEPEITP(...)
thread_safe_model = ThreadSafeModel(model)

# 现在可以在多线程中安全调用
```

---

## 🧪 测试最佳实践

### 单元测试

```python
import unittest
import torch

class TestEnhancedCEPEITP(unittest.TestCase):
    """EnhancedCEPEITP模型测试"""
    
    def setUp(self) -> None:
        """测试前准备"""
        self.model = EnhancedCEPEITP(
            input_dim=768,
            hidden_dims=[512, 256],
            output_dim=10
        )
        self.test_input = torch.randn(4, 768)
    
    def test_forward_shape(self) -> None:
        """测试前向传播输出形状"""
        output, metrics = self.model(self.test_input)
        
        self.assertEqual(output.shape, (4, 10))
        self.assertIn('cep_energies', metrics)
        self.assertIn('consciousness_metrics', metrics)
    
    def test_cep_energy_positive(self) -> None:
        """测试CEP能量为正"""
        output, metrics = self.model(self.test_input)
        
        cep_energies = metrics['cep_energies']
        self.assertGreater(cep_energies['total_energy'], 0)
    
    def test_consciousness_level_range(self) -> None:
        """测试意识水平在合理范围"""
        output, metrics = self.model(self.test_input)
        
        level = metrics['consciousness_metrics'].consciousness_level
        self.assertGreaterEqual(level, 0)
        self.assertLessEqual(level, 10)
    
    def test_gradient_flow(self) -> None:
        """测试梯度流动"""
        output, _ = self.model(self.test_input)
        loss = output.sum()
        loss.backward()
        
        # 检查所有参数都有梯度
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(
                param.grad,
                f"参数 {name} 没有梯度"
            )

if __name__ == '__main__':
    unittest.main()
```

---

### 集成测试

```python
def test_end_to_end_training() -> None:
    """端到端训练测试"""
    
    # 1. 准备数据
    train_data = create_dummy_dataset(1000)
    val_data = create_dummy_dataset(200)
    
    # 2. 创建模型
    model = EnhancedCEPEITP(
        input_dim=768,
        hidden_dims=[512, 256],
        output_dim=10
    )
    
    # 3. 训练
    trainer = Trainer(model, config)
    trainer.train(train_data, val_data, epochs=5)
    
    # 4. 验证
    val_loss, val_acc = trainer.evaluate(val_data)
    assert val_acc > 0.5, "验证准确率太低"
    
    # 5. 保存和加载
    model.save('test_checkpoint.pt')
    loaded_model = EnhancedCEPEITP.load('test_checkpoint.pt')
    
    # 6. 验证加载正确
    output1, _ = model(test_input)
    output2, _ = loaded_model(test_input)
    torch.testing.assert_close(output1, output2)
    
    print("✅ 端到端测试通过")
```

---

## 📚 文档字符串规范

### Google风格

```python
def train_model(
    model: nn.Module,
    train_data: DataLoader,
    val_data: DataLoader,
    epochs: int = 10,
    learning_rate: float = 1e-4
) -> Dict[str, List[float]]:
    """
    训练模型
    
    使用CEP约束训练神经网络模型，自动优化能量效率。
    
    Args:
        model: 要训练的PyTorch模型
        train_data: 训练数据加载器
        val_data: 验证数据加载器
        epochs: 训练轮数，默认10
        learning_rate: 学习率，默认1e-4
    
    Returns:
        包含训练和验证损失历史的字典:
        {
            'train_loss': [epoch1_loss, epoch2_loss, ...],
            'val_loss': [epoch1_loss, epoch2_loss, ...],
            'cep_energy': [epoch1_energy, ...]
        }
    
    Raises:
        ValueError: 如果epochs <= 0
        RuntimeError: 如果训练过程中出现CUDA错误
    
    Examples:
        >>> model = EnhancedCEPEITP(input_dim=768, ...)
        >>> train_loader = DataLoader(train_dataset, batch_size=32)
        >>> val_loader = DataLoader(val_dataset, batch_size=32)
        >>> history = train_model(model, train_loader, val_loader, epochs=5)
        >>> print(f"最终验证损失: {history['val_loss'][-1]}")
    
    Note:
        - 自动应用CEP能量约束
        - 自动保存最佳checkpoint
        - 支持早停（early stopping）
    
    See Also:
        - evaluate_model: 模型评估函数
        - save_checkpoint: 保存检查点
    """
    if epochs <= 0:
        raise ValueError(f"epochs必须>0，当前值: {epochs}")
    
    # 实现...
    ...
```

---

## 🎨 代码组织

### 模块结构

```
eit_p/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── base.py          # 基类
│   ├── enhanced_cep.py  # CEP模型
│   └── memristor.py     # 忆阻器网络
├── training/
│   ├── __init__.py
│   ├── trainer.py       # 训练器
│   └── optimizer.py     # 优化器
├── utils/
│   ├── __init__.py
│   ├── logger.py        # 日志
│   ├── config.py        # 配置
│   └── metrics.py       # 指标
└── evaluation/
    ├── __init__.py
    ├── metrics.py       # 评估指标
    └── consciousness.py # 意识检测
```

---

### 导入规范

```python
# ✅ 好的导入顺序

# 标准库
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# 第三方库
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 本地模块
from eit_p.models import EnhancedCEPEITP
from eit_p.training import Trainer
from eit_p.utils import setup_logging

# ❌ 避免的做法
from eit_p.models import *  # 不要用*
import torch as t  # 不要用缩写（除了约定俗成的np, pd等）
```

---

## 🔒 安全最佳实践

### 输入验证

```python
def validate_input(
    text: str,
    max_length: int = 1000,
    allowed_chars: Optional[str] = None
) -> str:
    """
    验证和清理输入
    
    Args:
        text: 输入文本
        max_length: 最大长度
        allowed_chars: 允许的字符，None表示不限制
    
    Returns:
        清理后的文本
    
    Raises:
        ValueError: 如果输入无效
    """
    if not isinstance(text, str):
        raise ValueError(f"输入必须是字符串，当前类型: {type(text)}")
    
    if len(text) == 0:
        raise ValueError("输入不能为空")
    
    if len(text) > max_length:
        raise ValueError(f"输入过长: {len(text)} > {max_length}")
    
    if allowed_chars:
        invalid = set(text) - set(allowed_chars)
        if invalid:
            raise ValueError(f"包含非法字符: {invalid}")
    
    # 清理
    cleaned = text.strip()
    
    return cleaned
```

---

### API安全

```python
from functools import wraps
import jwt

def require_auth(f):
    """API认证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': '缺少认证token'}), 401
        
        try:
            # 验证JWT token
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            request.user_id = payload['user_id']
        except jwt.InvalidTokenError:
            return jsonify({'error': '无效token'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

# 使用
@app.route('/api/inference')
@require_auth
def inference():
    user_id = request.user_id
    ...
```

---

## 📦 代码复用

### 基类设计

```python
from abc import ABC, abstractmethod

class BaseModel(ABC, nn.Module):
    """模型基类"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """前向传播（子类必须实现）"""
        pass
    
    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.get_config()
        }, path)
        self.logger.info(f"模型已保存到 {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """加载模型"""
        checkpoint = torch.load(path)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    @abstractmethod
    def get_config(self) -> Dict:
        """获取配置（子类必须实现）"""
        pass
```

---

## 🚀 性能优化

### 1. 批处理优化

```python
def batch_inference(
    model: nn.Module,
    inputs: List[str],
    batch_size: int = 32
) -> List[str]:
    """
    批量推理（比逐个推理快10倍）
    
    Args:
        model: 模型
        inputs: 输入列表
        batch_size: 批量大小
    
    Returns:
        输出列表
    """
    outputs = []
    
    # 分批处理
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        
        # 批量推理
        with torch.no_grad():
            batch_outputs = model(batch)
        
        outputs.extend(batch_outputs)
    
    return outputs
```

---

### 2. 缓存机制

```python
from functools import lru_cache

class CachedModel:
    """带缓存的模型"""
    
    def __init__(self, model: nn.Module, cache_size: int = 1000):
        self.model = model
        self.cache_size = cache_size
    
    @lru_cache(maxsize=1000)
    def cached_forward(self, input_hash: int) -> Tensor:
        """缓存的前向传播"""
        # 注意: 实际使用需要更复杂的缓存键
        return self.model(reconstruct_from_hash(input_hash))
```

---

### 3. 异步处理

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncInferenceService:
    """异步推理服务"""
    
    def __init__(self, model: nn.Module, max_workers: int = 4):
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def async_inference(self, input_text: str) -> str:
        """异步推理"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._sync_inference,
            input_text
        )
        return result
    
    def _sync_inference(self, input_text: str) -> str:
        """同步推理（在线程池中执行）"""
        with torch.no_grad():
            return self.model(input_text)

# 使用
async def main():
    service = AsyncInferenceService(model)
    
    # 并发处理多个请求
    tasks = [
        service.async_inference(text)
        for text in input_list
    ]
    results = await asyncio.gather(*tasks)
    return results
```

---

## 📊 监控和调试

### 梯度监控

```python
class GradientMonitor:
    """梯度监控器"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_norms: Dict[str, List[float]] = {}
    
    def monitor_gradients(self) -> Dict[str, float]:
        """监控梯度"""
        stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                if name not in self.gradient_norms:
                    self.gradient_norms[name] = []
                self.gradient_norms[name].append(grad_norm)
                
                stats[name] = grad_norm
        
        return stats
    
    def check_gradient_health(self) -> None:
        """检查梯度健康状况"""
        for name, param in self.model.named_parameters():
            if param.grad is None:
                logger.warning(f"参数 {name} 没有梯度！")
                continue
            
            grad_norm = param.grad.norm().item()
            
            # 梯度爆炸
            if grad_norm > 100:
                logger.warning(f"参数 {name} 梯度爆炸: {grad_norm:.2f}")
            
            # 梯度消失
            if grad_norm < 1e-7:
                logger.warning(f"参数 {name} 梯度消失: {grad_norm:.2e}")

# 使用
monitor = GradientMonitor(model)

for step in range(training_steps):
    loss.backward()
    monitor.check_gradient_health()
    optimizer.step()
```

---

## 🎯 代码审查清单

### 提交前检查

- [ ] 所有函数有类型注解
- [ ] 所有公开函数有文档字符串
- [ ] 没有print()调试语句（用logging）
- [ ] 没有硬编码的路径和参数
- [ ] 通过所有单元测试
- [ ] 代码符合PEP 8
- [ ] 没有未使用的导入
- [ ] 没有TODO注释
- [ ] Git commit message清晰
- [ ] 更新了相关文档

---

## 🌟 EIT-P特有的最佳实践

### 1. CEP参数管理

```python
@dataclass
class CEPConfig:
    """CEP配置（不可变）"""
    mass_term: float
    field_lambda: float
    entropy_lambda: float
    complexity_lambda: float
    
    def __post_init__(self):
        """验证配置"""
        assert all(v > 0 for v in [
            self.mass_term,
            self.field_lambda,
            self.entropy_lambda,
            self.complexity_lambda
        ]), "所有CEP参数必须为正"
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'mass_term': self.mass_term,
            'field_lambda': self.field_lambda,
            'entropy_lambda': self.entropy_lambda,
            'complexity_lambda': self.complexity_lambda
        }
```

---

### 2. 意识水平监控

```python
def monitor_consciousness_evolution(
    model: EnhancedCEPEITP,
    data_loader: DataLoader,
    log_interval: int = 100
) -> None:
    """
    监控训练过程中意识水平的演化
    """
    consciousness_history = []
    
    for step, batch in enumerate(data_loader):
        output, metrics = model(batch)
        level = metrics['consciousness_metrics'].consciousness_level
        
        consciousness_history.append({
            'step': step,
            'level': level,
            'timestamp': time.time()
        })
        
        if step % log_interval == 0:
            logger.info(
                f"步骤 {step}: 意识水平 = {level:.2f}"
            )
    
    # 可视化演化
    plot_consciousness_evolution(consciousness_history)
```

---

## 📖 参考资源

- [PEP 8 -- Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 484 -- Type Hints](https://peps.python.org/pep-0484/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PyTorch Best Practices](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

**更新日期**: 2025年10月8日  
**维护者**: EIT-P团队  
**版本**: v1.0

---

*遵循这些规范和最佳实践，能够编写高质量、可维护的EIT-P代码。*

