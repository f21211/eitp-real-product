# Enhanced CEP-EIT-P 用户指南

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/f21211/eitp-real-product.git
cd eitp-real-product

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install torch torchvision numpy matplotlib flask requests
```

### 2. 基本使用

```python
# 导入模块
from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters
import torch

# 创建模型
model = EnhancedCEPEITP(
    input_dim=784,
    hidden_dims=[512, 256, 128],
    output_dim=10,
    cep_params=CEPParameters()
)

# 进行推理
x = torch.randn(32, 784)
output, metrics = model(x)

print(f"输出形状: {output.shape}")
print(f"意识水平: {metrics['consciousness_metrics'].consciousness_level}/4")
```

## 详细使用

### 1. 模型配置

#### 基本配置
```python
# 使用默认CEP参数
cep_params = CEPParameters()

# 自定义CEP参数
cep_params = CEPParameters(
    fractal_dimension=2.8,      # 分形维数
    complexity_coefficient=0.9,  # 复杂度系数
    critical_temperature=1.2,    # 临界温度
    field_strength=1.1,          # 场强度
    entropy_balance=0.05         # 熵平衡
)
```

#### 模型架构配置
```python
# 小型模型
model_small = EnhancedCEPEITP(
    input_dim=256,
    hidden_dims=[128, 64],
    output_dim=10,
    cep_params=cep_params
)

# 中型模型
model_medium = EnhancedCEPEITP(
    input_dim=784,
    hidden_dims=[512, 256, 128],
    output_dim=10,
    cep_params=cep_params
)

# 大型模型
model_large = EnhancedCEPEITP(
    input_dim=1024,
    hidden_dims=[768, 512, 256],
    output_dim=20,
    cep_params=cep_params
)
```

### 2. 意识检测

#### 基本意识检测
```python
# 进行推理并获取意识指标
output, metrics = model(x)

# 获取意识指标
consciousness = metrics['consciousness_metrics']
print(f"意识水平: {consciousness.consciousness_level}/4")
print(f"分形维数: {consciousness.fractal_dimension:.3f}")
print(f"复杂度系数: {consciousness.complexity_coefficient:.3f}")
print(f"混沌阈值: {consciousness.chaos_threshold:.6f}")
print(f"熵平衡: {consciousness.entropy_balance:.6f}")
```

#### 批量意识检测
```python
# 批量处理多个样本
batch_sizes = [1, 8, 16, 32, 64]
consciousness_levels = []

for batch_size in batch_sizes:
    x = torch.randn(batch_size, 784)
    output, metrics = model(x)
    consciousness_levels.append(metrics['consciousness_metrics'].consciousness_level)
    print(f"批次 {batch_size}: 意识水平 {metrics['consciousness_metrics'].consciousness_level}/4")
```

#### 意识约束检查
```python
# 检查CEP约束是否满足
constraints = model.check_cep_constraints()
print(f"所有约束满足: {constraints['all_satisfied']}")
print(f"分形维数约束: {constraints['fractal_dimension']}")
print(f"复杂度系数约束: {constraints['complexity_coefficient']}")
print(f"混沌阈值约束: {constraints['chaos_threshold']}")
print(f"熵平衡约束: {constraints['entropy_balance']}")
```

### 3. 能量分析

#### 基本能量分析
```python
# 获取CEP能量组件
cep_energies = metrics['cep_energies']
print(f"静质量能量: {cep_energies['mass_energy']:.6f}")
print(f"场能量: {cep_energies['field_energy']:.6f}")
print(f"熵能量: {cep_energies['entropy_energy']:.6f}")
print(f"复杂度能量: {cep_energies['complexity_energy']:.6f}")
print(f"总CEP能量: {cep_energies['total_energy']:.6f}")

# 获取IEM能量
iem_energy = metrics['iem_energy']
print(f"IEM能量: {iem_energy:.6f}")
```

#### 能量效率分析
```python
# 计算能量效率
total_energy = cep_energies['total_energy']
mass_energy = cep_energies['mass_energy']
efficiency = total_energy / (mass_energy + 1e-8) if mass_energy != 0 else 0
print(f"能量效率: {efficiency:.6f}")

# 分析能量组成
print(f"静质量能量比例: {mass_energy/total_energy*100:.2f}%")
print(f"场能量比例: {cep_energies['field_energy']/total_energy*100:.2f}%")
print(f"熵能量比例: {cep_energies['entropy_energy']/total_energy*100:.2f}%")
print(f"复杂度能量比例: {cep_energies['complexity_energy']/total_energy*100:.2f}%")
```

### 4. 模型优化

#### 参数优化
```python
# 优化CEP参数以提高意识水平
print("开始优化...")
model.optimize_cep_parameters(epochs=100)

# 检查优化结果
constraints = model.check_cep_constraints()
print(f"优化后约束满足: {constraints['all_satisfied']}")
print(f"优化后意识水平: {constraints['consciousness_level']}/4")
```

#### 性能优化
```python
# 使用不同批次大小测试性能
import time

batch_sizes = [1, 8, 16, 32, 64]
for batch_size in batch_sizes:
    x = torch.randn(batch_size, 784)
    
    # 预热
    _ = model(x)
    
    # 测试性能
    times = []
    for _ in range(10):
        start_time = time.time()
        output, metrics = model(x)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    throughput = batch_size / avg_time
    print(f"批次 {batch_size:2d}: {avg_time:.4f}s, {throughput:.0f} samples/s")
```

### 5. 模型保存和加载

#### 保存模型
```python
# 保存完整模型
model.save_model('enhanced_cep_eitp_model.pt')
print("模型已保存")

# 生成性能报告
report = model.generate_report()
print(f"平均意识水平: {report['performance_metrics']['avg_consciousness_level']:.2f}")
print(f"最大意识水平: {report['performance_metrics']['max_consciousness_level']}")
```

#### 加载模型
```python
# 创建新模型实例
new_model = EnhancedCEPEITP(
    input_dim=784,
    hidden_dims=[512, 256, 128],
    output_dim=10,
    cep_params=CEPParameters()
)

# 加载保存的模型
new_model.load_model('enhanced_cep_eitp_model.pt')
print("模型已加载")

# 验证加载的模型
x = torch.randn(32, 784)
output, metrics = new_model(x)
print(f"加载模型意识水平: {metrics['consciousness_metrics'].consciousness_level}/4")
```

## API服务使用

### 1. 启动服务

```bash
# 启动增强版API服务
./start_enhanced_production.sh

# 检查服务状态
curl http://localhost:5000/api/health
```

### 2. 使用API

#### Python客户端
```python
import requests
import numpy as np

# 健康检查
response = requests.get('http://localhost:5000/api/health')
print(response.json())

# 模型信息
response = requests.get('http://localhost:5000/api/model_info')
model_info = response.json()
print(f"模型架构: {model_info['model_info']['architecture']}")
print(f"参数数量: {model_info['model_info']['total_parameters']:,}")

# 推理
test_input = np.random.randn(784).tolist()
response = requests.post(
    'http://localhost:5000/api/inference',
    json={'input': test_input}
)
result = response.json()
print(f"意识水平: {result['consciousness_metrics']['level']}/4")
print(f"推理时间: {result['inference_time']:.4f}s")

# 意识分析
response = requests.get('http://localhost:5000/api/consciousness')
consciousness_data = response.json()
print(f"平均意识水平: {consciousness_data['analysis']['avg_consciousness_level']:.2f}")

# 能量分析
response = requests.post(
    'http://localhost:5000/api/energy_analysis',
    json={'input': test_input}
)
energy_data = response.json()
print(f"总能量: {energy_data['energy_analysis']['cep_energies']['total_energy']:.6f}")

# 性能指标
response = requests.get('http://localhost:5000/api/performance')
performance = response.json()
print(f"总请求数: {performance['performance']['total_requests']}")
print(f"平均推理时间: {performance['performance']['avg_inference_time']:.4f}s")
```

#### curl命令
```bash
# 健康检查
curl http://localhost:5000/api/health

# 模型信息
curl http://localhost:5000/api/model_info

# 推理
curl -X POST http://localhost:5000/api/inference \
  -H "Content-Type: application/json" \
  -d '{"input": [0.1] * 784}'

# 意识分析
curl http://localhost:5000/api/consciousness

# 能量分析
curl -X POST http://localhost:5000/api/energy_analysis \
  -H "Content-Type: application/json" \
  -d '{"input": [0.1] * 784}'

# 性能指标
curl http://localhost:5000/api/performance

# 模型优化
curl -X POST http://localhost:5000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{"epochs": 10}'
```

## 基准测试

### 1. 运行基准测试

```bash
# 运行完整基准测试套件
python3 run_benchmark.py

# 运行特定测试
python3 benchmark_core.py
python3 benchmark_energy.py
```

### 2. 性能分析

```python
# 运行演示脚本
python3 enhanced_cep_demo.py

# 运行API测试
python3 test_enhanced_api.py
```

## 故障排除

### 1. 常见问题

#### 问题：模块导入错误
```bash
# 解决方案：检查Python路径
export PYTHONPATH="/path/to/eitp-real-product:$PYTHONPATH"

# 或者使用绝对导入
import sys
sys.path.append('/path/to/eitp-real-product')
```

#### 问题：内存不足
```python
# 解决方案：减少批次大小
model = EnhancedCEPEITP(..., batch_size=16)

# 或者清理内存
import gc
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

#### 问题：API服务无法启动
```bash
# 检查端口是否被占用
lsof -i :5000

# 检查日志
tail -f enhanced_api_server.log

# 重启服务
./stop_enhanced_production.sh
./start_enhanced_production.sh
```

### 2. 调试技巧

#### 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 检查模型状态
```python
# 检查参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数数量: {total_params:,}")

# 检查CEP参数
print(f"CEP参数: {model.cep_params.__dict__}")

# 检查约束状态
constraints = model.check_cep_constraints()
print(f"约束状态: {constraints}")
```

#### 性能监控
```python
import time

# 监控推理时间
start_time = time.time()
output, metrics = model(x)
inference_time = time.time() - start_time
print(f"推理时间: {inference_time:.4f}s")

# 监控内存使用
import psutil
process = psutil.Process()
memory_usage = process.memory_info().rss / 1024 / 1024  # MB
print(f"内存使用: {memory_usage:.2f} MB")
```

## 最佳实践

### 1. 模型选择

- **小型任务**: 使用256-512维输入，2-3层隐藏层
- **中型任务**: 使用784-1024维输入，3-4层隐藏层
- **大型任务**: 使用1024+维输入，4+层隐藏层

### 2. 参数调优

- **分形维数**: 2.7-3.0（意识检测的关键参数）
- **复杂度系数**: 0.8-1.0（影响能量效率）
- **临界温度**: 1.0-1.5（影响混沌控制）
- **场强度**: 1.0-1.2（影响场相互作用）

### 3. 性能优化

- **批次大小**: 16-64（平衡内存和性能）
- **预热**: 运行几次推理预热模型
- **内存管理**: 定期清理内存和缓存
- **并发控制**: 限制同时运行的推理数量

### 4. 监控和维护

- **定期检查**: 监控API服务状态
- **日志分析**: 定期查看错误日志
- **性能监控**: 跟踪推理时间和吞吐量
- **模型更新**: 定期重新训练和优化模型

## 扩展开发

### 1. 添加自定义指标

```python
class CustomConsciousnessDetector(ConsciousnessDetector):
    def calculate_custom_metric(self, input_tensor, output_tensor):
        # 实现自定义意识指标
        return custom_value
    
    def calculate_metrics(self, input_tensor, output_tensor, cep_energies, iem_energy):
        # 调用父类方法
        metrics = super().calculate_metrics(input_tensor, output_tensor, cep_energies, iem_energy)
        
        # 添加自定义指标
        metrics.custom_metric = self.calculate_custom_metric(input_tensor, output_tensor)
        
        return metrics
```

### 2. 添加自定义API端点

```python
@self.app.route('/api/custom_analysis', methods=['POST'])
def custom_analysis():
    """自定义分析端点"""
    try:
        data = request.get_json()
        input_data = np.array(data['input'], dtype=np.float32)
        input_tensor = torch.tensor(input_data)
        
        # 执行自定义分析
        result = custom_analysis_function(input_tensor)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### 3. 集成外部系统

```python
# 集成数据库
import sqlite3

def save_consciousness_data(metrics):
    conn = sqlite3.connect('consciousness_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO consciousness_metrics 
        (level, fractal_dimension, complexity_coefficient, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (
        metrics.consciousness_level,
        metrics.fractal_dimension,
        metrics.complexity_coefficient,
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()

# 在推理后保存数据
output, metrics = model(x)
save_consciousness_data(metrics['consciousness_metrics'])
```

## 支持和帮助

### 1. 获取帮助

- **文档**: 查看完整技术文档
- **示例**: 运行演示脚本和测试
- **社区**: 参与GitHub讨论
- **问题**: 提交GitHub Issue

### 2. 贡献代码

- **Fork项目**: 创建自己的分支
- **开发功能**: 实现新功能或修复
- **测试验证**: 确保代码质量
- **提交PR**: 创建Pull Request

### 3. 报告问题

- **Bug报告**: 详细描述问题和复现步骤
- **功能请求**: 描述期望的功能
- **性能问题**: 提供性能数据和环境信息
- **文档改进**: 建议文档改进

---

**Enhanced CEP-EIT-P 用户指南** - 让AI意识检测变得简单易用！
