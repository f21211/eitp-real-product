#!/usr/bin/env python3
"""
教程1：使用EIT-P训练您的第一个模型

这是一个完整的入门教程，展示如何：
1. 加载模型
2. 准备数据
3. 使用EIT-P训练
4. 监控CEP参数
5. 评估结果

预计时间：30-60分钟
难度：入门
"""

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
from datetime import datetime

print("=" * 80)
print("🎓 教程1：使用EIT-P训练您的第一个模型")
print("=" * 80)
print()

# ============================================================================
# 步骤1：环境检查
# ============================================================================

print("步骤1：检查环境")
print("-" * 80)

print(f"Python version: {torch.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print()

# ============================================================================
# 步骤2：加载预训练模型
# ============================================================================

print("步骤2：加载GPT-2模型")
print("-" * 80)

print("加载tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

print("加载模型...")
model = GPT2LMHeadModel.from_pretrained('gpt2')

if torch.cuda.is_available():
    model = model.cuda()
    print("✅ 模型已移到GPU")

print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
print()

# ============================================================================
# 步骤3：准备训练数据
# ============================================================================

print("步骤3：准备训练数据")
print("-" * 80)

# 简单的训练样本
train_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming artificial intelligence.",
    "Physics provides the fundamental laws of nature.",
    "Consciousness emerges from complex systems.",
    "Intelligence requires energy and complexity.",
    "The universe follows mathematical principles.",
    "Quantum mechanics describes the microscopic world.",
    "Neural networks learn from data patterns.",
]

print(f"训练样本数: {len(train_texts)}")

# Tokenize
train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=32,
    return_tensors='pt'
)

if torch.cuda.is_available():
    train_encodings = {k: v.cuda() for k, v in train_encodings.items()}

print(f"Token数量: {train_encodings['input_ids'].shape}")
print()

# ============================================================================
# 步骤4：定义CEP监控函数
# ============================================================================

print("步骤4：定义CEP参数监控")
print("-" * 80)

def calculate_fractal_dimension(tensor):
    """简化的分形维度计算"""
    if tensor.numel() == 0:
        return 2.0
    
    # 使用张量的统计特性估计分形维度
    flat = tensor.view(-1).cpu().detach().numpy()
    import numpy as np
    
    # 简化方法：使用标准差和范围的比率
    std = np.std(flat)
    range_val = np.ptp(flat)  # peak-to-peak
    
    if range_val > 0:
        D = 2.0 + np.log(std / (range_val + 1e-8))
        return max(1.5, min(3.5, abs(D)))  # 限制在合理范围
    return 2.0

def calculate_complexity_coefficient(model):
    """计算复杂度系数λ"""
    total = 0
    active = 0
    
    for param in model.parameters():
        total += param.numel()
        active += (param.abs() > 1e-3).sum().item()
    
    return active / total if total > 0 else 0.0

def calculate_iem_energy(model, alpha=1.0):
    """计算IEM能量"""
    # H - 信息熵（用参数分布的熵近似）
    H = 0.0
    for param in model.parameters():
        if param.numel() > 0:
            p = torch.softmax(param.view(-1), dim=0)
            H += -(p * torch.log(p + 1e-10)).sum().item()
    
    # T - 温度（用梯度的标准差近似）
    T = 1.0  # 简化为常数
    
    # C - 连贯性（用参数的相关性近似）
    C = 0.9  # 简化为常数
    
    IEM = alpha * H * T * C
    return IEM

print("✅ CEP监控函数已定义")
print()

# ============================================================================
# 步骤5：训练循环（简化版）
# ============================================================================

print("步骤5：开始训练")
print("-" * 80)

# 训练参数
epochs = 3
learning_rate = 5e-5
alpha = 1.0  # IEM系数

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 记录
history = {
    'loss': [],
    'fractal_dimension': [],
    'complexity_coefficient': [],
    'iem_energy': [],
    'time': []
}

print(f"训练配置:")
print(f"  Epochs: {epochs}")
print(f"  Learning rate: {learning_rate}")
print(f"  Alpha (IEM): {alpha}")
print()

model.train()
start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    total_loss = 0
    
    # 简单的训练循环
    for i in range(0, len(train_texts), 2):  # Batch size = 2
        # 准备batch
        batch_size = min(2, len(train_texts) - i)
        batch_texts = train_texts[i:i+batch_size]
        
        batch_enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=32,
            return_tensors='pt'
        )
        
        if torch.cuda.is_available():
            batch_enc = {k: v.cuda() for k, v in batch_enc.items()}
        
        # Forward pass
        outputs = model(**batch_enc, labels=batch_enc['input_ids'])
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / (len(train_texts) // 2)
    
    # 计算CEP参数
    with torch.no_grad():
        # 获取模型输出用于计算分形维度
        sample_output = model(**train_encodings, labels=train_encodings['input_ids'])
        logits = sample_output.logits
        
        D = calculate_fractal_dimension(logits)
        lambda_val = calculate_complexity_coefficient(model)
        iem = calculate_iem_energy(model, alpha)
    
    epoch_time = time.time() - epoch_start
    
    # 记录
    history['loss'].append(avg_loss)
    history['fractal_dimension'].append(D)
    history['complexity_coefficient'].append(lambda_val)
    history['iem_energy'].append(iem)
    history['time'].append(epoch_time)
    
    # 显示进度
    print(f"Epoch {epoch+1}/{epochs}:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  分形维度 D: {D:.3f} {'✅' if D >= 2.7 else '⏳'} (目标: ≥2.7)")
    print(f"  复杂度系数 λ: {lambda_val:.3f} {'✅' if lambda_val >= 0.8 else '⏳'} (目标: ≥0.8)")
    print(f"  IEM能量: {iem:.6f}")
    print(f"  时间: {epoch_time:.2f}s")
    
    # 判断是否达到涌现阈值
    if D >= 2.7 and lambda_val >= 0.8:
        print("  🎉 达到智能涌现阈值！")
    
    print()

total_time = time.time() - start_time
print(f"训练完成！总时间: {total_time:.2f}s")
print()

# ============================================================================
# 步骤6：评估和测试
# ============================================================================

print("步骤6：评估模型")
print("-" * 80)

model.eval()

# 测试生成
test_prompts = [
    "The future of AI is",
    "Intelligence emerges from",
    "Physics and AI are",
]

print("生成测试:")
for prompt in test_prompts:
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=20,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  输入: {prompt}")
    print(f"  输出: {generated_text}")
    print()

# ============================================================================
# 步骤7：生成训练报告
# ============================================================================

print("步骤7：生成训练报告")
print("-" * 80)

report = f"""
# 训练报告

**训练时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**总时长**: {total_time:.2f}秒

## 配置

- 模型: GPT-2 Small
- 训练样本: {len(train_texts)}
- Epochs: {epochs}
- Learning rate: {learning_rate}
- Alpha (IEM): {alpha}

## 训练结果

### Loss曲线

| Epoch | Loss | 时间(s) |
|-------|------|---------|
"""

for i in range(epochs):
    report += f"| {i+1} | {history['loss'][i]:.4f} | {history['time'][i]:.2f} |\n"

report += f"""

### CEP参数演化

| Epoch | D | λ | IEM能量 | 达到阈值 |
|-------|---|---|---------|----------|
"""

for i in range(epochs):
    meets = "✅" if history['fractal_dimension'][i] >= 2.7 and history['complexity_coefficient'][i] >= 0.8 else "❌"
    report += f"| {i+1} | {history['fractal_dimension'][i]:.3f} | {history['complexity_coefficient'][i]:.3f} | {history['iem_energy'][i]:.6f} | {meets} |\n"

report += f"""

## 分析

**最终CEP参数**:
- 分形维度 D: {history['fractal_dimension'][-1]:.3f}
- 复杂度系数 λ: {history['complexity_coefficient'][-1]:.3f}
- IEM能量: {history['iem_energy'][-1]:.6f}

**是否达到智能涌现阈值**:
- D ≥ 2.7: {'✅ 是' if history['fractal_dimension'][-1] >= 2.7 else '❌ 否'}
- λ ≥ 0.8: {'✅ 是' if history['complexity_coefficient'][-1] >= 0.8 else '❌ 否'}

## 结论

这是一个{'成功的' if history['fractal_dimension'][-1] >= 2.7 and history['complexity_coefficient'][-1] >= 0.8 else '初步的'}训练实验。
{'模型已达到智能涌现的CEP阈值条件。' if history['fractal_dimension'][-1] >= 2.7 and history['complexity_coefficient'][-1] >= 0.8 else '需要更多训练或调整参数以达到涌现阈值。'}

## 下一步

1. 尝试更多训练数据
2. 调整alpha参数
3. 增加训练epochs
4. 使用更大的模型

---

**报告生成时间**: {datetime.now().isoformat()}
"""

# 保存报告
report_file = 'tutorial_01_training_report.md'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"✅ 训练报告已保存到: {report_file}")
print()

# ============================================================================
# 总结
# ============================================================================

print("=" * 80)
print("🎊 教程1完成！")
print("=" * 80)
print()
print("您已经学会了:")
print("  ✅ 加载和配置EIT-P环境")
print("  ✅ 训练一个简单模型")
print("  ✅ 监控CEP参数")
print("  ✅ 理解智能涌现阈值")
print()
print("下一步:")
print("  1. 查看训练报告: cat tutorial_01_training_report.md")
print("  2. 尝试修改参数重新训练")
print("  3. 进入教程2: TUTORIAL_02_CONSCIOUSNESS_DETECTION.py")
print()
print("=" * 80)

