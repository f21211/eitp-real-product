import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoConfig

from .coherence_evaluator import CoherenceEvaluator
from .long_range_tester import LongRangeTester
from .thermodynamic_evaluator import ThermodynamicEvaluator
from ..regularization.chaos import ChaosRegularizer
from ..regularization.entropy import EntropyRegularizer
from .self_healing_tester import SelfHealingTester

import json
from datetime import datetime
from typing import Dict

class EmergenceEvaluator:
    """涌现评估器：计算PRD中指定的关键指标，包括连贯性、长程准确率等。支持指标导出。"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        self.coherence_scorer = CoherenceEvaluator(device=device)
        self.long_range_tester = LongRangeTester(device=device)
        self.thermo_evaluator = ThermodynamicEvaluator()
        self.chaos_reg = ChaosRegularizer()
        self.entropy_reg = EntropyRegularizer(model.config.hidden_size)
        self.self_healing_tester = SelfHealingTester(device=device)

    def evaluate_all(self, test_data):
        if not test_data:
            return {"error": "No test data provided"}
        metrics = {}
        metrics['coherence_score'] = self.evaluate_coherence(test_data)
        metrics['long_range_accuracy'] = self.evaluate_long_range_accuracy(test_data)
        metrics['thermodynamic_efficiency'] = self.evaluate_thermodynamic_efficiency()
        metrics['lyapunov_exponent'] = self.evaluate_lyapunov_exponent(test_data)
        metrics['differential_entropy'] = self.evaluate_differential_entropy(test_data)
        metrics['self_healing_rate'] = self.evaluate_self_healing(test_data)
        metrics.update(self.evaluate_internal_validation(test_data))
        return metrics

    def evaluate_coherence(self, test_data):
        # 实现连贯性评估
        texts = self.generate_texts(test_data)
        # 使用批量评分提高效率
        scores = self.coherence_scorer.batch_score([t for t in texts if t], batch_size=8)
        return sum(scores) / len(scores) if scores else 0.0

    def evaluate_long_range_accuracy(self, test_data):
        # 实现长程依赖测试
        results = self.long_range_tester.test(self.model, test_data)
        return results.get('accuracy', 0.0)

    def evaluate_thermodynamic_efficiency(self):
        # 实现热力学效率计算
        return self.thermo_evaluator.compute_efficiency(self.model)

    def evaluate_lyapunov_exponent(self, test_data):
        # 计算最大Lyapunov指数
        x = test_data.get('input').to(self.device)
        return self.chaos_reg.estimate_lyapunov_exponent(self.model, x)

    def evaluate_differential_entropy(self, test_data):
        # 计算微分熵
        inputs = test_data.get('input').to(self.device)
        outputs = self.model(inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        return self.entropy_reg.compute_entropy(hidden_states.flatten(0, 1))

    def evaluate_self_healing(self, test_data):
        # 测试自修复能力
        return self.self_healing_tester.test(self.model, test_data)

    def generate_texts(self, test_data):
        # 生成文本用于评估
        generated = []
        inputs_list = test_data.get('inputs', [])
        if not inputs_list:
            return generated

        for input in inputs_list:
            try:
                input_tensor = torch.tensor(input).unsqueeze(0).to(self.device)
                output = self.model.generate(input_tensor, max_length=50, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
                text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                generated.append(text)
            except Exception as e:
                generated.append("")
        return generated

    def evaluate_internal_validation(self, test_data):
        """计算内部验证指标，如Λmax、H(Z)。"""
        metrics = {}
        x = test_data.get('input').to(self.device)
        outputs = self.model(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        metrics['max_lyapunov'] = self.chaos_reg.estimate_lyapunov_exponent(self.model, x).item()
        metrics['differential_entropy'] = self.entropy_reg.compute_entropy(hidden_states.flatten(0, 1)).item()
        metrics['criticality_score'] = abs(metrics['max_lyapunov']) < 0.1  # 简单阈值
        
        return metrics

    def export_metrics(self, metrics: Dict, output_path: str = 'evaluation_results.json'):
        """导出评估指标到JSON文件。

        Args:
            metrics: 评估指标字典
            output_path: 输出文件路径
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
        print(f"Metrics exported to {output_path}")
