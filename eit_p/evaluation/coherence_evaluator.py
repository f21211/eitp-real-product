
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Union

class CoherenceEvaluator:
    """连贯性评估器：基于LLM-as-a-Judge（PRD FR4.1），计算生成文本的逻辑一致性分数。"""

    def __init__(self, judge_model: str = 'gpt2', device: str = 'cuda'):
        self.device = device
        self.judge_model = AutoModelForCausalLM.from_pretrained(judge_model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(judge_model)
        self.judge_model.eval()

    def score(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        """计算连贯性分数（基于困惑度，较低困惑表示更高连贯性）。

        Args:
            text: 单文本或文本列表

        Returns:
            score: 分数（0-1，1表示完美连贯）或分数列表
        """
        if isinstance(text, str):
            text = [text]

        scores = []
        for t in text:
            if not t.strip():
                scores.append(0.0)
                continue

            try:
                inputs = self.tokenizer(t, return_tensors='pt').to(self.device)
                with torch.no_grad():
                    outputs = self.judge_model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss.item()

                # 困惑度 = exp(loss)
                perplexity = torch.exp(torch.tensor(loss)).item()

                # 分数 = 1 / (1 + perplexity) （归一化到[0,1]）
                score = 1 / (1 + perplexity)
                scores.append(score)
            except Exception as e:
                scores.append(0.0)  # 错误处理

        return scores if len(scores) > 1 else scores[0]

    def batch_score(self, texts: List[str], batch_size: int = 4) -> List[float]:
        """批量评估连贯性分数。"""
        batched_scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batched_scores.extend(self.score(batch))
        return batched_scores
