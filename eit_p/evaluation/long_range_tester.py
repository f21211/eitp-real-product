
import torch
import torch.nn.functional as F
from typing import Dict

class LongRangeTester:
    """长程依赖测试器：实现PRD 5.1中的Long-Range Accuracy测试，目标>70% on 8k tokens。"""

    def __init__(self, max_seq_len: int = 8192, num_steps: int = 10, device: str = 'cuda'):
        self.max_seq_len = max_seq_len
        self.num_steps = num_steps
        self.device = device

    def test(self, model, test_data: Dict) -> Dict[str, float]:
        """执行长程依赖测试。

        Args:
            model: 要测试的模型
            test_data: 测试数据，包含'input_ids'和'labels'

        Returns:
            results: 包含'accuracy'的字典
        """
        if 'input_ids' not in test_data or 'labels' not in test_data:
            return {'accuracy': 0.0, 'error': 'Invalid test data'}

        input_ids = test_data['input_ids'].to(self.device)
        labels = test_data['labels'].to(self.device)

        try:
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                # 多步键检索：模拟长程依赖
                preds = torch.argmax(logits, dim=-1)
                accuracy = (preds == labels).float().mean().item()

                # 长程特定：仅计算长序列部分的准确率
                long_range_acc = (preds[:, -self.max_seq_len//2:] == labels[:, -self.max_seq_len//2:]).float().mean().item()

            return {'accuracy': accuracy, 'long_range_accuracy': long_range_acc}
        except Exception as e:
            return {'accuracy': 0.0, 'error': str(e)}
