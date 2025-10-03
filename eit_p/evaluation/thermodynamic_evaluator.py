
class ThermodynamicEvaluator:
    def __init__(self):
        pass

    def compute_efficiency(self, model, input_data, flops_estimator):
        # 计算FLOPs
        flops = flops_estimator.estimate(model, input_data)
        
        # 计算信息增益
        info_gain = self.compute_information_gain(model, input_data)
        
        # 效率 = info_gain / flops
        efficiency = info_gain / (flops + 1e-8)
        
        return efficiency.item()

    def compute_information_gain(self, model, input_data):
        with torch.no_grad():
            outputs = model(input_data)
            hidden_states = outputs.hidden_states[-1]
            # 使用熵作为信息增益的代理
            entropy = self.compute_entropy(hidden_states)
            return -entropy  # 负熵作为增益

    def compute_entropy(self, tensor):
        # 简单熵计算
        p = F.softmax(tensor.view(-1, tensor.size(-1)), dim=-1)
        entropy = - (p * p.log()).sum(dim=-1).mean()
        return entropy.item()
