
import pytest
import torch
from eit_p.regularization.path_norm import PathNormRegularizer
from eit_p.losses.total_loss import TotalLoss
from transformers import AutoModelForCausalLM

@pytest.fixture
def sample_model():
    return AutoModelForCausalLM.from_pretrained('gpt2', output_hidden_states=True)

def test_path_norm_regularizer(sample_model):
    reg = PathNormRegularizer()
    losses = reg(sample_model)
    assert 'total_loss' in losses
    assert losses['total_loss'] >= 0, "损失应非负"

def test_total_loss(sample_model):
    total_loss = TotalLoss(vocab_size=50257, hidden_dim=768)
    input_ids = torch.randint(0, 50257, (2, 10))
    labels = input_ids.clone()
    loss_dict = total_loss(sample_model, input_ids, labels)
    assert 'total_loss' in loss_dict
    assert loss_dict['total_loss'] > 0, "总损失应正值"

def test_edge_case_empty_input(sample_model):
    total_loss = TotalLoss(vocab_size=50257, hidden_dim=768)
    input_ids = torch.tensor([]).reshape(0, 0)  # 空输入
    labels = input_ids.clone()
    with pytest.raises(ValueError):  # 预期错误
        total_loss(sample_model, input_ids, labels)
