import math
import torch
import torch.nn as nn
from src.core.lora import LoRALayer, LinearWithLoRA, inject_lora, count_parameters


def test_lora_layer_output_shape():
    layer = LoRALayer(64, 32, rank=4, alpha=8)
    x = torch.randn(10, 64)
    assert layer(x).shape == (10, 32)


def test_lora_layer_b_initialized_to_zero():
    layer = LoRALayer(64, 32, rank=4, alpha=8)
    assert layer.B.data.abs().max().item() == 0.0


def test_lora_layer_initial_output_is_zero():
    layer = LoRALayer(64, 32, rank=4, alpha=8)
    x = torch.randn(5, 64)
    assert layer(x).abs().max().item() == 0.0


def test_lora_layer_scaling():
    layer = LoRALayer(4, 4, rank=2, alpha=4)
    assert layer.scaling == 2.0  # alpha / rank = 4 / 2


def test_linear_with_lora_freezes_base():
    linear = nn.Linear(64, 32)
    wrapped = LinearWithLoRA(linear, rank=4, alpha=8)
    for param in wrapped.linear.parameters():
        assert not param.requires_grad


def test_linear_with_lora_lora_params_trainable():
    linear = nn.Linear(64, 32)
    wrapped = LinearWithLoRA(linear, rank=4, alpha=8)
    assert wrapped.lora.A.requires_grad
    assert wrapped.lora.B.requires_grad


def test_linear_with_lora_output_shape():
    linear = nn.Linear(64, 32)
    wrapped = LinearWithLoRA(linear, rank=4, alpha=8)
    x = torch.randn(8, 64)
    assert wrapped(x).shape == (8, 32)


def test_inject_lora_replaces_target_layers():
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
    inject_lora(model, target_modules=["0"], rank=4, alpha=8)
    assert isinstance(model[0], LinearWithLoRA)
    assert isinstance(model[2], nn.Linear)  # not replaced


def test_inject_lora_freezes_all_base_params():
    model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 16))
    inject_lora(model, target_modules=["0", "1"], rank=4, alpha=8)
    for name, param in model.named_parameters():
        if "lora" not in name:
            assert not param.requires_grad, f"{name} should be frozen"


def test_inject_lora_returns_dict():
    model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 16))
    result = inject_lora(model, target_modules=["0", "1"], rank=4, alpha=8)
    assert isinstance(result, dict)
    assert len(result) == 2


def test_count_parameters_trainable_pct_low_after_lora():
    model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10))
    inject_lora(model, target_modules=["0", "2"], rank=8, alpha=16)
    counts = count_parameters(model)
    assert counts["trainable_pct"] < 10.0
    assert counts["trainable"] > 0
    assert counts["frozen"] > counts["trainable"]


def test_count_parameters_keys():
    model = nn.Linear(16, 8)
    counts = count_parameters(model)
    assert set(counts.keys()) == {"total", "trainable", "frozen", "trainable_pct"}
