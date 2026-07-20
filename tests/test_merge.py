import torch
import torch.nn as nn
from src.core.lora import LinearWithLoRA, inject_lora
from src.core.merge import merge_lora_weights


def _lora_model(seed: int = 42):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
    inject_lora(model, target_modules=["0", "2"], rank=4, alpha=8)
    return model


def test_merge_removes_all_lora_layers():
    model = _lora_model()
    merge_lora_weights(model)
    for module in model.modules():
        assert not isinstance(module, LinearWithLoRA)


def test_merge_preserves_output_within_tolerance():
    model = _lora_model()
    x = torch.randn(5, 16)
    with torch.no_grad():
        out_before = model(x).clone()
    merge_lora_weights(model)
    with torch.no_grad():
        out_after = model(x)
    max_diff = (out_before - out_after).abs().max().item()
    assert max_diff < 1e-5, f"Max diff {max_diff:.2e} exceeds 1e-5"


def test_merge_restores_all_params_as_trainable():
    model = _lora_model()
    merge_lora_weights(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    all_params = list(model.parameters())
    assert len(trainable) == len(all_params)


def test_merge_after_training_preserves_output():
    from src.core.trainer import train_lora

    torch.manual_seed(0)
    model = _lora_model()
    x_data = torch.randn(50, 16)
    y_data = torch.randn(50, 4)
    train_lora(model, {"inputs": x_data, "targets": y_data}, epochs=5, lr=1e-3, batch_size=16)

    x_test = torch.randn(5, 16)
    with torch.no_grad():
        out_before = model(x_test).clone()
    merge_lora_weights(model)
    with torch.no_grad():
        out_after = model(x_test)
    max_diff = (out_before - out_after).abs().max().item()
    assert max_diff < 1e-5, f"Max diff after training: {max_diff:.2e}"
