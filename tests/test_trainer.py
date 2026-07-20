import torch
import torch.nn as nn
from src.core.lora import inject_lora
from src.core.trainer import train_lora


def _model_and_data(seed: int = 42):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8))
    x = torch.randn(100, 32)
    y = torch.randn(100, 8)
    return model, {"inputs": x, "targets": y}


def test_train_returns_list_length_equals_epochs():
    model, data = _model_and_data()
    inject_lora(model, target_modules=["0", "2"], rank=4, alpha=8)
    losses = train_lora(model, data, epochs=3, lr=1e-3, batch_size=16)
    assert isinstance(losses, list)
    assert len(losses) == 3


def test_train_losses_are_positive_floats():
    model, data = _model_and_data()
    inject_lora(model, target_modules=["0"], rank=4, alpha=8)
    losses = train_lora(model, data, epochs=2)
    assert all(isinstance(l, float) for l in losses)
    assert all(l > 0 for l in losses)


def test_train_loss_decreases_over_epochs():
    model, data = _model_and_data()
    inject_lora(model, target_modules=["0", "2"], rank=4, alpha=8)
    losses = train_lora(model, data, epochs=30, lr=5e-3, batch_size=16)
    assert losses[-1] < losses[0]


def test_train_frozen_weights_unchanged():
    model, data = _model_and_data()
    inject_lora(model, target_modules=["0"], rank=4, alpha=8)
    frozen_before = model[0].linear.weight.data.clone()
    train_lora(model, data, epochs=3, lr=1e-3, batch_size=16)
    assert torch.allclose(model[0].linear.weight.data, frozen_before)


def test_train_lora_params_change():
    model, data = _model_and_data()
    inject_lora(model, target_modules=["0"], rank=4, alpha=8)
    a_before = model[0].lora.A.data.clone()
    train_lora(model, data, epochs=3, lr=1e-3, batch_size=16)
    assert not torch.allclose(model[0].lora.A.data, a_before)
