import math

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A @ self.B) * self.scaling


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def inject_lora(
    model: nn.Module, target_modules: list[str], rank: int = 8, alpha: int = 16
) -> dict:
    for param in model.parameters():
        param.requires_grad = False

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules)
    ]

    lora_layers = {}
    for name, module in targets:
        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        child_name = parts[-1]
        parent = dict(model.named_modules())[parent_name] if parent_name else model
        lora_linear = LinearWithLoRA(module, rank, alpha)
        setattr(parent, child_name, lora_linear)
        lora_layers[name] = lora_linear

    return lora_layers


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": round(100 * trainable / total, 4) if total > 0 else 0.0,
    }
