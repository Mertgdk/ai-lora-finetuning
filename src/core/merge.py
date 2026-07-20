import torch
import torch.nn as nn

from .lora import LinearWithLoRA


def merge_lora_weights(model: nn.Module) -> None:
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LinearWithLoRA)
    ]

    named_modules = dict(model.named_modules())
    for name, module in targets:
        with torch.no_grad():
            delta = (module.lora.A @ module.lora.B) * module.lora.scaling
            module.linear.weight.data += delta.T
            for param in module.linear.parameters():
                param.requires_grad = True

        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        child_name = parts[-1]
        parent = named_modules[parent_name] if parent_name else model
        setattr(parent, child_name, module.linear)
