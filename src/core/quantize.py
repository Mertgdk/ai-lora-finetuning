import torch


def quantize_to_nf4(
    tensor: torch.Tensor, block_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = tensor.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales


def dequantize_from_nf4(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    original_shape: torch.Size,
) -> torch.Tensor:
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)
