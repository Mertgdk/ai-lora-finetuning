import torch
from src.core.quantize import quantize_to_nf4, dequantize_from_nf4


def test_quantize_output_shapes():
    t = torch.randn(128)  # 2 blocks of 64
    q, s = quantize_to_nf4(t, block_size=64)
    assert q.shape == (2, 64)
    assert s.shape == (2, 1)


def test_quantize_values_in_range():
    t = torch.randn(128)
    q, _ = quantize_to_nf4(t, block_size=64)
    assert q.min().item() >= -8
    assert q.max().item() <= 7


def test_quantize_dtype_is_int8():
    t = torch.randn(128)
    q, _ = quantize_to_nf4(t, block_size=64)
    assert q.dtype == torch.int8


def test_dequantize_output_shape():
    t = torch.randn(128)
    q, s = quantize_to_nf4(t, block_size=64)
    out = dequantize_from_nf4(q, s, t.shape)
    assert out.shape == t.shape


def test_dequantize_approximate_reconstruction():
    torch.manual_seed(0)
    t = torch.randn(256)  # 4 blocks of 64
    q, s = quantize_to_nf4(t, block_size=64)
    out = dequantize_from_nf4(q, s, t.shape)
    mse = ((t - out) ** 2).mean().item()
    assert mse < 0.05


def test_quantize_2d_tensor():
    t = torch.randn(8, 64)  # 8 blocks of 64 (512 total)
    q, s = quantize_to_nf4(t, block_size=64)
    assert q.shape == (8, 64)
    out = dequantize_from_nf4(q, s, t.shape)
    assert out.shape == (8, 64)


def test_memory_compression_ratio():
    t = torch.randn(256)
    q, s = quantize_to_nf4(t, block_size=64)
    original_bytes = t.numel() * 4  # float32
    quantized_bytes = q.numel() * 1  # int8
    assert quantized_bytes < original_bytes / 2
