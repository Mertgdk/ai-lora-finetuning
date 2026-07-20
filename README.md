# LLM Fine-tuning Pipeline (LoRA / QLoRA)

LoRA and simulated QLoRA implemented from scratch in pure PyTorch. No GPU required. No model downloads.

## What it builds

- `LoRALayer` -- low-rank adapter: y = (x @ A @ B) * (alpha/r)
- `LinearWithLoRA` -- wraps a frozen Linear with a trainable LoRA adapter
- `inject_lora` -- replaces target Linear layers in any nn.Module with LoRA-wrapped versions
- `count_parameters` -- reports total / trainable / frozen params and trainable %
- `quantize_to_nf4` -- simulates 4-bit NF4 block quantization
- `dequantize_from_nf4` -- reconstructs from int8 blocks
- `train_lora` -- AdamW training loop, only updates LoRA A/B matrices
- `merge_lora_weights` -- bakes W' = W + (alpha/r)*BA back into the base linear, removes adapters
- Streamlit UI -- interactive pipeline: configure rank/alpha, train, visualize, verify merge

## Key result

A 256->512->512->10 MLP has ~800k parameters. With LoRA r=8 on 2 layers, only ~8k parameters train (~1%). After training, merge_lora_weights produces a model with identical outputs (max diff < 1e-5).

## Setup

```bash
uv sync
```

## Run

```bash
uv run python -m streamlit run src/app.py --server.port 8505
```

## Test

```bash
uv run pytest tests/ -v
```

## What LoRA does (math)

Standard linear: `y = Wx`

LoRA adds: `y = Wx + (alpha/r) * BAx`

Where B is (d_out x r) and A is (r x d_in). For r=8 on a 256x512 layer:
- Original trainable: 131,072
- LoRA trainable: (256x8) + (8x512) = 2,048 + 4,096 = 6,144 (4.7%)
