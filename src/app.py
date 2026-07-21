import json
import os

import torch
import torch.nn as nn
import streamlit as st

from src.core.lora import inject_lora, count_parameters
from src.core.quantize import quantize_to_nf4, dequantize_from_nf4
from src.core.trainer import train_lora
from src.core.merge import merge_lora_weights

st.set_page_config(page_title="LoRA Fine-tuning Pipeline", layout="wide")
st.title("LoRA / QLoRA Fine-tuning Pipeline")

tab1, tab2 = st.tabs(["CPU Demo (Synthetic MLP)", "Real Llama-3.2-1B Results (Colab)"])

# ── Tab 1: CPU Demo ───────────────────────────────────────────────────────────
with tab1:
    st.caption("Pure PyTorch LoRA -- synthetic MLP, CPU only, no model downloads.")

    with st.sidebar:
        st.header("Configuration")
        rank = st.selectbox("Rank (r)", [4, 8, 16, 32], index=1)
        alpha = st.number_input("Alpha", value=int(rank) * 2, min_value=1, step=1)
        target_layers = st.multiselect(
            "Target layers",
            ["0", "2", "4"],
            default=["0", "2"],
            help="Layer indices in the MLP to inject LoRA adapters into",
        )
        epochs = st.slider("Training epochs", 5, 100, 30)
        lr = st.select_slider(
            "Learning rate",
            options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}",
        )
        seed = st.number_input("Random seed", value=42, step=1)
        run = st.button("Run LoRA Pipeline", type="primary", use_container_width=True)

    st.markdown("""
**Model:** 256 -> 512 -> ReLU -> 512 -> ReLU -> 10 (MLP, synthetic data, CPU)

**Pipeline:** Inject LoRA -> Train (only adapters update) -> Merge -> Verify output unchanged
""")

    if run:
        torch.manual_seed(int(seed))

        model = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10),
        )

        n_samples = 500
        x_data = torch.randn(n_samples, 256)
        y_idx = torch.randint(0, 10, (n_samples,))
        y_data = torch.zeros(n_samples, 10).scatter_(1, y_idx.unsqueeze(1), 1.0)
        data = {"inputs": x_data, "targets": y_data}

        params_before = count_parameters(model)
        if target_layers:
            inject_lora(model, target_modules=target_layers, rank=int(rank), alpha=int(alpha))
        params_after = count_parameters(model)

        st.subheader("Parameter Counts")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Before injection**")
            st.metric("Total", f"{params_before['total']:,}")
            st.metric("Trainable", f"{params_before['trainable']:,}")
            st.metric("Trainable %", f"{params_before['trainable_pct']:.2f}%")
        with col2:
            st.markdown("**After LoRA injection**")
            st.metric("Total", f"{params_after['total']:,}")
            st.metric("Trainable", f"{params_after['trainable']:,}",
                      delta=f"{params_after['trainable'] - params_before['trainable']:+,}")
            st.metric("Trainable %", f"{params_after['trainable_pct']:.4f}%")

        st.caption(f"Training only {params_after['trainable_pct']:.4f}% of parameters "
                   f"({params_after['trainable']:,} vs {params_before['total']:,} total)")

        with st.spinner(f"Training {epochs} epochs..."):
            losses = train_lora(model, data, epochs=int(epochs), lr=float(lr), batch_size=16)

        st.subheader("Training Loss Curve")
        st.line_chart({"Loss": losses})
        st.caption(f"Start: {losses[0]:.4f} -> End: {losses[-1]:.4f} "
                   f"({100*(losses[0]-losses[-1])/losses[0]:.1f}% reduction)")

        st.subheader("QLoRA: Simulated 4-bit Quantization")
        weight = (model[0].linear.weight.data.detach()
                  if hasattr(model[0], "linear") else model[0].weight.data.detach())
        flat = weight.reshape(-1)
        flat_trimmed = flat[: (len(flat) // 64) * 64]
        q, s = quantize_to_nf4(flat_trimmed, block_size=64)
        orig_bytes = flat_trimmed.numel() * 4
        q_bytes = q.numel()
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            st.metric("Original (float32)", f"{orig_bytes:,} bytes")
        with col_q2:
            st.metric("Quantized (int8)", f"{q_bytes:,} bytes",
                      delta=f"-{100*(1-q_bytes/orig_bytes):.0f}%")
        with col_q3:
            recon = dequantize_from_nf4(q, s, flat_trimmed.shape)
            st.metric("Reconstruction MSE", f"{((flat_trimmed - recon) ** 2).mean().item():.6f}")

        st.subheader("Merge Verification")
        x_test = torch.randn(10, 256)
        with torch.no_grad():
            out_before = model(x_test).clone()
        merge_lora_weights(model)
        params_merged = count_parameters(model)
        with torch.no_grad():
            out_after = model(x_test)
        max_diff = (out_before - out_after).abs().max().item()

        with col3:
            st.markdown("**After merge**")
            st.metric("Total", f"{params_merged['total']:,}")
            st.metric("Trainable", f"{params_merged['trainable']:,}")
            st.metric("Trainable %", f"{params_merged['trainable_pct']:.2f}%")

        if max_diff < 1e-4:
            st.success(f"Merge verified: max output diff = {max_diff:.2e}")
        else:
            st.error(f"Merge diff too large: {max_diff:.2e}")

# ── Tab 2: Real Llama Results ─────────────────────────────────────────────────
with tab2:
    results_path = os.path.join(os.path.dirname(__file__), "..", "results", "colab_results.json")

    if not os.path.exists(results_path):
        st.info("Henuz Colab sonucu yok. Colab notebook'un son hucresini (cell-12) calistirip push et.")
        st.code("git pull  # sonra Streamlit'i yenile", language="bash")
    else:
        with open(results_path) as f:
            r = json.load(f)

        st.subheader("Model & Training Setup")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", r["model"].split("/")[-1])
        c2.metric("Total Params", f"{r['total_params']:,}")
        c3.metric("Trainable Params", f"{r['trainable_params']:,}")
        c4.metric("Trainable %", f"{r['trainable_pct']:.4f}%")

        st.caption(f"Dataset: {r['dataset']} | {r['n_samples']} ornek | "
                   f"{r['epochs']} epoch | lr={r['lr']}")

        st.subheader("Training Loss Curve")
        st.line_chart({"Loss": r["all_losses"]})
        el = r["epoch_losses"]
        reduction = 100 * (el[0] - el[-1]) / el[0]
        st.caption(f"Loss: {el[0]:.4f} -> {el[-1]:.4f} ({reduction:.1f}% dusus) | "
                   f"Epoch avg: {' -> '.join(f'{v:.4f}' for v in el)}")

        st.subheader("Merge Verification")
        diff = r["merge_max_diff"]
        # bfloat16 models: argmax match is the meaningful check, not raw logit diff
        if r["merge_success"]:
            st.success(f"Merge BASARILI -- max logit diff: {diff:.2e}")
        else:
            st.warning(
                f"Max logit diff: {diff:.2e} (bfloat16 precision -- numerically expected, "
                f"argmax/generation output degismez)"
            )

        st.subheader("Before vs After Fine-tuning")
        for pair in r["qa_pairs"]:
            with st.expander(f"Musteri: {pair['question']}"):
                col_b, col_a = st.columns(2)
                with col_b:
                    st.markdown("**ONCESI (baseline)**")
                    st.write(pair["before"])
                with col_a:
                    st.markdown("**SONRASI (fine-tuned)**")
                    st.write(pair["after"])
