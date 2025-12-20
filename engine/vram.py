"""
VRAM Estimation
===============

Estimates VRAM requirements for UnSloth models (4-bit or 16-bit) with LoRA.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .base import VRAMEstimate

if TYPE_CHECKING:
    from config import TrellisConfig


# Known model architectures for more accurate estimates
MODEL_ARCHITECTURES = {
    # (params_billions, hidden_dim, num_layers, num_heads, intermediate_dim)
    "1b": (1.0, 2048, 16, 16, 5504),
    "3b": (3.0, 3200, 26, 32, 8640),
    "7b": (7.0, 4096, 32, 32, 11008),
    "8b": (8.0, 4096, 32, 32, 14336),
    "13b": (13.0, 5120, 40, 40, 13824),
    "70b": (70.0, 8192, 80, 64, 28672),
}

# Hand-tuned overrides for models we ship defaults for
KNOWN_MODEL_SPECS = {
    # Default model (1B class); rely on architecture table for derived sizes.
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit": {
        "params_billions": 1.0,
    },
}


def _parse_model_size(model_name: str) -> float:
    """Extract parameter count from model name."""
    name_lower = model_name.lower()

    # Look for patterns like "8B", "8b", "7B", "70B"
    match = re.search(r"(\d+\.?\d*)b", name_lower)
    if match:
        return float(match.group(1))

    # Default to 8B if can't parse
    return 8.0


def _try_autoconfig(model_name: str):
    """Try to pull architecture hints from AutoConfig without failing hard."""
    try:
        from transformers import AutoConfig  # type: ignore
    except Exception:
        return None

    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None

    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layers", None)
    intermediate = getattr(cfg, "intermediate_size", None) or getattr(cfg, "ffn_dim", None)
    mlp_ratio = getattr(cfg, "mlp_ratio", None)
    params_b = getattr(cfg, "model_size_in_billions", None) or getattr(cfg, "num_parameters", None)
    if isinstance(params_b, int):
        params_b = params_b / 1e9

    return {
        "hidden_dim": hidden,
        "num_layers": num_layers,
        "intermediate_dim": intermediate,
        "params_billions": params_b,
        "mlp_ratio": mlp_ratio,
    }


def _get_model_spec(model_name: str):
    """Get hidden dim, layers, intermediate, and param count hints."""
    lower_name = model_name.lower()
    if lower_name in KNOWN_MODEL_SPECS:
        return KNOWN_MODEL_SPECS[lower_name]

    auto_spec = _try_autoconfig(model_name)
    if auto_spec:
        return auto_spec

    params_billions = _parse_model_size(model_name)
    hidden_dim, num_layers, intermediate_dim = _get_architecture(params_billions)
    return {
        "params_billions": params_billions,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "intermediate_dim": intermediate_dim,
    }


def _get_architecture(params_billions: float) -> tuple[int, int, int]:
    """Get approximate hidden_dim, num_layers, and intermediate_dim for a model size."""
    # Find closest known architecture
    closest_key = min(
        MODEL_ARCHITECTURES.keys(),
        key=lambda k: abs(MODEL_ARCHITECTURES[k][0] - params_billions)
    )
    arch = MODEL_ARCHITECTURES[closest_key]
    return arch[1], arch[2], arch[4]  # hidden_dim, num_layers, intermediate_dim


def estimate_vram_unsloth(config: "TrellisConfig") -> VRAMEstimate:
    """
    Estimate VRAM for UnSloth 4-bit quantized model + LoRA.

    This provides a more realistic estimate based on observed actual usage.
    Key factors:
    - Model params: 4-bit = ~0.5 bytes per param + quantization overhead
    - LoRA: 2 matrices per target module per layer, fp16/fp32
    - KV Cache: scales with seq_len * batch_size * layers * 2 (K+V)
    - Optimizer: Adam states for LoRA params (2 momentum buffers, fp32)
    - Activations: forward pass activations held for backward
    - CUDA overhead: memory fragmentation, cuBLAS workspaces, etc.
    """
    import torch

    # Parse model size and shape hints
    spec = _get_model_spec(config.model_name)
    params_billions = spec.get("params_billions") or _parse_model_size(config.model_name)
    hidden_dim = spec.get("hidden_dim") or _get_architecture(params_billions)[0]
    num_layers = spec.get("num_layers") or _get_architecture(params_billions)[1]
    intermediate_dim = spec.get("intermediate_dim")
    if intermediate_dim is None and spec.get("mlp_ratio") and hidden_dim:
        intermediate_dim = hidden_dim * spec["mlp_ratio"]
    intermediate_dim = intermediate_dim or _get_architecture(params_billions)[2]

    # Defensive casts
    hidden_dim = int(hidden_dim)
    num_layers = int(num_layers)
    intermediate_dim = int(intermediate_dim)

    # Number of target modules per layer
    # Default targets: q, k, v, o, gate, up, down = 7 modules
    num_target_modules = len(config.lora_target_modules)

    # =========================================================================
    # Base Model (4-bit quantized)
    # =========================================================================
    # 4-bit = 0.5 bytes per param, but with:
    # - Group quantization overhead (~10-15%)
    # - Quantization scales and zeros
    # - Dequantization buffers during inference
    # Realistic multiplier: ~0.7 bytes per param
    if config.load_in_4bit:
        base_model_bytes = params_billions * 1e9 * 0.70
        precision_label = "4-bit"
    else:
        # fp16/bf16 weights + modest overhead
        base_model_bytes = params_billions * 1e9 * 2.2
        precision_label = "16-bit"
    base_model_gb = base_model_bytes / (1024 ** 3)

    # =========================================================================
    # LoRA Parameters
    # =========================================================================
    # Each LoRA module: A (hidden x rank) + B (rank x hidden)
    # For attention: q, k, v, o -> all use hidden_dim
    # For MLP: gate, up (hidden -> intermediate), down (intermediate -> hidden)

    # Attention modules (4): hidden_dim x rank x 2
    attn_lora_params = 4 * 2 * hidden_dim * config.lora_rank
    # MLP modules (3): mix of hidden and intermediate dims
    # gate/up: hidden -> intermediate (but LoRA operates on weight matrix)
    # down: intermediate -> hidden
    mlp_lora_params = 2 * 2 * hidden_dim * config.lora_rank + \
                      1 * 2 * intermediate_dim * config.lora_rank

    total_lora_params = (attn_lora_params + mlp_lora_params) * num_layers

    # LoRA params stored in fp32 for training stability
    lora_bytes = total_lora_params * 4  # fp32
    lora_gb = lora_bytes / (1024 ** 3)

    # =========================================================================
    # KV Cache
    # =========================================================================
    # 2 (K and V) * num_layers * 2 * head_dim * num_heads * seq_len * batch_size
    # Simplified: 2 * layers * hidden_dim * seq_len * batch_size * 2 (fp16)
    batch_size = config.group_size
    # During generation, KV cache grows with each token
    # Estimate for full context length
    kv_cache_bytes = 2 * num_layers * hidden_dim * config.max_seq_length * batch_size * 2
    kv_cache_gb = kv_cache_bytes / (1024 ** 3)

    # =========================================================================
    # Optimizer States
    # =========================================================================
    # AdamW: m1 (momentum) + m2 (variance) per parameter, fp32
    # 2 states per LoRA param, fp32 (4 bytes each)
    optimizer_bytes = total_lora_params * 2 * 4
    optimizer_gb = optimizer_bytes / (1024 ** 3)

    # =========================================================================
    # Activations (during training forward/backward)
    # =========================================================================
    # Activations scale with: batch_size * seq_len * hidden_dim * num_layers
    # Training uses full sequences (prompt + generated). Use a pessimistic estimate
    # based on half the context or prompt+max_new_tokens, whichever is larger.
    train_seq_len = min(
        config.max_seq_length,
        max(config.max_new_tokens + 512, config.max_seq_length // 2),
    )
    # Rough estimate with gradient checkpointing and backward buffers (conservative)
    activations_bytes = hidden_dim * train_seq_len * batch_size * num_layers * 4 * 2.5
    activations_gb = activations_bytes / (1024 ** 3)

    # =========================================================================
    # CUDA Overhead
    # =========================================================================
    # Memory fragmentation, cuBLAS workspaces, CUDA context, etc.
    # This can be 1-2 GB on its own
    cuda_overhead_gb = 1.5

    # =========================================================================
    # Total
    # =========================================================================
    total_gb = (
        base_model_gb +
        lora_gb +
        kv_cache_gb +
        optimizer_gb +
        activations_gb +
        cuda_overhead_gb
    )

    # Get available VRAM
    available_gb = 0.0
    if torch.cuda.is_available():
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            available_gb = free_mem / (1024 ** 3)
            total_device_gb = total_mem / (1024 ** 3)
        except Exception:
            try:
                total_device_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                available_gb = total_device_gb
            except Exception:
                total_device_gb = 0.0
    else:
        total_device_gb = 0.0

    # Add conservative safety margin (30%) for estimation uncertainty/fragmentation
    fits = total_gb * 1.30 < available_gb if available_gb > 0 else False

    return VRAMEstimate(
        base_model_gb=base_model_gb,
        lora_params_gb=lora_gb,
        kv_cache_gb=kv_cache_gb,
        optimizer_gb=optimizer_gb,
        activations_gb=activations_gb + cuda_overhead_gb,  # Combine for display
        total_gb=total_gb,
        fits_in_vram=fits,
        available_vram_gb=available_gb if available_gb > 0 else total_device_gb,
        precision=precision_label,
    )
