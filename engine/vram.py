"""
VRAM Estimation
===============

Estimates VRAM requirements for UnSloth 4-bit quantized models with LoRA.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .base import VRAMEstimate

if TYPE_CHECKING:
    from config import TrellisConfig


# Known model architectures for more accurate estimates
MODEL_ARCHITECTURES = {
    # (params_billions, hidden_dim, num_layers, num_heads)
    "1b": (1.0, 2048, 16, 16),
    "3b": (3.0, 3200, 26, 32),
    "7b": (7.0, 4096, 32, 32),
    "8b": (8.0, 4096, 32, 32),
    "13b": (13.0, 5120, 40, 40),
    "70b": (70.0, 8192, 80, 64),
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


def _get_architecture(params_billions: float) -> tuple[int, int]:
    """Get approximate hidden_dim and num_layers for a model size."""
    # Find closest known architecture
    closest_key = min(
        MODEL_ARCHITECTURES.keys(),
        key=lambda k: abs(MODEL_ARCHITECTURES[k][0] - params_billions)
    )
    arch = MODEL_ARCHITECTURES[closest_key]
    return arch[1], arch[2]  # hidden_dim, num_layers


def estimate_vram_unsloth(config: "TrellisConfig") -> VRAMEstimate:
    """
    Estimate VRAM for UnSloth 4-bit quantized model + LoRA.

    Approximations:
    - Model params: 4-bit = ~0.5 bytes per param
    - LoRA: 2 matrices per layer, fp16
    - KV Cache: scales with seq_len * batch_size * layers
    - Optimizer: ~2x LoRA param size (Adam states)
    - Activations: rough estimate based on batch and seq_len
    """
    import torch

    # Parse model size
    params_billions = _parse_model_size(config.model_name)
    hidden_dim, num_layers = _get_architecture(params_billions)

    # Number of target modules per layer
    # Default targets: q, k, v, o, gate, up, down = 7 modules
    num_target_modules = len(config.lora_target_modules)

    # Base model (4-bit quantized)
    # 4-bit = 0.5 bytes per param, but with group quantization overhead
    # Realistic: ~0.6 bytes per param
    base_model_bytes = params_billions * 1e9 * 0.6
    base_model_gb = base_model_bytes / (1024 ** 3)

    # LoRA parameters
    # Each LoRA: A (hidden x rank) + B (rank x hidden) per module per layer
    # Using fp16 (2 bytes)
    lora_params_per_layer = num_target_modules * 2 * hidden_dim * config.lora_rank
    total_lora_params = lora_params_per_layer * num_layers
    lora_bytes = total_lora_params * 2  # fp16
    lora_gb = lora_bytes / (1024 ** 3)

    # KV Cache
    # 2 (K and V) * num_layers * 2 (key_dim ~ hidden/heads * heads) * seq_len * batch_size * 2 (fp16)
    # Simplified: 2 * layers * hidden_dim * seq_len * batch_size * 2
    batch_size = config.group_size
    kv_cache_bytes = 2 * num_layers * hidden_dim * config.max_seq_length * batch_size * 2
    kv_cache_gb = kv_cache_bytes / (1024 ** 3)

    # Optimizer states (AdamW: m1 + m2 per parameter)
    # 2 states per LoRA param, fp32 (4 bytes)
    optimizer_bytes = total_lora_params * 2 * 4
    optimizer_gb = optimizer_bytes / (1024 ** 3)

    # Activations (rough estimate)
    # During forward pass, activations scale with hidden_dim * seq_len * batch
    # During backward, gradients are similar
    activations_bytes = hidden_dim * config.max_seq_length * batch_size * 4 * 2  # fp32, 2x for backward
    activations_gb = activations_bytes / (1024 ** 3)

    # Total
    total_gb = base_model_gb + lora_gb + kv_cache_gb + optimizer_gb + activations_gb

    # Get available VRAM
    available_gb = 0.0
    if torch.cuda.is_available():
        try:
            available_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass

    # Add safety margin (15%)
    fits = total_gb * 1.15 < available_gb if available_gb > 0 else False

    return VRAMEstimate(
        base_model_gb=base_model_gb,
        lora_params_gb=lora_gb,
        kv_cache_gb=kv_cache_gb,
        optimizer_gb=optimizer_gb,
        activations_gb=activations_gb,
        total_gb=total_gb,
        fits_in_vram=fits,
        available_vram_gb=available_gb,
    )
