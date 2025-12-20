"""
Trellis Configuration
=====================

All tunable parameters in one place. Supports JSON serialization for session persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TrellisConfig:
    """All tunable parameters in one place."""

    # Model
    model_name: str = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    append_think_tag: bool = True
    think_tag: str = "<think>"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Generation
    group_size: int = 4
    max_new_tokens: int = 256
    temperature: float = 1.2
    min_p: float = 0.1

    # Training
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    kl_beta: float = 0.03  # KL anchor strength; 0 disables

    # Undo settings
    max_undos: Optional[int] = None  # None = unlimited (will warn about disk usage)

    # Paths
    save_dir: str = "./trellis_sessions"

    # Default dataset
    default_dataset: str = "abhayesian/introspection-prompts"
    default_split: str = "train"

    # Optional prompt wrapping
    system_prompt: Optional[str] = None
    prompt_prefix: str = ""
    prompt_suffix: str = ""

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        data = asdict(self)
        # Convert tuple to list for JSON serialization
        data["lora_target_modules"] = list(data["lora_target_modules"])
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrellisConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        # Convert list back to tuple
        if "lora_target_modules" in data:
            data["lora_target_modules"] = tuple(data["lora_target_modules"])
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        data = asdict(self)
        data["lora_target_modules"] = list(data["lora_target_modules"])
        return data
