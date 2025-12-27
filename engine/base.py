"""
Base Engine
===========

Abstract base class for training engines, allowing future extension to
different backends (different LoRA implementations, different optimizers, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import TrellisConfig
    from state.checkpoint import Checkpoint


class BaseEngine(ABC):
    """
    Abstract base class for training engines.

    Engines handle:
    - Model loading and configuration
    - Generation (sampling)
    - Training step (policy gradient update)
    - State management (checkpointing, restoration)
    """

    @abstractmethod
    def load_model(self) -> str:
        """
        Initialize model, tokenizer, LoRA, optimizer.

        Returns:
            Status message
        """
        ...

    @abstractmethod
    def generate_options(self, prompt: str) -> list[str]:
        """
        Generate GROUP_SIZE continuations for the given prompt.

        Args:
            prompt: The input prompt

        Returns:
            List of generated continuations
        """
        ...

    def generate_options_streaming(self, prompt: str):
        """
        Stream GROUP_SIZE continuations for the given prompt.

        Default implementation falls back to a single synchronous yield.
        """
        yield self.generate_options(prompt)

    @abstractmethod
    def train_step(self, choice_idx: int) -> tuple[str, dict]:
        """
        Perform one preference update.

        Args:
            choice_idx: 0..GROUP_SIZE-1 for chosen, GROUP_SIZE for reject-all

        Returns:
            (status_message, metrics_dict)
        """
        ...

    @abstractmethod
    def compute_drift(self) -> float:
        """
        L2 distance of LoRA weights from zero (distance from base model).

        Returns:
            Drift value
        """
        ...

    @abstractmethod
    def get_adapter_state(self) -> dict:
        """
        Snapshot adapter weights (CPU-resident).

        Returns:
            State dictionary
        """
        ...

    @abstractmethod
    def get_optimizer_state(self) -> dict:
        """
        Snapshot optimizer state (CPU-resident).

        Returns:
            State dictionary
        """
        ...

    @abstractmethod
    def restore_state(self, adapter_state: dict, optimizer_state: dict) -> None:
        """
        Restore adapter and optimizer from snapshots.

        Args:
            adapter_state: Adapter weights dictionary
            optimizer_state: Optimizer state dictionary
        """
        ...

    @abstractmethod
    def save_adapter(self, path: str) -> None:
        """
        Save current LoRA adapter to disk in HuggingFace format.

        Args:
            path: Directory to save adapter
        """
        ...

    @abstractmethod
    def merge_and_save(self, path: str) -> str:
        """
        Merge LoRA into base model and save full model.

        Args:
            path: Directory to save merged model

        Returns:
            Status message
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether model is loaded and ready."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name for dropdown display."""
        ...
