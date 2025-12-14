"""
Checkpoint
==========

A restorable snapshot of training state, including adapter weights,
optimizer state, and cached completions for undo.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


@dataclass
class Checkpoint:
    """A restorable snapshot of training state."""

    id: str
    step_count: int
    created_at: str

    # What led here (for history display and undo)
    prompt: Optional[str] = None
    choice: Optional[str] = None
    chosen_response: Optional[str] = None
    options: list[str] = field(default_factory=list)  # Cached completions for undo

    # Metrics at this point
    drift_from_base: float = 0.0

    # Tensor data paths (on disk)
    adapter_state_path: Optional[str] = None
    optimizer_state_path: Optional[str] = None

    # In-memory state (before persistence, cleared after)
    _adapter_state: Optional[dict] = field(default=None, repr=False)
    _optimizer_state: Optional[dict] = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        step_count: int,
        prompt: Optional[str] = None,
        choice: Optional[str] = None,
        chosen_response: Optional[str] = None,
        options: Optional[list[str]] = None,
        drift_from_base: float = 0.0,
        adapter_state: Optional[dict] = None,
        optimizer_state: Optional[dict] = None,
    ) -> "Checkpoint":
        """Create a new checkpoint with generated ID and timestamp."""
        return cls(
            id=uuid.uuid4().hex[:8],
            step_count=step_count,
            created_at=datetime.now().isoformat(),
            prompt=prompt,
            choice=choice,
            chosen_response=chosen_response,
            options=options or [],
            drift_from_base=drift_from_base,
            _adapter_state=adapter_state,
            _optimizer_state=optimizer_state,
        )

    def persist(self, checkpoint_dir: Path) -> None:
        """
        Write tensors and metadata to disk, then clear in-memory state.

        Directory structure:
            checkpoint_dir/
                meta.json
                adapter.pt
                optimizer.pt
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        meta = {
            "id": self.id,
            "step_count": self.step_count,
            "created_at": self.created_at,
            "prompt": self.prompt,
            "choice": self.choice,
            "chosen_response": self.chosen_response,
            "options": self.options,
            "drift_from_base": self.drift_from_base,
        }
        with open(checkpoint_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save tensors
        if self._adapter_state is not None:
            adapter_path = checkpoint_dir / "adapter.pt"
            torch.save(self._adapter_state, adapter_path)
            self.adapter_state_path = str(adapter_path)
            self._adapter_state = None  # Free memory

        if self._optimizer_state is not None:
            optimizer_path = checkpoint_dir / "optimizer.pt"
            torch.save(self._optimizer_state, optimizer_path)
            self.optimizer_state_path = str(optimizer_path)
            self._optimizer_state = None  # Free memory

    def load_tensors(self, device: str = "cuda") -> tuple[dict, dict]:
        """
        Lazy-load tensors from disk.

        Returns:
            (adapter_state, optimizer_state) dictionaries
        """
        adapter_state = {}
        optimizer_state = {}

        if self.adapter_state_path and Path(self.adapter_state_path).exists():
            adapter_state = torch.load(
                self.adapter_state_path,
                map_location=device,
                weights_only=True,
            )

        if self.optimizer_state_path and Path(self.optimizer_state_path).exists():
            optimizer_state = torch.load(
                self.optimizer_state_path,
                map_location=device,
                weights_only=True,
            )

        return adapter_state, optimizer_state

    @classmethod
    def load_from_dir(cls, checkpoint_dir: Path) -> "Checkpoint":
        """Load a checkpoint from a directory."""
        meta_path = checkpoint_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        checkpoint = cls(
            id=meta["id"],
            step_count=meta["step_count"],
            created_at=meta["created_at"],
            prompt=meta.get("prompt"),
            choice=meta.get("choice"),
            chosen_response=meta.get("chosen_response"),
            options=meta.get("options", []),
            drift_from_base=meta.get("drift_from_base", 0.0),
        )

        # Set paths if files exist
        adapter_path = checkpoint_dir / "adapter.pt"
        if adapter_path.exists():
            checkpoint.adapter_state_path = str(adapter_path)

        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            checkpoint.optimizer_state_path = str(optimizer_path)

        return checkpoint

    def to_dict(self) -> dict:
        """Convert to dictionary for index serialization (no tensor data)."""
        return {
            "id": self.id,
            "step_count": self.step_count,
            "created_at": self.created_at,
            "prompt": self.prompt,
            "choice": self.choice,
            "drift_from_base": self.drift_from_base,
            "adapter_state_path": self.adapter_state_path,
            "optimizer_state_path": self.optimizer_state_path,
        }
