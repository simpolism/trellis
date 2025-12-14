"""
Linear Undo Stack
=================

Linear undo history with disk persistence, replacing the previous tree structure.
Supports undo/redo navigation through training history.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from .checkpoint import Checkpoint


class LinearUndoStack:
    """
    Linear undo history with disk persistence.

    Unlike a tree structure, this is strictly linear:
    - push() adds a new checkpoint
    - undo() moves back one step
    - redo() moves forward if available
    - trim() enforces max_undos limit by removing oldest

    Checkpoints are persisted to disk so sessions can be resumed with
    full undo history intact.
    """

    def __init__(self, save_dir: Path, max_undos: Optional[int] = None):
        """
        Initialize the undo stack.

        Args:
            save_dir: Directory for persisting checkpoints
            max_undos: Maximum number of undo states to keep (None = unlimited)
        """
        self.save_dir = Path(save_dir)
        self.checkpoints_dir = self.save_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.max_undos = max_undos

        self.checkpoints: list[Checkpoint] = []
        self.current_index: int = -1  # Points to current position

        self._index_path = self.save_dir / "undo_stack.json"

    def push(self, checkpoint: Checkpoint) -> None:
        """
        Add a new checkpoint after the current position.

        If we're not at the end (after an undo), this truncates the "future".
        Persists the checkpoint to disk immediately.
        """
        # Truncate any "future" if we pushed after undo
        if self.current_index < len(self.checkpoints) - 1:
            # Delete orphaned checkpoint directories
            for orphan in self.checkpoints[self.current_index + 1:]:
                self._delete_checkpoint_dir(orphan)
            self.checkpoints = self.checkpoints[:self.current_index + 1]

        # Add new checkpoint
        self.checkpoints.append(checkpoint)
        self.current_index += 1

        # Persist to disk
        checkpoint_dir = self._get_checkpoint_dir(checkpoint)
        checkpoint.persist(checkpoint_dir)

        # Trim oldest if over limit
        if self.max_undos is not None and len(self.checkpoints) > self.max_undos + 1:
            self._trim_oldest()

        self._save_index()

    def undo(self) -> Optional[Checkpoint]:
        """
        Move back one step in history.

        Returns:
            The checkpoint to restore, or None if at beginning
        """
        if not self.can_undo():
            return None

        self.current_index -= 1
        self._save_index()
        return self.checkpoints[self.current_index]

    def redo(self) -> Optional[Checkpoint]:
        """
        Move forward one step if available.

        Returns:
            The checkpoint to restore, or None if at end
        """
        if not self.can_redo():
            return None

        self.current_index += 1
        self._save_index()
        return self.checkpoints[self.current_index]

    def current(self) -> Optional[Checkpoint]:
        """Get current checkpoint without moving."""
        if 0 <= self.current_index < len(self.checkpoints):
            return self.checkpoints[self.current_index]
        return None

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self.current_index > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self.current_index < len(self.checkpoints) - 1

    def get_history(self) -> list[Checkpoint]:
        """Get all checkpoints up to and including current position."""
        return self.checkpoints[:self.current_index + 1]

    def get_all(self) -> list[Checkpoint]:
        """Get all checkpoints including any 'future' after undo."""
        return self.checkpoints.copy()

    def _get_checkpoint_dir(self, checkpoint: Checkpoint) -> Path:
        """Get the directory path for a checkpoint."""
        dir_name = f"{checkpoint.step_count:05d}_{checkpoint.id}"
        return self.checkpoints_dir / dir_name

    def _delete_checkpoint_dir(self, checkpoint: Checkpoint) -> None:
        """Delete a checkpoint's directory from disk."""
        checkpoint_dir = self._get_checkpoint_dir(checkpoint)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

    def _trim_oldest(self) -> None:
        """Remove the oldest checkpoint (index 1, preserving root at 0)."""
        if len(self.checkpoints) > 1:
            # Don't delete root (index 0), delete the next oldest
            oldest = self.checkpoints[1]
            self._delete_checkpoint_dir(oldest)
            self.checkpoints.pop(1)
            self.current_index -= 1  # Adjust current index

    def _save_index(self) -> None:
        """Persist stack metadata to JSON."""
        data = {
            "current_index": self.current_index,
            "max_undos": self.max_undos,
            "checkpoints": [
                {
                    "id": cp.id,
                    "step_count": cp.step_count,
                    "dir": str(self._get_checkpoint_dir(cp).relative_to(self.save_dir)),
                }
                for cp in self.checkpoints
            ],
        }
        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, save_dir: Path) -> "LinearUndoStack":
        """
        Restore stack from disk for session resume.

        Args:
            save_dir: Directory containing undo_stack.json and checkpoints/

        Returns:
            Reconstructed LinearUndoStack with full history
        """
        save_dir = Path(save_dir)
        index_path = save_dir / "undo_stack.json"

        if not index_path.exists():
            raise FileNotFoundError(f"No undo stack index at {index_path}")

        with open(index_path) as f:
            data = json.load(f)

        stack = cls(save_dir=save_dir, max_undos=data.get("max_undos"))
        stack.current_index = data["current_index"]

        # Load checkpoints from directories
        for cp_info in data["checkpoints"]:
            checkpoint_dir = save_dir / cp_info["dir"]
            if checkpoint_dir.exists():
                checkpoint = Checkpoint.load_from_dir(checkpoint_dir)
                stack.checkpoints.append(checkpoint)
            else:
                raise FileNotFoundError(f"Checkpoint directory missing: {checkpoint_dir}")

        return stack

    def estimate_disk_usage(self) -> tuple[int, str]:
        """
        Estimate total disk usage of all checkpoints.

        Returns:
            (bytes, human_readable_string)
        """
        total_bytes = 0
        for checkpoint in self.checkpoints:
            checkpoint_dir = self._get_checkpoint_dir(checkpoint)
            if checkpoint_dir.exists():
                for file in checkpoint_dir.iterdir():
                    total_bytes += file.stat().st_size

        # Human readable
        if total_bytes < 1024:
            human = f"{total_bytes} B"
        elif total_bytes < 1024 ** 2:
            human = f"{total_bytes / 1024:.1f} KB"
        elif total_bytes < 1024 ** 3:
            human = f"{total_bytes / (1024 ** 2):.1f} MB"
        else:
            human = f"{total_bytes / (1024 ** 3):.2f} GB"

        return total_bytes, human
