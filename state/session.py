"""
Session State
=============

Manages session save/resume with all necessary state for continuation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import TrellisConfig


@dataclass
class SessionState:
    """
    Serializable session state for save/resume.

    A session directory contains:
        session.json      - This metadata
        config.json       - TrellisConfig
        undo_stack.json   - LinearUndoStack index
        checkpoints/      - Checkpoint directories
        journal/          - Journal logs
    """

    version: str = "2.0"
    created_at: str = ""
    last_modified: str = ""

    # User-friendly session name
    name: Optional[str] = None

    # Paths relative to session directory
    config_path: str = "config.json"
    undo_stack_path: str = "undo_stack.json"
    journal_path: str = "journal/session_log.md"

    # Prompt source state
    prompt_source_dataset_id: Optional[str] = None
    prompt_source_subset: Optional[str] = None
    prompt_source_split: str = "train"
    prompt_source_index: int = 0

    # Training state
    current_step: int = 0

    @classmethod
    def create_new(cls, session_dir: Path, config: TrellisConfig) -> "SessionState":
        """Create a new session state and save config."""
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = session_dir / "config.json"
        config.save(str(config_path))

        # Create journal directory
        journal_dir = session_dir / "journal"
        journal_dir.mkdir(exist_ok=True)

        now = datetime.now().isoformat()
        state = cls(
            created_at=now,
            last_modified=now,
        )

        state.save(session_dir / "session.json")
        return state

    def save(self, path: Path) -> None:
        """Save session state to JSON."""
        self.last_modified = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SessionState":
        """Load session state from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def update_prompt_source(
        self,
        dataset_id: Optional[str],
        subset: Optional[str],
        split: str,
        index: int,
    ) -> None:
        """Update prompt source state."""
        self.prompt_source_dataset_id = dataset_id
        self.prompt_source_subset = subset
        self.prompt_source_split = split
        self.prompt_source_index = index

    def update_step(self, step: int) -> None:
        """Update current training step."""
        self.current_step = step

    @staticmethod
    def discover_sessions(base_dir: str) -> list[tuple[str, "SessionState", str]]:
        """
        Find all valid sessions in a directory.

        Args:
            base_dir: Directory to scan for session subdirectories

        Returns:
            List of (session_name, SessionState, full_path) tuples, sorted by last_modified
        """
        base_path = Path(base_dir)
        sessions = []

        if not base_path.exists():
            return sessions

        for item in base_path.iterdir():
            if item.is_dir():
                session_file = item / "session.json"
                if session_file.exists():
                    try:
                        state = SessionState.load(session_file)
                        sessions.append((item.name, state, str(item)))
                    except Exception:
                        # Skip invalid sessions
                        continue

        # Sort by last_modified, most recent first
        sessions.sort(key=lambda x: x[1].last_modified, reverse=True)
        return sessions

    @staticmethod
    def generate_session_name() -> str:
        """Generate a unique session name based on timestamp."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def load_session(session_dir: str) -> tuple[SessionState, TrellisConfig]:
    """
    Load a complete session from disk.

    Args:
        session_dir: Path to session directory

    Returns:
        (SessionState, TrellisConfig) tuple

    Raises:
        FileNotFoundError: If session files are missing
    """
    session_path = Path(session_dir)
    session_file = session_path / "session.json"
    config_file = session_path / "config.json"

    if not session_file.exists():
        raise FileNotFoundError(f"Session file not found: {session_file}")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    state = SessionState.load(session_file)
    config = TrellisConfig.load(str(config_file))

    return state, config
